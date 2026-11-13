"""Unit/integration tests with synthetic 500 Hz WFDB records, fully in RAM."""
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ptbxl_500hz_pipeline as p


def fixture_bytes():
    store = {}
    ids = [1, 2, 3, 4]
    patients = [42, 42, 77, 100]
    codes = ["{'NORM': 0}", "{'MI': 50, 'STTC': 70}", "{'CD': 0, 'HYP': 0}", "{'RHYTHM': 20}"]
    metadata = pd.DataFrame({
        "ecg_id": ids, "patient_id": patients, "strat_fold": [1, 1, 9, 10],
        "filename_hr": [f"records500/00000/{v:05d}_hr" for v in ids],
        "filename_lr": [f"records100/00000/{v:05d}_lr" for v in ids],
        "scp_codes": codes,
    })
    store["ptbxl_database.csv"] = metadata.to_csv(index=False).encode()
    scp = pd.DataFrame({
        "code": ["NORM", "MI", "STTC", "CD", "HYP", "RHYTHM"],
        "diagnostic": [1, 1, 1, 1, 1, 0],
        "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", ""],
    }).set_index("code")
    store["scp_statements.csv"] = scp.to_csv().encode()
    for index in ids:
        base = f"records500/00000/{index:05d}_hr"
        header = [f"{index:05d}_hr 12 500 5000"]
        for lead_i, lead in enumerate(p.LEADS):
            header.append(f"{index:05d}_hr.dat 16 1000.0({lead_i})/mV 16 0 0 0 0 {lead}")
        store[base + ".hea"] = ("\n".join(header) + "\n").encode()
        samples = ((np.arange(5000 * 12).reshape(5000, 12) % 1000) - 300).astype("<i2")
        store[base + ".dat"] = samples.tobytes()
    return store


def test_raw_decoder_calibrates_and_preserves_all_5000_samples():
    store = fixture_bytes()
    header = p.parse_wfdb_header(store["records500/00000/00001_hr.hea"])
    ecg = p.decode_wfdb_dat(store["records500/00000/00001_hr.dat"], header)
    assert header.sample_rate == 500
    assert ecg.shape == (5000, 12)
    assert ecg.dtype == np.float32
    np.testing.assert_allclose(ecg[0], (np.arange(12) - 300 - np.arange(12)) / 1000,
                               atol=1e-7)
    np.testing.assert_allclose(ecg[100, 0], ((100 * 12) % 1000 - 300) / 1000)


def test_in_memory_pipeline_never_fetches_records100():
    store = fixture_bytes()
    requests_seen = []

    def fake_fetch(base, path):
        requests_seen.append(path)
        return store[path]

    with patch.object(p, "fetch_bytes", side_effect=fake_fetch):
        df = p.build_dataframe(version="1.0.3", workers=2, batch_size=2, verbose=False)

    assert len(df) == 4
    assert df.columns[-1] == "target"
    assert len(set(df.ecg_id)) == 4
    assert df.user_id.tolist() == [42, 42, 77, 100]
    assert df.strat_fold.tolist() == [1, 1, 9, 10]
    assert df.fs_hz.eq(500).all()
    assert df.target.tolist() == ["NORM", "MI|STTC", "CD|HYP", "UNLABELED"]
    assert df.target_mi_vs_norm.tolist()[:2] == [0, 1]
    assert df.target_mi_vs_norm.isna().sum() == 2
    assert [ecg.shape for ecg in df.ecg] == [(5000, 12)] * 4
    assert all((not path.startswith("records100/")) for path in requests_seen)
    assert requests_seen.count("ptbxl_database.csv") == 1
    assert requests_seen.count("scp_statements.csv") == 1


def test_wide_and_pickle_roundtrip(tmp_path):
    store = fixture_bytes()
    with patch.object(p, "fetch_bytes", side_effect=lambda _, path: store[path]):
        df = p.build_dataframe(limit=2, workers=2, verbose=False)
    wide = p.to_wide_dataframe(df)
    assert wide.shape[1] == 60000 + df.shape[1] - 1
    assert wide.columns[-1] == "target"
    assert wide.loc[0, "I_t0100"] == pytest.approx(df.ecg.iloc[0][100, 0])
    output = tmp_path / "test.pkl"
    p.save_dataframe(df, output)
    loaded = pd.read_pickle(output)
    np.testing.assert_array_equal(loaded.ecg.iloc[0], df.ecg.iloc[0])
    assert loaded.target.tolist() == ["NORM", "MI|STTC"]


def test_rejects_100hz_record():
    store = fixture_bytes()
    fake_header = store["records500/00000/00001_hr.hea"].replace(b"12 500 5000", b"12 100 1000")
    with patch.object(p, "fetch_bytes", side_effect=lambda base, path: fake_header):
        with pytest.raises(ValueError, match="Unexpected signal dimensions"):
            p.download_ecg("https://example.com/", "records500/00000/00001_hr")
