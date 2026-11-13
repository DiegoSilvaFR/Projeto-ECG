import argparse
import ast
import io
import posixpath
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FS_HZ = 500
SAMPLES = 5000
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BASE_URL_TEMPLATE = "https://physionet.org/files/ptb-xl/{version}/"
_tls = threading.local()


def _session() -> requests.Session:
    """One keep-alive HTTP session per thread (no disk cache)."""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        retries = Retry(
            total=5, connect=5, read=4, backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET",)),
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2))
        _tls.session = s
    return _tls.session


def fetch_bytes(base_url: str, relative_path: str) -> bytes:
    """Retrieve raw bytes in RAM, without temporary downloads on disk."""
    if relative_path.startswith("/") or "\\" in relative_path or ".." in relative_path.split("/"):
        raise ValueError(f"Unsafe path: {relative_path!r}")
    response = _session().get(base_url + relative_path, timeout=(15, 120))
    response.raise_for_status()
    return response.content


@dataclass(frozen=True)
class WFDBHeader:
    sample_rate: int
    samples: int
    channels: int
    dat_filename: str
    gain: np.ndarray
    baseline: np.ndarray
    lead_indices: np.ndarray


def parse_wfdb_header(data: bytes) -> WFDBHeader:
    """Parse the PTB-XL single-file, unskewed WFDB format-16 header."""
    lines = [line.strip() for line in data.decode("ascii").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("Empty WFDB header")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("Incomplete WFDB record header")
    channels = int(first[1])
    rate = float(first[2].split("/")[0])
    samples = int(first[3])
    if not rate.is_integer() or channels != len(LEADS) or len(lines) < channels + 1:
        raise ValueError("Unsupported sampling rate, lead count, or header length")

    filenames, gains, baselines, names = [], [], [], []
    gain_pattern = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:\((-?\d+)\))?/([^\s]+)"
    )
    for line in lines[1:channels + 1]:
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"Incomplete signal specification: {line!r}")
        filename, fmt, gain_text = parts[:3]
        if fmt != "16":
            raise NotImplementedError(f"Only WFDB format 16 is supported, got {fmt!r}")
        gain_match = gain_pattern.fullmatch(gain_text)
        if not gain_match:
            raise ValueError(f"Unsupported gain field: {gain_text!r}")
        gain, baseline_in_gain, units = gain_match.groups()
        if units.lower() != "mv":
            raise NotImplementedError(f"Unsupported amplitude units: {units!r}")
        filenames.append(filename)
        gains.append(float(gain))
        baselines.append(int(baseline_in_gain) if baseline_in_gain is not None else int(parts[4]))
        names.append(parts[8].upper())

    if len(set(filenames)) != 1:
        raise NotImplementedError("PTB-XL reader expects one interleaved .dat file")
    if any(not np.isfinite(g) or g <= 0 for g in gains):
        raise ValueError("ADC gains must be positive and finite")
    lead_names = [lead.upper() for lead in LEADS]
    if len(set(names)) != channels or set(names) != set(lead_names):
        raise ValueError(f"Unexpected/missing lead names: {names}")
    indices = np.array([names.index(lead) for lead in lead_names], dtype=np.intp)
    return WFDBHeader(
        int(rate), samples, channels, filenames[0],
        np.asarray(gains, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32),
        indices,
    )


def decode_wfdb_dat(dat: bytes, header: WFDBHeader) -> np.ndarray:
    """Calibrate interleaved little-endian int16 samples to mV, (5000,12)."""
    raw = np.frombuffer(dat, dtype="<i2")  # zero-copy view over response bytes
    expected = header.samples * header.channels
    if raw.size != expected:
        raise ValueError(f"Corrupt .dat: expected {expected} 16-bit values, got {raw.size}")
    ecg = raw.reshape(header.samples, header.channels).astype(np.float32)
    # WFDB physical amplitude: (digital_value - adc_baseline) / adc_gain.
    ecg -= header.baseline[None, :]
    ecg /= header.gain[None, :]
    # Reorder to the documented lead order, regardless of header ordering.
    return np.ascontiguousarray(ecg[:, header.lead_indices])


def download_ecg(base_url: str, record_path: str) -> np.ndarray:
    """Use exactly the high-rate record path supplied by the metadata."""
    if not record_path.startswith("records500/") or not record_path.endswith("_hr"):
        raise ValueError(f"Not a PTB-XL 500 Hz record: {record_path!r}")
    header = parse_wfdb_header(fetch_bytes(base_url, record_path + ".hea"))
    if header.sample_rate != FS_HZ or header.samples != SAMPLES:
        raise ValueError(f"Unexpected signal dimensions at {record_path}: "
                         f"{header.sample_rate} Hz, {header.samples} samples")
    dat_path = posixpath.join(posixpath.dirname(record_path), header.dat_filename)
    return decode_wfdb_dat(fetch_bytes(base_url, dat_path), header)


def load_metadata(base_url: str) -> pd.DataFrame:
    """Read CSV metadata and map SCP diagnostic codes to five superclasses."""
    db = pd.read_csv(io.BytesIO(fetch_bytes(base_url, "ptbxl_database.csv")))
    statements = pd.read_csv(io.BytesIO(fetch_bytes(base_url, "scp_statements.csv")), index_col=0)
    required = {"ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"}
    if not required.issubset(db.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(db.columns))}")
    required_scp = {"diagnostic", "diagnostic_class"}
    if not required_scp.issubset(statements.columns):
        raise ValueError(f"Missing SCP columns: {sorted(required_scp - set(statements.columns))}")
    diagnostic = statements[pd.to_numeric(statements["diagnostic"], errors="coerce").eq(1)]
    code_to_class = diagnostic["diagnostic_class"].dropna().to_dict()

    def get_labels(value: str) -> tuple[str, ...]:
        codes = ast.literal_eval(value)
        if not isinstance(codes, dict):
            raise ValueError("scp_codes must be a dictionary of SCP code likelihoods")
        # An SCP likelihood of zero means 'unknown', not that the code is absent.
        classes = {code_to_class[code] for code in codes if code in code_to_class}
        return tuple(label for label in CLASSES if label in classes)

    if db["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ecg_id in metadata")
    db["diagnostic_labels"] = db["scp_codes"].map(get_labels)
    return db


def _build_row(item: dict, ecg: np.ndarray) -> dict:
    labels = tuple(item["diagnostic_labels"])
    row = {
        "ecg_id": int(item["ecg_id"]),
        "user_id": int(item["patient_id"]),
        "strat_fold": int(item["strat_fold"]),
        "fs_hz": FS_HZ,
    }
    row.update({cls: int(cls in labels) for cls in CLASSES})
    # Experimental convenience: MI (possibly alongside STTC/CD/HYP) vs *pure* NORM.
    # Records with any other diagnosis, ambiguous coexistence, or no diagnosis
    # are excluded from this optional binary target (NA).
    if labels == ("NORM",):
        row["target_mi_vs_norm"] = 0
    elif "MI" in labels and "NORM" not in labels:
        row["target_mi_vs_norm"] = 1
    else:
        row["target_mi_vs_norm"] = None
    row["ecg"] = ecg
    row["target"] = "|".join(labels) if labels else "UNLABELED"
    return row


def build_dataframe(
    *, version: str = "1.0.3", workers: int = 8, batch_size: int = 16,
    limit: int | None = None, offset: int = 0, verbose: bool = True,
) -> pd.DataFrame:
    """Load native 500 Hz PTB-XL files to RAM, one row per ECG (no downsampling).

    The complete 21,799-record array payload needs ~5.23 GB of RAM. The
    additional peak usage depends on pandas and other data kept in memory.
    Optional offset+limit allow smaller independent extraction jobs.
    """
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must have the form X.Y.Z")
    if workers < 1 or batch_size < 1 or offset < 0 or (limit is not None and limit < 1):
        raise ValueError("workers/batch_size/limit must be positive and offset >= 0")
    base_url = BASE_URL_TEMPLATE.format(version=version)
    metadata = load_metadata(base_url).sort_values("ecg_id")
    if limit is None:
        metadata = metadata.iloc[offset:]
    else:
        metadata = metadata.iloc[offset:offset + limit]
    items = metadata.to_dict("records")
    if not items:
        raise ValueError("No ECG records in selected offset/limit interval")
    rows = []
    if verbose:
        print(f"Reading {len(items):,} PTB-XL records at native 500 Hz; no resampling")
    # Bound the futures and input buffers to one small batch at a time.
    # Threaded HTTP I/O usually helps more than GPU for this workload.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            signals = executor.map(lambda item: download_ecg(base_url, str(item["filename_hr"])), batch)
            for item, ecg in zip(batch, signals):
                if ecg.shape != (SAMPLES, len(LEADS)) or ecg.dtype != np.float32:
                    raise ValueError(f"Invalid ECG decoded for ecg_id={item['ecg_id']}")
                rows.append(_build_row(item, ecg))
            if verbose:
                print(f"Processed {min(start + batch_size, len(items)):,}/{len(items):,}",
                      end="\r", flush=True)
    if verbose:
        print()
    df = pd.DataFrame.from_records(rows)
    # Ensure a nullable integer binary target and target always last.
    df["target_mi_vs_norm"] = pd.array(df["target_mi_vs_norm"], dtype="Int8")
    return df[[c for c in df.columns if c != "target"] + ["target"]]


def to_wide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optional 60,000-column matrix; requires substantial extra RAM for full data."""
    if len(df) == 0:
        raise ValueError("Cannot expand an empty DataFrame")
    signals = np.stack(df["ecg"].to_numpy())  # (N, 5000, 12)
    lead_major = np.ascontiguousarray(signals.transpose(0, 2, 1))
    features = lead_major.reshape(len(df), -1)
    names = [f"{lead}_t{t:04d}" for lead in LEADS for t in range(SAMPLES)]
    wide = pd.DataFrame(features, index=df.index, columns=names)
    metadata = df.drop(columns=["ecg", "target"])
    result = pd.concat([metadata, wide], axis=1, copy=False)
    result["target"] = df["target"]
    return result


def save_dataframe(df: pd.DataFrame, out_file: str | Path) -> None:
    """Save the processed DataFrame only, never raw waveform downloads."""
    path = Path(out_file)
    if path.suffix.lower() == ".pkl":
        df.to_pickle(path)
    elif path.suffix.lower() == ".parquet":
        # Parquet's list type does not store a 2D ndarray cell directly.
        # Flatten each ECG in time-major order before serializing.
        if "ecg" in df:
            df = df.copy()
            df["ecg"] = [arr.ravel() for arr in df["ecg"]]
            print("Parquet ECG column is a flat time-major list of 60,000 values; "
                  "reshape each row to (5000, 12) after loading.")
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output extension must be .pkl or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="1.0.3", help="PTB-XL version (default: 1.0.3)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel HTTP download workers")
    parser.add_argument("--batch-size", type=int, default=16, help="Bounded batch size")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N sorted ECGs")
    parser.add_argument("--limit", type=int, help="Process at most N ECGs")
    parser.add_argument("--layout", choices=("compact", "wide"), default="compact")
    parser.add_argument("--out", default="ptbxl_500hz.parquet", help="Saved DataFrame, .pkl or .parquet")
    args = parser.parse_args()

    df = build_dataframe(version=args.version, workers=args.workers, batch_size=args.batch_size,
                         offset=args.offset, limit=args.limit)
    if args.layout == "wide":
        print("Warning: wide layout creates 60,000 float columns and may need > 12 GB RAM")
        df = to_wide_dataframe(df)
    save_dataframe(df, args.out)
    print(f"Saved {len(df):,} ECGs with {len(df.columns):,} columns to {Path(args.out).resolve()}")
    print("Patient count:", df["user_id"].nunique())
    print("Multi-label class counts:\n" + df[list(CLASSES)].sum().to_string())
    print("Example labels:\n" + df[["ecg_id", "user_id", "strat_fold", "target"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
