import argparse
import ast
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Union

import numpy as np
import pandas as pd

FS_HZ = 500
SAMPLES = 5000
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class WFDBHeader:
    sample_rate: int
    samples: int
    channels: int
    dat_filename: str
    gain: np.ndarray
    baseline: np.ndarray
    lead_indices: np.ndarray


def _looks_like_dataset_root(folder: Path) -> bool:
    return (folder / "ptbxl_database.csv").is_file() and (
        folder / "scp_statements.csv"
    ).is_file() and (folder / "records500").is_dir()


def find_dataset_root(folder: str | Path = ".") -> Path:
    """Find PTB-XL under repo root, extracted dataset root, or records500/.

    This intentionally never downloads or modifies source files.
    """
    src = Path(folder).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {src}")
    if _looks_like_dataset_root(src):
        return src
    if src.name == "records500" and _looks_like_dataset_root(src.parent):
        return src.parent

    # Common unpacked layout in the supplied Projeto-ECG GitHub repository.
    common = src / "ptb-xl-1.0.3" / (
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
    if _looks_like_dataset_root(common):
        return common.resolve()

    # Generic fallback for other nested extraction names.
    roots = sorted({p.parent.resolve() for p in src.rglob("ptbxl_database.csv")
                    if _looks_like_dataset_root(p.parent)})
    if not roots:
        raise FileNotFoundError(
            "Could not find a PTB-XL dataset containing ptbxl_database.csv, "
            "scp_statements.csv and records500/. Point --data-dir at the "
            "repository root or the extracted PTB-XL dataset directory."
        )
    if len(roots) > 1:
        raise ValueError(
            "Multiple PTB-XL datasets found. Select one via --data-dir: "
            + ", ".join(map(str, roots))
        )
    return roots[0]


def parse_wfdb_header(data: bytes) -> WFDBHeader:
    """Parse the PTB-XL single-file WFDB format-16 header (without WFDB lib)."""
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
        baselines.append(int(baseline_in_gain) if baseline_in_gain is not None
                         else int(parts[4]))
        names.append(parts[8].upper())

    if len(set(filenames)) != 1:
        raise NotImplementedError("PTB-XL reader expects one interleaved .dat file")
    if any(not np.isfinite(g) or g <= 0 for g in gains):
        raise ValueError("ADC gains must be positive and finite")
    expected_leads = [lead.upper() for lead in LEADS]
    if len(set(names)) != channels or set(names) != set(expected_leads):
        raise ValueError(f"Unexpected/missing lead names: {names}")
    indices = np.array([names.index(lead) for lead in expected_leads], dtype=np.intp)
    return WFDBHeader(
        int(rate), samples, channels, filenames[0],
        np.asarray(gains, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32), indices,
    )


def decode_wfdb_dat(dat: Union[bytes, np.ndarray], header: WFDBHeader) -> np.ndarray:
    """Decode interleaved little-endian int16 into (5000, 12) float32 mV."""
    raw = np.frombuffer(dat, dtype="<i2") if isinstance(dat, bytes) else np.asarray(dat)
    if raw.dtype != np.dtype("<i2"):
        raise TypeError(f"Expected little-endian int16 waveform, got {raw.dtype}")
    expected = header.samples * header.channels
    if raw.size != expected:
        raise ValueError(f"Corrupt .dat: expected {expected} 16-bit values, got {raw.size}")
    # Copy into owned float32 storage: mmap/bytes buffer need not remain open.
    ecg = raw.reshape(header.samples, header.channels).astype(np.float32)
    ecg -= header.baseline[None, :]
    ecg /= header.gain[None, :]
    return np.ascontiguousarray(ecg[:, header.lead_indices])


def _safe_record_file(root: Path, relative: str) -> Path:
    """Enforce metadata's original records500 paths, including on Windows."""
    name = relative.replace("\\", "/")
    rel = PurePosixPath(name)
    if (rel.is_absolute() or not rel.parts or rel.parts[0] != "records500"
            or ".." in rel.parts or not rel.name.endswith("_hr")):
        raise ValueError(f"Not a safe PTB-XL 500 Hz record path: {relative!r}")
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to((root / "records500").resolve()):
        raise ValueError(f"Record resolves outside records500: {relative!r}")
    return path


def read_local_ecg(root: str | Path, record_path: str,
                   *, read_mode: str = "memmap") -> np.ndarray:
    """Read native high-rate ECG from existing local WFDB .hea and .dat files.

    memmap: zero-copy OS-backed raw int16 view (default).
    frombuffer: Path.read_bytes + np.frombuffer raw int16 view.
    Both calibrate to float32 and leave original files untouched.
    """
    root = Path(root).resolve()
    base = _safe_record_file(root, record_path)
    header_path = base.with_suffix(".hea")
    if not header_path.is_file():
        raise FileNotFoundError(f"Missing ECG header: {header_path}")
    header = parse_wfdb_header(header_path.read_bytes())
    if header.sample_rate != FS_HZ or header.samples != SAMPLES:
        raise ValueError(f"Unexpected signal dimensions at {base}: "
                         f"{header.sample_rate} Hz, {header.samples} samples")
    if Path(header.dat_filename).name != header.dat_filename:
        raise ValueError(f"Unsafe .dat filename in header: {header.dat_filename!r}")
    dat_path = base.parent / header.dat_filename
    if not dat_path.is_file():
        raise FileNotFoundError(f"Missing ECG waveform: {dat_path}")
    expected_size = header.samples * header.channels * np.dtype("<i2").itemsize
    size = dat_path.stat().st_size
    if size != expected_size:
        raise ValueError(f"Invalid .dat size ({size} bytes; expected {expected_size}): "
                         f"{dat_path}. Ensure the actual data are present, "
                         "not a Git LFS pointer or an incomplete file.")

    if read_mode == "frombuffer":
        return decode_wfdb_dat(dat_path.read_bytes(), header)
    if read_mode != "memmap":
        raise ValueError("read_mode must be 'memmap' or 'frombuffer'")
    raw = np.memmap(dat_path, dtype="<i2", mode="r")
    try:
        return decode_wfdb_dat(raw, header)
    finally:
        # Fully detach from OS-backed mapping. The returned ECG is an owned array.
        raw._mmap.close()


def load_metadata(root: str | Path) -> pd.DataFrame:
    """Read local CSVs and map SCP diagnostic codes to five superclasses."""
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    required = {"ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"}
    if not required.issubset(db.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(db.columns))}")
    required_scp = {"diagnostic", "diagnostic_class"}
    if not required_scp.issubset(statements.columns):
        raise ValueError(f"Missing SCP columns: {sorted(required_scp - set(statements.columns))}")
    if db["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ecg_id in metadata")
    diagnostic = statements[pd.to_numeric(statements["diagnostic"], errors="coerce").eq(1)]
    code_to_class = diagnostic["diagnostic_class"].dropna().to_dict()

    def labels(value: str) -> tuple[str, ...]:
        codes = ast.literal_eval(value)
        if not isinstance(codes, dict):
            raise ValueError("scp_codes must be a dictionary of SCP code likelihoods")
        classes = {code_to_class[code] for code in codes if code in code_to_class}
        return tuple(label for label in CLASSES if label in classes)

    db["diagnostic_labels"] = db["scp_codes"].map(labels)
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
    *, data_dir: str | Path = ".", workers: int = 4, batch_size: int = 32,
    limit: int | None = None, offset: int = 0, read_mode: str = "memmap",
    verbose: bool = True,
) -> pd.DataFrame:
    """Return complete/local-subset DataFrame, without changing source files.

    The complete waveform payload occupies roughly 5.23 GB float32, excluding
    object overhead. Use offset/limit to process smaller independent shards.
    """
    if workers < 1 or batch_size < 1 or offset < 0 or (limit is not None and limit < 1):
        raise ValueError("workers/batch_size/limit must be positive; offset >= 0")
    if read_mode not in {"memmap", "frombuffer"}:
        raise ValueError("read_mode must be 'memmap' or 'frombuffer'")
    root = find_dataset_root(data_dir)
    db = load_metadata(root).sort_values("ecg_id")
    db = db.iloc[offset:(offset + limit if limit is not None else None)]
    items = db.to_dict("records")
    if not items:
        raise ValueError("No ECG records in selected offset/limit interval")
    if verbose:
        print(f"PTB-XL local root: {root}")
        print(f"Reading {len(items):,} 500 Hz records (no downloading/resampling)")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            # executor.map preserves the metadata order; peak pending jobs are
            # bounded by batch_size rather than the full ~22k-record dataset.
            signals = executor.map(
                lambda item: read_local_ecg(root, str(item["filename_hr"]),
                                            read_mode=read_mode), batch
            )
            for item, ecg in zip(batch, signals):
                if ecg.shape != (SAMPLES, len(LEADS)) or ecg.dtype != np.float32:
                    raise ValueError(f"Invalid ECG for ecg_id={item['ecg_id']}")
                rows.append(_build_row(item, ecg))
            if verbose:
                print(f"Processed {min(start + batch_size, len(items)):,}/{len(items):,}",
                      end="\r", flush=True)
    if verbose:
        print()
    df = pd.DataFrame.from_records(rows)
    df["target_mi_vs_norm"] = pd.array(df["target_mi_vs_norm"], dtype="Int8")
    return df[[c for c in df.columns if c != "target"] + ["target"]]


def to_wide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optional 60,000 float columns; needs ample memory on a full dataset."""
    if len(df) == 0:
        raise ValueError("Cannot expand an empty DataFrame")
    signals = np.stack(df["ecg"].to_numpy())
    features = np.ascontiguousarray(signals.transpose(0, 2, 1)).reshape(len(df), -1)
    names = [f"{lead}_t{t:04d}" for lead in LEADS for t in range(SAMPLES)]
    wide = pd.DataFrame(features, index=df.index, columns=names)
    result = pd.concat([df.drop(columns=["ecg", "target"]), wide], axis=1, copy=False)
    result["target"] = df["target"]
    return result


def save_dataframe(df: pd.DataFrame, out_file: str | Path) -> None:
    """Save the final DataFrame (never writes raw ECG files)."""
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pkl":
        df.to_pickle(path)
    elif path.suffix.lower() == ".parquet":
        if "ecg" in df:
            df = df.copy()
            df["ecg"] = [signal.ravel() for signal in df["ecg"]]
            print("Parquet stores each ECG as a flat 60,000-element vector. "
                  "Reshape to (5000, 12) after reading.")
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output extension must be .pkl or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="ptb-xl-1.0.3",
                        help="Repository root, PTB-XL extracted root, or records500/ (default: .)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel local-file readers")
    parser.add_argument("--batch-size", type=int, default=64, help="Bound pending reads")
    parser.add_argument("--read-mode", choices=("memmap", "frombuffer"), default="memmap",
                        help="Read local int16 using NumPy memmap or frombuffer")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N ECGs (sorted ecg_id)")
    parser.add_argument("--limit", type=int, help="Process at most N ECGs")
    parser.add_argument("--layout", choices=("compact", "wide"), default="compact")
    parser.add_argument("--out", default="ptbxl_500hz.parquet", help="Output .pkl or .parquet")
    args = parser.parse_args()

    df = build_dataframe(data_dir=args.data_dir, workers=args.workers,
                         batch_size=args.batch_size, read_mode=args.read_mode,
                         offset=args.offset, limit=args.limit)
    if args.layout == "wide":
        print("Warning: 60,000 columns; processing all ECGs may require >12 GB RAM")
        df = to_wide_dataframe(df)
    save_dataframe(df, args.out)
    print(f"Saved {len(df):,} ECGs × {len(df.columns):,} columns to {Path(args.out).resolve()}")
    print("Unique patients:", df["user_id"].nunique())
    print("Diagnostic superclass counts:\n" + df[list(CLASSES)].sum().to_string())
    print("Example targets:\n" + df[["ecg_id", "user_id", "strat_fold", "target"]]
          .head().to_string(index=False))


if __name__ == "__main__":
    main()
