# PTB-XL native 500 Hz pipeline

Uses **only the original** `records500/` files (no `records100/` access, Butterworth filter, interpolation, or downsampling). Every ECG is **10 seconds × 500 Hz × 12 leads** = `float32` array with shape `(5000, 12)`, measured in **millivolts**. The resulting `pandas.DataFrame` has one row per ECG, `user_id` from `patient_id`, multi-label diagnostics, and `target` as the final column.

## Install

```bash
pip install numpy pandas requests pytest
# Optional, only if saving to Parquet:
pip install pyarrow
```

CUDA/CuPy is **not** required or used: without filtering, decoding is a small vectorized NumPy operation, and network transfer is the likely bottleneck. HTTP concurrency helps more than unnecessary CPU↔GPU copies. You can move the final tensor to your GPU for model training.

## Quick preview

```bash
python ptbxl_500hz_pipeline.py --limit 50 --workers 8 --out ptbxl_preview_500hz.pkl
```

## Full dataset

```bash
python ptbxl_500hz_pipeline.py --version 1.0.3 --workers 8 --batch-size 16 --out ptbxl_500hz.pkl
```

To use the original release of your assignment (instead of latest corrected 1.0.3):

```bash
python ptbxl_500hz_pipeline.py --version 1.0.0 --workers 8 --out ptbxl_v100_500hz.pkl
```

All raw `.hea` and `.dat` files are retrieved with `requests` and held in **RAM**. Each `.dat` is decoded by `np.frombuffer(..., dtype='<i2')` with no temporary waveform files. The only file written is the optional **final processed** DataFrame.

For machines with limited RAM, process independent shards using `--offset` and `--limit`, e.g., `--offset 0 --limit 2000 --out shard_00.pkl`, then `--offset 2000 --limit 2000 --out shard_01.pkl`. A full in-memory DataFrame still needs enough RAM to hold the entire corpus.

## DataFrame schema (compact by default)

| ecg_id | user_id | strat_fold | fs_hz | NORM | MI | STTC | CD | HYP | target_mi_vs_norm | ecg | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 42 | 1 | 500 | 1 | 0 | 0 | 0 | 0 | 0 | ndarray(5000,12) | NORM |
| 2 | 42 | 1 | 500 | 0 | 1 | 1 | 0 | 0 | 1 | ndarray(5000,12) | MI\|STTC |

These are **synthetic example records**, not real PTB-XL rows. `target` can include combinations such as `MI|STTC`, `CD|HYP`, and `UNLABELED`. The five binary columns give independent *diagnostic-superclass* flags. `target_mi_vs_norm` is an **optional** derived target: `0` for pure NORM; `1` if MI is among the labels and NORM is absent; otherwise `pd.NA`. MI+STTC is therefore `1`, but no diagnostic combination is silently discarded from the main dataset.

For **strictly exclusive** NORM-vs-MI, instead select rows with `target` exactly `NORM` or `MI` (you will drop MI+STTC, etc.).

## Python usage

```python
import numpy as np
import pandas as pd
from ptbxl_500hz_pipeline import build_dataframe, CLASSES, to_wide_dataframe

df = build_dataframe(limit=100)  # or pd.read_pickle('ptbxl_500hz.pkl')
X = np.stack(df['ecg'].to_numpy())   # (N, 5000, 12), float32, mV
Y = df[list(CLASSES)].to_numpy(dtype=np.int8)  # (N, 5) multi-hot
users = df['user_id'].to_numpy()

train = df['strat_fold'].between(1, 8)
val = df['strat_fold'].eq(9)
test = df['strat_fold'].eq(10)

print(df['target'].value_counts())
print(df[list(CLASSES)].sum())

# If torch is installed and a CUDA device is available:
# import torch
# X_gpu = torch.from_numpy(X[:32]).to('cuda')  # transfer batches only

# Optional extremely memory-intensive expanded table (60,000 signal columns):
# df_wide = to_wide_dataframe(df.head(10))
```

The official `strat_fold` respects patients (all recordings from one patient are kept in the same fold), so it prevents patient leakage when used as recommended (1–8 train, 9 validation, 10 test). 

## Important memory note

PTB-XL v1.0.3 contains **21,799 ECGs**. At 5000×12 float32 values each, the *raw numeric portion of the processed in-memory dataset* is about **5.23 GB (4.87 GiB)**, plus DataFrame overhead and temporary buffers. `np.stack(df['ecg'])` allocates **another 5.23 GB**; keep the compact object-array DataFrame during ingestion, use minibatches during training, and avoid the 60,000-column wide form unless RAM permits. Downloaded 16-bit `.dat` values have about half the float32 footprint before decoding. The entire corpus is not continuous 24/7 signal—it contains separate 10-second ECGs.

`--layout wide` produces 60,000 numeric columns. It is only recommended on small subsets or machines with ample RAM. In compact `.parquet`, each `ecg` cell becomes a flattened time-major vector of 60,000 values; restore with:

```python
df = pd.read_parquet('ptbxl_500hz.parquet')
df['ecg'] = [np.asarray(x, dtype=np.float32).reshape(5000, 12) for x in df['ecg']]
```

## Testing

```bash
pytest -q test_ptbxl_500hz_pipeline.py
```

Offline synthetic tests verify WFDB int16 byte decoding, mV calibration, 5000×12 output, multi-label class mapping, patient IDs, labels, saved DataFrames, and *absence of any `records100/` requests*. The full remote PhysioNet run has not been executed in this environment.

Dataset documentation: https://physionet.org/content/ptb-xl/1.0.3/
