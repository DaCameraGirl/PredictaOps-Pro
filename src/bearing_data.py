"""Load and feature-extract the NASA/IMS real bearing run-to-failure vibration data (Test 2).

Each raw file is a 1-second, 20kHz vibration snapshot (20480 rows) with one column
per bearing (4 bearings, 1 accelerometer each), taken every 10 minutes for ~7 days
until bearing 1 failed (outer race defect). We don't feed raw waveforms to the model;
we reduce each snapshot to standard vibration-analysis statistics per bearing.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "ims_test2"
BEARING_COLS = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
FEATURE_NAMES = ["mean", "std", "rms", "kurtosis", "skew", "peak_to_peak", "crest_factor"]

FAILED_BEARING = "bearing_1"
FAILURE_MODE = "outer race defect"


def _parse_timestamp(filename: str) -> datetime:
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def _snapshot_features(signal: np.ndarray) -> dict:
    mean = signal.mean()
    std = signal.std()
    rms = np.sqrt(np.mean(signal**2))
    centered = signal - mean
    kurtosis = np.mean(centered**4) / (std**4) if std > 0 else 0.0
    skew = np.mean(centered**3) / (std**3) if std > 0 else 0.0
    peak_to_peak = signal.max() - signal.min()
    peak = max(abs(signal.max()), abs(signal.min()))
    crest_factor = peak / rms if rms > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "kurtosis": kurtosis,
        "skew": skew,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
    }


def build_feature_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per (bearing, timestamp) with vibration-analysis features, sorted by time."""
    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.name != "README.md" and p.name != "readme.txt")
    rows = []
    for path in files:
        timestamp = _parse_timestamp(path.name)
        snapshot = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
        for i, bearing in enumerate(BEARING_COLS):
            feats = _snapshot_features(snapshot[:, i])
            rows.append({"bearing": bearing, "timestamp": timestamp, **feats})
    df = pd.DataFrame(rows).sort_values(["bearing", "timestamp"]).reset_index(drop=True)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL in remaining snapshots (x10 minutes) until bearing 1's real recorded failure.

    Only bearing 1 has a known failure time; the other three were still healthy when
    the experiment ended, so their true RUL is unknown (right-censored) and they're
    dropped from the labeled set rather than given a fabricated target.
    """
    failed = df[df["bearing"] == FAILED_BEARING].sort_values("timestamp").reset_index(drop=True)
    failed["RUL"] = len(failed) - 1 - failed.index
    return failed
