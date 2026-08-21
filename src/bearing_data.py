"""Load, validate, and feature-extract the NASA/IMS real bearing run-to-failure vibration data (Test 2).

Each raw file is a 1-second, 20kHz vibration snapshot (20480 rows) with one column
per bearing (4 bearings, 1 accelerometer each), taken every 10 minutes for ~7 days
until bearing 1 failed (outer race defect). We don't feed raw waveforms to the model;
we reduce each snapshot to standard vibration-analysis statistics per bearing.

Dataset source: NASA Prognostics Center of Excellence Data Set Repository, "Bearing
Data Set" (IMS, University of Cincinnati, sponsored by Rexnord Corp.), Test 2.
Original repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Mirror used by scripts/download_data.py: https://github.com/RicardoPSLopes/IMS-DATASET
"""
import hashlib
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ims_test2"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CACHE = PROCESSED_DIR / "ims_test2_features.csv"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

BEARING_COLS = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
FEATURE_NAMES = ["mean", "std", "rms", "kurtosis", "skew", "peak_to_peak", "crest_factor"]

FAILED_BEARING = "bearing_1"
FAILURE_MODE = "outer race defect"

EXPECTED_N_SNAPSHOTS = 984
EXPECTED_N_CHANNELS = 4
EXPECTED_SAMPLES_PER_SNAPSHOT = 20480
EXPECTED_SAMPLING_RATE_HZ = 20000
EXPECTED_INTERVAL_MINUTES = 10
IGNORED_FILENAMES = {"README.md", "readme.txt"}


class DatasetValidationError(Exception):
    """Raised when the raw dataset doesn't match the documented IMS Test 2 shape."""


def _parse_timestamp(filename: str) -> datetime:
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def _raw_files(raw_dir: Path) -> list[Path]:
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.name not in IGNORED_FILENAMES]
    # Every filename in this dataset *is* its own real recording timestamp (not an
    # arbitrary label), so parsing it is the authoritative ordering, not a filename
    # guess — there is no separate timestamp column in the files themselves. We still
    # verify parse success explicitly rather than assuming it, and check for gaps.
    parsed = []
    for path in files:
        try:
            ts = _parse_timestamp(path.name)
        except ValueError as exc:
            raise DatasetValidationError(f"unparseable snapshot filename: {path.name}") from exc
        parsed.append((ts, path))
    parsed.sort(key=lambda item: item[0])
    return [path for _, path in parsed]


def validate_raw_dataset(raw_dir: Path = RAW_DIR) -> dict:
    """Structural validation against the documented IMS Test 2 shape. Raises
    DatasetValidationError with a specific reason on any mismatch."""
    if not raw_dir.exists():
        raise DatasetValidationError(
            f"raw data directory not found: {raw_dir}. Run scripts/download_data.py first."
        )

    files = _raw_files(raw_dir)
    if len(files) != EXPECTED_N_SNAPSHOTS:
        raise DatasetValidationError(
            f"expected {EXPECTED_N_SNAPSHOTS} snapshots, found {len(files)}"
        )

    timestamps = [_parse_timestamp(p.name) for p in files]
    if len(set(timestamps)) != len(timestamps):
        raise DatasetValidationError("duplicate snapshot timestamps detected")
    if timestamps != sorted(timestamps):
        raise DatasetValidationError("snapshot filenames did not sort into chronological order")

    gaps_minutes = [(b - a).total_seconds() / 60 for a, b in itertools.pairwise(timestamps)]
    irregular_gaps = [
        (i, g) for i, g in enumerate(gaps_minutes) if abs(g - EXPECTED_INTERVAL_MINUTES) > 1
    ]

    malformed = []
    for path in files:
        try:
            snap = pd.read_csv(path, sep=r"\s+", header=None)
        except Exception as exc:
            malformed.append(f"{path.name}: unreadable ({exc})")
            continue
        if snap.shape != (EXPECTED_SAMPLES_PER_SNAPSHOT, EXPECTED_N_CHANNELS):
            expected = (EXPECTED_SAMPLES_PER_SNAPSHOT, EXPECTED_N_CHANNELS)
            malformed.append(f"{path.name}: shape {snap.shape}, expected {expected}")
        elif snap.isna().any().any():
            malformed.append(f"{path.name}: contains missing values")

    if malformed:
        raise DatasetValidationError(f"{len(malformed)} malformed snapshot(s), e.g. {malformed[0]}")

    return {
        "n_snapshots": len(files),
        "n_channels": EXPECTED_N_CHANNELS,
        "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "timestamp_min": timestamps[0].isoformat(),
        "timestamp_max": timestamps[-1].isoformat(),
        "irregular_gaps": irregular_gaps,  # non-fatal: reported, not rejected
    }


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
    files = _raw_files(raw_dir)
    rows = []
    for path in files:
        timestamp = _parse_timestamp(path.name)
        snapshot = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
        for i, bearing in enumerate(BEARING_COLS):
            feats = _snapshot_features(snapshot[:, i])
            rows.append({"bearing": bearing, "timestamp": timestamp, **feats})
    df = pd.DataFrame(rows).sort_values(["bearing", "timestamp"]).reset_index(drop=True)
    return df


def raw_dataset_checksum(raw_dir: Path = RAW_DIR) -> str:
    """SHA-256 over sorted (filename, content) pairs — stable regardless of filesystem order."""
    digest = hashlib.sha256()
    for path in _raw_files(raw_dir):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL in remaining snapshots (x10 minutes) until bearing 1's real recorded failure.

    Only bearing 1 has a known failure time; the other three were still healthy when
    the experiment ended, so their true RUL is unknown (right-censored) and they're
    dropped from the labeled set rather than given a fabricated target.
    """
    failed = df[df["bearing"] == FAILED_BEARING].sort_values("timestamp").reset_index(drop=True)
    failed["RUL"] = len(failed) - 1 - failed.index
    return failed
