"""Load, validate, and feature-extract NASA/IMS real bearing run-to-failure vibration data.

Each raw file is a 1-second, 20kHz vibration snapshot (20480 rows) with one column
per bearing (4 bearings, 1 accelerometer each), taken every 10 minutes for ~7 days
until the run's documented failure. We don't feed raw waveforms to the model; we
reduce each snapshot to standard vibration-analysis statistics per bearing.

Dataset source: NASA Prognostics Center of Excellence Data Set Repository, "Bearing
Data Set" (IMS, University of Cincinnati, sponsored by Rexnord Corp.), Test 2.
Original repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Mirror used by scripts/download_data.py: https://github.com/RicardoPSLopes/IMS-DATASET
"""
import hashlib
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

BEARING_COLS = ("bearing_1", "bearing_2", "bearing_3", "bearing_4")
FEATURE_NAMES = ["mean", "std", "rms", "kurtosis", "skew", "peak_to_peak", "crest_factor"]

IGNORED_FILENAMES = {"README.md", "readme.txt"}


@dataclass(frozen=True)
class FailureSpec:
    bearing: str
    failure_timestamp: datetime
    failure_mode: str


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset_name: str
    raw_dir: Path
    features_cache: Path
    metadata_path: Path
    bearing_cols: tuple[str, ...]
    failures: tuple[FailureSpec, ...]
    expected_n_snapshots: int
    expected_n_channels: int
    expected_samples_per_snapshot: int
    sampling_rate_hz: int
    expected_interval_minutes: int
    ignored_filenames: frozenset[str] = frozenset(IGNORED_FILENAMES)


IMS_TEST2 = RunSpec(
    run_id="ims_test2",
    dataset_name="NASA/IMS Bearing Data Set, Test 2",
    raw_dir=DATA_DIR / "raw" / "ims_test2",
    features_cache=PROCESSED_DIR / "ims_test2_features.csv",
    metadata_path=PROCESSED_DIR / "dataset_metadata.json",
    bearing_cols=BEARING_COLS,
    failures=(
        FailureSpec(
            bearing="bearing_1",
            failure_timestamp=datetime(2004, 2, 19, 6, 22, 39),
            failure_mode="outer race defect",
        ),
    ),
    expected_n_snapshots=984,
    expected_n_channels=4,
    expected_samples_per_snapshot=20480,
    sampling_rate_hz=20000,
    expected_interval_minutes=10,
)

RUN_SPECS = {IMS_TEST2.run_id: IMS_TEST2}
DEFAULT_RUN = IMS_TEST2

# Backward-compatible names for the existing Test 2 dashboard and artifacts.
RAW_DIR = IMS_TEST2.raw_dir
FEATURES_CACHE = IMS_TEST2.features_cache
METADATA_PATH = IMS_TEST2.metadata_path
FAILED_BEARING = IMS_TEST2.failures[0].bearing
FAILURE_MODE = IMS_TEST2.failures[0].failure_mode
EXPECTED_N_SNAPSHOTS = IMS_TEST2.expected_n_snapshots
EXPECTED_N_CHANNELS = IMS_TEST2.expected_n_channels
EXPECTED_SAMPLES_PER_SNAPSHOT = IMS_TEST2.expected_samples_per_snapshot
EXPECTED_SAMPLING_RATE_HZ = IMS_TEST2.sampling_rate_hz
EXPECTED_INTERVAL_MINUTES = IMS_TEST2.expected_interval_minutes


class DatasetValidationError(Exception):
    """Raised when the raw dataset doesn't match the documented IMS Test 2 shape."""


def get_run_spec(run_id: str) -> RunSpec:
    try:
        return RUN_SPECS[run_id]
    except KeyError as exc:
        known = ", ".join(sorted(RUN_SPECS))
        raise ValueError(f"unknown run {run_id!r}; expected one of: {known}") from exc


def _parse_timestamp(filename: str) -> datetime:
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def _raw_files(raw_dir: Path, run_spec: RunSpec = DEFAULT_RUN) -> list[Path]:
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.name not in run_spec.ignored_filenames]
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


def validate_raw_dataset(raw_dir: Path | None = None, run_spec: RunSpec = DEFAULT_RUN) -> dict:
    """Structural validation against the run's documented shape. Raises
    DatasetValidationError with a specific reason on any mismatch."""
    raw_dir = raw_dir or run_spec.raw_dir
    if not raw_dir.exists():
        raise DatasetValidationError(
            f"raw data directory not found: {raw_dir}. Run scripts/download_data.py first."
        )

    files = _raw_files(raw_dir, run_spec)
    if len(files) != run_spec.expected_n_snapshots:
        raise DatasetValidationError(
            f"expected {run_spec.expected_n_snapshots} snapshots, found {len(files)}"
        )

    timestamps = [_parse_timestamp(p.name) for p in files]
    if len(set(timestamps)) != len(timestamps):
        raise DatasetValidationError("duplicate snapshot timestamps detected")
    if timestamps != sorted(timestamps):
        raise DatasetValidationError("snapshot filenames did not sort into chronological order")

    gaps_minutes = [(b - a).total_seconds() / 60 for a, b in itertools.pairwise(timestamps)]
    irregular_gaps = [
        (i, g) for i, g in enumerate(gaps_minutes) if abs(g - run_spec.expected_interval_minutes) > 1
    ]

    malformed = []
    for path in files:
        try:
            snap = pd.read_csv(path, sep=r"\s+", header=None)
        except Exception as exc:
            malformed.append(f"{path.name}: unreadable ({exc})")
            continue
        if snap.shape != (run_spec.expected_samples_per_snapshot, run_spec.expected_n_channels):
            expected = (run_spec.expected_samples_per_snapshot, run_spec.expected_n_channels)
            malformed.append(f"{path.name}: shape {snap.shape}, expected {expected}")
        elif snap.isna().any().any():
            malformed.append(f"{path.name}: contains missing values")

    if malformed:
        raise DatasetValidationError(f"{len(malformed)} malformed snapshot(s), e.g. {malformed[0]}")

    return {
        "run_id": run_spec.run_id,
        "n_snapshots": len(files),
        "n_channels": run_spec.expected_n_channels,
        "sampling_rate_hz": run_spec.sampling_rate_hz,
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


def build_feature_table(raw_dir: Path | None = None, run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    """One row per (bearing, timestamp) with vibration-analysis features, sorted by time."""
    raw_dir = raw_dir or run_spec.raw_dir
    files = _raw_files(raw_dir, run_spec)
    rows = []
    for path in files:
        timestamp = _parse_timestamp(path.name)
        snapshot = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
        for i, bearing in enumerate(run_spec.bearing_cols):
            feats = _snapshot_features(snapshot[:, i])
            rows.append({"run_id": run_spec.run_id, "bearing": bearing, "timestamp": timestamp, **feats})
    df = pd.DataFrame(rows).sort_values(["run_id", "bearing", "timestamp"]).reset_index(drop=True)
    return df


def raw_dataset_checksum(raw_dir: Path | None = None, run_spec: RunSpec = DEFAULT_RUN) -> str:
    """SHA-256 over sorted (filename, content) pairs — stable regardless of filesystem order."""
    raw_dir = raw_dir or run_spec.raw_dir
    digest = hashlib.sha256()
    for path in _raw_files(raw_dir, run_spec):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def normalize_feature_table(df: pd.DataFrame, run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    """Return a feature table with run identity present without rewriting old caches."""
    normalized = df.copy()
    if "run_id" not in normalized.columns:
        normalized.insert(0, "run_id", run_spec.run_id)
    return normalized


def load_feature_table(run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    table = pd.read_csv(run_spec.features_cache, parse_dates=["timestamp"])
    return normalize_feature_table(table, run_spec)


def add_rul(df: pd.DataFrame, run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    """RUL in remaining snapshots until each documented failure timestamp.

    Bearings not listed in the run spec are right-censored and dropped from the
    labeled set rather than given a fabricated target.
    """
    df = normalize_feature_table(df, run_spec)
    labeled = []
    for failure in run_spec.failures:
        trajectory = df[
            (df["run_id"] == run_spec.run_id)
            & (df["bearing"] == failure.bearing)
            & (df["timestamp"] <= failure.failure_timestamp)
        ].sort_values("timestamp").reset_index(drop=True)
        if trajectory.empty or trajectory.iloc[-1]["timestamp"] != failure.failure_timestamp:
            raise DatasetValidationError(
                f"failure timestamp {failure.failure_timestamp.isoformat()} not found for {failure.bearing}"
            )
        trajectory["RUL"] = len(trajectory) - 1 - trajectory.index
        trajectory["failure_timestamp"] = failure.failure_timestamp.isoformat()
        trajectory["failure_mode"] = failure.failure_mode
        trajectory["trajectory_id"] = f"{run_spec.run_id}:{failure.bearing}"
        labeled.append(trajectory)

    if not labeled:
        return pd.DataFrame(columns=[*df.columns, "RUL", "failure_timestamp", "failure_mode", "trajectory_id"])
    return pd.concat(labeled, ignore_index=True)
