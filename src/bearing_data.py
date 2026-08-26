"""Load, validate, and feature-extract NASA/IMS bearing run-to-failure data.

The IMS archive contains three independent test-to-failure runs. Snapshot filenames
are acquisition timestamps; each snapshot contains 20,480 vibration samples at
20 kHz. Test 1 has two accelerometer channels per bearing, while Tests 2 and 3 have
one channel per bearing.

Authoritative source:
NASA Prognostics Center of Excellence, "Bearing Data Set" (IMS, University of
Cincinnati, supported by Rexnord Corp.).
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
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
    """Documented failed bearing and the run-end snapshot used as the RUL=0 label endpoint."""

    bearing: str
    endpoint_timestamp: datetime
    failure_mode: str


@dataclass(frozen=True)
class ChannelSpec:
    """Map one physical snapshot column to a bearing and sensor identity."""

    channel_index: int
    bearing: str
    sensor_id: str


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
    channel_map: tuple[ChannelSpec, ...] = ()
    allowed_interval_minutes: tuple[int, ...] = ()
    source_note: str = ""


def _single_sensor_channels() -> tuple[ChannelSpec, ...]:
    return tuple(
        ChannelSpec(channel_index=i, bearing=bearing, sensor_id="sensor_1")
        for i, bearing in enumerate(BEARING_COLS)
    )


IMS_TEST1 = RunSpec(
    run_id="ims_test1",
    dataset_name="NASA/IMS Bearing Data Set, Test 1",
    raw_dir=DATA_DIR / "raw" / "ims_test1",
    features_cache=PROCESSED_DIR / "ims_test1_features.csv",
    metadata_path=PROCESSED_DIR / "ims_test1_metadata.json",
    bearing_cols=BEARING_COLS,
    failures=(
        FailureSpec(
            bearing="bearing_3",
            endpoint_timestamp=datetime(2003, 11, 25, 23, 39, 56),
            failure_mode="inner race defect",
        ),
        FailureSpec(
            bearing="bearing_4",
            endpoint_timestamp=datetime(2003, 11, 25, 23, 39, 56),
            failure_mode="rolling element defect",
        ),
    ),
    expected_n_snapshots=2156,
    expected_n_channels=8,
    expected_samples_per_snapshot=20480,
    sampling_rate_hz=20000,
    expected_interval_minutes=10,
    allowed_interval_minutes=(5, 10),
    channel_map=(
        ChannelSpec(0, "bearing_1", "sensor_x"),
        ChannelSpec(1, "bearing_1", "sensor_y"),
        ChannelSpec(2, "bearing_2", "sensor_x"),
        ChannelSpec(3, "bearing_2", "sensor_y"),
        ChannelSpec(4, "bearing_3", "sensor_x"),
        ChannelSpec(5, "bearing_3", "sensor_y"),
        ChannelSpec(6, "bearing_4", "sensor_x"),
        ChannelSpec(7, "bearing_4", "sensor_y"),
    ),
    source_note=(
        "Recording ended 2003-11-25 23:39:56; source documentation reports "
        "bearing 3 inner-race and bearing 4 rolling-element damage at experiment end."
    ),
)

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
            endpoint_timestamp=datetime(2004, 2, 19, 6, 22, 39),
            failure_mode="outer race defect",
        ),
    ),
    expected_n_snapshots=984,
    expected_n_channels=4,
    expected_samples_per_snapshot=20480,
    sampling_rate_hz=20000,
    expected_interval_minutes=10,
    allowed_interval_minutes=(10,),
    channel_map=_single_sensor_channels(),
    source_note=(
        "Recording ended 2004-02-19 06:22:39; source documentation reports "
        "bearing 1 outer-race failure at experiment end."
    ),
)

IMS_TEST3 = RunSpec(
    run_id="ims_test3",
    dataset_name="NASA/IMS Bearing Data Set, Test 3",
    raw_dir=DATA_DIR / "raw" / "ims_test3",
    features_cache=PROCESSED_DIR / "ims_test3_features.csv",
    metadata_path=PROCESSED_DIR / "ims_test3_metadata.json",
    bearing_cols=BEARING_COLS,
    failures=(
        FailureSpec(
            bearing="bearing_3",
            endpoint_timestamp=datetime(2004, 4, 4, 19, 1, 57),
            failure_mode="outer race defect",
        ),
    ),
    expected_n_snapshots=4448,
    expected_n_channels=4,
    expected_samples_per_snapshot=20480,
    sampling_rate_hz=20000,
    expected_interval_minutes=10,
    allowed_interval_minutes=(10,),
    channel_map=_single_sensor_channels(),
    source_note=(
        "Recording ended 2004-04-04 19:01:57; source documentation reports "
        "bearing 3 outer-race failure at experiment end."
    ),
)

RUN_SPECS = {
    IMS_TEST1.run_id: IMS_TEST1,
    IMS_TEST2.run_id: IMS_TEST2,
    IMS_TEST3.run_id: IMS_TEST3,
}
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
    """Raised when a raw run or feature cache violates its documented run contract."""


def get_run_spec(run_id: str) -> RunSpec:
    try:
        return RUN_SPECS[run_id]
    except KeyError as exc:
        known = ", ".join(sorted(RUN_SPECS))
        raise ValueError(f"unknown run {run_id!r}; expected one of: {known}") from exc


def _channels_for(run_spec: RunSpec) -> tuple[ChannelSpec, ...]:
    if run_spec.channel_map:
        return run_spec.channel_map
    return tuple(
        ChannelSpec(channel_index=i, bearing=bearing, sensor_id="sensor_1")
        for i, bearing in enumerate(run_spec.bearing_cols)
    )


def _allowed_intervals_for(run_spec: RunSpec) -> tuple[int, ...]:
    if run_spec.allowed_interval_minutes:
        return run_spec.allowed_interval_minutes
    return (run_spec.expected_interval_minutes,)


def validate_run_spec(run_spec: RunSpec) -> list[str]:
    """Validate metadata before it is trusted by raw-data processing."""
    errors: list[str] = []
    channels = _channels_for(run_spec)

    if not run_spec.run_id:
        errors.append("run_id must not be empty")
    if len(channels) != run_spec.expected_n_channels:
        errors.append(
            f"{run_spec.run_id}: channel map has {len(channels)} entries; "
            f"expected {run_spec.expected_n_channels}"
        )

    indexes = [channel.channel_index for channel in channels]
    if len(indexes) != len(set(indexes)):
        errors.append(f"{run_spec.run_id}: duplicate channel indexes")
    if indexes and (min(indexes) < 0 or max(indexes) >= run_spec.expected_n_channels):
        errors.append(f"{run_spec.run_id}: channel index outside snapshot shape")

    sensor_keys = [(channel.bearing, channel.sensor_id) for channel in channels]
    if len(sensor_keys) != len(set(sensor_keys)):
        errors.append(f"{run_spec.run_id}: duplicate bearing/sensor mapping")

    for channel in channels:
        if channel.bearing not in run_spec.bearing_cols:
            errors.append(f"{run_spec.run_id}: channel maps unknown bearing {channel.bearing!r}")

    seen_failures: set[str] = set()
    for failure in run_spec.failures:
        if failure.bearing not in run_spec.bearing_cols:
            errors.append(f"{run_spec.run_id}: failure maps unknown bearing {failure.bearing!r}")
        if failure.bearing in seen_failures:
            errors.append(f"{run_spec.run_id}: duplicate failure for {failure.bearing}")
        seen_failures.add(failure.bearing)

    if not _allowed_intervals_for(run_spec):
        errors.append(f"{run_spec.run_id}: no allowed recording interval")

    for value_name, value in (
        ("expected_n_snapshots", run_spec.expected_n_snapshots),
        ("expected_n_channels", run_spec.expected_n_channels),
        ("expected_samples_per_snapshot", run_spec.expected_samples_per_snapshot),
        ("sampling_rate_hz", run_spec.sampling_rate_hz),
    ):
        if value <= 0:
            errors.append(f"{run_spec.run_id}: {value_name} must be positive")

    return errors


def _parse_timestamp(filename: str) -> datetime:
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def _raw_files(raw_dir: Path, run_spec: RunSpec = DEFAULT_RUN) -> list[Path]:
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.name not in run_spec.ignored_filenames]
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
    """Structural validation against a documented IMS run."""
    spec_errors = validate_run_spec(run_spec)
    if spec_errors:
        raise DatasetValidationError("; ".join(spec_errors))

    raw_dir = raw_dir or run_spec.raw_dir
    if not raw_dir.exists():
        raise DatasetValidationError(
            f"raw data directory not found: {raw_dir}. "
            "Extract the selected IMS run there before preparing features."
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

    allowed_intervals = _allowed_intervals_for(run_spec)
    gaps_minutes = [(b - a).total_seconds() / 60 for a, b in itertools.pairwise(timestamps)]
    irregular_gaps = [
        (i, g)
        for i, g in enumerate(gaps_minutes)
        if all(abs(g - allowed) > 1 for allowed in allowed_intervals)
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
        "allowed_interval_minutes": list(allowed_intervals),
        "irregular_gaps": irregular_gaps,
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
    """One row per physical sensor and timestamp with vibration-analysis features."""
    raw_dir = raw_dir or run_spec.raw_dir
    files = _raw_files(raw_dir, run_spec)
    rows = []
    for path in files:
        timestamp = _parse_timestamp(path.name)
        snapshot = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
        for channel in _channels_for(run_spec):
            feats = _snapshot_features(snapshot[:, channel.channel_index])
            rows.append(
                {
                    "run_id": run_spec.run_id,
                    "bearing": channel.bearing,
                    "sensor_id": channel.sensor_id,
                    "channel_index": channel.channel_index,
                    "timestamp": timestamp,
                    **feats,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["run_id", "bearing", "sensor_id", "timestamp"]
    ).reset_index(drop=True)


def raw_dataset_checksum(raw_dir: Path | None = None, run_spec: RunSpec = DEFAULT_RUN) -> str:
    """SHA-256 over sorted (filename, content) pairs — stable regardless of filesystem order."""
    raw_dir = raw_dir or run_spec.raw_dir
    digest = hashlib.sha256()
    for path in _raw_files(raw_dir, run_spec):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def normalize_feature_table(df: pd.DataFrame, run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    """Normalize cache identity while preserving the committed legacy Test 2 CSV."""
    normalized = df.copy()
    if "run_id" not in normalized.columns:
        normalized.insert(0, "run_id", run_spec.run_id)
    elif set(normalized["run_id"].astype(str).unique()) != {run_spec.run_id}:
        found = ", ".join(sorted(normalized["run_id"].astype(str).unique()))
        raise DatasetValidationError(
            f"feature cache run_id mismatch for {run_spec.run_id}: found {found}"
        )

    channels = _channels_for(run_spec)
    by_bearing: dict[str, list[ChannelSpec]] = {}
    for channel in channels:
        by_bearing.setdefault(channel.bearing, []).append(channel)

    if "sensor_id" not in normalized.columns or "channel_index" not in normalized.columns:
        if any(len(mapped) != 1 for mapped in by_bearing.values()):
            raise DatasetValidationError(
                f"legacy feature cache for {run_spec.run_id} lacks sensor identity; rebuild it"
            )

        sensor_lookup = {bearing: mapped[0].sensor_id for bearing, mapped in by_bearing.items()}
        channel_lookup = {
            bearing: mapped[0].channel_index for bearing, mapped in by_bearing.items()
        }
        if "sensor_id" not in normalized.columns:
            normalized.insert(
                normalized.columns.get_loc("bearing") + 1,
                "sensor_id",
                normalized["bearing"].map(sensor_lookup),
            )
        if "channel_index" not in normalized.columns:
            normalized.insert(
                normalized.columns.get_loc("sensor_id") + 1,
                "channel_index",
                normalized["bearing"].map(channel_lookup),
            )

    if normalized["sensor_id"].isna().any() or normalized["channel_index"].isna().any():
        raise DatasetValidationError(f"feature cache contains unmapped sensor rows for {run_spec.run_id}")
    return normalized


def load_feature_table(run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    table = pd.read_csv(run_spec.features_cache, parse_dates=["timestamp"])
    return normalize_feature_table(table, run_spec)


def add_rul(df: pd.DataFrame, run_spec: RunSpec = DEFAULT_RUN) -> pd.DataFrame:
    """Label documented failed bearings using the documented run-end snapshot.

    Sensor rows from the same physical bearing share one bearing-level RUL target.
    Bearings not listed in the run spec remain right-censored and are not labeled.
    """
    df = normalize_feature_table(df, run_spec)
    labeled = []
    for failure in run_spec.failures:
        trajectory = df[
            (df["run_id"] == run_spec.run_id)
            & (df["bearing"] == failure.bearing)
            & (df["timestamp"] <= failure.endpoint_timestamp)
        ].copy()
        timestamps = sorted(pd.to_datetime(trajectory["timestamp"]).unique())
        endpoint = pd.Timestamp(failure.endpoint_timestamp)
        if not timestamps or pd.Timestamp(timestamps[-1]) != endpoint:
            raise DatasetValidationError(
                f"label endpoint {failure.endpoint_timestamp.isoformat()} "
                f"not found for {failure.bearing}"
            )

        rul_by_timestamp = {
            pd.Timestamp(timestamp): len(timestamps) - 1 - index
            for index, timestamp in enumerate(timestamps)
        }
        trajectory["RUL"] = pd.to_datetime(trajectory["timestamp"]).map(rul_by_timestamp)
        trajectory["label_endpoint_timestamp"] = failure.endpoint_timestamp.isoformat()
        trajectory["failure_mode"] = failure.failure_mode
        trajectory["trajectory_id"] = f"{run_spec.run_id}:{failure.bearing}"
        labeled.append(
            trajectory.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
        )

    if not labeled:
        return pd.DataFrame(
            columns=[
                *df.columns,
                "RUL",
                "label_endpoint_timestamp",
                "failure_mode",
                "trajectory_id",
            ]
        )
    return pd.concat(labeled, ignore_index=True)
