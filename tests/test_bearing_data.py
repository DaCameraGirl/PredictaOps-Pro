from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bearing_data import (
    BEARING_COLS,
    DEFAULT_RUN,
    DatasetValidationError,
    FailureSpec,
    RunSpec,
    _snapshot_features,
    add_rul,
    build_feature_table,
    get_run_spec,
    normalize_feature_table,
    validate_raw_dataset,
)


def _tiny_run_spec(tmp_path, *, run_id="tiny_run", failures=None, expected_n_snapshots=3) -> RunSpec:
    start = datetime(2004, 1, 1)
    if failures is None:
        failures = (
            FailureSpec(
                bearing="bearing_1",
                failure_timestamp=start + timedelta(minutes=10 * (expected_n_snapshots - 1)),
                failure_mode="test failure",
            ),
        )
    return RunSpec(
        run_id=run_id,
        dataset_name="Tiny structural validation run",
        raw_dir=tmp_path,
        features_cache=tmp_path / f"{run_id}_features.csv",
        metadata_path=tmp_path / f"{run_id}_metadata.json",
        bearing_cols=BEARING_COLS,
        failures=failures,
        expected_n_snapshots=expected_n_snapshots,
        expected_n_channels=4,
        expected_samples_per_snapshot=4,
        sampling_rate_hz=20000,
        expected_interval_minutes=10,
    )


def _write_snapshot(path, rows=4, cols=4) -> None:
    path.write_text("\n".join(" ".join("0" for _ in range(cols)) for _ in range(rows)))


def test_snapshot_features_on_known_sine_wave():
    """A pure sine wave has textbook-known RMS and crest factor, independent of this
    codebase — a real ground truth to check the feature math against, not just
    "does it run"."""
    t = np.linspace(0, 1, 20480, endpoint=False)
    amplitude = 2.0
    signal = amplitude * np.sin(2 * np.pi * 50 * t)

    feats = _snapshot_features(signal)

    assert feats["mean"] == pytest.approx(0.0, abs=1e-6)
    assert feats["rms"] == pytest.approx(amplitude / np.sqrt(2), rel=1e-3)
    assert feats["peak_to_peak"] == pytest.approx(2 * amplitude, rel=1e-3)
    assert feats["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-2)


def test_snapshot_features_handles_constant_signal_without_dividing_by_zero():
    signal = np.zeros(20480)
    feats = _snapshot_features(signal)
    assert feats["kurtosis"] == 0.0
    assert feats["skew"] == 0.0
    assert feats["crest_factor"] == 0.0


def test_add_rul_counts_down_to_exactly_zero_at_last_recorded_snapshot():
    rows = []
    start = datetime(2004, 1, 1)
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=10)
    for bearing in BEARING_COLS:
        for i in range(10):
            rows.append({"run_id": run_spec.run_id, "bearing": bearing, "timestamp": start + timedelta(minutes=10 * i)})
    df = pd.DataFrame(rows)

    labeled = add_rul(df, run_spec)

    assert set(labeled["bearing"].unique()) == {"bearing_1"}
    assert labeled.sort_values("timestamp")["RUL"].iloc[-1] == 0
    assert labeled.sort_values("timestamp")["RUL"].iloc[0] == 9
    assert (labeled["RUL"] >= 0).all()


def test_add_rul_only_labels_the_bearing_with_a_known_failure():
    """Bearings 2-4 never failed in this test — right-censored data must never get a
    fabricated RUL label."""
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=1)
    rows = [{"run_id": run_spec.run_id, "bearing": b, "timestamp": datetime(2004, 1, 1)} for b in BEARING_COLS]
    labeled = add_rul(pd.DataFrame(rows), run_spec)
    assert "bearing_2" not in labeled["bearing"].values
    assert "bearing_3" not in labeled["bearing"].values
    assert "bearing_4" not in labeled["bearing"].values


def test_run_selection_returns_immutable_test2_spec():
    run_spec = get_run_spec("ims_test2")
    assert run_spec == DEFAULT_RUN
    assert run_spec.run_id == "ims_test2"

    with pytest.raises(ValueError, match="unknown run"):
        get_run_spec("ims_test1")


def test_existing_test2_cache_is_normalized_with_run_identity_without_rewrite(feature_table):
    assert "run_id" in feature_table.columns
    assert set(feature_table["run_id"]) == {"ims_test2"}

    old_cache_shape = feature_table.drop(columns=["run_id"])
    normalized = normalize_feature_table(old_cache_shape, DEFAULT_RUN)
    assert normalized.columns[0] == "run_id"
    assert set(normalized["run_id"]) == {"ims_test2"}


def test_build_feature_table_adds_run_identity_to_new_rows(tmp_path):
    run_spec = _tiny_run_spec(tmp_path, expected_n_snapshots=2)
    for i in range(2):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"))

    table = build_feature_table(run_spec=run_spec)

    assert "run_id" in table.columns
    assert set(table["run_id"]) == {run_spec.run_id}


def test_documented_failure_labeling_uses_exact_failure_timestamp():
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=3)
    rows = []
    start = datetime(2004, 1, 1)
    for bearing in BEARING_COLS:
        for i in range(4):
            rows.append({"run_id": run_spec.run_id, "bearing": bearing, "timestamp": start + timedelta(minutes=10 * i)})

    labeled = add_rul(pd.DataFrame(rows), run_spec)

    assert len(labeled) == 3
    assert labeled["timestamp"].max() == run_spec.failures[0].failure_timestamp
    assert set(labeled["failure_timestamp"]) == {run_spec.failures[0].failure_timestamp.isoformat()}
    assert set(labeled["failure_mode"]) == {"test failure"}
    assert set(labeled["trajectory_id"]) == {f"{run_spec.run_id}:bearing_1"}


def test_missing_exact_failure_timestamp_is_rejected():
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=3)
    rows = [
        {"run_id": run_spec.run_id, "bearing": "bearing_1", "timestamp": datetime(2004, 1, 1)},
        {
            "run_id": run_spec.run_id,
            "bearing": "bearing_1",
            "timestamp": datetime(2004, 1, 1) + timedelta(minutes=10),
        },
    ]

    with pytest.raises(DatasetValidationError, match="failure timestamp"):
        add_rul(pd.DataFrame(rows), run_spec)


def test_validate_raw_dataset_rejects_wrong_snapshot_count(tmp_path):
    (tmp_path / "2004.02.12.10.32.39").write_text("0 0 0 0\n")
    with pytest.raises(DatasetValidationError, match=r"expected .* snapshots"):
        validate_raw_dataset(tmp_path)


def test_validate_raw_dataset_rejects_unparseable_filename(tmp_path):
    (tmp_path / "not-a-timestamp.txt").write_text("0 0 0 0\n")
    with pytest.raises(DatasetValidationError, match="unparseable"):
        validate_raw_dataset(tmp_path)


def test_validate_raw_dataset_accepts_custom_structural_spec(tmp_path):
    run_spec = _tiny_run_spec(tmp_path)

    for i in range(3):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"))

    result = validate_raw_dataset(run_spec=run_spec)

    assert result["run_id"] == run_spec.run_id
    assert result["n_snapshots"] == 3


def test_validate_raw_dataset_rejects_malformed_snapshot_shape(tmp_path):
    run_spec = _tiny_run_spec(tmp_path)

    for i in range(3):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"), cols=3 if i == 1 else 4)

    with pytest.raises(DatasetValidationError, match="malformed"):
        validate_raw_dataset(run_spec=run_spec)


def test_feature_table_has_no_missing_values(feature_table):
    from bearing_data import FEATURE_NAMES

    assert not feature_table[FEATURE_NAMES].isna().any().any()


def test_feature_table_has_no_duplicate_bearing_timestamp_pairs(feature_table):
    dupes = feature_table.duplicated(subset=["bearing", "timestamp"])
    assert not dupes.any()
