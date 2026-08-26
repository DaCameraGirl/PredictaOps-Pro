from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bearing_data import (
    ALL_RUN_SPECS,
    BEARING_COLS,
    DEFAULT_RUN,
    RUN_SPECS,
    ChannelSpec,
    DatasetValidationError,
    FailureSpec,
    IMS_TEST1,
    IMS_TEST3,
    RunSpec,
    _snapshot_features,
    add_rul,
    build_feature_table,
    get_run_spec,
    normalize_feature_table,
    validate_raw_dataset,
    validate_run_spec,
)


def _tiny_run_spec(
    tmp_path,
    *,
    run_id="tiny_run",
    failures=None,
    expected_n_snapshots=3,
    bearing_cols=BEARING_COLS,
    expected_n_channels=4,
    channel_map=(),
    allowed_interval_minutes=(),
) -> RunSpec:
    start = datetime(2004, 1, 1)
    if failures is None:
        failures = (
            FailureSpec(
                bearing=bearing_cols[0],
                endpoint_timestamp=start + timedelta(minutes=10 * (expected_n_snapshots - 1)),
                failure_mode="test failure",
            ),
        )
    return RunSpec(
        run_id=run_id,
        dataset_name="Tiny structural validation run",
        raw_dir=tmp_path,
        features_cache=tmp_path / f"{run_id}_features.csv",
        metadata_path=tmp_path / f"{run_id}_metadata.json",
        bearing_cols=bearing_cols,
        failures=failures,
        expected_n_snapshots=expected_n_snapshots,
        expected_n_channels=expected_n_channels,
        expected_samples_per_snapshot=4,
        sampling_rate_hz=20000,
        expected_interval_minutes=10,
        channel_map=channel_map,
        allowed_interval_minutes=allowed_interval_minutes,
    )


def _write_snapshot(path, rows=4, cols=4) -> None:
    path.write_text("\n".join(" ".join("0" for _ in range(cols)) for _ in range(rows)))


def test_snapshot_features_on_known_sine_wave():
    """A pure sine wave gives independent ground truth for feature math."""
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


def test_add_rul_counts_down_to_exactly_zero_at_label_endpoint():
    rows = []
    start = datetime(2004, 1, 1)
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=10)
    for bearing in BEARING_COLS:
        for i in range(10):
            rows.append(
                {
                    "run_id": run_spec.run_id,
                    "bearing": bearing,
                    "timestamp": start + timedelta(minutes=10 * i),
                }
            )
    df = pd.DataFrame(rows)

    labeled = add_rul(df, run_spec)

    assert set(labeled["bearing"].unique()) == {"bearing_1"}
    assert labeled.sort_values("timestamp")["RUL"].iloc[-1] == 0
    assert labeled.sort_values("timestamp")["RUL"].iloc[0] == 9
    assert (labeled["RUL"] >= 0).all()


def test_add_rul_only_labels_the_bearing_with_a_known_failure():
    """Right-censored bearings must never get a fabricated RUL label."""
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=1)
    rows = [
        {"run_id": run_spec.run_id, "bearing": b, "timestamp": datetime(2004, 1, 1)}
        for b in BEARING_COLS
    ]
    labeled = add_rul(pd.DataFrame(rows), run_spec)
    assert "bearing_2" not in labeled["bearing"].values
    assert "bearing_3" not in labeled["bearing"].values
    assert "bearing_4" not in labeled["bearing"].values


def test_verified_run_catalog_contains_all_three_ims_experiments():
    assert set(ALL_RUN_SPECS) == {"ims_test1", "ims_test2", "ims_test3"}
    assert set(RUN_SPECS) == {"ims_test2"}
    assert get_run_spec("ims_test2") == DEFAULT_RUN
    assert get_run_spec("ims_test1") == IMS_TEST1
    assert get_run_spec("ims_test3") == IMS_TEST3

    with pytest.raises(ValueError, match="unknown run"):
        get_run_spec("ims_test4")


def test_test1_and_test3_metadata_matches_documented_layout():
    assert validate_run_spec(IMS_TEST1) == []
    assert IMS_TEST1.expected_n_snapshots == 2156
    assert IMS_TEST1.expected_n_channels == 8
    assert IMS_TEST1.allowed_interval_minutes == (5, 10)
    assert [(f.bearing, f.failure_mode) for f in IMS_TEST1.failures] == [
        ("bearing_3", "inner race defect"),
        ("bearing_4", "rolling element defect"),
    ]
    assert len([c for c in IMS_TEST1.channel_map if c.bearing == "bearing_3"]) == 2

    assert validate_run_spec(IMS_TEST3) == []
    assert IMS_TEST3.expected_n_snapshots == 4448
    assert IMS_TEST3.expected_n_channels == 4
    assert [(f.bearing, f.failure_mode) for f in IMS_TEST3.failures] == [
        ("bearing_3", "outer race defect"),
    ]


def test_existing_test2_cache_is_normalized_without_rewrite(feature_table):
    assert "run_id" in feature_table.columns
    assert "sensor_id" in feature_table.columns
    assert "channel_index" in feature_table.columns
    assert set(feature_table["run_id"]) == {"ims_test2"}

    legacy = feature_table.drop(columns=["run_id", "sensor_id", "channel_index"])
    normalized = normalize_feature_table(legacy, DEFAULT_RUN)

    assert normalized.columns[0] == "run_id"
    assert set(normalized["run_id"]) == {"ims_test2"}
    assert set(normalized["sensor_id"]) == {"sensor_1"}
    assert set(normalized["channel_index"]) == {0, 1, 2, 3}


def test_feature_cache_with_wrong_run_identity_is_rejected():
    table = pd.DataFrame(
        {
            "run_id": ["ims_test3"],
            "bearing": ["bearing_1"],
            "sensor_id": ["sensor_1"],
            "channel_index": [0],
            "timestamp": [datetime(2004, 1, 1)],
        }
    )
    with pytest.raises(DatasetValidationError, match="run_id mismatch"):
        normalize_feature_table(table, DEFAULT_RUN)


def test_build_feature_table_adds_run_and_sensor_identity(tmp_path):
    run_spec = _tiny_run_spec(tmp_path, expected_n_snapshots=2)
    for i in range(2):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"))

    table = build_feature_table(run_spec=run_spec)

    assert set(table["run_id"]) == {run_spec.run_id}
    assert set(table["sensor_id"]) == {"sensor_1"}
    assert set(table["channel_index"]) == {0, 1, 2, 3}


def test_multi_sensor_bearing_rows_share_bearing_level_rul(tmp_path):
    channels = (
        ChannelSpec(0, "bearing_1", "sensor_x"),
        ChannelSpec(1, "bearing_1", "sensor_y"),
    )
    endpoint = datetime(2004, 1, 1, 0, 10)
    run_spec = _tiny_run_spec(
        tmp_path,
        expected_n_snapshots=2,
        bearing_cols=("bearing_1",),
        expected_n_channels=2,
        channel_map=channels,
        failures=(FailureSpec("bearing_1", endpoint, "test failure"),),
    )
    for i in range(2):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"), cols=2)

    table = build_feature_table(run_spec=run_spec)
    labeled = add_rul(table, run_spec)

    assert len(table) == 4
    assert set(table["sensor_id"]) == {"sensor_x", "sensor_y"}
    assert labeled.groupby("timestamp")["RUL"].nunique().max() == 1
    assert labeled.groupby("timestamp")["RUL"].first().tolist() == [1, 0]
    assert set(labeled["trajectory_id"]) == {f"{run_spec.run_id}:bearing_1"}


def test_documented_failure_labeling_uses_run_end_endpoint():
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=3)
    rows = []
    start = datetime(2004, 1, 1)
    for bearing in BEARING_COLS:
        for i in range(4):
            rows.append(
                {
                    "run_id": run_spec.run_id,
                    "bearing": bearing,
                    "timestamp": start + timedelta(minutes=10 * i),
                }
            )

    labeled = add_rul(pd.DataFrame(rows), run_spec)

    assert len(labeled) == 3
    assert labeled["timestamp"].max() == run_spec.failures[0].endpoint_timestamp
    assert set(labeled["label_endpoint_timestamp"]) == {
        run_spec.failures[0].endpoint_timestamp.isoformat()
    }
    assert set(labeled["failure_mode"]) == {"test failure"}
    assert set(labeled["trajectory_id"]) == {f"{run_spec.run_id}:bearing_1"}


def test_missing_label_endpoint_is_rejected():
    run_spec = _tiny_run_spec(Path("."), expected_n_snapshots=3)
    rows = [
        {"run_id": run_spec.run_id, "bearing": "bearing_1", "timestamp": datetime(2004, 1, 1)},
        {
            "run_id": run_spec.run_id,
            "bearing": "bearing_1",
            "timestamp": datetime(2004, 1, 1) + timedelta(minutes=10),
        },
    ]

    with pytest.raises(DatasetValidationError, match="label endpoint"):
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


def test_validate_raw_dataset_accepts_documented_five_and_ten_minute_intervals(tmp_path):
    run_spec = _tiny_run_spec(
        tmp_path,
        expected_n_snapshots=3,
        allowed_interval_minutes=(5, 10),
        failures=(
            FailureSpec(
                "bearing_1",
                datetime(2004, 1, 1, 0, 15),
                "test failure",
            ),
        ),
    )
    for ts in [
        datetime(2004, 1, 1, 0, 0),
        datetime(2004, 1, 1, 0, 5),
        datetime(2004, 1, 1, 0, 15),
    ]:
        _write_snapshot(tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S"))

    result = validate_raw_dataset(run_spec=run_spec)

    assert result["allowed_interval_minutes"] == [5, 10]
    assert result["irregular_gaps"] == []


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
