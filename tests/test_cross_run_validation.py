from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import cross_run_validation
from bearing_data import FEATURE_NAMES, FailureSpec, RunSpec
from cross_run_validation import (
    AGGREGATED_FEATURES,
    CrossRunValidationError,
    MissingPreparedRunError,
    aggregate_physical_bearing_features,
    label_documented_failure_trajectories,
    leave_one_run_out_folds,
    load_prepared_failure_data,
    run_leave_one_run_out_validation,
)


def _run_spec(tmp_path: Path, run_id: str, failures: tuple[FailureSpec, ...]) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        dataset_name=f"Synthetic {run_id}",
        raw_dir=tmp_path / run_id,
        features_cache=tmp_path / f"{run_id}_features.csv",
        metadata_path=tmp_path / f"{run_id}_metadata.json",
        bearing_cols=("bearing_1", "bearing_2", "bearing_3", "bearing_4"),
        failures=failures,
        expected_n_snapshots=3,
        expected_n_channels=4,
        expected_samples_per_snapshot=4,
        sampling_rate_hz=20000,
        expected_interval_minutes=10,
    )


def _sensor_feature_rows(
    run_id: str,
    bearing: str,
    timestamps: list[datetime],
    sensor_ids: tuple[str, ...] = ("sensor_1",),
) -> list[dict]:
    rows = []
    for time_index, timestamp in enumerate(timestamps):
        for sensor_index, sensor_id in enumerate(sensor_ids):
            row = {
                "run_id": run_id,
                "bearing": bearing,
                "sensor_id": sensor_id,
                "timestamp": timestamp,
            }
            for feature_index, feature in enumerate(FEATURE_NAMES):
                row[feature] = float(time_index + sensor_index + feature_index + 1)
            rows.append(row)
    return rows


def _validation_rows(run_id: str, trajectory_ids: tuple[str, ...]) -> list[dict]:
    rows = []
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        for row_index in range(4):
            row = {
                "run_id": run_id,
                "bearing": trajectory_id.split(":")[-1],
                "timestamp": datetime(2004, 1, 1) + timedelta(hours=row_index),
                "sensor_count": 1,
                "trajectory_id": trajectory_id,
                "RUL_hours": float(3 - row_index + trajectory_index),
                "life_fraction_remaining": float(3 - row_index) / 3,
            }
            for feature_index, feature in enumerate(AGGREGATED_FEATURES):
                row[feature] = float(feature_index + row_index + trajectory_index)
            rows.append(row)
    return rows


def test_sensor_aggregation_emits_one_row_per_physical_bearing_timestamp():
    timestamps = [datetime(2004, 1, 1), datetime(2004, 1, 1, 0, 10)]
    table = pd.DataFrame(
        _sensor_feature_rows(
            "ims_test1",
            "bearing_3",
            timestamps,
            sensor_ids=("sensor_x", "sensor_y"),
        )
    )

    aggregated = aggregate_physical_bearing_features(table)

    assert len(aggregated) == 2
    assert set(aggregated["sensor_count"]) == {2}
    assert aggregated.loc[0, "rms_sensor_max_abs"] == pytest.approx(4.0)
    assert "rms_sensor_mean" not in AGGREGATED_FEATURES
    assert "sensor_count" not in AGGREGATED_FEATURES


def test_sensor_aggregation_rejects_duplicate_sensor_timestamp_rows():
    timestamp = datetime(2004, 1, 1)
    rows = _sensor_feature_rows("ims_test1", "bearing_3", [timestamp])
    table = pd.DataFrame([rows[0], rows[0]])

    with pytest.raises(CrossRunValidationError, match="duplicate sensor rows"):
        aggregate_physical_bearing_features(table)


def test_hour_based_rul_uses_actual_timestamps_across_five_and_ten_minute_gaps(tmp_path):
    timestamps = [
        datetime(2004, 1, 1, 0, 0),
        datetime(2004, 1, 1, 0, 5),
        datetime(2004, 1, 1, 0, 15),
    ]
    endpoint = timestamps[-1]
    spec = _run_spec(
        tmp_path,
        "ims_test1",
        (FailureSpec("bearing_3", endpoint, "inner race defect"),),
    )
    table = pd.DataFrame(
        _sensor_feature_rows(
            "ims_test1",
            "bearing_3",
            timestamps,
            sensor_ids=("sensor_x", "sensor_y"),
        )
    )
    aggregated = aggregate_physical_bearing_features(table)

    labeled = label_documented_failure_trajectories(aggregated, {"ims_test1": spec})

    assert labeled["RUL_hours"].tolist() == pytest.approx([0.25, 10 / 60, 0.0])
    assert set(labeled["trajectory_id"]) == {"ims_test1:bearing_3"}
    assert set(labeled["failure_mode"]) == {"inner race defect"}


def test_test1_failure_bearings_stay_together_in_held_out_run():
    labeled = pd.DataFrame(
        [
            *_validation_rows(
                "ims_test1",
                ("ims_test1:bearing_3", "ims_test1:bearing_4"),
            ),
            *_validation_rows("ims_test2", ("ims_test2:bearing_1",)),
            *_validation_rows("ims_test3", ("ims_test3:bearing_3",)),
        ]
    )

    folds = {held_out: (train, test) for held_out, train, test in leave_one_run_out_folds(labeled)}
    train, test = folds["ims_test1"]

    assert "ims_test1" not in set(train["run_id"])
    assert set(test["run_id"]) == {"ims_test1"}
    assert set(test["trajectory_id"]) == {"ims_test1:bearing_3", "ims_test1:bearing_4"}


def test_leave_one_run_out_validation_reports_baseline_and_disjoint_folds():
    labeled = pd.DataFrame(
        [
            *_validation_rows(
                "ims_test1",
                ("ims_test1:bearing_3", "ims_test1:bearing_4"),
            ),
            *_validation_rows("ims_test2", ("ims_test2:bearing_1",)),
            *_validation_rows("ims_test3", ("ims_test3:bearing_3",)),
        ]
    )

    result = run_leave_one_run_out_validation(
        labeled,
        model_factory=lambda: DummyRegressor(strategy="mean"),
    )

    assert result["validation_method"] == "leave-one-entire-IMS-run-out"
    assert result["target"] == "RUL_hours_to_documented_experiment_endpoint"
    assert result["n_runs"] == 3
    assert {fold["held_out_run"] for fold in result["folds"]} == {
        "ims_test1",
        "ims_test2",
        "ims_test3",
    }
    for fold in result["folds"]:
        assert fold["held_out_run"] not in fold["train_runs"]
        assert "baseline_mae_hours" in fold
        assert fold["beats_baseline_mae"] is False
    assert result["overall"]["beats_baseline_mae"] is False


def test_missing_prepared_runs_fail_loudly_before_validation(tmp_path, monkeypatch):
    specs = {
        "ims_test1": _run_spec(
            tmp_path,
            "ims_test1",
            (FailureSpec("bearing_3", datetime(2004, 1, 1), "inner race defect"),),
        ),
        "ims_test3": _run_spec(
            tmp_path,
            "ims_test3",
            (FailureSpec("bearing_3", datetime(2004, 1, 1), "outer race defect"),),
        ),
    }
    monkeypatch.setattr(cross_run_validation, "get_run_spec", lambda run_id: specs[run_id])

    with pytest.raises(MissingPreparedRunError, match="prepare_bearing_run.py"):
        load_prepared_failure_data(["ims_test1", "ims_test3"])
