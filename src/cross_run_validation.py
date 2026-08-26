"""Leakage-safe cross-run validation for prepared NASA/IMS bearing runs.

This module is deliberately separate from ``train_bearing.py``. The existing served
model is a Test-2-only, single-trajectory model. Cross-run evaluation instead:

1. collapses multiple accelerometer sensors back to one physical-bearing row per
   timestamp,
2. labels only documented failed bearings,
3. measures RUL in hours from the documented experiment endpoint, and
4. holds out an entire IMS experiment at a time.

No cross-run model artifact is written or served from here. The output is validation
evidence only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from bearing_data import FEATURE_NAMES, RunSpec, get_run_spec, load_feature_table

AGGREGATED_FEATURES = tuple(
    name
    for feature in FEATURE_NAMES
    for name in (f"{feature}_sensor_mean", f"{feature}_sensor_max_abs")
)


class CrossRunValidationError(ValueError):
    """Raised when cross-run validation would violate its data contract."""


class MissingPreparedRunError(FileNotFoundError):
    """Raised when a requested run has not been feature-extracted yet."""


def make_cross_run_model():
    """Fixed model configuration for validation; no held-out-run tuning occurs."""
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def aggregate_physical_bearing_features(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse sensor rows into one row per physical bearing and timestamp.

    Test 1 has two orthogonal accelerometers per bearing; Tests 2 and 3 have one.
    Treating Test 1's two sensors as two independent target trajectories would
    duplicate labels and overstate sample independence. We therefore retain sensor
    evidence through two orientation-agnostic summaries while emitting one target
    row per physical bearing/time point.

    ``sensor_count`` is retained as provenance only and is intentionally excluded
    from ``AGGREGATED_FEATURES`` so the model cannot trivially identify Test 1 from
    its two-sensor layout.
    """
    required = {"run_id", "bearing", "sensor_id", "timestamp", *FEATURE_NAMES}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise CrossRunValidationError(f"feature table missing columns: {', '.join(missing)}")

    normalized = table.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    duplicate_sensor_rows = normalized.duplicated(
        subset=["run_id", "bearing", "sensor_id", "timestamp"]
    )
    if duplicate_sensor_rows.any():
        raise CrossRunValidationError("duplicate sensor rows for the same run/bearing/timestamp")

    rows: list[dict] = []
    grouped = normalized.groupby(["run_id", "bearing", "timestamp"], sort=True)
    for (run_id, bearing, timestamp), sensor_rows in grouped:
        row: dict[str, object] = {
            "run_id": str(run_id),
            "bearing": str(bearing),
            "timestamp": pd.Timestamp(timestamp),
            "sensor_count": int(sensor_rows["sensor_id"].nunique()),
        }
        for feature in FEATURE_NAMES:
            values = sensor_rows[feature].astype(float).to_numpy()
            row[f"{feature}_sensor_mean"] = float(values.mean())
            row[f"{feature}_sensor_max_abs"] = float(np.max(np.abs(values)))
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=["run_id", "bearing", "timestamp", "sensor_count", *AGGREGATED_FEATURES]
        )
    return result.sort_values(["run_id", "bearing", "timestamp"]).reset_index(drop=True)


def label_documented_failure_trajectories(
    aggregated: pd.DataFrame,
    run_specs: Mapping[str, RunSpec],
) -> pd.DataFrame:
    """Attach hour-based RUL only to bearings with documented experiment-end failures."""
    labeled: list[pd.DataFrame] = []
    for run_id, run_spec in run_specs.items():
        run_rows = aggregated[aggregated["run_id"] == run_id]
        if run_rows.empty:
            raise CrossRunValidationError(f"no aggregated feature rows for {run_id}")

        for failure in run_spec.failures:
            trajectory = run_rows[run_rows["bearing"] == failure.bearing].copy()
            trajectory = trajectory[trajectory["timestamp"] <= failure.endpoint_timestamp]
            if trajectory.empty:
                raise CrossRunValidationError(
                    f"no rows for documented failure trajectory {run_id}:{failure.bearing}"
                )

            endpoint = pd.Timestamp(failure.endpoint_timestamp)
            if trajectory["timestamp"].max() != endpoint:
                raise CrossRunValidationError(
                    f"label endpoint {endpoint.isoformat()} not found for {run_id}:{failure.bearing}"
                )

            trajectory["RUL_hours"] = (
                endpoint - trajectory["timestamp"]
            ).dt.total_seconds() / 3600.0
            trajectory["trajectory_id"] = f"{run_id}:{failure.bearing}"
            trajectory["failure_mode"] = failure.failure_mode
            trajectory["label_endpoint_timestamp"] = endpoint.isoformat()
            max_rul = float(trajectory["RUL_hours"].max())
            trajectory["life_fraction_remaining"] = (
                trajectory["RUL_hours"] / max_rul if max_rul > 0 else 0.0
            )
            labeled.append(trajectory)

    if not labeled:
        raise CrossRunValidationError("no documented failure trajectories available")
    return pd.concat(labeled, ignore_index=True).sort_values(
        ["run_id", "bearing", "timestamp"]
    ).reset_index(drop=True)


def load_prepared_failure_data(run_ids: Sequence[str]) -> tuple[pd.DataFrame, dict[str, RunSpec]]:
    """Load prepared caches and return one labeled row per physical bearing/timestamp."""
    if len(set(run_ids)) != len(run_ids):
        raise CrossRunValidationError("run ids must be unique")
    if len(run_ids) < 2:
        raise CrossRunValidationError("cross-run validation requires at least two independent runs")

    specs = {run_id: get_run_spec(run_id) for run_id in run_ids}
    missing_paths = [spec.features_cache for spec in specs.values() if not spec.features_cache.exists()]
    if missing_paths:
        formatted = ", ".join(str(path) for path in missing_paths)
        raise MissingPreparedRunError(
            "prepared feature cache(s) missing: "
            f"{formatted}. Prepare each run with scripts/prepare_bearing_run.py first."
        )

    tables = [load_feature_table(spec) for spec in specs.values()]
    combined = pd.concat(tables, ignore_index=True)
    aggregated = aggregate_physical_bearing_features(combined)
    return label_documented_failure_trajectories(aggregated, specs), specs


def leave_one_run_out_folds(labeled: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return folds where no row from the held-out run appears in training."""
    run_ids = sorted(labeled["run_id"].astype(str).unique())
    if len(run_ids) < 2:
        raise CrossRunValidationError("leave-one-run-out validation requires at least two runs")

    folds = []
    for held_out_run in run_ids:
        train = labeled[labeled["run_id"] != held_out_run].copy()
        test = labeled[labeled["run_id"] == held_out_run].copy()
        if train.empty or test.empty:
            raise CrossRunValidationError(f"empty train/test fold for held-out run {held_out_run}")
        if held_out_run in set(train["run_id"].astype(str)):
            raise AssertionError("run leakage guard tripped")
        if set(test["run_id"].astype(str)) != {held_out_run}:
            raise AssertionError("held-out fold contains rows from another run")
        folds.append((held_out_run, train, test))
    return folds


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    dangerous = residual > 0
    return {
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "rmse_hours": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "median_ae_hours": float(median_absolute_error(y_true, y_pred)),
        "dangerous_overprediction_pct": float(np.mean(dangerous) * 100),
        "mean_dangerous_overprediction_hours": (
            float(residual[dangerous].mean()) if dangerous.any() else 0.0
        ),
    }


def run_leave_one_run_out_validation(
    labeled: pd.DataFrame,
    *,
    model_factory: Callable[[], object] = make_cross_run_model,
) -> dict:
    """Fit on complete runs and evaluate on an entirely unseen IMS experiment."""
    fold_results: list[dict] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_baseline: list[np.ndarray] = []

    for held_out_run, train, test in leave_one_run_out_folds(labeled):
        model = model_factory()
        model.fit(train[list(AGGREGATED_FEATURES)], train["RUL_hours"])
        prediction = np.asarray(model.predict(test[list(AGGREGATED_FEATURES)]), dtype=float)
        truth = test["RUL_hours"].to_numpy(dtype=float)
        baseline = np.full(len(test), float(train["RUL_hours"].mean()))

        model_metrics = _fold_metrics(truth, prediction)
        baseline_metrics = _fold_metrics(truth, baseline)
        fold_result = {
            "held_out_run": held_out_run,
            "train_runs": sorted(train["run_id"].astype(str).unique().tolist()),
            "train_trajectories": sorted(train["trajectory_id"].astype(str).unique().tolist()),
            "held_out_trajectories": sorted(test["trajectory_id"].astype(str).unique().tolist()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            **model_metrics,
            "baseline_mae_hours": baseline_metrics["mae_hours"],
            "baseline_rmse_hours": baseline_metrics["rmse_hours"],
            "beats_baseline_mae": model_metrics["mae_hours"] < baseline_metrics["mae_hours"],
        }

        late_life = test["life_fraction_remaining"].to_numpy(dtype=float) <= 0.25
        if late_life.any():
            fold_result["late_life_mae_hours"] = float(
                mean_absolute_error(truth[late_life], prediction[late_life])
            )
        else:
            fold_result["late_life_mae_hours"] = None

        fold_results.append(fold_result)
        all_true.append(truth)
        all_pred.append(prediction)
        all_baseline.append(baseline)

    combined_true = np.concatenate(all_true)
    combined_pred = np.concatenate(all_pred)
    combined_baseline = np.concatenate(all_baseline)
    overall = _fold_metrics(combined_true, combined_pred)
    overall_baseline = _fold_metrics(combined_true, combined_baseline)
    overall["baseline_mae_hours"] = overall_baseline["mae_hours"]
    overall["baseline_rmse_hours"] = overall_baseline["rmse_hours"]
    overall["beats_baseline_mae"] = overall["mae_hours"] < overall_baseline["mae_hours"]

    return {
        "validation_method": "leave-one-entire-IMS-run-out",
        "target": "RUL_hours_to_documented_experiment_endpoint",
        "feature_aggregation": "one physical bearing/timestamp; sensor mean + sensor max-absolute",
        "baseline": "training-fold mean RUL_hours",
        "model_features": list(AGGREGATED_FEATURES),
        "n_runs": len(fold_results),
        "folds": fold_results,
        "overall": overall,
    }


def validate_prepared_runs(
    run_ids: Sequence[str],
    *,
    model_factory: Callable[[], object] = make_cross_run_model,
) -> dict:
    labeled, _ = load_prepared_failure_data(run_ids)
    return run_leave_one_run_out_validation(labeled, model_factory=model_factory)


def default_output_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "cross_run" / "cross_run_metrics.json"
