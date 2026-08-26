"""Build features, validate a selected run, backtest, and train the RUL regressor.

Only one bearing in this test actually failed, so there's no second unit to hold
out the way a turbofan model would hold out whole engines. A random split across
this single trajectory would leak the shape of the near future into training, since
adjacent-in-time snapshots are nearly identical, so validation uses a chronological,
expanding-window walk-forward backtest instead: at each step the model only sees
data strictly earlier than what it's asked to predict. This is slower and reports
worse numbers than a random split, but the numbers are honest.

There is no fitted preprocessing step here (no scaler, no imputer, no encoder) —
XGBoost trains directly on the raw extracted features — so there's no separate
"fit only on the training fold" concern beyond the RUL clip, which is a fixed
constant chosen from a visual read of the data, not fit from labels.

Run: python src/train_bearing.py --run ims_test2
"""
import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from bearing_data import (
    DEFAULT_RUN,
    FEATURE_NAMES,
    RUN_SPECS,
    add_rul,
    build_feature_table,
    get_run_spec,
    load_feature_table,
    raw_dataset_checksum,
    validate_raw_dataset,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# RMS/kurtosis stay flat for the first ~600 snapshots; cap so training isn't dominated
# by the flat healthy stretch.
RUL_CLIP = 400
SNAPSHOT_MINUTES = 10
N_FOLDS = 4
MIN_TRAIN_FRACTION = 0.4  # first fold trains on this much history before predicting anything


class MultiTrajectoryTrainingError(ValueError):
    """Raised when the single-trajectory trainer is given multiple failure paths."""


def make_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def make_baseline_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Simplest defensible baseline: predict every test point as the training fold's
    mean RUL. Any model earning its complexity has to beat this, not just beat zero."""
    return np.full(len(test), train["RUL"].mean())


def asymmetric_score(y_true, y_pred, scale=RUL_CLIP):
    """Predicting more time than a bearing actually has is the dangerous error
    (maintenance gets scheduled too late), so it's penalized harder than the
    reverse. diff > 0 means the model over-promised remaining life. The two
    divisors below are the classic PHM08 constants (13 early, 10 late), scaled
    from their original 125-cycle target range up to ours so the exponential
    doesn't just overflow on a wider scale."""
    k_early, k_late = 13 * scale / 125, 10 * scale / 125
    diff = y_pred - y_true
    return float(np.mean(np.where(diff < 0, np.exp(-diff / k_early) - 1, np.exp(diff / k_late) - 1)))


def horizon_bucket(y_true: np.ndarray) -> np.ndarray:
    """Early/middle/late life, by true RUL tercile, for the fold's *own* true values."""
    t1, t2 = np.percentile(y_true, [33.3, 66.7])
    return np.where(y_true >= t2, "early-life", np.where(y_true >= t1, "mid-life", "late-life"))


def walk_forward_backtest(labeled: pd.DataFrame):
    """Returns (y_true, y_pred, y_baseline, fold_meta) across all folds, strictly
    chronological: fold i only ever trains on data whose timestamp precedes every
    timestamp in fold i's test slice. Re-sorts defensively rather than trusting the
    caller already sorted it — a shuffled input here would silently reintroduce the
    exact leakage this function exists to prevent."""
    if "trajectory_id" in labeled.columns and labeled["trajectory_id"].nunique() > 1:
        trajectories = ", ".join(sorted(labeled["trajectory_id"].unique()))
        raise MultiTrajectoryTrainingError(
            "single-trajectory walk-forward training cannot flatten multiple independent "
            f"failure trajectories: {trajectories}"
        )

    labeled = labeled.sort_values("timestamp").reset_index(drop=True)
    n = len(labeled)
    fold_bounds = np.linspace(int(n * MIN_TRAIN_FRACTION), n, N_FOLDS + 1, dtype=int)
    all_true, all_pred, all_base = [], [], []
    fold_meta = []
    for i in range(N_FOLDS):
        train_end, test_end = fold_bounds[i], fold_bounds[i + 1]
        if test_end <= train_end:
            continue
        train = labeled.iloc[:train_end]
        test = labeled.iloc[train_end:test_end]
        assert train["timestamp"].max() < test["timestamp"].min(), (
            "leakage guard tripped: a training row is not strictly earlier than the test fold"
        )
        model = make_model()
        model.fit(train[FEATURE_NAMES], train["RUL"])
        pred = model.predict(test[FEATURE_NAMES])
        all_true.append(test["RUL"].to_numpy())
        all_pred.append(pred)
        all_base.append(make_baseline_prediction(train, test))
        fold_meta.append(
            {
                "fold": i,
                "train_rows": int(train_end),
                "test_rows": int(test_end - train_end),
                "train_start": train["timestamp"].min().isoformat(),
                "train_end": train["timestamp"].max().isoformat(),
                "test_start": test["timestamp"].min().isoformat(),
                "test_end": test["timestamp"].max().isoformat(),
            }
        )
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_base),
        fold_meta,
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=sorted(RUN_SPECS),
        default=DEFAULT_RUN.run_id,
        help="documented bearing run to train against",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    run_spec = get_run_spec(args.run)

    if run_spec.features_cache.exists():
        table = load_feature_table(run_spec)
        print(f"loaded cached features from {run_spec.features_cache} ({len(table)} rows)")
    else:
        print(f"no cached features, building {run_spec.run_id} from raw data in {run_spec.raw_dir} ...")
        validation = validate_raw_dataset(run_spec=run_spec)
        if validation["irregular_gaps"]:
            print(f"warning: {len(validation['irregular_gaps'])} irregular timestamp gap(s) in raw data")
        table = build_feature_table(run_spec=run_spec)
        run_spec.features_cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(run_spec.features_cache, index=False)

        run_spec.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        run_spec.metadata_path.write_text(json.dumps({
            "run_id": run_spec.run_id,
            "dataset": run_spec.dataset_name,
            "documented_failures": [
                {
                    "bearing": failure.bearing,
                    "failure_timestamp": failure.failure_timestamp.isoformat(),
                    "failure_mode": failure.failure_mode,
                }
                for failure in run_spec.failures
            ],
            "generated_at": datetime.now(UTC).isoformat(),
            "code_version": git_commit(),
            **validation,
            "raw_sha256": raw_dataset_checksum(run_spec=run_spec),
        }, indent=2))
        print(f"wrote dataset metadata to {run_spec.metadata_path}")

    labeled = add_rul(table, run_spec)
    labeled["RUL"] = labeled["RUL"].clip(upper=RUL_CLIP)

    y_true, y_pred, y_base, fold_meta = walk_forward_backtest(labeled)
    residual = y_pred - y_true  # true = pred - residual, for building the interval below

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    medae = median_absolute_error(y_true, y_pred)
    score = asymmetric_score(y_true, y_pred)
    resid_lo, resid_hi = np.percentile(residual, [10, 90])

    baseline_mae = mean_absolute_error(y_true, y_base)
    baseline_rmse = mean_squared_error(y_true, y_base) ** 0.5

    buckets = horizon_bucket(y_true)
    horizon_mae = {
        b: float(mean_absolute_error(y_true[buckets == b], y_pred[buckets == b]))
        for b in ["early-life", "mid-life", "late-life"]
        if (buckets == b).any()
    }

    late_mask = residual > 0  # model over-promised remaining life: the dangerous direction
    pct_late = float(np.mean(late_mask) * 100)
    mean_late_magnitude = float(residual[late_mask].mean()) if late_mask.any() else 0.0
    mean_early_magnitude = float(-residual[~late_mask].mean()) if (~late_mask).any() else 0.0

    print(f"walk-forward backtest MAE: {mae:.1f} snapshots (~{mae * SNAPSHOT_MINUTES / 60:.1f} h), n={len(y_true)}")
    print(f"walk-forward backtest RMSE: {rmse:.1f} snapshots (~{rmse * SNAPSHOT_MINUTES / 60:.1f} h)")
    print(f"walk-forward backtest median AE: {medae:.1f} snapshots")
    print(f"baseline (fold-mean predictor) MAE: {baseline_mae:.1f}, RMSE: {baseline_rmse:.1f} snapshots")
    print(f"model beats baseline on MAE: {mae < baseline_mae}, on RMSE: {rmse < baseline_rmse}")
    print(f"asymmetric late-prediction penalty score (mean per point): {score:.2f} (lower is better)")
    print(f"80% empirical residual interval: [{resid_lo:.1f}, {resid_hi:.1f}] snapshots")
    print(f"late (dangerous) predictions: {pct_late:.1f}% of points, "
          f"mean magnitude {mean_late_magnitude:.1f} snapshots")
    print(f"early (conservative) predictions: {100 - pct_late:.1f}% of points, "
          f"mean magnitude {mean_early_magnitude:.1f} snapshots")

    # Backtest above is for honest metrics only. The model actually served by the app
    # is fit on the whole trajectory, same as a real deployment would use all history
    # collected so far.
    final_model = make_model()
    final_model.fit(labeled[FEATURE_NAMES], labeled["RUL"])

    # Reproducibility check: a freshly loaded copy of what we're about to save must
    # predict identically to the in-memory model, or the artifact isn't trustworthy.
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "bearing_rul_model.joblib"
    joblib.dump(final_model, model_path)
    reloaded = joblib.load(model_path)
    check_row = labeled[FEATURE_NAMES].iloc[[0]]
    original_pred = final_model.predict(check_row)[0]
    reloaded_pred = reloaded.predict(check_row)[0]
    assert abs(original_pred - reloaded_pred) < 1e-6, "reloaded model prediction does not match original"
    print(f"reload consistency check passed ({original_pred:.4f} == {reloaded_pred:.4f})")

    (MODEL_DIR / "bearing_feature_cols.json").write_text(json.dumps(FEATURE_NAMES))
    metrics = {
        "validation_method": "chronological walk-forward, expanding window, no shuffling",
        "n_folds": N_FOLDS,
        "fold_boundaries": fold_meta,
        "n_backtest_points": len(y_true),
        "mae_snapshots": mae,
        "mae_hours": mae * SNAPSHOT_MINUTES / 60,
        "rmse_snapshots": rmse,
        "rmse_hours": rmse * SNAPSHOT_MINUTES / 60,
        "median_ae_snapshots": medae,
        "baseline_mae_snapshots": baseline_mae,
        "baseline_rmse_snapshots": baseline_rmse,
        "model_beats_baseline_mae": bool(mae < baseline_mae),
        "model_beats_baseline_rmse": bool(rmse < baseline_rmse),
        "horizon_mae_snapshots": horizon_mae,
        "asymmetric_score": score,
        "pct_late_predictions": pct_late,
        "mean_late_magnitude_snapshots": mean_late_magnitude,
        "mean_early_magnitude_snapshots": mean_early_magnitude,
        "interval_80_residual_low": float(resid_lo),
        "interval_80_residual_high": float(resid_hi),
        "interval_note": (
            "Global empirical uncertainty range derived from walk-forward residuals; "
            "not a conditionally calibrated guarantee."
        ),
        "snapshot_minutes": SNAPSHOT_MINUTES,
        "rul_clip": RUL_CLIP,
        "scope_statement": (
            "Trained and backtested on one confirmed failure trajectory (bearing 1, "
            "outer race defect). This demonstrates trajectory fitting and honest "
            "backtesting on real data, not a validated general bearing-failure model."
        ),
    }
    (MODEL_DIR / "bearing_metrics.json").write_text(json.dumps(metrics, indent=2))

    model_metadata = {
        "feature_order": FEATURE_NAMES,
        "training_range": {
            "start": labeled["timestamp"].min().isoformat(),
            "end": labeled["timestamp"].max().isoformat(),
            "n_rows": len(labeled),
        },
        "code_version": git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "rul_clip": RUL_CLIP,
        "algorithm": "xgboost.XGBRegressor",
        # XGBoost's default `missing` hyperparameter is literally float('nan'), the
        # sentinel it uses for missing values, which isn't valid strict JSON.
        "hyperparameters": {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in make_model().get_params().items()
        },
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2, default=str))
    print(f"saved model + metrics + metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
