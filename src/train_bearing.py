"""Train the RUL regressor on bearing 1's real vibration feature trajectory.

Only one bearing in this test actually failed, so there's no second unit to hold
out the way the turbofan model held out whole engines. A random split across this
single trajectory would leak the shape of the near future into training, since
adjacent-in-time snapshots are nearly identical, so validation uses a chronological,
expanding-window walk-forward backtest instead: at each step the model only sees
data strictly earlier than what it's asked to predict. This is slower and reports
worse numbers than a random split, but the numbers are honest.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bearing_data import FEATURE_NAMES, add_rul, build_feature_table

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "ims_test2_features.csv"

RUL_CLIP = 400  # RMS/kurtosis stay flat for the first ~600 snapshots; cap so training isn't dominated by the flat healthy stretch
SNAPSHOT_MINUTES = 10
N_FOLDS = 4
MIN_TRAIN_FRACTION = 0.4  # first fold trains on this much history before predicting anything


def make_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


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


def walk_forward_backtest(labeled: pd.DataFrame):
    n = len(labeled)
    fold_bounds = np.linspace(int(n * MIN_TRAIN_FRACTION), n, N_FOLDS + 1, dtype=int)
    all_true, all_pred = [], []
    for i in range(N_FOLDS):
        train_end, test_end = fold_bounds[i], fold_bounds[i + 1]
        if test_end <= train_end:
            continue
        train = labeled.iloc[:train_end]
        test = labeled.iloc[train_end:test_end]
        model = make_model()
        model.fit(train[FEATURE_NAMES], train["RUL"])
        all_true.append(test["RUL"].to_numpy())
        all_pred.append(model.predict(test[FEATURE_NAMES]))
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    if DATA_CACHE.exists():
        table = pd.read_csv(DATA_CACHE, parse_dates=["timestamp"])
    else:
        table = build_feature_table()
        DATA_CACHE.parent.mkdir(exist_ok=True)
        table.to_csv(DATA_CACHE, index=False)

    labeled = add_rul(table)
    labeled["RUL"] = labeled["RUL"].clip(upper=RUL_CLIP)

    y_true, y_pred = walk_forward_backtest(labeled)
    residual = y_pred - y_true  # true = pred - residual, for building the interval below
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    score = asymmetric_score(y_true, y_pred)
    resid_lo, resid_hi = np.percentile(residual, [10, 90])

    print(f"walk-forward backtest MAE: {mae:.1f} snapshots (~{mae * SNAPSHOT_MINUTES / 60:.1f} h), n={len(y_true)}")
    print(f"walk-forward backtest RMSE: {rmse:.1f} snapshots (~{rmse * SNAPSHOT_MINUTES / 60:.1f} h)")
    print(f"asymmetric late-prediction penalty score (mean per point): {score:.2f} (lower is better)")
    print(f"80% empirical residual interval: [{resid_lo:.1f}, {resid_hi:.1f}] snapshots")

    # Backtest above is for honest metrics only. The model actually served by the app
    # is fit on the whole trajectory, same as a real deployment would use all history
    # collected so far.
    final_model = make_model()
    final_model.fit(labeled[FEATURE_NAMES], labeled["RUL"])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "bearing_rul_model.joblib")
    (MODEL_DIR / "bearing_feature_cols.json").write_text(json.dumps(FEATURE_NAMES))
    metrics = {
        "validation_method": "chronological walk-forward, expanding window, no shuffling",
        "n_backtest_points": int(len(y_true)),
        "mae_snapshots": mae,
        "mae_hours": mae * SNAPSHOT_MINUTES / 60,
        "rmse_snapshots": rmse,
        "rmse_hours": rmse * SNAPSHOT_MINUTES / 60,
        "asymmetric_score": score,
        "interval_80_residual_low": float(resid_lo),
        "interval_80_residual_high": float(resid_hi),
        "snapshot_minutes": SNAPSHOT_MINUTES,
        "rul_clip": RUL_CLIP,
        "scope_statement": (
            "Trained and backtested on one confirmed failure trajectory (bearing 1, "
            "outer race defect). This demonstrates trajectory fitting and honest "
            "backtesting on real data, not a validated general bearing-failure model."
        ),
    }
    (MODEL_DIR / "bearing_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
