"""Train the RUL regressor on bearing 1's real vibration feature trajectory."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from bearing_data import FEATURE_NAMES, add_rul, build_feature_table

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "ims_test2_features.csv"

RUL_CLIP = 400  # RMS/kurtosis stay flat for the first ~600 snapshots; cap so training isn't dominated by the flat healthy stretch
VAL_FRACTION = 0.2


def main():
    if DATA_CACHE.exists():
        table = pd.read_csv(DATA_CACHE, parse_dates=["timestamp"])
    else:
        table = build_feature_table()
        DATA_CACHE.parent.mkdir(exist_ok=True)
        table.to_csv(DATA_CACHE, index=False)

    labeled = add_rul(table)
    labeled["RUL"] = labeled["RUL"].clip(upper=RUL_CLIP)

    # Only one bearing in this test actually failed, so there's no second unit to hold
    # out the way the turbofan model held out whole engines. Instead we hold out a
    # random slice of snapshots from across this bearing's full lifetime, so training
    # still sees examples from every stage of degradation, healthy through failing.
    # This validates interpolation within the observed range, not forecasting an
    # unseen future pattern, which tree models can't do from a single trajectory.
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(labeled))
    n_val = int(len(labeled) * VAL_FRACTION)
    val_idx, fit_idx = shuffled[:n_val], shuffled[n_val:]
    fit_df = labeled.iloc[fit_idx]
    val_df = labeled.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(fit_df[FEATURE_NAMES], fit_df["RUL"])

    val_pred = model.predict(val_df[FEATURE_NAMES])
    val_rmse = mean_squared_error(val_df["RUL"], val_pred) ** 0.5
    print(f"random held-out snapshots validation RMSE: {val_rmse:.1f} snapshots (~{val_rmse * 10:.0f} min)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "bearing_rul_model.joblib")
    (MODEL_DIR / "bearing_feature_cols.json").write_text(json.dumps(FEATURE_NAMES))
    (MODEL_DIR / "bearing_metrics.json").write_text(json.dumps({"val_rmse_snapshots": val_rmse}, indent=2))
    print(f"saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
