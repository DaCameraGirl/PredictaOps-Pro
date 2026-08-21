"""Train and evaluate the RUL regressor, then save the artifact for serving."""
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from data_loader import load_test, load_test_rul, load_train
from features import add_train_rul, last_cycle_per_unit, signal_feature_cols

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 scoring function: penalizes late (unsafe) predictions harder than early ones."""
    diff = y_pred - y_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


def unit_split(units: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, int(len(shuffled) * val_frac))
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def main():
    train_raw = load_train()
    feature_cols = signal_feature_cols(train_raw)
    labeled = add_train_rul(train_raw)

    train_units, val_units = unit_split(labeled["unit"].unique())
    fit_df = labeled[labeled["unit"].isin(train_units)]
    val_df = labeled[labeled["unit"].isin(val_units)]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(fit_df[feature_cols], fit_df["RUL"])

    val_pred = model.predict(val_df[feature_cols])
    val_rmse = mean_squared_error(val_df["RUL"], val_pred) ** 0.5
    print(f"held-out unit validation RMSE: {val_rmse:.2f} cycles")

    test_last = last_cycle_per_unit(load_test())
    true_rul = load_test_rul()
    test_pred = model.predict(test_last[feature_cols])
    test_true = true_rul.loc[test_last["unit"]].to_numpy()

    test_rmse = mean_squared_error(test_true, test_pred) ** 0.5
    test_score = nasa_score(test_true, test_pred)
    print(f"official test set RMSE: {test_rmse:.2f} cycles")
    print(f"official test set PHM08 score: {test_score:.1f} (lower is better)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "rul_model.joblib")
    (MODEL_DIR / "feature_cols.json").write_text(json.dumps(feature_cols))
    metrics = {
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "test_phm08_score": test_score,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved model + feature list + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
