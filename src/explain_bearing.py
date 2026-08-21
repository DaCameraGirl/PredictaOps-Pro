"""SHAP-based explanations for bearing RUL predictions."""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RECONCILIATION_TOLERANCE = 1e-2  # snapshots

logger = logging.getLogger(__name__)


class BearingRulExplainer:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "bearing_rul_model.joblib")
        self.feature_cols = json.loads((MODEL_DIR / "bearing_feature_cols.json").read_text())
        self._explainer = shap.TreeExplainer(self.model)

    def _raw_predict(self, row: pd.DataFrame) -> float:
        return float(self.model.predict(row[self.feature_cols])[0])

    def predict(self, row: pd.DataFrame) -> float:
        """RUL can't be negative; the model has no such constraint built in, so clip here."""
        return max(0.0, self._raw_predict(row))

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        x = row[self.feature_cols]
        raw_pred = self._raw_predict(row)

        try:
            shap_values = self._explainer(x)
            base_value = float(shap_values.base_values[0])
            per_feature = shap_values.values[0]
            reconciled_total = base_value + float(per_feature.sum())
            reconciliation = {
                "model_output": raw_pred,
                "base_value_plus_shap_sum": reconciled_total,
                "difference": abs(raw_pred - reconciled_total),
                "within_tolerance": abs(raw_pred - reconciled_total) < RECONCILIATION_TOLERANCE,
            }
            contributions = sorted(
                (
                    {
                        "feature": col,
                        "value": float(x.iloc[0][col]),
                        "shap_value": float(per_feature[i]),
                    }
                    for i, col in enumerate(self.feature_cols)
                ),
                key=lambda c: abs(c["shap_value"]),
                reverse=True,
            )
            return {
                "predicted_rul": max(0.0, raw_pred),
                "average_model_output": base_value,
                "top_contributors": contributions[:top_k],
                "reconciliation": reconciliation,
                "shap_unavailable": False,
            }
        except Exception:
            logger.exception("SHAP explanation failed; falling back to prediction-only response")
            return {
                "predicted_rul": max(0.0, raw_pred),
                "average_model_output": None,
                "top_contributors": [],
                "reconciliation": None,
                "shap_unavailable": True,
            }
