"""SHAP-based explanations for individual RUL predictions."""
import json
from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class RulExplainer:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "rul_model.joblib")
        self.feature_cols = json.loads((MODEL_DIR / "feature_cols.json").read_text())
        self._explainer = shap.TreeExplainer(self.model)

    def predict(self, row: pd.DataFrame) -> float:
        return float(self.model.predict(row[self.feature_cols])[0])

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        """Top-k sensors pushing this prediction away from the average RUL, with direction."""
        x = row[self.feature_cols]
        shap_values = self._explainer(x)
        contributions = sorted(
            (
                {
                    "feature": col,
                    "value": float(x.iloc[0][col]),
                    "shap_value": float(shap_values.values[0][i]),
                }
                for i, col in enumerate(self.feature_cols)
            ),
            key=lambda c: abs(c["shap_value"]),
            reverse=True,
        )
        return {
            "predicted_rul": self.predict(row),
            "base_value": float(shap_values.base_values[0]),
            "top_contributors": contributions[:top_k],
        }
