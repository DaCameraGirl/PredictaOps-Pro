"""SHAP-based explanations for bearing RUL predictions."""
import contextlib
import importlib
import json
import logging
import os
import platform
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RECONCILIATION_TOLERANCE = 1e-2  # snapshots

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_native_loader_error_dialogs():
    if platform.system() != "Windows":
        yield
        return

    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfault_errorbox = 0x0002
        sem_noopenfileerrorbox = 0x8000
        flags = sem_failcriticalerrors | sem_nogpfault_errorbox | sem_noopenfileerrorbox
        kernel32 = ctypes.windll.kernel32
        previous_mode = kernel32.SetErrorMode(flags)
        kernel32.SetErrorMode(previous_mode | flags)
    except Exception:
        previous_mode = None

    try:
        yield
    finally:
        if previous_mode is not None:
            kernel32.SetErrorMode(previous_mode)


def _should_skip_shap() -> str | None:
    if os.environ.get("PMS_DISABLE_SHAP") == "1":
        return "PMS_DISABLE_SHAP=1"
    return None


def _build_shap_explainer(model):
    with _suppress_native_loader_error_dialogs():
        shap = importlib.import_module("shap")
        return shap.TreeExplainer(model)


class BearingRulExplainer:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "bearing_rul_model.joblib")
        self.feature_cols = json.loads((MODEL_DIR / "bearing_feature_cols.json").read_text())
        self._explainer = None
        self.shap_error = None
        skip_reason = _should_skip_shap()
        if skip_reason:
            self.shap_error = skip_reason
            logger.warning(
                "SHAP unavailable; explanations will fall back to prediction-only responses: %s",
                skip_reason,
            )
            return
        try:
            self._explainer = _build_shap_explainer(self.model)
        except Exception as exc:
            self.shap_error = str(exc)
            logger.warning(
                "SHAP unavailable; explanations will fall back to prediction-only responses: %s",
                exc,
            )

    def _raw_predict(self, row: pd.DataFrame) -> float:
        return float(self.model.predict(row[self.feature_cols])[0])

    def predict(self, row: pd.DataFrame) -> float:
        """RUL can't be negative; the model has no such constraint built in, so clip here."""
        return max(0.0, self._raw_predict(row))

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> dict:
        x = row[self.feature_cols]
        raw_pred = self._raw_predict(row)

        if self._explainer is None:
            return {
                "predicted_rul": max(0.0, raw_pred),
                "average_model_output": None,
                "top_contributors": [],
                "reconciliation": None,
                "shap_unavailable": True,
                "shap_error": self.shap_error,
            }

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
        except Exception as exc:
            logger.exception("SHAP explanation failed; falling back to prediction-only response")
            return {
                "predicted_rul": max(0.0, raw_pred),
                "average_model_output": None,
                "top_contributors": [],
                "reconciliation": None,
                "shap_unavailable": True,
                "shap_error": str(exc),
            }
