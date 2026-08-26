"""FastAPI backend serving bearing RUL predictions, SHAP explanations, vibration
analysis, health-state classification, and maintenance decision support."""
import csv
import io
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from bearing_data import (
    BEARING_COLS,
    DEFAULT_RUN,
    FAILED_BEARING,
    FAILURE_MODE,
    FEATURE_NAMES,
    METADATA_PATH,
    load_feature_table,
)
from bearing_profiling import feature_trend, summarize
from degradation_signal import DegradationSignal
from explain_bearing import BearingRulExplainer
from vibration_analysis import analyze
from waveform_cache import WaveformCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="Predictive Maintenance Studio")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_explainer = BearingRulExplainer()
_table = load_feature_table(DEFAULT_RUN)
_timestamps = sorted(_table["timestamp"].unique())
_degradation = DegradationSignal(_table)
_metrics = json.loads((MODEL_DIR / "bearing_metrics.json").read_text())
_model_metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text())
_dataset_metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None
try:
    _waveforms = WaveformCache()
except FileNotFoundError:
    _waveforms = None
    logger.warning("waveform cache not found; /api/waveform will 503")

_failed = _table[_table["bearing"] == FAILED_BEARING].sort_values("timestamp").reset_index(drop=True)
_failed["true_rul"] = len(_failed) - 1 - _failed.index

ACTION_LABELS = {
    "continue_monitoring": "Continue monitoring",
    "increase_inspection_frequency": "Increase inspection frequency",
    "schedule_inspection": "Schedule inspection",
    "prepare_planned_maintenance": "Prepare planned maintenance",
    "immediate_human_review": "Immediate human review",
}
HUMAN_VERIFICATION_NOTICE = (
    "This recommendation is generated from transparent threshold rules over model "
    "output and requires human verification before any maintenance action is taken."
)


def _row_at(bearing: str, index: int) -> pd.DataFrame:
    if index < 0 or index >= len(_timestamps):
        raise HTTPException(status_code=404, detail="snapshot index out of range")
    ts = _timestamps[index]
    match = _table[(_table["bearing"] == bearing) & (_table["timestamp"] == ts)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"bearing {bearing} not found")
    return match.iloc[[0]]


def _interval(predicted_rul: float) -> dict:
    """80% interval built from walk-forward backtest residuals: true = pred - residual.
    This is a global range derived from historical residuals, not a per-prediction,
    conditionally calibrated interval."""
    lo = max(0.0, predicted_rul - _metrics["interval_80_residual_high"])
    hi = max(0.0, predicted_rul - _metrics["interval_80_residual_low"])
    return {"low": round(lo, 1), "high": round(hi, 1), "note": _metrics["interval_note"]}


def _true_rul_at(bearing: str, ts) -> int | None:
    if bearing != FAILED_BEARING:
        return None
    true_row = _failed[_failed["timestamp"] == ts]
    return int(true_row.iloc[0]["true_rul"]) if not true_row.empty else None


def _recommend_action(health_state: str, interval_high: float) -> dict:
    if health_state in ("insufficient_evidence", "healthy"):
        code = "continue_monitoring"
        evidence = "No sustained deviation from this bearing's healthy baseline."
    elif health_state == "watch":
        code = "increase_inspection_frequency"
        evidence = "Deviation detected but below the warning threshold; watch for persistence."
    elif health_state == "warning":
        code = "schedule_inspection"
        evidence = "Sustained deviation from the healthy baseline crossed the warning threshold."
    elif interval_high is not None and interval_high < 24:
        code = "immediate_human_review"
        evidence = "Critical deviation, and the model's own uncertainty range suggests very little time may remain."
    else:
        code = "prepare_planned_maintenance"
        evidence = "Sustained deviation crossed the critical threshold."
    return {
        "action": code,
        "action_label": ACTION_LABELS[code],
        "evidence": evidence,
        "requires_human_verification": True,
        "disclaimer": HUMAN_VERIFICATION_NOTICE,
    }


def _prediction_payload(bearing: str, row: pd.DataFrame, ts) -> dict:
    predicted = _explainer.predict(row)
    degradation = _degradation.evaluate(bearing, ts)
    true_rul = _true_rul_at(bearing, ts)
    interval = _interval(predicted)
    payload = {
        "bearing": bearing,
        "predicted_rul": round(predicted, 1),
        "interval_80": interval,
        "has_ground_truth": bearing == FAILED_BEARING,
        **degradation,
        "recommendation": _recommend_action(degradation["health_state"], interval["high"]),
    }
    if true_rul is not None:
        payload["true_rul"] = true_rul
    return payload


@app.get("/api/health")
def health():
    return {"status": "ok", "n_snapshots": len(_timestamps), "model_loaded": _explainer.model is not None}


@app.get("/api/profile")
def profile():
    return {
        **summarize(_table),
        "model": _metrics,
        "model_metadata": _model_metadata,
        "dataset_metadata": _dataset_metadata,
    }


@app.get("/api/timeline")
def timeline():
    return {
        "n_snapshots": len(_timestamps),
        "timestamps": [ts.isoformat() for ts in _timestamps],
        "default_index": len(_timestamps) - 50,
        "snapshot_minutes": _metrics["snapshot_minutes"],
    }


@app.get("/api/snapshot/{index}")
def snapshot(index: int):
    ts = _timestamps[index] if 0 <= index < len(_timestamps) else None
    if ts is None:
        raise HTTPException(status_code=404, detail="snapshot index out of range")

    bearings = [_prediction_payload(bearing, _row_at(bearing, index), ts) for bearing in BEARING_COLS]
    return {"index": index, "timestamp": ts.isoformat(), "bearings": bearings}


@app.get("/api/snapshot/{index}/bearing/{bearing_id}")
def bearing_detail(index: int, bearing_id: str):
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    row = _row_at(bearing_id, index)
    explanation = _explainer.explain(row)
    payload = _prediction_payload(bearing_id, row, _timestamps[index])
    return {
        "timestamp": _timestamps[index].isoformat(),
        "failure_mode": FAILURE_MODE if bearing_id == FAILED_BEARING else None,
        "event_timeline": _degradation.timeline(bearing_id),
        **explanation,
        **payload,
    }


@app.get("/api/bearing1-trend")
def bearing1_trend():
    """Predicted vs. real RUL across bearing 1's whole life, sampled for the chart."""
    step = max(1, len(_failed) // 200)
    sampled = _failed.iloc[::step]
    points = []
    for _, row in sampled.iterrows():
        row_df = _table[(_table["bearing"] == FAILED_BEARING) & (_table["timestamp"] == row["timestamp"])].iloc[[0]]
        predicted = _explainer.predict(row_df)
        points.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "true_rul": int(row["true_rul"]),
                "predicted_rul": round(predicted, 1),
            }
        )
    return points


@app.get("/api/feature-trend/{bearing_id}")
def feature_trend_endpoint(bearing_id: str):
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    return {
        "bearing": bearing_id,
        "baseline_snapshots": 100,
        "points": feature_trend(_table, bearing_id),
    }


@app.get("/api/waveform/{index}/bearing/{bearing_id}")
def waveform(index: int, bearing_id: str):
    if _waveforms is None:
        raise HTTPException(status_code=503, detail="waveform cache not available on this deployment")
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    if index < 0 or index >= len(_timestamps):
        raise HTTPException(status_code=404, detail="snapshot index out of range")
    ts_iso = _timestamps[index].isoformat()
    return analyze(_waveforms, bearing_id, ts_iso)


@app.get("/api/export/trajectory/{bearing_id}.csv")
def export_trajectory_csv(bearing_id: str):
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    sub = _table[_table["bearing"] == bearing_id].sort_values("timestamp")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", *FEATURE_NAMES])
    for _, row in sub.iterrows():
        writer.writerow([row["timestamp"].isoformat(), *[row[f] for f in FEATURE_NAMES]])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={bearing_id}_trajectory.csv"},
    )


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
