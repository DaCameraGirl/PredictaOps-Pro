"""FastAPI backend serving bearing RUL predictions, SHAP explanations, and dataset profiling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from bearing_data import BEARING_COLS, FAILED_BEARING, FAILURE_MODE
from bearing_profiling import summarize
from explain_bearing import BearingRulExplainer

DATA_CACHE = Path(__file__).resolve().parent.parent / "data" / "ims_test2_features.csv"

app = FastAPI(title="Predictive Maintenance Studio")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_explainer = BearingRulExplainer()
_table = pd.read_csv(DATA_CACHE, parse_dates=["timestamp"])
_timestamps = sorted(_table["timestamp"].unique())

_failed = _table[_table["bearing"] == FAILED_BEARING].sort_values("timestamp").reset_index(drop=True)
_failed["true_rul"] = len(_failed) - 1 - _failed.index


def _row_at(bearing: str, index: int) -> pd.DataFrame:
    if index < 0 or index >= len(_timestamps):
        raise HTTPException(status_code=404, detail="snapshot index out of range")
    ts = _timestamps[index]
    match = _table[(_table["bearing"] == bearing) & (_table["timestamp"] == ts)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"bearing {bearing} not found")
    return match.iloc[[0]]


def _risk(predicted_rul: float) -> str:
    return "high" if predicted_rul < 50 else "medium" if predicted_rul < 200 else "low"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/profile")
def profile():
    return summarize(_table)


@app.get("/api/timeline")
def timeline():
    return {
        "n_snapshots": len(_timestamps),
        "timestamps": [ts.isoformat() for ts in _timestamps],
        "default_index": len(_timestamps) - 50,
    }


@app.get("/api/snapshot/{index}")
def snapshot(index: int):
    ts = _timestamps[index] if 0 <= index < len(_timestamps) else None
    if ts is None:
        raise HTTPException(status_code=404, detail="snapshot index out of range")

    bearings = []
    for bearing in BEARING_COLS:
        row = _row_at(bearing, index)
        predicted = _explainer.predict(row)
        entry = {
            "bearing": bearing,
            "predicted_rul": round(predicted, 1),
            "risk": _risk(predicted),
            "has_ground_truth": bearing == FAILED_BEARING,
        }
        if bearing == FAILED_BEARING:
            true_row = _failed[_failed["timestamp"] == ts]
            if not true_row.empty:
                entry["true_rul"] = int(true_row.iloc[0]["true_rul"])
        bearings.append(entry)

    return {"index": index, "timestamp": ts.isoformat(), "bearings": bearings}


@app.get("/api/snapshot/{index}/bearing/{bearing_id}")
def bearing_detail(index: int, bearing_id: str):
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    row = _row_at(bearing_id, index)
    explanation = _explainer.explain(row)
    return {
        "bearing": bearing_id,
        "timestamp": _timestamps[index].isoformat(),
        "failure_mode": FAILURE_MODE if bearing_id == FAILED_BEARING else None,
        **explanation,
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


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
