"""FastAPI backend serving RUL predictions, SHAP explanations, and dataset profiling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data_loader import load_test, load_train
from explain import RulExplainer
from features import last_cycle_per_unit
from profiling import summarize

app = FastAPI(title="Predictive Maintenance Studio")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_explainer = RulExplainer()
_test_last = last_cycle_per_unit(load_test())
_train_summary = None


def _row_for_unit(unit_id: int):
    match = _test_last[_test_last.unit == unit_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"unit {unit_id} not found")
    return match.iloc[[0]]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/profile")
def profile():
    global _train_summary
    if _train_summary is None:
        summary = summarize(load_train())
        _train_summary = {
            "n_rows": summary["n_rows"],
            "n_units": summary["n_units"],
            "constant_sensors": summary["constant_sensors"],
            "lifetime_cycles": summary["lifetime_cycles"],
            "top_degradation_sensors": summary["top_degradation_sensors"],
        }
    return _train_summary


@app.get("/api/engines")
def list_engines():
    rows = []
    for unit_id in sorted(_test_last["unit"].unique()):
        row = _row_for_unit(int(unit_id))
        predicted_rul = _explainer.predict(row)
        rows.append(
            {
                "unit": int(unit_id),
                "cycle": int(row.iloc[0]["cycle"]),
                "predicted_rul": round(predicted_rul, 1),
                "risk": "high" if predicted_rul < 30 else "medium" if predicted_rul < 75 else "low",
            }
        )
    return rows


@app.get("/api/engines/{unit_id}")
def engine_detail(unit_id: int):
    row = _row_for_unit(unit_id)
    explanation = _explainer.explain(row)
    return {
        "unit": unit_id,
        "cycle": int(row.iloc[0]["cycle"]),
        **explanation,
    }


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
