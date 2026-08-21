# Predictive Maintenance Studio

ABB Accelerator 2026 — Theme 1: Agentic Predictive Maintenance Studio.

Predicts remaining useful life (RUL) for a fleet of turbofan engines from
the NASA C-MAPSS FD001 dataset, explains each prediction with SHAP, and
serves both through a dashboard.

## What it does

- **Profiling** (`src/profiling.py`): scans the raw sensor data, flags
  constant/no-signal sensors, and reports which sensors correlate most
  with degradation.
- **Feature + label engineering** (`src/features.py`): drops dead
  sensors, computes a clipped RUL label per training row.
- **Model** (`src/train.py`): XGBoost regressor, evaluated with RMSE and
  the PHM08 competition scoring function on the official held-out test
  set.
- **Explainability** (`src/explain.py`): SHAP TreeExplainer, returns the
  top sensors driving each engine's prediction and in which direction.
- **API + dashboard** (`app/`): FastAPI backend, single-page dashboard
  showing the fleet sorted by urgency and a per-engine SHAP breakdown.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python src/train.py           # trains model, writes models/rul_model.joblib
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Data

`data/` contains the NASA C-MAPSS FD001 subset (100 train engines, 100
test engines, single operating condition, single fault mode), sourced
from the public PCoE prognostics data repository.
