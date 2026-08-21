# Predictive Maintenance Studio

ABB Accelerator 2026 — Theme 1: Agentic Predictive Maintenance Studio.

Predicts remaining useful life (RUL) for a set of real motor bearings from
the NASA/IMS bearing run-to-failure vibration dataset, explains each
prediction with SHAP, and serves both through a dashboard.

## What it does

- **Feature extraction** (`src/bearing_data.py`): reads the raw 20kHz
  accelerometer snapshots (one per bearing, every 10 minutes, for ~7 days)
  and reduces each to standard vibration-analysis statistics (RMS,
  kurtosis, skew, peak-to-peak, crest factor).
- **Profiling** (`src/bearing_profiling.py`): reports which of those
  features correlate most with the real recorded degradation.
- **Model** (`src/train_bearing.py`): XGBoost regressor trained on bearing
  1, the one bearing that actually failed (outer race defect) during the
  test. RUL is clipped at 400 snapshots since the vibration signal stays
  flat for roughly the first 600 snapshots of this bearing's life. Because
  there's only one real failure trajectory (not 100 independent engines
  like a simulated dataset would give you), validation holds out a random
  slice of snapshots from across that trajectory rather than a held-out
  unit — see the comment in `train_bearing.py` for why.
- **Explainability** (`src/explain_bearing.py`): SHAP TreeExplainer,
  returns the top vibration features driving each bearing's prediction.
- **API + dashboard** (`app/`): FastAPI backend, single-page dashboard
  with a time slider over the ~7-day test, a live 4-bearing risk list, a
  per-bearing SHAP breakdown, and a predicted-vs-actual RUL trend chart
  for bearing 1's whole life.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python src/train_bearing.py   # extracts features (first run only), trains model
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Data

The trained model (`models/bearing_rul_model.joblib`) and the extracted
feature table (`data/ims_test2_features.csv`) are committed, so the app
runs without needing the raw data. If you want to regenerate features
from scratch, the raw dataset (~525MB, 984 files) is the NASA/IMS Test 2
bearing set, mirrored at
https://github.com/RicardoPSLopes/IMS-DATASET — clone it and copy its
`data/` folder into this project's `data/ims_test2/`, then delete
`data/ims_test2_features.csv` and rerun `train_bearing.py`.
