# Predictive Maintenance Studio

ABB Accelerator 2026 — Theme 1: Agentic Predictive Maintenance Studio.

Predicts remaining useful life (RUL) for a set of real motor bearings from
the NASA/IMS bearing run-to-failure vibration dataset, explains each
prediction with SHAP, and serves both through a dashboard.

The target is the full production Predictive Maintenance Studio, built through
reviewable production slices rather than one untestable mega-change. See
[`ROADMAP.md`](ROADMAP.md) for the platform target and PR sequence.

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
  flat for roughly the first 600 snapshots of this bearing's life.
  Validated with a chronological, expanding-window walk-forward backtest
  (train only on the past, predict the next block, roll forward) since
  there's only one real failure trajectory to learn from and a random
  split across it would leak near-future shape into training. Reports
  MAE/RMSE in snapshots and hours, plus an asymmetric score that
  penalizes late (over-optimistic) predictions harder than early ones.
  The model actually served by the app is refit on the full trajectory
  afterward, same as a real deployment would use all history collected
  so far — see `train_bearing.py` for the reasoning.
- **Degradation signal** (`src/degradation_signal.py`): a separate,
  simpler statistical check (deviation from each bearing's own healthy
  baseline) for "is this degrading at all," kept independent from the
  RUL regression's point estimate, since the two claims have very
  different reliability.
- **Model abstention** (`app/main.py`): RUL predictions are only emitted
  when the asset is inside the validated RUL domain. For right-censored
  bearings or snapshots with insufficient baseline history, the API says
  `unsupported` or `insufficient_evidence`, reports what evidence is
  known, and demotes the raw model number to diagnostic context.
- **Explainability** (`src/explain_bearing.py`): SHAP TreeExplainer,
  returns the top vibration features driving each bearing's prediction.
- **API + dashboard** (`app/`): FastAPI backend, single-page dashboard
  with a time slider over the ~7-day test, a live 4-bearing risk list
  with 80% prediction intervals, a per-bearing SHAP breakdown, and a
  time-ordered predicted-vs-actual RUL trend chart for bearing 1's whole
  life. Bearings 2-4 never failed during the test, so their estimates are
  explicitly labeled as extrapolated and not independently verifiable,
  never presented as known values.

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
