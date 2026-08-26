# Predictive Maintenance Studio

ABB Accelerator 2026 — Theme 1: Agentic Predictive Maintenance Studio.

A vibration-based predictive-maintenance studio built on the NASA/IMS bearing test-to-failure experiments. The current served model predicts remaining useful life (RUL) on IMS Test 2, explains predictions with SHAP when available, and exposes the evidence through a FastAPI dashboard. The data layer now also has verified run definitions for IMS Tests 1 and 3 so the project can move to leakage-safe cross-run validation without inventing failure labels.

## Current technical status

The project deliberately separates what is **served today** from what is **prepared for the next validation stage**:

- **IMS Test 2 — served model:** 984 snapshots, 4 channels, one sensor per bearing. Bearing 1 has the documented outer-race failure at the experiment endpoint. The existing XGBoost RUL model and dashboard remain based on this run.
- **IMS Test 1 — data foundation:** 2,156 snapshots, 8 channels, two sensors per bearing. Bearing 3 has a documented inner-race defect and bearing 4 a rolling-element defect at the experiment endpoint. The first 43 recordings include 5-minute intervals; later recordings are 10 minutes apart.
- **IMS Test 3 — data foundation:** 4,448 snapshots, 4 channels, one sensor per bearing. Bearing 3 has the documented outer-race failure at the experiment endpoint.

Tests 1 and 3 are **not yet used to train the served RUL model**. That is intentional. The next modeling stage is a separate cross-run validator that holds out entire test runs/trajectories instead of mixing neighboring time-series rows across train and test.

The target is the full production Predictive Maintenance Studio, built through
reviewable production slices rather than one untestable mega-change. See
[`ROADMAP.md`](ROADMAP.md) for the platform target and PR sequence.

## What it does

- **Run-aware data contracts** (`src/bearing_data.py`): immutable metadata for the three documented IMS test runs, including channel-to-bearing mapping, sensor identity, structural validation, run-specific cache paths, and documented failure endpoints/modes.
- **Feature extraction** (`src/bearing_data.py`): converts 20 kHz accelerometer snapshots into vibration statistics including RMS, standard deviation, kurtosis, skew, peak-to-peak, and crest factor. Multi-sensor Test 1 data retains sensor identity instead of pretending each sensor is a separate bearing.
- **Profiling** (`src/bearing_profiling.py`): reports which vibration features correlate most strongly with the recorded degradation.
- **Current RUL model** (`src/train_bearing.py`): XGBoost trained only on the documented Test 2 failure trajectory. Validation is chronological expanding-window walk-forward rather than a random split. The current Test 2 target remains clipped at 400 snapshots for the existing model; that assumption is not applied to Tests 1 or 3.
- **Leakage guards:** the current single-trajectory trainer only exposes `ims_test2`. Newly registered runs can be validated and feature-extracted, but cannot accidentally flow through the old trainer before cross-run validation exists.
- **Degradation signal** (`src/degradation_signal.py`): an independent baseline-deviation signal for whether a bearing appears to be degrading, separate from the RUL point estimate.
- **Model abstention** (`app/main.py`): RUL predictions are only emitted when the asset is inside the validated RUL domain. For right-censored bearings or snapshots with insufficient baseline history, the API says `unsupported` or `insufficient_evidence`, reports what evidence is known, and demotes the raw model number to diagnostic context.
- **Explainability** (`src/explain_bearing.py`): SHAP TreeExplainer when SHAP is available. If SHAP import/initialization fails, prediction serving stays alive and reports the explanation as unavailable rather than crashing the API.
- **API + dashboard** (`app/`): FastAPI backend and single-page studio with playback, bearing risk/health state, waveform and FFT views, feature trends, anomaly/spike context, maintenance decision support, RUL history, exports, explicit abstention, and explicit labeling of censored bearings whose true failure time is unknown.

## Run the existing studio

The committed Test 2 feature cache and trained model are enough to run the app; raw vibration files are not required for normal use.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Rebuild the current Test 2 model

The existing downloader is intentionally Test-2-only:

```bash
python scripts/download_data.py
python src/train_bearing.py --run ims_test2
```

Raw Test 2 files go under `data/raw/ims_test2/`. The feature cache is `data/processed/ims_test2_features.csv`, and the existing dashboard/model artifacts remain backward-compatible.

## Prepare IMS Test 1 or Test 3

Tests 1 and 3 have verified metadata and independent cache paths, but the repository does not automatically download the much larger full IMS archive. After obtaining the original NASA/IMS data and placing a run's timestamp-named snapshot files in the expected directory, prepare it with:

```bash
python scripts/prepare_bearing_run.py --run ims_test1
python scripts/prepare_bearing_run.py --run ims_test3
```

Expected raw directories:

```text
data/raw/ims_test1/
data/raw/ims_test2/
data/raw/ims_test3/
```

Prepared outputs are independent:

```text
data/processed/ims_test1_features.csv
data/processed/ims_test2_features.csv
data/processed/ims_test3_features.csv
```

The preparation command validates snapshot count/shape/timestamps, preserves Test 1 sensor identity, writes run metadata/checksums, and does **not** train a model.

## Dataset provenance and RUL labels

Authoritative source: NASA Prognostics Center of Excellence, **Bearing Data Set**, collected by the IMS Center at the University of Cincinnati with support from Rexnord Corp.

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

For supervised RUL labeling, the final documented snapshot of a failed-bearing experiment is used as the **RUL = 0 label endpoint**. This does not claim that the exact physical instant of fault onset is known. Bearings without a documented failure in a run remain right-censored and are not assigned fabricated failure labels.

## SHAP on Windows

SHAP is attempted normally on every supported platform. If the local SHAP/Numba/LLVM stack cannot initialize, the app falls back to prediction-only explanations and reports why SHAP is unavailable. To explicitly disable SHAP:

```powershell
$env:PMS_DISABLE_SHAP = "1"
uvicorn app.main:app --reload
```

## Validation

Pull requests run Ruff, the full pytest suite including browser E2E coverage, and a Docker build in GitHub Actions.

```bash
python -m ruff check src app tests scripts
python -m pytest tests -v
```

## What the current results do — and do not — prove

The Test 2 studio is useful for demonstrating a complete vibration-to-maintenance workflow, but one failed bearing is not enough to claim broad industrial RUL generalization. Bearings 2–4 in Test 2 are right-censored, so their RUL estimates are extrapolations rather than independently verifiable ground truth.

The newly registered Test 1 and Test 3 failures provide the foundation for the next, more defensible question: **can a model trained on complete failure trajectories generalize to an entirely held-out run?** Until that validator is implemented and real run caches are prepared, the dashboard should not present cross-run performance claims.

## Path to production

1. **Multi-run data foundation — in progress:** Test 1/Test 3 metadata, channel layouts, failure endpoints, per-run validation, and independent feature caches are now supported.
2. **Cross-run validation — next:** train on one or more complete failure runs and hold out an entire run/trajectory; never random-split neighboring time-series rows.
3. **Uncertainty:** replace the current global residual interval with calibration based on genuinely independent trajectories once enough calibration data exists.
4. **Dashboard provenance:** show current run, validation mode, training runs, held-out run, failed vs. censored assets, and model limitations directly in the UI.
5. **Streaming / monitoring:** add ingestion, model monitoring, and retraining triggers only after the validation story is credible.
6. **Narrow maintenance assistant:** only then add an evidence-citing agent that summarizes model signals and uncertainty without inventing risk, cost, or maintenance facts.

That ordering is intentional: credible data and validation first, agentic features second.
