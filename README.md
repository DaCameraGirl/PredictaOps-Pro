<p align="center">
  <img src="docs/assets/predictaops-hero.svg" width="100%" alt="PredictaOps Pro monitoring a conveyor line and warning about bearing degradation before an unplanned line stop" />
</p>

<h1 align="center">PredictaOps Pro</h1>

<p align="center"><strong>Know what's breaking before it breaks.</strong></p>

<p align="center">
Machine health and downtime-prevention software for maintenance, reliability, warehouse, packaging, and manufacturing operations.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-95.9%25-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 95.9%" />
  <img src="https://img.shields.io/badge/HTML-3.9%25-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML 3.9%" />
  <img src="https://img.shields.io/badge/Other-0.2%25-475569?style=for-the-badge" alt="Other 0.2%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-7C3AED?style=flat-square" alt="SHAP Explainability" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square" alt="SQLAlchemy" />
  <img src="https://img.shields.io/badge/Alembic-Migrations-0F766E?style=flat-square" alt="Alembic" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions" />
</p>

## Keep the line moving

A failing bearing is rarely just a bearing problem. On a conveyor, packaging, processing, or production line, one machine failure can stop upstream and downstream work, leave operators waiting on repairs, delay orders, and put time-sensitive product at risk.

**PredictaOps Pro is built around one operational question: _what needs attention before it becomes downtime?_**

| Before the failure | Before the line stops | Before the disruption spreads |
| --- | --- | --- |
| Detect vibration and degradation changes | Estimate remaining useful life only when the model has enough evidence | Turn supported evidence into alerts, cases, inspections, and work orders |
| Preserve sensor and machine provenance | Abstain when the model is outside its validated domain | Keep human approval in the maintenance loop |
| Surface waveform, FFT, feature trends, and anomalies | Explain model evidence when SHAP is available | Preserve traceable maintenance outcomes and CMMS sync history |

> [!IMPORTANT]
> PredictaOps Pro does **not** invent confidence just to produce a number. When an asset is outside the validated RUL domain or there is not enough evidence, the serving contract returns `unsupported` or `insufficient_evidence` instead of presenting a weak prediction as fact.

### From sensor signal to maintenance action

```text
Machine / conveyor sensor
        ↓
Canonical ingestion + provenance
        ↓
Waveform + vibration analytics
        ↓
Degradation / anomaly evidence
        ↓
Supported RUL prediction OR explicit abstention
        ↓
Alert → maintenance case → inspection → work order → CMMS sync
```

### Built for the people who lose time when equipment stops

- **Maintenance teams** that need earlier evidence instead of another emergency repair.
- **Reliability engineers** who need traceable machine-health signals and defensible model behavior.
- **Warehouse and distribution operations** where conveyors, motors, bearings, and sortation equipment can become throughput bottlenecks.
- **Manufacturing and packaging lines** where one failed component can idle an entire section of production.
- **Operations leaders** who care less about an ML score than whether the line keeps moving safely and predictably.

The current validation foundation uses the NASA/IMS bearing test-to-failure experiments. The broader platform architecture is intentionally industrial and maps data through organization → site → asset → component → sensor rather than treating the lab dataset as the product boundary.

### Challenge context

Originally developed for **ABB Accelerator 2026 — Theme 1: Agentic Predictive Maintenance Studio**. PredictaOps Pro is an independent project and is not presented as an ABB-owned or ABB-endorsed product.

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

## Platform Core database

Production Slice 6 introduces the persistent platform registry underneath the existing IMS
studio. The current dashboard still reads the committed IMS feature/model
artifacts, while the new database layer provides the production identity model
that later ingestion, serving, security, and operations slices build on.

Local development defaults to SQLite at `data/platform_core.db` when
`PMS_DATABASE_URL` is unset. Production should set `PMS_DATABASE_URL` to a
PostgreSQL connection string, for example:

```text
PMS_DATABASE_URL=postgresql+psycopg://user:password@host:5432/predictive_maintenance
```

Run migrations and register the documented NASA/IMS runs as normal platform
entities:

```bash
alembic upgrade head
python scripts/bootstrap_platform.py
```

The bootstrap maps the existing dataset into the same hierarchy a plant uses:

```text
NASA/IMS Bearing Data Set -> IMS Bearing Test Rigs -> IMS Test 2 Machine -> Bearing 1 -> Sensor 1
```

Useful platform endpoints:

```text
GET  /api/platform/health
POST /api/platform/bootstrap/ims
GET  /api/platform/inventory
```

## Industrial ingestion

Production Slice 7 adds the canonical ingestion layer on top of the Platform Core
registry. CSV, Parquet, REST, MQTT, OPC-UA, ABB, and replay adapters all produce
the same internal ingestion contract before validation, unit normalization,
UTC timestamp normalization, sensor resolution, persistence, and receipt
generation.

Native connector classes in `src/industrial_ingestion/connectors.py` provide
MQTT broker subscription, OPC-UA data-change subscription/polling, and
credentialed ABB API fetch paths. The API routes below are HTTP bridge endpoints
for pushing source payloads into the same adapter pipeline.

Scalar readings persist to `machine_readings`. Waveforms are first-class
`waveform_records` with sample count, sampling rate, checksum, storage URI, and
provenance; waveform samples are landed outside `MachineReading.payload`.

Useful ingestion endpoints:

```text
POST /api/ingestion/sources
POST /api/ingestion/{organization_id}/rest
POST /api/ingestion/{organization_id}/mqtt
POST /api/ingestion/{organization_id}/opcua
POST /api/ingestion/{organization_id}/abb
POST /api/ingestion/{organization_id}/files/csv
POST /api/ingestion/{organization_id}/files/parquet
POST /api/ingestion/{organization_id}/replay/{batch_id}
GET  /api/ingestion/{organization_id}/health
```

## Analytics pipeline

Production Slice 8 computes deterministic analytics from canonical
`machine_readings` and `waveform_records`. It persists feature records with
source provenance, validates waveform integrity before reading samples, computes
time-domain and FFT features, scores baseline anomalies and degradation trends,
and stores evidence-backed sensor health states. This layer does not train,
register, promote, or serve ML models; that remains Production Slice 9.

Useful analytics endpoints:

```text
POST /api/analytics/{organization_id}/batches/{batch_id}/compute
POST /api/analytics/{organization_id}/sensors/{sensor_id}/recompute
GET  /api/analytics/{organization_id}/health
```

## ML platform

Production Slice 9 adds the model-development control plane. Dataset versions
are immutable snapshots built from canonical analytics feature records; experiment
runs capture reproducible training config, code version, validation method,
metrics, baseline comparison, uncertainty evidence, and abstention policy. Model
registries track immutable model versions, promotion stages, explicit human
approval before production promotion, and rollback events.

This slice does **not** perform live per-asset model resolution, drift monitoring,
or retraining triggers. Those are Production Slice 10 responsibilities.

Useful ML platform endpoints:

```text
POST /api/ml/{organization_id}/dataset-versions
GET  /api/ml/{organization_id}/dataset-versions
POST /api/ml/{organization_id}/experiments
GET  /api/ml/{organization_id}/experiments
GET  /api/ml/{organization_id}/experiments/{experiment_run_id}
POST /api/ml/{organization_id}/registries
GET  /api/ml/{organization_id}/registries
POST /api/ml/{organization_id}/model-versions
POST /api/ml/{organization_id}/model-versions/{model_version_id}/promote
POST /api/ml/{organization_id}/registries/{registry_id}/rollback
```

## Production serving

Production Slice 10 turns approved registry models into live inference contracts.
Serving bindings attach approved production model versions to organization, site,
asset, component, or sensor scopes. Prediction requests resolve the most specific
active binding for a sensor, verify the model artifact SHA-256 before loading,
construct feature vectors from canonical analytics records, enforce schema and
training-domain compatibility, and persist both the model-resolution decision and
prediction evidence.

Supported RUL predictions are returned only when the platform can prove the
approved model version, dataset snapshot, feature values, artifact checksum, and
abstention policy that produced the result. Otherwise the API persists and
returns `unsupported` or `insufficient_evidence` with the known evidence. Drift
and data-quality monitors create retraining triggers, but replacement production
models still require the Slice 9 human approval flow.

Useful production-serving endpoints:

```text
POST /api/serving/{organization_id}/bindings
POST /api/serving/{organization_id}/predict/rul
GET  /api/serving/{organization_id}/predictions
GET  /api/serving/{organization_id}/health
```

## Maintenance operations

Production Slice 11 converts persisted machine evidence into a traceable human
workflow. Prediction-driven alert evaluation stores the original serving
evidence snapshot, applies an explicit caller-supplied maintenance rule, and
keeps unsupported or insufficient-evidence predictions on a human-review path
without inventing RUL claims.

Alerts can be acknowledged, resolved, or dismissed by active organization
members. Alerts can open first-class maintenance cases with source hierarchy
preserved from the alert; manual cases require an active human opener and valid
asset/component/sensor ancestry. Case resolution must use the dedicated
resolution endpoint so outcome, summary, and resolver evidence are persisted
before a resolved case can be closed. Cases support append-only human-authored
technician notes, requested inspections that cannot override the case hierarchy
without an explicit future reassignment path and require human findings before
completion, and local work orders that begin as drafts. Work orders require
explicit human approval before work starts and explicit completion details before
completion.

CMMS synchronization is an explicit active-member action. The default production
adapter truthfully returns `not_configured` and never fabricates an external ID.
A deterministic test adapter exercises successful sync and idempotent retry
behavior without adding vendor credentials or claiming a vendor integration.
Create idempotency is scoped by provider, and a work order bound to one
successful external provider cannot silently switch providers. Later CMMS
operations without an explicit provider automatically use the bound provider.
External create success requires a non-empty external ID; update, cancel, and
close preserve the existing bound external ID when the adapter does not echo it
and fail closed if the adapter returns a different ID. Adapter timeouts and
runtime failures are persisted as `timeout` or `failed` sync records with
client-safe error details and without mutating any existing CMMS provider or
external ID binding.

Useful maintenance endpoints:

```text
POST /api/maintenance/{organization_id}/alerts/evaluate-prediction
GET  /api/maintenance/{organization_id}/alerts
GET  /api/maintenance/{organization_id}/alerts/{alert_id}
POST /api/maintenance/{organization_id}/alerts/{alert_id}/acknowledge
POST /api/maintenance/{organization_id}/alerts/{alert_id}/resolve
POST /api/maintenance/{organization_id}/alerts/{alert_id}/case
POST /api/maintenance/{organization_id}/cases
GET  /api/maintenance/{organization_id}/cases
GET  /api/maintenance/{organization_id}/cases/{case_id}
POST /api/maintenance/{organization_id}/cases/{case_id}/transition
POST /api/maintenance/{organization_id}/cases/{case_id}/notes
GET  /api/maintenance/{organization_id}/cases/{case_id}/notes
POST /api/maintenance/{organization_id}/cases/{case_id}/inspections
POST /api/maintenance/{organization_id}/inspections/{inspection_id}/start
POST /api/maintenance/{organization_id}/inspections/{inspection_id}/complete
POST /api/maintenance/{organization_id}/inspections/{inspection_id}/cancel
POST /api/maintenance/{organization_id}/cases/{case_id}/work-orders
POST /api/maintenance/{organization_id}/work-orders/{work_order_id}/approve
POST /api/maintenance/{organization_id}/work-orders/{work_order_id}/start
POST /api/maintenance/{organization_id}/work-orders/{work_order_id}/complete
POST /api/maintenance/{organization_id}/work-orders/{work_order_id}/cancel
POST /api/maintenance/{organization_id}/work-orders/{work_order_id}/cmms-sync
GET  /api/maintenance/{organization_id}/work-orders/{work_order_id}/cmms-sync
POST /api/maintenance/{organization_id}/cases/{case_id}/resolve
GET  /api/maintenance/{organization_id}/health
```

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

Pull requests run Ruff, the full pytest suite including browser E2E coverage, a
PostgreSQL migration/platform-stack job through Maintenance Operations, and a
Docker build in GitHub Actions.

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