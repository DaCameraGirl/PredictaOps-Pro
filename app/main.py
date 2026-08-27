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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from analytics_pipeline.service import AnalyticsService
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
from industrial_ingestion.contracts import SourceRegistration
from industrial_ingestion.service import IngestionService
from ml_platform.contracts import (
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    PromoteModelVersion,
    RegistryCreate,
    RollbackModelVersion,
)
from ml_platform.service import MLPlatformService
from platform_core.config import database_settings, safe_database_label
from platform_core.database import SessionLocal, check_database
from platform_core.models import Base
from platform_core.services import PlatformService, get_platform_inventory
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
VALIDATED_RUL_DOMAIN = (
    "IMS Test 2 bearing 1, documented outer-race defect, same rig, sensors, "
    "sampling cadence, and extracted vibration feature schema."
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


def _rul_support(bearing: str, degradation: dict) -> dict:
    if degradation["health_state"] == "insufficient_evidence":
        return {
            "rul_prediction_supported": False,
            "prediction_status": "insufficient_evidence",
            "prediction_status_label": "Insufficient evidence",
            "abstention_reason": (
                "This asset has not accumulated enough baseline history for a "
                "defensible remaining-life claim. The degradation signal reports "
                "what is known so far instead."
            ),
            "validated_domain": VALIDATED_RUL_DOMAIN,
            "known_evidence": [
                "RUL model artifact loaded successfully.",
                "Healthy-baseline window is still being established.",
                "No supported remaining-life claim is emitted for this snapshot.",
            ],
        }
    if bearing != FAILED_BEARING:
        return {
            "rul_prediction_supported": False,
            "prediction_status": "unsupported",
            "prediction_status_label": "Unsupported",
            "abstention_reason": (
                "No failure was observed for this bearing in the source run, so "
                "there is no validated remaining-life label for this asset. The "
                "system can report degradation evidence, but it must abstain from "
                "claiming a verified RUL prediction."
            ),
            "validated_domain": VALIDATED_RUL_DOMAIN,
            "known_evidence": [
                "Same IMS Test 2 rig and sensor cadence as the trained trajectory.",
                "This bearing was right-censored: it was still running when recording ended.",
                "The degradation state is computed from this bearing's own baseline.",
            ],
        }
    return {
        "rul_prediction_supported": True,
        "prediction_status": "supported",
        "prediction_status_label": "Supported",
        "abstention_reason": None,
        "validated_domain": VALIDATED_RUL_DOMAIN,
        "known_evidence": [
            "Documented failure trajectory with ground-truth RUL labels.",
            "Failure mode matches the model's validated domain.",
            "Prediction interval is derived from chronological walk-forward residuals.",
        ],
    }


def _prediction_payload(bearing: str, row: pd.DataFrame, ts) -> dict:
    diagnostic_output = _explainer.predict(row)
    degradation = _degradation.evaluate(bearing, ts)
    true_rul = _true_rul_at(bearing, ts)
    support = _rul_support(bearing, degradation)
    interval = _interval(diagnostic_output) if support["rul_prediction_supported"] else None
    payload = {
        "bearing": bearing,
        "predicted_rul": round(diagnostic_output, 1) if support["rul_prediction_supported"] else None,
        "diagnostic_model_output_rul": round(diagnostic_output, 1),
        "interval_80": interval,
        "has_ground_truth": bearing == FAILED_BEARING,
        **support,
        **degradation,
        "recommendation": _recommend_action(
            degradation["health_state"],
            interval["high"] if interval is not None else None,
        ),
    }
    if true_rul is not None:
        payload["true_rul"] = true_rul
    return payload


@app.get("/api/health")
def health():
    return {"status": "ok", "n_snapshots": len(_timestamps), "model_loaded": _explainer.model is not None}


@app.get("/api/platform/health")
def platform_health():
    settings = database_settings()
    database_label = safe_database_label(settings.url)
    with SessionLocal() as session:
        try:
            check_database(session)
            existing_tables = set(inspect(session.bind).get_table_names())
            expected_tables = set(Base.metadata.tables)
            missing_tables = sorted(expected_tables.difference(existing_tables))
        except SQLAlchemyError as exc:
            return {
                "status": "unhealthy",
                "database_url": database_label,
                "migrated": False,
                "error": str(exc),
            }
        return {
            "status": "ok" if not missing_tables else "unmigrated",
            "database_url": database_label,
            "migrated": not missing_tables,
            "missing_tables": missing_tables,
        }


@app.post("/api/platform/bootstrap/ims")
def bootstrap_platform_ims():
    with SessionLocal() as session:
        try:
            summary = PlatformService(session).bootstrap_ims_registry()
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return summary.model_dump()


@app.get("/api/platform/inventory")
def platform_inventory():
    with SessionLocal() as session:
        try:
            return get_platform_inventory(session)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ingestion/sources")
def register_ingestion_source(registration: SourceRegistration):
    with SessionLocal() as session:
        try:
            source = IngestionService(session).register_source(registration)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {
            "id": source.id,
            "organization_id": source.organization_id,
            "name": source.name,
            "source_type": source.source_type,
            "status": source.status,
        }


def _ingest_payload(organization_id: str, source_type: str, payload, source_name: str, **options):
    with SessionLocal() as session:
        try:
            receipt = IngestionService(session).ingest(
                organization_id,
                source_type=source_type,
                payload=payload,
                source_name=source_name,
                **options,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return receipt.model_dump()


async def _json_body(request: Request):
    try:
        return await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON request body") from exc


@app.post("/api/ingestion/{organization_id}/rest")
async def ingest_rest(organization_id: str, request: Request, source_name: str = "REST Push"):
    return _ingest_payload(organization_id, "rest", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/mqtt")
async def ingest_mqtt(
    organization_id: str,
    request: Request,
    source_name: str = "MQTT Bridge",
    topic: str | None = None,
):
    return _ingest_payload(organization_id, "mqtt", await request.body(), source_name, topic=topic)


@app.post("/api/ingestion/{organization_id}/opcua")
async def ingest_opcua(organization_id: str, request: Request, source_name: str = "OPC-UA Bridge"):
    return _ingest_payload(organization_id, "opcua", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/abb")
async def ingest_abb(organization_id: str, request: Request, source_name: str = "ABB Adapter"):
    return _ingest_payload(organization_id, "abb", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/files/csv")
async def ingest_csv_file(
    organization_id: str,
    request: Request,
    source_name: str = "CSV File",
    source_uri: str | None = None,
):
    return _ingest_payload(organization_id, "csv", await request.body(), source_name, source_uri=source_uri)


@app.post("/api/ingestion/{organization_id}/files/parquet")
async def ingest_parquet_file(
    organization_id: str,
    request: Request,
    source_name: str = "Parquet File",
    source_uri: str | None = None,
):
    return _ingest_payload(organization_id, "parquet", await request.body(), source_name, source_uri=source_uri)


@app.post("/api/ingestion/{organization_id}/replay/{batch_id}")
def replay_ingestion_batch(organization_id: str, batch_id: str, source_name: str = "Replay"):
    with SessionLocal() as session:
        try:
            receipt = IngestionService(session).replay_batch(organization_id, batch_id, source_name=source_name)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return receipt.model_dump()


@app.get("/api/ingestion/{organization_id}/health")
def ingestion_health(organization_id: str):
    with SessionLocal() as session:
        try:
            return IngestionService(session).health(organization_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/analytics/{organization_id}/batches/{batch_id}/compute")
def compute_analytics_batch(organization_id: str, batch_id: str):
    with SessionLocal() as session:
        try:
            receipt = AnalyticsService(session).compute_batch(organization_id, batch_id)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return receipt.model_dump()


@app.post("/api/analytics/{organization_id}/sensors/{sensor_id}/recompute")
def recompute_analytics_sensor(organization_id: str, sensor_id: str):
    with SessionLocal() as session:
        try:
            receipt = AnalyticsService(session).recompute_sensor(organization_id, sensor_id)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return receipt.model_dump()


@app.get("/api/analytics/{organization_id}/health")
def analytics_health(organization_id: str):
    with SessionLocal() as session:
        try:
            return AnalyticsService(session).health(organization_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ml/{organization_id}/dataset-versions")
def create_ml_dataset_version(organization_id: str, request: DatasetVersionCreate):
    with SessionLocal() as session:
        try:
            dataset = MLPlatformService(session).create_dataset_version(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _dataset_version_payload(dataset)


@app.get("/api/ml/{organization_id}/dataset-versions")
def list_ml_dataset_versions(organization_id: str):
    with SessionLocal() as session:
        try:
            datasets = MLPlatformService(session).list_dataset_versions(organization_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {"dataset_versions": [_dataset_version_payload(dataset) for dataset in datasets]}


@app.post("/api/ml/{organization_id}/experiments")
def run_ml_experiment(organization_id: str, request: ExperimentCreate):
    with SessionLocal() as session:
        try:
            experiment = MLPlatformService(session).run_experiment(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _experiment_payload(experiment)


@app.get("/api/ml/{organization_id}/experiments")
def list_ml_experiments(organization_id: str):
    with SessionLocal() as session:
        try:
            experiments = MLPlatformService(session).list_experiments(organization_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {"experiments": [_experiment_payload(experiment) for experiment in experiments]}


@app.get("/api/ml/{organization_id}/experiments/{experiment_run_id}")
def get_ml_experiment(organization_id: str, experiment_run_id: str):
    with SessionLocal() as session:
        try:
            experiment = MLPlatformService(session).get_experiment(organization_id, experiment_run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _experiment_payload(experiment)


@app.post("/api/ml/{organization_id}/registries")
def create_ml_registry(organization_id: str, request: RegistryCreate):
    with SessionLocal() as session:
        try:
            registry = MLPlatformService(session).create_registry(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _registry_payload(registry)


@app.get("/api/ml/{organization_id}/registries")
def list_ml_registries(organization_id: str):
    with SessionLocal() as session:
        try:
            return {"registries": MLPlatformService(session).list_registries(organization_id)}
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ml/{organization_id}/model-versions")
def register_ml_model_version(organization_id: str, request: ModelVersionCreate):
    with SessionLocal() as session:
        try:
            model_version = MLPlatformService(session).register_model_version(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _model_version_payload(model_version)


@app.post("/api/ml/{organization_id}/model-versions/{model_version_id}/promote")
def promote_ml_model_version(organization_id: str, model_version_id: str, request: PromoteModelVersion):
    with SessionLocal() as session:
        try:
            model_version = MLPlatformService(session).promote_model_version(
                organization_id,
                model_version_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _model_version_payload(model_version)


@app.post("/api/ml/{organization_id}/registries/{registry_id}/rollback")
def rollback_ml_model_version(organization_id: str, registry_id: str, request: RollbackModelVersion):
    with SessionLocal() as session:
        try:
            model_version = MLPlatformService(session).rollback_model_version(organization_id, registry_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _model_version_payload(model_version)


def _dataset_version_payload(dataset) -> dict:
    return {
        "id": dataset.id,
        "organization_id": dataset.organization_id,
        "name": dataset.name,
        "version": dataset.version,
        "status": dataset.status,
        "source_algorithm_version": dataset.source_algorithm_version,
        "target_name": dataset.target_name,
        "target_unit": dataset.target_unit,
        "feature_names": dataset.feature_names,
        "row_count": dataset.row_count,
        "validation_group_count": dataset.validation_group_count,
        "fingerprint": dataset.fingerprint,
        "filters": dataset.filters,
        "provenance": dataset.provenance,
    }


def _experiment_payload(experiment) -> dict:
    return {
        "id": experiment.id,
        "organization_id": experiment.organization_id,
        "dataset_version_id": experiment.dataset_version_id,
        "name": experiment.name,
        "status": experiment.status,
        "algorithm": experiment.algorithm,
        "validation_method": experiment.validation_method,
        "code_version": experiment.code_version,
        "training_config": experiment.training_config,
        "metrics": experiment.metrics,
        "baseline_metrics": experiment.baseline_metrics,
        "uncertainty": experiment.uncertainty,
        "abstention_policy": experiment.abstention_policy,
        "artifact_uri": experiment.artifact_uri,
        "artifact_sha256": experiment.artifact_sha256,
        "provenance": experiment.provenance,
    }


def _registry_payload(registry) -> dict:
    return {
        "id": registry.id,
        "organization_id": registry.organization_id,
        "name": registry.name,
        "task": registry.task,
        "status": registry.status,
        "description": registry.description,
    }


def _model_version_payload(model_version) -> dict:
    return {
        "id": model_version.id,
        "organization_id": model_version.organization_id,
        "registry_id": model_version.registry_id,
        "experiment_run_id": model_version.experiment_run_id,
        "dataset_version_id": model_version.dataset_version_id,
        "version": model_version.version,
        "stage": model_version.stage,
        "approval_status": model_version.approval_status,
        "artifact_uri": model_version.artifact_uri,
        "artifact_sha256": model_version.artifact_sha256,
        "metrics": model_version.metrics,
        "baseline_metrics": model_version.baseline_metrics,
        "uncertainty": model_version.uncertainty,
        "abstention_policy": model_version.abstention_policy,
        "provenance": model_version.provenance,
        "approved_by_user_id": model_version.approved_by_user_id,
        "approved_at": model_version.approved_at.isoformat() if model_version.approved_at else None,
    }


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
