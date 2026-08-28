"""FastAPI backend serving bearing RUL predictions, SHAP explanations, vibration
analysis, health-state classification, and maintenance decision support."""
import csv
import io
import json
import logging
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, select
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
from enterprise_security.contracts import (
    IdentityProviderCreate,
    IdentityProviderUpdate,
    MembershipChange,
    MembershipStatusChange,
    SecretReferenceCreate,
    SecretReferenceUpdate,
    ServicePrincipalCreate,
    ServicePrincipalUpdate,
    UserIdentityOnboard,
)
from enterprise_security.permissions import (
    ANALYTICS_READ,
    ANALYTICS_RUN,
    AUDIT_READ,
    INGESTION_MANAGE,
    INGESTION_WRITE,
    MAINTENANCE_CMMS_SYNC,
    MAINTENANCE_MANAGE,
    MAINTENANCE_READ,
    MAINTENANCE_WORK_ORDER_APPROVE,
    ML_EXPERIMENT_RUN,
    ML_MODEL_PROMOTE_PRODUCTION,
    ML_MODEL_PROMOTE_VALIDATED,
    ML_MODEL_REGISTER,
    ML_READ,
    PLATFORM_READ,
    PREDICTION_READ,
    SECRETS_MANAGE,
    SECURITY_MANAGE,
    SERVING_BIND,
    SERVING_PREDICT,
)
from enterprise_security.service import (
    MAX_AUDIT_HTTP_PATH_LENGTH,
    AuthenticationError,
    AuthorizationError,
    OidcTokenVerifier,
    SecurityService,
    audit_event_payload,
    identity_provider_payload,
    secret_reference_payload,
    security_settings,
    service_principal_payload,
)
from explain_bearing import BearingRulExplainer
from industrial_ingestion.contracts import SourceRegistration
from industrial_ingestion.service import IngestionService
from maintenance_operations.contracts import (
    AlertAcknowledgeRequest,
    AlertResolveRequest,
    CaseCreate,
    CaseCreateFromAlertRequest,
    CaseTransitionRequest,
    CmmsSyncRequest,
    InspectionCancelRequest,
    InspectionCompleteRequest,
    InspectionRequestCreate,
    InspectionStartRequest,
    NoteCreate,
    PredictionAlertEvaluationRequest,
    ResolutionCreate,
    WorkOrderApproveRequest,
    WorkOrderCancelRequest,
    WorkOrderCompleteRequest,
    WorkOrderCreate,
    WorkOrderStartRequest,
)
from maintenance_operations.service import (
    MaintenanceOperationsService,
    alert_payload,
    case_payload,
    cmms_sync_payload,
    inspection_payload,
    note_payload,
    resolution_payload,
    work_order_payload,
)
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
from platform_core.models import Asset, Base, Component, Organization, Sensor, Site
from platform_core.services import PlatformService, get_platform_inventory
from production_serving.contracts import PredictionRequest, ServingBindingCreate
from production_serving.service import ProductionServingService
from vibration_analysis import analyze
from waveform_cache import WaveformCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SECURITY_SETTINGS = security_settings()
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
OIDC_VERIFIER = OidcTokenVerifier(http_timeout_seconds=SECURITY_SETTINGS.oidc_http_timeout_seconds)

app = FastAPI(
    title="Predictive Maintenance Studio",
    docs_url=None if SECURITY_SETTINGS.environment == "production" and not SECURITY_SETTINGS.docs_enabled else "/docs",
    redoc_url=(
        None if SECURITY_SETTINGS.environment == "production" and not SECURITY_SETTINGS.docs_enabled else "/redoc"
    ),
    openapi_url=None
    if SECURITY_SETTINGS.environment == "production" and not SECURITY_SETTINGS.docs_enabled
    else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SECURITY_SETTINGS.cors_allowed_origins),
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    if len(request.url.path) > MAX_AUDIT_HTTP_PATH_LENGTH:
        return JSONResponse(status_code=414, content={"detail": "URI path is too long"})
    supplied_request_id = request.headers.get("X-Request-ID")
    if supplied_request_id is not None and not REQUEST_ID_PATTERN.fullmatch(supplied_request_id):
        return JSONResponse(
            status_code=400,
            content={"detail": "X-Request-ID must be 1-64 characters of letters, digits, '.', '_', ':', or '-'"},
        )
    request_id = supplied_request_id or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

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


def _enterprise_security_enabled() -> bool:
    return SECURITY_SETTINGS.mode == "enterprise"


def _reject_enterprise_legacy_endpoint() -> None:
    if _enterprise_security_enabled():
        raise HTTPException(status_code=403, detail="legacy IMS demo endpoint is disabled in enterprise mode")


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("invalid or missing authentication")
    return token


def _security_service(session) -> SecurityService:
    return SecurityService(session, verifier=OIDC_VERIFIER)


def _authorize(
    session,
    request: Request,
    organization_id: str,
    permission: str,
    *,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
):
    if not _enterprise_security_enabled():
        if action.startswith("security.") and action != "security.me":
            raise HTTPException(
                status_code=403,
                detail="enterprise security administration requires enterprise security mode",
            )
        return None
    security = _security_service(session)
    try:
        context = security.authenticate_bearer(
            organization_id,
            _bearer_token(request),
            request_id=getattr(request.state, "request_id", uuid4().hex),
        )
        security.require_permission(
            context,
            permission,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            http_method=request.method,
            http_path=request.url.path,
        )
        return context
    except AuthenticationError as exc:
        session.rollback()
        raise HTTPException(status_code=401, detail="invalid or missing authentication") from exc
    except AuthorizationError as exc:
        session.commit()
        raise HTTPException(status_code=403, detail="not authorized for this organization") from exc


def _authenticated_user_id(context, fallback_user_id: str | None) -> str:
    if not _enterprise_security_enabled():
        if fallback_user_id is None:
            raise HTTPException(status_code=400, detail="user id is required when enterprise security is disabled")
        return fallback_user_id
    try:
        return context.require_user()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="human user principal required") from exc


def _membership_payload(membership) -> dict:
    return {
        "id": membership.id,
        "organization_id": membership.organization_id,
        "user_id": membership.user_id,
        "role": membership.role,
        "lifecycle_state": membership.lifecycle_state,
        "created_at": membership.created_at,
        "updated_at": membership.updated_at,
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
    if _enterprise_security_enabled():
        raise HTTPException(status_code=403, detail="IMS bootstrap is disabled as an unauthenticated enterprise API")
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
    if _enterprise_security_enabled():
        raise HTTPException(status_code=403, detail="use organization-scoped inventory in enterprise mode")
    with SessionLocal() as session:
        try:
            return get_platform_inventory(session)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/platform/{organization_id}/inventory")
def organization_platform_inventory(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, PLATFORM_READ, action="platform.inventory.read")
            organization = session.get(Organization, organization_id)
            if organization is None:
                raise HTTPException(status_code=404, detail="organization not found")
            sites = list(
                session.scalars(
                    select(Site)
                    .where(Site.organization_id == organization_id)
                    .order_by(Site.slug)
                )
            )
            assets = list(
                session.scalars(
                    select(Asset)
                    .where(Asset.organization_id == organization_id)
                    .order_by(Asset.slug)
                )
            )
            components = list(
                session.scalars(
                    select(Component)
                    .where(Component.organization_id == organization_id)
                    .order_by(Component.slug)
                )
            )
            sensors = list(
                session.scalars(
                    select(Sensor)
                    .where(Sensor.organization_id == organization_id)
                    .order_by(Sensor.slug)
                )
            )
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {
            "organizations": [
                {"id": organization.id, "slug": organization.slug, "name": organization.name},
            ],
            "sites": [
                {"id": site.id, "slug": site.slug, "name": site.name, "timezone": site.timezone}
                for site in sites
            ],
            "assets": [
                {
                    "id": asset.id,
                    "site_id": asset.site_id,
                    "slug": asset.slug,
                    "name": asset.name,
                    "asset_type": asset.asset_type,
                }
                for asset in assets
            ],
            "components": [
                {
                    "id": component.id,
                    "asset_id": component.asset_id,
                    "slug": component.slug,
                    "name": component.name,
                    "component_type": component.component_type,
                }
                for component in components
            ],
            "sensors": [
                {
                    "id": sensor.id,
                    "component_id": sensor.component_id,
                    "slug": sensor.slug,
                    "name": sensor.name,
                    "sensor_type": sensor.sensor_type,
                    "unit": sensor.unit,
                    "sampling_rate_hz": sensor.sampling_rate_hz,
                    "channel_name": sensor.channel_name,
                    "axis": sensor.axis,
                    "manufacturer": sensor.manufacturer,
                    "model": sensor.model,
                }
                for sensor in sensors
            ],
        }


@app.get("/api/security/{organization_id}/me")
def get_current_security_context(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(session, request, organization_id, PLATFORM_READ, action="security.me")
            session.commit()
            if context is None:
                return {"security_mode": "disabled"}
            return context.model_dump(exclude={"subject"})
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/security/{organization_id}/identity-providers")
def create_identity_provider(organization_id: str, request: IdentityProviderCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, SECURITY_MANAGE, action="security.idp.create")
            idp = _security_service(session).create_identity_provider(
                organization_id,
                request,
                allow_development_targets=SECURITY_SETTINGS.environment != "production",
            )
            session.commit()
            return identity_provider_payload(idp)
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/security/{organization_id}/identity-providers")
def list_identity_providers(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, SECURITY_MANAGE, action="security.idp.list")
            idps = _security_service(session).list_identity_providers(organization_id)
            session.commit()
            return {"identity_providers": [identity_provider_payload(idp) for idp in idps]}
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.patch("/api/security/{organization_id}/identity-providers/{identity_provider_id}")
def update_identity_provider(
    organization_id: str,
    identity_provider_id: str,
    request: IdentityProviderUpdate,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            _authorize(
                session,
                http_request,
                organization_id,
                SECURITY_MANAGE,
                action="security.idp.update",
                resource_type="identity_provider",
                resource_id=identity_provider_id,
            )
            idp = _security_service(session).update_identity_provider(organization_id, identity_provider_id, request)
            session.commit()
            return identity_provider_payload(idp)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/security/{organization_id}/user-identities")
def onboard_user_identity(organization_id: str, request: UserIdentityOnboard, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                SECURITY_MANAGE,
                action="security.user_identity.onboard",
            )
            result = _security_service(session).onboard_user_identity(organization_id, request, actor=context)
            session.commit()
            user = result["user"]
            return {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "lifecycle_state": user.lifecycle_state,
                },
                "membership": _membership_payload(result["membership"]),
                "identity": {
                    "id": result["identity"].id,
                    "organization_id": result["identity"].organization_id,
                    "user_id": result["identity"].user_id,
                    "identity_provider_id": result["identity"].identity_provider_id,
                    "issuer": result["identity"].issuer,
                    "subject": result["identity"].subject,
                    "last_seen_at": result["identity"].last_seen_at,
                },
            }
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/security/{organization_id}/memberships")
def change_membership_role(organization_id: str, request: MembershipChange, http_request: Request):
    with SessionLocal() as session:
        security = _security_service(session)
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                PLATFORM_READ,
                action="security.membership.auth",
                resource_type="membership",
                resource_id=request.user_id,
            )
            membership = security.change_membership_role(organization_id, request, actor=context)
            session.commit()
            return _membership_payload(membership)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.patch("/api/security/{organization_id}/memberships/{user_id}")
def change_membership_status(
    organization_id: str,
    user_id: str,
    request: MembershipStatusChange,
    http_request: Request,
):
    with SessionLocal() as session:
        security = _security_service(session)
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                PLATFORM_READ,
                action="security.membership.auth",
                resource_type="membership",
                resource_id=user_id,
            )
            membership = security.change_membership_status(organization_id, user_id, request, actor=context)
            session.commit()
            return _membership_payload(membership)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/security/{organization_id}/service-principals")
def create_service_principal(organization_id: str, request: ServicePrincipalCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(
                session,
                http_request,
                organization_id,
                SECURITY_MANAGE,
                action="security.service_principal.create",
            )
            principal = _security_service(session).create_service_principal(organization_id, request)
            session.commit()
            return service_principal_payload(principal)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/security/{organization_id}/service-principals")
def list_service_principals(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, SECURITY_MANAGE, action="security.service_principal.list")
            principals = _security_service(session).list_service_principals(organization_id)
            session.commit()
            return {"service_principals": [service_principal_payload(principal) for principal in principals]}
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.patch("/api/security/{organization_id}/service-principals/{principal_id}")
def update_service_principal(
    organization_id: str,
    principal_id: str,
    request: ServicePrincipalUpdate,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            _authorize(
                session,
                http_request,
                organization_id,
                SECURITY_MANAGE,
                action="security.service_principal.update",
                resource_type="service_principal",
                resource_id=principal_id,
            )
            principal = _security_service(session).update_service_principal(organization_id, principal_id, request)
            session.commit()
            return service_principal_payload(principal)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/security/{organization_id}/secret-references")
def create_secret_reference(organization_id: str, request: SecretReferenceCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                SECRETS_MANAGE,
                action="security.secret.create",
            )
            secret = _security_service(session).create_secret_reference(
                organization_id,
                request,
                created_by_user_id=_authenticated_user_id(context, None),
            )
            session.commit()
            return secret_reference_payload(secret)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/security/{organization_id}/secret-references")
def list_secret_references(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, SECRETS_MANAGE, action="security.secret.list")
            secrets = _security_service(session).list_secret_references(organization_id)
            session.commit()
            return {"secret_references": [secret_reference_payload(secret) for secret in secrets]}
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.patch("/api/security/{organization_id}/secret-references/{secret_id}")
def update_secret_reference(
    organization_id: str,
    secret_id: str,
    request: SecretReferenceUpdate,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            _authorize(
                session,
                http_request,
                organization_id,
                SECRETS_MANAGE,
                action="security.secret.update",
                resource_type="secret_reference",
                resource_id=secret_id,
            )
            secret = _security_service(session).update_secret_reference(organization_id, secret_id, request)
            session.commit()
            return secret_reference_payload(secret)
        except AuthorizationError as exc:
            session.commit()
            raise HTTPException(status_code=403, detail="not authorized for this organization") from exc
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/security/{organization_id}/audit-events")
def list_security_audit_events(organization_id: str, request: Request, limit: int = 100, offset: int = 0):
    with SessionLocal() as session:
        try:
            security = _security_service(session)
            _authorize(session, request, organization_id, AUDIT_READ, action="security.audit.list")
            events = security.list_audit_events(organization_id, limit=limit, offset=offset)
            session.commit()
            return {"audit_events": [audit_event_payload(event) for event in events]}
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ingestion/sources")
def register_ingestion_source(registration: SourceRegistration):
    if _enterprise_security_enabled():
        raise HTTPException(status_code=403, detail="use organization-scoped ingestion source registration")
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


@app.post("/api/ingestion/{organization_id}/sources")
def register_organization_ingestion_source(organization_id: str, registration: SourceRegistration, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, INGESTION_MANAGE, action="ingestion.source.register")
            scoped_registration = registration.model_copy(update={"organization_id": organization_id})
            source = IngestionService(session).register_source(scoped_registration)
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
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.rest.write")
        session.commit()
    return _ingest_payload(organization_id, "rest", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/mqtt")
async def ingest_mqtt(
    organization_id: str,
    request: Request,
    source_name: str = "MQTT Bridge",
    topic: str | None = None,
):
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.mqtt.write")
        session.commit()
    return _ingest_payload(organization_id, "mqtt", await request.body(), source_name, topic=topic)


@app.post("/api/ingestion/{organization_id}/opcua")
async def ingest_opcua(organization_id: str, request: Request, source_name: str = "OPC-UA Bridge"):
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.opcua.write")
        session.commit()
    return _ingest_payload(organization_id, "opcua", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/abb")
async def ingest_abb(organization_id: str, request: Request, source_name: str = "ABB Adapter"):
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.abb.write")
        session.commit()
    return _ingest_payload(organization_id, "abb", await _json_body(request), source_name)


@app.post("/api/ingestion/{organization_id}/files/csv")
async def ingest_csv_file(
    organization_id: str,
    request: Request,
    source_name: str = "CSV File",
    source_uri: str | None = None,
):
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.csv.write")
        session.commit()
    return _ingest_payload(organization_id, "csv", await request.body(), source_name, source_uri=source_uri)


@app.post("/api/ingestion/{organization_id}/files/parquet")
async def ingest_parquet_file(
    organization_id: str,
    request: Request,
    source_name: str = "Parquet File",
    source_uri: str | None = None,
):
    with SessionLocal() as session:
        _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.parquet.write")
        session.commit()
    return _ingest_payload(organization_id, "parquet", await request.body(), source_name, source_uri=source_uri)


@app.post("/api/ingestion/{organization_id}/replay/{batch_id}")
def replay_ingestion_batch(organization_id: str, batch_id: str, request: Request, source_name: str = "Replay"):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, INGESTION_WRITE, action="ingestion.replay")
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
def ingestion_health(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, INGESTION_MANAGE, action="ingestion.health.read")
            health_payload = IngestionService(session).health(organization_id)
            session.commit()
            return health_payload
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/analytics/{organization_id}/batches/{batch_id}/compute")
def compute_analytics_batch(organization_id: str, batch_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ANALYTICS_RUN, action="analytics.batch.compute")
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
def recompute_analytics_sensor(organization_id: str, sensor_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ANALYTICS_RUN, action="analytics.sensor.recompute")
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
def analytics_health(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ANALYTICS_READ, action="analytics.health.read")
            health_payload = AnalyticsService(session).health(organization_id)
            session.commit()
            return health_payload
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ml/{organization_id}/dataset-versions")
def create_ml_dataset_version(organization_id: str, request: DatasetVersionCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, ML_EXPERIMENT_RUN, action="ml.dataset.create")
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
def list_ml_dataset_versions(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ML_READ, action="ml.dataset.list")
            datasets = MLPlatformService(session).list_dataset_versions(organization_id)
            session.commit()
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {"dataset_versions": [_dataset_version_payload(dataset) for dataset in datasets]}


@app.post("/api/ml/{organization_id}/experiments")
def run_ml_experiment(organization_id: str, request: ExperimentCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, ML_EXPERIMENT_RUN, action="ml.experiment.run")
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
def list_ml_experiments(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ML_READ, action="ml.experiment.list")
            experiments = MLPlatformService(session).list_experiments(organization_id)
            session.commit()
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {"experiments": [_experiment_payload(experiment) for experiment in experiments]}


@app.get("/api/ml/{organization_id}/experiments/{experiment_run_id}")
def get_ml_experiment(organization_id: str, experiment_run_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ML_READ, action="ml.experiment.get")
            experiment = MLPlatformService(session).get_experiment(organization_id, experiment_run_id)
            session.commit()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _experiment_payload(experiment)


@app.post("/api/ml/{organization_id}/registries")
def create_ml_registry(organization_id: str, request: RegistryCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, ML_MODEL_REGISTER, action="ml.registry.create")
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
def list_ml_registries(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, ML_READ, action="ml.registry.list")
            session.commit()
            return {"registries": MLPlatformService(session).list_registries(organization_id)}
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/ml/{organization_id}/model-versions")
def register_ml_model_version(organization_id: str, request: ModelVersionCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, ML_MODEL_REGISTER, action="ml.model.register")
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
def promote_ml_model_version(
    organization_id: str,
    model_version_id: str,
    request: PromoteModelVersion,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            permission = (
                ML_MODEL_PROMOTE_PRODUCTION
                if request.target_stage == "production"
                else ML_MODEL_PROMOTE_VALIDATED
            )
            context = _authorize(session, http_request, organization_id, permission, action="ml.model.promote")
            if _enterprise_security_enabled():
                request = request.model_copy(update={"approved_by_user_id": _authenticated_user_id(context, None)})
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
def rollback_ml_model_version(
    organization_id: str,
    registry_id: str,
    request: RollbackModelVersion,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                ML_MODEL_PROMOTE_PRODUCTION,
                action="ml.model.rollback",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"approved_by_user_id": _authenticated_user_id(context, None)})
            model_version = MLPlatformService(session).rollback_model_version(organization_id, registry_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _model_version_payload(model_version)


@app.post("/api/serving/{organization_id}/bindings")
def create_serving_binding(organization_id: str, request: ServingBindingCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(session, http_request, organization_id, SERVING_BIND, action="serving.binding.create")
            if _enterprise_security_enabled():
                request = request.model_copy(update={"approved_by_user_id": _authenticated_user_id(context, None)})
            binding = ProductionServingService(session).bind_model(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return _serving_binding_payload(binding)


@app.post("/api/serving/{organization_id}/predict/rul")
def predict_production_rul(organization_id: str, request: PredictionRequest, http_request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, SERVING_PREDICT, action="serving.predict.rul")
            prediction = ProductionServingService(session).predict_rul(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return prediction.model_dump()


@app.get("/api/serving/{organization_id}/predictions")
def list_production_predictions(organization_id: str, request: Request, sensor_id: str | None = None):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, PREDICTION_READ, action="serving.prediction.list")
            predictions = ProductionServingService(session).prediction_history(organization_id, sensor_id=sensor_id)
            session.commit()
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return {"predictions": predictions}


@app.get("/api/serving/{organization_id}/health")
def production_serving_health(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, PREDICTION_READ, action="serving.health.read")
            health_payload = ProductionServingService(session).health(organization_id)
            session.commit()
            return health_payload
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/maintenance/{organization_id}/alerts/evaluate-prediction")
def evaluate_maintenance_alert_from_prediction(
    organization_id: str,
    request: PredictionAlertEvaluationRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            _authorize(session, http_request, organization_id, MAINTENANCE_MANAGE, action="maintenance.alert.evaluate")
            result = MaintenanceOperationsService(session).evaluate_prediction_alert(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return result.model_dump()


@app.get("/api/maintenance/{organization_id}/alerts")
def list_maintenance_alerts(organization_id: str, request: Request, status: str | None = None):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.alert.list")
            alerts = MaintenanceOperationsService(session).list_alerts(organization_id, status=status)
            session.commit()
            return {"alerts": alerts}
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/maintenance/{organization_id}/alerts/{alert_id}")
def get_maintenance_alert(organization_id: str, alert_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.alert.get")
            alert = MaintenanceOperationsService(session).get_alert(organization_id, alert_id)
            session.commit()
            return alert
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/maintenance/{organization_id}/alerts/{alert_id}/acknowledge")
def acknowledge_maintenance_alert(
    organization_id: str,
    alert_id: str,
    request: AlertAcknowledgeRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.alert.acknowledge",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"acknowledged_by_user_id": _authenticated_user_id(context, None)})
            alert = MaintenanceOperationsService(session).acknowledge_alert(organization_id, alert_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return alert_payload(alert)


@app.post("/api/maintenance/{organization_id}/alerts/{alert_id}/resolve")
def resolve_maintenance_alert(organization_id: str, alert_id: str, request: AlertResolveRequest, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.alert.resolve",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"resolved_by_user_id": _authenticated_user_id(context, None)})
            alert = MaintenanceOperationsService(session).resolve_alert(organization_id, alert_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return alert_payload(alert)


@app.post("/api/maintenance/{organization_id}/cases")
def open_maintenance_case(organization_id: str, request: CaseCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.case.create",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"opened_by_user_id": _authenticated_user_id(context, None)})
            case = MaintenanceOperationsService(session).open_case(organization_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return case_payload(case)


@app.post("/api/maintenance/{organization_id}/alerts/{alert_id}/case")
def open_maintenance_case_from_alert(
    organization_id: str,
    alert_id: str,
    request: CaseCreateFromAlertRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.case.create_from_alert",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"opened_by_user_id": _authenticated_user_id(context, None)})
            case = MaintenanceOperationsService(session).open_case_from_alert(organization_id, alert_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return case_payload(case)


@app.get("/api/maintenance/{organization_id}/cases")
def list_maintenance_cases(organization_id: str, request: Request, status: str | None = None):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.case.list")
            cases = MaintenanceOperationsService(session).list_cases(organization_id, status=status)
            session.commit()
            return {"cases": cases}
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.get("/api/maintenance/{organization_id}/cases/{case_id}")
def get_maintenance_case(organization_id: str, case_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.case.get")
            case = MaintenanceOperationsService(session).get_case(organization_id, case_id)
            session.commit()
            return case
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/maintenance/{organization_id}/cases/{case_id}/transition")
def transition_maintenance_case(
    organization_id: str,
    case_id: str,
    request: CaseTransitionRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.case.transition",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"actor_user_id": _authenticated_user_id(context, None)})
            case = MaintenanceOperationsService(session).transition_case(organization_id, case_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return case_payload(case)


@app.post("/api/maintenance/{organization_id}/cases/{case_id}/inspections")
def request_maintenance_inspection(
    organization_id: str,
    case_id: str,
    request: InspectionRequestCreate,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.inspection.request",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"requested_by_user_id": _authenticated_user_id(context, None)})
            inspection = MaintenanceOperationsService(session).request_inspection(organization_id, case_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return inspection_payload(inspection)


@app.post("/api/maintenance/{organization_id}/inspections/{inspection_id}/start")
def start_maintenance_inspection(
    organization_id: str,
    inspection_id: str,
    request: InspectionStartRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.inspection.start",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"started_by_user_id": _authenticated_user_id(context, None)})
            inspection = MaintenanceOperationsService(session).start_inspection(organization_id, inspection_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return inspection_payload(inspection)


@app.post("/api/maintenance/{organization_id}/inspections/{inspection_id}/complete")
def complete_maintenance_inspection(
    organization_id: str,
    inspection_id: str,
    request: InspectionCompleteRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.inspection.complete",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"performed_by_user_id": _authenticated_user_id(context, None)})
            inspection = MaintenanceOperationsService(session).complete_inspection(
                organization_id,
                inspection_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return inspection_payload(inspection)


@app.post("/api/maintenance/{organization_id}/inspections/{inspection_id}/cancel")
def cancel_maintenance_inspection(
    organization_id: str,
    inspection_id: str,
    request: InspectionCancelRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.inspection.cancel",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"cancelled_by_user_id": _authenticated_user_id(context, None)})
            inspection = MaintenanceOperationsService(session).cancel_inspection(
                organization_id,
                inspection_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return inspection_payload(inspection)


@app.post("/api/maintenance/{organization_id}/cases/{case_id}/notes")
def add_maintenance_note(organization_id: str, case_id: str, request: NoteCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.note.add",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"author_user_id": _authenticated_user_id(context, None)})
            note = MaintenanceOperationsService(session).add_note(organization_id, case_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return note_payload(note)


@app.get("/api/maintenance/{organization_id}/cases/{case_id}/notes")
def list_maintenance_notes(organization_id: str, case_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.note.list")
            notes = MaintenanceOperationsService(session).list_notes(organization_id, case_id)
            session.commit()
            return {"notes": notes}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/maintenance/{organization_id}/cases/{case_id}/work-orders")
def create_maintenance_work_order(organization_id: str, case_id: str, request: WorkOrderCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.work_order.create",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"requested_by_user_id": _authenticated_user_id(context, None)})
            work_order = MaintenanceOperationsService(session).create_work_order(organization_id, case_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return work_order_payload(work_order)


@app.post("/api/maintenance/{organization_id}/work-orders/{work_order_id}/approve")
def approve_maintenance_work_order(
    organization_id: str,
    work_order_id: str,
    request: WorkOrderApproveRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_WORK_ORDER_APPROVE,
                action="maintenance.work_order.approve",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"approved_by_user_id": _authenticated_user_id(context, None)})
            work_order = MaintenanceOperationsService(session).approve_work_order(
                organization_id,
                work_order_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return work_order_payload(work_order)


@app.post("/api/maintenance/{organization_id}/work-orders/{work_order_id}/start")
def start_maintenance_work_order(
    organization_id: str,
    work_order_id: str,
    request: WorkOrderStartRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.work_order.start",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"started_by_user_id": _authenticated_user_id(context, None)})
            work_order = MaintenanceOperationsService(session).start_work_order(organization_id, work_order_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return work_order_payload(work_order)


@app.post("/api/maintenance/{organization_id}/work-orders/{work_order_id}/complete")
def complete_maintenance_work_order(
    organization_id: str,
    work_order_id: str,
    request: WorkOrderCompleteRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.work_order.complete",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"completed_by_user_id": _authenticated_user_id(context, None)})
            work_order = MaintenanceOperationsService(session).complete_work_order(
                organization_id,
                work_order_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return work_order_payload(work_order)


@app.post("/api/maintenance/{organization_id}/work-orders/{work_order_id}/cancel")
def cancel_maintenance_work_order(
    organization_id: str,
    work_order_id: str,
    request: WorkOrderCancelRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.work_order.cancel",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"cancelled_by_user_id": _authenticated_user_id(context, None)})
            work_order = MaintenanceOperationsService(session).cancel_work_order(
                organization_id,
                work_order_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return work_order_payload(work_order)


@app.post("/api/maintenance/{organization_id}/work-orders/{work_order_id}/cmms-sync")
def sync_maintenance_work_order_to_cmms(
    organization_id: str,
    work_order_id: str,
    request: CmmsSyncRequest,
    http_request: Request,
):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_CMMS_SYNC,
                action="maintenance.cmms.sync",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"initiated_by_user_id": _authenticated_user_id(context, None)})
            sync = MaintenanceOperationsService(session).sync_work_order_to_cmms(
                organization_id,
                work_order_id,
                request,
            )
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return cmms_sync_payload(sync)


@app.get("/api/maintenance/{organization_id}/work-orders/{work_order_id}/cmms-sync")
def list_maintenance_work_order_cmms_syncs(organization_id: str, work_order_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.cmms.list")
            syncs = MaintenanceOperationsService(session).list_cmms_sync_records(organization_id, work_order_id)
            session.commit()
            return {"sync_records": syncs}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


@app.post("/api/maintenance/{organization_id}/cases/{case_id}/resolve")
def resolve_maintenance_case(organization_id: str, case_id: str, request: ResolutionCreate, http_request: Request):
    with SessionLocal() as session:
        try:
            context = _authorize(
                session,
                http_request,
                organization_id,
                MAINTENANCE_MANAGE,
                action="maintenance.case.resolve",
            )
            if _enterprise_security_enabled():
                request = request.model_copy(update={"resolved_by_user_id": _authenticated_user_id(context, None)})
            resolution = MaintenanceOperationsService(session).resolve_case(organization_id, case_id, request)
            session.commit()
        except ValueError as exc:
            session.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc
        return resolution_payload(resolution)


@app.get("/api/maintenance/{organization_id}/health")
def maintenance_operations_health(organization_id: str, request: Request):
    with SessionLocal() as session:
        try:
            _authorize(session, request, organization_id, MAINTENANCE_READ, action="maintenance.health.read")
            health_payload = MaintenanceOperationsService(session).health(organization_id)
            session.commit()
            return health_payload
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail=f"platform database unavailable: {exc}") from exc


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


def _serving_binding_payload(binding) -> dict:
    return {
        "id": binding.id,
        "organization_id": binding.organization_id,
        "registry_id": binding.registry_id,
        "model_version_id": binding.model_version_id,
        "scope_type": binding.scope_type,
        "scope_id": binding.scope_id,
        "status": binding.status,
        "approved_by_user_id": binding.approved_by_user_id,
        "activated_at": binding.activated_at.isoformat() if binding.activated_at else None,
        "reason": binding.reason,
        "provenance": binding.provenance,
    }


@app.get("/api/profile")
def profile():
    _reject_enterprise_legacy_endpoint()
    return {
        **summarize(_table),
        "model": _metrics,
        "model_metadata": _model_metadata,
        "dataset_metadata": _dataset_metadata,
    }


@app.get("/api/timeline")
def timeline():
    _reject_enterprise_legacy_endpoint()
    return {
        "n_snapshots": len(_timestamps),
        "timestamps": [ts.isoformat() for ts in _timestamps],
        "default_index": len(_timestamps) - 50,
        "snapshot_minutes": _metrics["snapshot_minutes"],
    }


@app.get("/api/snapshot/{index}")
def snapshot(index: int):
    _reject_enterprise_legacy_endpoint()
    ts = _timestamps[index] if 0 <= index < len(_timestamps) else None
    if ts is None:
        raise HTTPException(status_code=404, detail="snapshot index out of range")

    bearings = [_prediction_payload(bearing, _row_at(bearing, index), ts) for bearing in BEARING_COLS]
    return {"index": index, "timestamp": ts.isoformat(), "bearings": bearings}


@app.get("/api/snapshot/{index}/bearing/{bearing_id}")
def bearing_detail(index: int, bearing_id: str):
    _reject_enterprise_legacy_endpoint()
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
    _reject_enterprise_legacy_endpoint()
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
    _reject_enterprise_legacy_endpoint()
    if bearing_id not in BEARING_COLS:
        raise HTTPException(status_code=404, detail=f"unknown bearing {bearing_id}")
    return {
        "bearing": bearing_id,
        "baseline_snapshots": 100,
        "points": feature_trend(_table, bearing_id),
    }


@app.get("/api/waveform/{index}/bearing/{bearing_id}")
def waveform(index: int, bearing_id: str):
    _reject_enterprise_legacy_endpoint()
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
    _reject_enterprise_legacy_endpoint()
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
