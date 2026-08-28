import importlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from alembic import command
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
    WorkOrderCompleteRequest,
    WorkOrderCreate,
    WorkOrderStartRequest,
)
from maintenance_operations.service import CmmsAdapterResult, DeterministicCmmsAdapter, MaintenanceOperationsService
from ml_platform.artifact_store import ModelArtifactStore
from ml_platform.contracts import (
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    PromoteModelVersion,
    RegistryCreate,
)
from ml_platform.service import MLPlatformService
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.database import make_engine
from platform_core.models import (
    Base,
    CmmsSyncRecord,
    MaintenanceAlert,
    MaintenanceCase,
    MaintenanceResolution,
    PredictionRecord,
)
from platform_core.repositories import PlatformRepository
from production_serving.contracts import PredictionRequest, ServingBindingCreate
from production_serving.service import ProductionServingService

ROOT = Path(__file__).resolve().parent.parent


class AlternateDeterministicCmmsAdapter(DeterministicCmmsAdapter):
    provider_name = "alternate-test"


class NoEchoUpdateCmmsAdapter:
    provider_name = "no-echo-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        if operation == "create":
            return CmmsAdapterResult(status="succeeded", external_id="A-123", external_status="created")
        return CmmsAdapterResult(status="succeeded", external_id=None, external_status=operation)


class MissingCreateExternalIdCmmsAdapter:
    provider_name = "missing-create-id-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        return CmmsAdapterResult(status="succeeded", external_id=None, external_status=operation)


class MismatchedExternalIdCmmsAdapter:
    provider_name = "mismatched-id-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        if operation == "create":
            return CmmsAdapterResult(status="succeeded", external_id="A-123", external_status="created")
        return CmmsAdapterResult(status="succeeded", external_id="B-999", external_status=operation)


class TimeoutCmmsAdapter:
    provider_name = "timeout-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        raise TimeoutError("timeout while using credential secret-token")


class CreateThenTimeoutCmmsAdapter:
    provider_name = "create-then-timeout-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        if operation == "create":
            return CmmsAdapterResult(status="succeeded", external_id="A-123", external_status="created")
        raise TimeoutError("timeout while updating secret-token")


class FailingCmmsAdapter:
    provider_name = "failing-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        raise RuntimeError("adapter failed with password=secret-token")


class UnsupportedStatusCmmsAdapter:
    provider_name = "unsupported-status-test"

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def sync(self, operation, work_order, *, idempotency_key):
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        return CmmsAdapterResult(
            status="made_up_status",
            external_id="SHOULD-NOT-BIND",
            metadata={"idempotency_key": idempotency_key},
        )


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    external_url = os.environ.get("PMS_PLATFORM_CORE_TEST_DATABASE_URL")
    if external_url:
        url = external_url
    else:
        db_path = tmp_path / "platform.db"
        url = f"sqlite:///{db_path.as_posix()}"
        monkeypatch.setenv("PMS_DATABASE_URL", url)

    cfg = Config(str(ROOT / "alembic.ini"))
    if external_url:
        clean_engine = make_engine(url)
        try:
            Base.metadata.drop_all(clean_engine)
            with clean_engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        finally:
            clean_engine.dispose()
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr("platform_core.database.engine", engine)
    monkeypatch.setattr("platform_core.database.SessionLocal", session_factory)
    try:
        yield engine, session_factory
    finally:
        if external_url:
            Base.metadata.drop_all(engine)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        engine.dispose()


@pytest.fixture
def maintenance_fixture(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_operational_context(session)
        session.commit()
        return fixture


def _seed_operational_context(session):
    repo = PlatformRepository(session)
    org = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
    site = repo.create_site(org.id, SiteCreate(slug="atlanta", name="Atlanta Plant"))
    asset = repo.create_asset(
        org.id,
        AssetCreate(site_id=site.id, slug="pump-p-104", name="Pump P-104", asset_type="pump"),
    )
    component = repo.create_component(
        org.id,
        ComponentCreate(asset_id=asset.id, slug="bearing", name="Drive-End Bearing", component_type="bearing"),
    )
    sensor = repo.create_sensor(
        org.id,
        SensorCreate(
            component_id=component.id,
            slug="vs-017",
            name="VS-017",
            sensor_type="accelerometer",
            unit="g",
        ),
    )
    other_asset = repo.create_asset(
        org.id,
        AssetCreate(site_id=site.id, slug="pump-p-105", name="Pump P-105", asset_type="pump"),
    )
    other_component = repo.create_component(
        org.id,
        ComponentCreate(asset_id=other_asset.id, slug="bearing", name="Drive-End Bearing", component_type="bearing"),
    )
    other_sensor = repo.create_sensor(
        org.id,
        SensorCreate(
            component_id=other_component.id,
            slug="vs-117",
            name="VS-117",
            sensor_type="accelerometer",
            unit="g",
        ),
    )
    technician = repo.create_user(
        UserCreate(email="tech@example.com", full_name="Technician", external_subject="oidc:tech")
    )
    manager = repo.create_user(
        UserCreate(email="manager@example.com", full_name="Manager", external_subject="oidc:manager")
    )
    outsider = repo.create_user(
        UserCreate(email="outsider@example.com", full_name="Outsider", external_subject="oidc:outsider")
    )
    other_user = repo.create_user(
        UserCreate(email="other@example.com", full_name="Other User", external_subject="oidc:other")
    )
    repo.add_membership(org.id, technician.id, "technician")
    repo.add_membership(org.id, manager.id, "engineer")

    other_org = repo.create_organization(OrganizationCreate(slug="other", name="Other Manufacturing"))
    repo.add_membership(other_org.id, other_user.id, "technician")

    return {
        "organization_id": org.id,
        "other_organization_id": other_org.id,
        "site_id": site.id,
        "asset_id": asset.id,
        "component_id": component.id,
        "sensor_id": sensor.id,
        "same_tenant_other_asset_id": other_asset.id,
        "same_tenant_other_component_id": other_component.id,
        "same_tenant_other_sensor_id": other_sensor.id,
        "technician_id": technician.id,
        "manager_id": manager.id,
        "outsider_id": outsider.id,
        "other_user_id": other_user.id,
        "supported_low_id": _prediction(session, org.id, sensor.id, status="supported", rul=12.0),
        "supported_high_id": _prediction(session, org.id, sensor.id, status="supported", rul=250.0),
        "unsupported_id": _prediction(
            session,
            org.id,
            sensor.id,
            status="unsupported",
            code="OUT_OF_TRAINING_DOMAIN",
            reason="live feature vector is outside the validated training domain",
        ),
        "insufficient_id": _prediction(
            session,
            org.id,
            sensor.id,
            status="insufficient_evidence",
            code="STALE_FEATURES",
            reason="live analytics features are stale",
        ),
    }


def _prediction(
    session,
    organization_id: str,
    sensor_id: str,
    *,
    status: str,
    rul: float | None = None,
    code: str | None = None,
    reason: str | None = None,
) -> str:
    repo = PlatformRepository(session)
    resolution = repo.create_production_model_resolution(
        organization_id,
        binding_id=None,
        registry_id=None,
        model_version_id=None,
        dataset_version_id=None,
        sensor_id=sensor_id,
        status="resolved" if status == "supported" else "abstained",
        reason_code="SUPPORTED" if status == "supported" else code or status.upper(),
        reason=reason or "prediction produced supported RUL",
        artifact_sha256="abc123" if status == "supported" else None,
        feature_schema=["scalar.rms", "scalar.kurtosis"],
        abstention_policy={"max_feature_age_minutes": 60},
        evidence={"serving_reference_time": datetime.now(UTC).isoformat()},
    )
    prediction = repo.create_prediction_record(
        organization_id,
        model_resolution_id=resolution.id,
        registry_id=None,
        model_version_id=None,
        dataset_version_id=None,
        sensor_id=sensor_id,
        observed_at=datetime.now(UTC),
        prediction_status=status,
        predicted_rul_hours=rul,
        abstention_code=code,
        uncertainty={"prediction_interval_80": {"lower": 8.0, "upper": 18.0}} if status == "supported" else None,
        feature_vector={"scalar.rms": 2.1, "scalar.kurtosis": 4.2} if status == "supported" else None,
        feature_record_ids=["feature-1", "feature-2"] if status == "supported" else [],
        abstention_reason=reason,
        provenance={
            "serving_slice": "production-slice-10",
            "binding_id": "binding-1" if status == "supported" else None,
            "artifact_sha256": resolution.artifact_sha256,
            "feature_record_ids": ["feature-1", "feature-2"] if status == "supported" else [],
            "request_kind": "live",
            "serving_reference_time": datetime.now(UTC).isoformat(),
        },
    )
    return prediction.id


def _service_alert(service, fixture, *, prediction_id=None, threshold=24.0, rule_id="rul-under-24h"):
    return service.evaluate_prediction_alert(
        fixture["organization_id"],
        PredictionAlertEvaluationRequest(
            prediction_id=prediction_id or fixture["supported_low_id"],
            rule_id=rule_id,
            rule_name="RUL below threshold",
            rul_threshold_hours=threshold,
            priority="high",
            recommended_action="Human review required before scheduling maintenance.",
        ),
    )


def _active_case(service, fixture):
    result = _service_alert(service, fixture)
    return service.open_case_from_alert(
        fixture["organization_id"],
        result.alert["id"],
        CaseCreateFromAlertRequest(opened_by_user_id=fixture["technician_id"]),
    )


def _draft_work_order(service, fixture):
    case = _active_case(service, fixture)
    return service.create_work_order(
        fixture["organization_id"],
        case.id,
        WorkOrderCreate(
            requested_by_user_id=fixture["technician_id"],
            title="Inspect drive-end bearing",
            requested_work="Inspect drive-end bearing and document findings.",
            priority="high",
        ),
    )


def _seed_real_serving_features(session, fixture) -> None:
    repo = PlatformRepository(session)
    base_time = datetime.now(UTC) - timedelta(minutes=5)
    run_a = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "maintenance-real-serving"},
    )
    run_b = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["same_tenant_other_sensor_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "maintenance-real-serving"},
    )
    for sensor_id, run_id, group, offset in [
        (fixture["sensor_id"], run_a.id, "bearing-a", 0.0),
        (fixture["same_tenant_other_sensor_id"], run_b.id, "bearing-b", 10.0),
    ]:
        for index in range(4):
            repo.create_analytics_feature(
                fixture["organization_id"],
                run_id=run_id,
                sensor_id=sensor_id,
                batch_id=None,
                source_kind="scalar",
                source_record_id=f"{group}-{index}",
                observed_at=base_time + timedelta(minutes=index),
                feature_name="scalar.rms",
                value=float(index + offset),
                unit="g",
                quality="good",
                algorithm_version="analytics-v1",
                provenance={
                    "target_rul_hours": float(8 - index - offset / 10),
                    "validation_group": group,
                },
            )


def _real_serving_prediction(session, fixture, tmp_path) -> str:
    _seed_real_serving_features(session, fixture)
    ml_service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
    dataset = ml_service.create_dataset_version(
        fixture["organization_id"],
        DatasetVersionCreate(name="maintenance-serving-features", version="v1", feature_names=["scalar.rms"]),
    )
    experiment = ml_service.run_experiment(
        fixture["organization_id"],
        ExperimentCreate(
            dataset_version_id=dataset.id,
            name="maintenance serving experiment",
            training_config={"n_estimators": 5, "random_state": 17},
        ),
    )
    registry = ml_service.create_registry(
        fixture["organization_id"],
        RegistryCreate(name="maintenance-bearing-rul", task="rul_regression"),
    )
    model_version = ml_service.register_model_version(
        fixture["organization_id"],
        ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.0"),
    )
    ml_service.promote_model_version(
        fixture["organization_id"],
        model_version.id,
        PromoteModelVersion(target_stage="validated"),
    )
    ml_service.promote_model_version(
        fixture["organization_id"],
        model_version.id,
        PromoteModelVersion(
            target_stage="production",
            approved_by_user_id=fixture["manager_id"],
            reason="approved for maintenance integration test",
        ),
    )
    ProductionServingService(session).bind_model(
        fixture["organization_id"],
        ServingBindingCreate(
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="sensor",
            scope_id=fixture["sensor_id"],
            approved_by_user_id=fixture["manager_id"],
            reason="serve maintenance integration test",
        ),
    )
    prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
        fixture["organization_id"],
        PredictionRequest(sensor_id=fixture["sensor_id"], registry_id=registry.id),
    )
    assert prediction.prediction_status == "supported"
    return prediction.id


def test_migration_creates_maintenance_operations_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert {
        "maintenance_alerts",
        "maintenance_cases",
        "maintenance_notes",
        "maintenance_inspections",
        "maintenance_work_orders",
        "cmms_sync_records",
    }.issubset(tables)


def test_supported_prediction_crossing_threshold_creates_evidence_backed_alert(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        result = _service_alert(service, maintenance_fixture, threshold=24.0)
        session.commit()

        assert result.created is True
        assert result.alert["alert_kind"] == "rul_threshold"
        assert result.alert["evidence_snapshot"]["prediction_status"] == "supported"
        assert result.alert["evidence_snapshot"]["predicted_rul_hours"] == 12.0
        assert result.alert["evidence_snapshot"]["feature_record_ids"] == ["feature-1", "feature-2"]
        snapshot = result.alert["evidence_snapshot"]
        assert snapshot["prediction_provenance"]["serving_slice"] == "production-slice-10"
        assert snapshot["model_resolution"]["artifact_sha256"] == "abc123"
        assert snapshot["model_resolution"]["feature_schema"] == ["scalar.rms", "scalar.kurtosis"]
        assert result.alert["evidence"]["rule"]["rul_threshold_hours"] == 24.0
        assert result.alert["evidence"]["maintenance_fact"] is False


def test_supported_prediction_not_crossing_threshold_does_not_create_false_alert(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        result = _service_alert(
            service,
            maintenance_fixture,
            prediction_id=maintenance_fixture["supported_high_id"],
            threshold=24.0,
        )
        session.commit()

        assert result.alert is None
        assert result.created is False
        assert "did not cross" in result.reason
        assert session.scalar(select(func.count()).select_from(MaintenanceAlert)) == 0


def test_abstained_prediction_creates_only_evidence_review_without_rul_claim(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        result = _service_alert(
            service,
            maintenance_fixture,
            prediction_id=maintenance_fixture["insufficient_id"],
            threshold=24.0,
            rule_id="review-stale-evidence",
        )
        session.commit()

        assert result.alert["alert_kind"] == "evidence_review"
        assert result.alert["source_reason_code"] == "STALE_FEATURES"
        assert "predicted_rul_hours" not in result.alert["evidence_snapshot"]
        assert "imminent" not in result.alert["summary"].lower()
        assert result.alert["evidence_snapshot"]["prediction_status"] == "insufficient_evidence"


def test_repeated_prediction_rule_evaluation_reuses_active_alert(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        first = _service_alert(service, maintenance_fixture)
        second = _service_alert(service, maintenance_fixture)
        session.commit()

        assert first.alert["id"] == second.alert["id"]
        assert second.created is False
        assert session.scalar(select(func.count()).select_from(MaintenanceAlert)) == 1


def test_cross_tenant_paths_and_human_membership_are_enforced(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        with pytest.raises(ValueError, match="prediction does not exist inside this organization"):
            service.evaluate_prediction_alert(
                maintenance_fixture["other_organization_id"],
                PredictionAlertEvaluationRequest(
                    prediction_id=maintenance_fixture["supported_low_id"],
                    rule_id="other-org-rule",
                    rul_threshold_hours=24.0,
                ),
            )
        case = _active_case(service, maintenance_fixture)

        with pytest.raises(ValueError, match="active member"):
            service.acknowledge_alert(
                maintenance_fixture["organization_id"],
                case.alert_id,
                AlertAcknowledgeRequest(acknowledged_by_user_id=maintenance_fixture["outsider_id"]),
            )
        with pytest.raises(ValueError, match="case must belong"):
            service.request_inspection(
                maintenance_fixture["other_organization_id"],
                case.id,
                InspectionRequestCreate(
                    requested_by_user_id=maintenance_fixture["other_user_id"],
                    requested_reason="bad tenant",
                ),
            )
        with pytest.raises(ValueError, match="case must belong"):
            service.create_work_order(
                maintenance_fixture["other_organization_id"],
                case.id,
                WorkOrderCreate(
                    requested_by_user_id=maintenance_fixture["other_user_id"],
                    title="Wrong tenant",
                    requested_work="Should fail.",
                ),
            )


def test_alert_acknowledgement_and_resolution_are_explicit_human_actions(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        alert = _service_alert(service, maintenance_fixture).alert
        acknowledged = service.acknowledge_alert(
            maintenance_fixture["organization_id"],
            alert["id"],
            AlertAcknowledgeRequest(
                acknowledged_by_user_id=maintenance_fixture["technician_id"],
                note="I will review this evidence.",
            ),
        )
        assert acknowledged.status == "acknowledged"
        assert acknowledged.acknowledged_by_user_id == maintenance_fixture["technician_id"]
        assert acknowledged.acknowledged_at is not None
        resolved = service.resolve_alert(
            maintenance_fixture["organization_id"],
            alert["id"],
            AlertResolveRequest(
                resolved_by_user_id=maintenance_fixture["manager_id"],
                disposition="dismissed",
                reason="Reviewed by technician; no case opened for this training fixture.",
            ),
        )
        session.commit()

        assert resolved.status == "dismissed"
        assert resolved.resolved_by_user_id == maintenance_fixture["manager_id"]
        assert resolved.disposition_reason


def test_case_creation_from_alert_is_idempotent_while_active(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        alert = _service_alert(service, maintenance_fixture).alert
        request = CaseCreateFromAlertRequest(opened_by_user_id=maintenance_fixture["technician_id"])
        first = service.open_case_from_alert(maintenance_fixture["organization_id"], alert["id"], request)
        second = service.open_case_from_alert(maintenance_fixture["organization_id"], alert["id"], request)
        session.commit()

        assert first.id == second.id
        assert session.scalar(select(func.count()).select_from(MaintenanceCase)) == 1
        source_snapshot = first.evidence["source_evidence_snapshot"]
        assert source_snapshot["prediction_record_id"] == maintenance_fixture["supported_low_id"]


def test_case_resolution_must_use_first_class_resolution_record(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        case = _active_case(service, maintenance_fixture)

        with pytest.raises(ValueError, match="dedicated case resolution endpoint"):
            service.transition_case(
                maintenance_fixture["organization_id"],
                case.id,
                CaseTransitionRequest(actor_user_id=maintenance_fixture["manager_id"], target_status="resolved"),
            )
        assert session.scalar(select(func.count()).select_from(MaintenanceResolution)) == 0

        service.resolve_case(
            maintenance_fixture["organization_id"],
            case.id,
            ResolutionCreate(
                resolved_by_user_id=maintenance_fixture["manager_id"],
                outcome="monitor",
                summary="Resolved after human review.",
            ),
        )
        closed = service.transition_case(
            maintenance_fixture["organization_id"],
            case.id,
            CaseTransitionRequest(actor_user_id=maintenance_fixture["manager_id"], target_status="closed"),
        )
        session.commit()

        assert session.scalar(select(func.count()).select_from(MaintenanceResolution)) == 1
        assert closed.status == "closed"


def test_case_and_inspection_reject_same_tenant_wrong_hierarchy(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        alert = _service_alert(service, maintenance_fixture).alert

        with pytest.raises(ValueError, match="hierarchy must match source evidence"):
            service.open_case(
                maintenance_fixture["organization_id"],
                CaseCreate(
                    alert_id=alert["id"],
                    title="Bad hierarchy case",
                    opened_by_user_id=maintenance_fixture["technician_id"],
                    sensor_id=maintenance_fixture["same_tenant_other_sensor_id"],
                ),
            )

        with pytest.raises(ValueError, match="hierarchy does not match"):
            service.open_case(
                maintenance_fixture["organization_id"],
                CaseCreate(
                    title="Manual bad hierarchy case",
                    opened_by_user_id=maintenance_fixture["technician_id"],
                    asset_id=maintenance_fixture["asset_id"],
                    sensor_id=maintenance_fixture["same_tenant_other_sensor_id"],
                ),
            )

        case = service.open_case_from_alert(
            maintenance_fixture["organization_id"],
            alert["id"],
            CaseCreateFromAlertRequest(opened_by_user_id=maintenance_fixture["technician_id"]),
        )
        with pytest.raises(ValueError, match="hierarchy must match source evidence"):
            service.request_inspection(
                maintenance_fixture["organization_id"],
                case.id,
                InspectionRequestCreate(
                    requested_by_user_id=maintenance_fixture["technician_id"],
                    requested_reason="Should not point at a different bearing.",
                    sensor_id=maintenance_fixture["same_tenant_other_sensor_id"],
                ),
            )


def test_manual_case_requires_active_human_opener(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    app_main = importlib.reload(importlib.import_module("app.main"))
    client = TestClient(app_main.app)
    response = client.post(
        f"/api/maintenance/{maintenance_fixture['organization_id']}/cases",
        json={"title": "Manual case", "priority": "medium"},
    )
    assert response.status_code == 422

    with pytest.raises(ValidationError):
        CaseCreate(title="Manual case")

    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        with pytest.raises(ValueError, match="active member"):
            service.open_case(
                maintenance_fixture["organization_id"],
                CaseCreate(
                    title="Manual case",
                    opened_by_user_id=maintenance_fixture["outsider_id"],
                ),
            )


def test_alert_snapshot_uses_real_slice_10_prediction_resolution(migrated_db, maintenance_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        prediction_id = _real_serving_prediction(session, maintenance_fixture, tmp_path)
        service = MaintenanceOperationsService(session)
        result = service.evaluate_prediction_alert(
            maintenance_fixture["organization_id"],
            PredictionAlertEvaluationRequest(
                prediction_id=prediction_id,
                rule_id="real-serving-rul-under-999h",
                rul_threshold_hours=999.0,
            ),
        )
        session.commit()

        snapshot = result.alert["evidence_snapshot"]
        prediction = session.get(PredictionRecord, prediction_id)
        assert result.created is True
        assert snapshot["prediction_record_id"] == prediction_id
        assert snapshot["model_version_id"] == prediction.model_version_id
        assert snapshot["dataset_version_id"] == prediction.dataset_version_id
        assert snapshot["model_resolution_id"] == prediction.model_resolution_id
        assert snapshot["feature_record_ids"]
        assert snapshot["prediction_provenance"]["serving_slice"] == "production-slice-10"
        assert snapshot["prediction_provenance"]["artifact_sha256"] == snapshot["model_resolution"]["artifact_sha256"]
        assert snapshot["model_resolution"]["id"] == prediction.model_resolution_id


def test_technician_notes_are_append_only_and_human_authored(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        case = _active_case(service, maintenance_fixture)
        first = service.add_note(
            maintenance_fixture["organization_id"],
            case.id,
            NoteCreate(author_user_id=maintenance_fixture["technician_id"], body="Checked pump housing."),
        )
        second = service.add_note(
            maintenance_fixture["organization_id"],
            case.id,
            NoteCreate(author_user_id=maintenance_fixture["technician_id"], body="Vibration manually rechecked."),
        )
        with pytest.raises(ValueError, match="human-authored"):
            service.add_note(
                maintenance_fixture["organization_id"],
                case.id,
                NoteCreate(
                    author_user_id=maintenance_fixture["technician_id"],
                    body="The model says this was inspected.",
                    note_kind="ai_generated",
                ),
            )
        session.commit()

        notes = service.list_notes(maintenance_fixture["organization_id"], case.id)
        assert {note["id"] for note in notes} == {first.id, second.id}
        assert {note["body"] for note in notes} == {"Checked pump housing.", "Vibration manually rechecked."}
        assert all(note["metadata"]["human_authored"] is True for note in notes)


def test_inspection_lifecycle_requires_transitions_and_human_findings(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        case = _active_case(service, maintenance_fixture)
        inspection = service.request_inspection(
            maintenance_fixture["organization_id"],
            case.id,
            InspectionRequestCreate(
                requested_by_user_id=maintenance_fixture["technician_id"],
                requested_reason="Unsupported evidence requires human check.",
            ),
        )
        with pytest.raises(ValueError, match="invalid inspection transition"):
            service.complete_inspection(
                maintenance_fixture["organization_id"],
                inspection.id,
                InspectionCompleteRequest(
                    performed_by_user_id=maintenance_fixture["technician_id"],
                    condition="watch",
                    findings="Cannot complete before starting.",
                ),
            )
        service.start_inspection(
            maintenance_fixture["organization_id"],
            inspection.id,
            InspectionStartRequest(started_by_user_id=maintenance_fixture["technician_id"]),
        )
        completed = service.complete_inspection(
            maintenance_fixture["organization_id"],
            inspection.id,
            InspectionCompleteRequest(
                performed_by_user_id=maintenance_fixture["technician_id"],
                condition="degraded",
                findings="Audible bearing noise and elevated temperature.",
                recommended_follow_up="Schedule bearing replacement.",
            ),
        )
        with pytest.raises(ValueError, match="invalid inspection transition"):
            service.cancel_inspection(
                maintenance_fixture["organization_id"],
                inspection.id,
                InspectionCancelRequest(
                    cancelled_by_user_id=maintenance_fixture["technician_id"],
                    reason="Too late.",
                ),
            )
        session.commit()

        assert completed.status == "completed"
        assert completed.findings.startswith("Audible")
        assert completed.evidence_metadata["observation_kind"] == "technician_observation"
        assert completed.evidence_metadata["maintenance_fact"] is True


def test_work_order_lifecycle_requires_human_approval_and_completion(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        work_order = _draft_work_order(service, maintenance_fixture)
        original_prediction = session.get(PredictionRecord, maintenance_fixture["supported_low_id"])
        original_prediction_snapshot = {
            "status": original_prediction.prediction_status,
            "rul": original_prediction.predicted_rul_hours,
            "provenance": dict(original_prediction.provenance),
        }

        with pytest.raises(ValueError, match="invalid work-order transition"):
            service.start_work_order(
                maintenance_fixture["organization_id"],
                work_order.id,
                WorkOrderStartRequest(started_by_user_id=maintenance_fixture["technician_id"]),
            )
        with pytest.raises(ValueError, match="active member"):
            service.approve_work_order(
                maintenance_fixture["organization_id"],
                work_order.id,
                WorkOrderApproveRequest(approved_by_user_id=maintenance_fixture["outsider_id"]),
            )
        approved = service.approve_work_order(
            maintenance_fixture["organization_id"],
            work_order.id,
            WorkOrderApproveRequest(
                approved_by_user_id=maintenance_fixture["manager_id"],
                note="Approved after human review.",
            ),
        )
        with pytest.raises(ValueError, match="invalid work-order transition"):
            service.complete_work_order(
                maintenance_fixture["organization_id"],
                work_order.id,
                WorkOrderCompleteRequest(
                    completed_by_user_id=maintenance_fixture["technician_id"],
                    completion_notes="Cannot complete before start.",
                ),
            )
        started = service.start_work_order(
            maintenance_fixture["organization_id"],
            work_order.id,
            WorkOrderStartRequest(started_by_user_id=maintenance_fixture["technician_id"]),
        )
        completed = service.complete_work_order(
            maintenance_fixture["organization_id"],
            work_order.id,
            WorkOrderCompleteRequest(
                completed_by_user_id=maintenance_fixture["technician_id"],
                completion_notes="Bearing replaced and vibration checked.",
                work_performed="Replaced drive-end bearing.",
            ),
        )
        session.commit()

        session.refresh(original_prediction)
        assert work_order.status == "completed"
        assert approved.approved_by_user_id == maintenance_fixture["manager_id"]
        assert started.started_at is not None
        assert completed.work_performed == "Replaced drive-end bearing."
        assert original_prediction.prediction_status == original_prediction_snapshot["status"]
        assert original_prediction.predicted_rul_hours == original_prediction_snapshot["rul"]
        assert original_prediction.provenance == original_prediction_snapshot["provenance"]


def test_disabled_cmms_adapter_persists_truthful_not_configured_state(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(operation="create", initiated_by_user_id=maintenance_fixture["technician_id"]),
        )
        session.commit()

        assert sync.status == "not_configured"
        assert sync.initiator_type == "user"
        assert sync.initiated_by_user_id == maintenance_fixture["technician_id"]
        assert sync.external_id is None
        assert sync.error_category == "not_configured"
        assert work_order.cmms_provider == "disabled"
        assert work_order.cmms_external_id is None
        assert work_order.cmms_state == "not_configured"


def test_deterministic_cmms_adapter_and_create_idempotency(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = DeterministicCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        request = CmmsSyncRequest(
            operation="create",
            provider_name=adapter.provider_name,
            initiated_by_user_id=maintenance_fixture["technician_id"],
        )

        first = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        second = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        session.commit()

        assert first.status == "succeeded"
        assert first.external_id is not None
        assert second.status == "skipped"
        assert second.external_id == first.external_id
        assert first.initiated_by_user_id == maintenance_fixture["technician_id"]
        assert second.initiated_by_user_id == maintenance_fixture["technician_id"]
        assert len(adapter.calls) == 1
        assert session.scalar(select(func.count()).select_from(CmmsSyncRecord)) == 2


def test_cmms_sync_requires_active_member_and_keeps_bound_provider_for_later_operations(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        primary = DeterministicCmmsAdapter()
        alternate = AlternateDeterministicCmmsAdapter()
        service = MaintenanceOperationsService(
            session,
            cmms_adapters={primary.provider_name: primary, alternate.provider_name: alternate},
        )
        work_order = _draft_work_order(service, maintenance_fixture)

        with pytest.raises(ValueError, match="active member"):
            service.sync_work_order_to_cmms(
                maintenance_fixture["organization_id"],
                work_order.id,
                CmmsSyncRequest(
                    operation="create",
                    provider_name=primary.provider_name,
                    initiated_by_user_id=maintenance_fixture["outsider_id"],
                ),
            )

        create_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=primary.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        original_provider = work_order.cmms_provider
        original_external_id = work_order.cmms_external_id

        for operation in ["create", "update", "cancel", "close"]:
            with pytest.raises(ValueError, match="already bound to a different CMMS provider"):
                service.sync_work_order_to_cmms(
                    maintenance_fixture["organization_id"],
                    work_order.id,
                    CmmsSyncRequest(
                        operation=operation,
                        provider_name=alternate.provider_name,
                        initiated_by_user_id=maintenance_fixture["technician_id"],
                    ),
                )
        update_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert create_sync.status == "succeeded"
        assert update_sync.status == "succeeded"
        assert update_sync.provider_name == primary.provider_name
        assert update_sync.external_id == original_external_id
        assert work_order.cmms_provider == original_provider
        assert work_order.cmms_external_id == original_external_id
        assert len(primary.calls) == 2
        assert primary.calls[-1]["operation"] == "update"
        assert len(alternate.calls) == 0
        assert session.scalar(select(func.count()).select_from(CmmsSyncRecord)) == 2


def test_cmms_update_without_bound_provider_uses_disabled_adapter_truthfully(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = MaintenanceOperationsService(session)
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert sync.status == "not_configured"
        assert sync.provider_name == "disabled"
        assert sync.external_id is None
        assert work_order.cmms_provider == "disabled"
        assert work_order.cmms_external_id is None


def test_cmms_bound_provider_is_used_when_no_provider_is_supplied(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        primary = DeterministicCmmsAdapter()
        alternate = AlternateDeterministicCmmsAdapter()
        service = MaintenanceOperationsService(
            session,
            cmms_adapters={primary.provider_name: primary, alternate.provider_name: alternate},
        )
        work_order = _draft_work_order(service, maintenance_fixture)
        create_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=primary.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        update_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        with pytest.raises(ValueError, match="already bound to a different CMMS provider"):
            service.sync_work_order_to_cmms(
                maintenance_fixture["organization_id"],
                work_order.id,
                CmmsSyncRequest(
                    operation="close",
                    provider_name=alternate.provider_name,
                    initiated_by_user_id=maintenance_fixture["technician_id"],
                ),
            )
        session.commit()

        assert work_order.cmms_provider == primary.provider_name
        assert work_order.cmms_external_id == create_sync.external_id
        assert update_sync.provider_name == primary.provider_name
        assert update_sync.external_id == create_sync.external_id
        assert [call["operation"] for call in primary.calls] == ["create", "update"]
        assert len(alternate.calls) == 0


def test_cmms_non_create_success_without_external_id_preserves_bound_identifier(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        primary = NoEchoUpdateCmmsAdapter()
        alternate = AlternateDeterministicCmmsAdapter()
        service = MaintenanceOperationsService(
            session,
            cmms_adapters={primary.provider_name: primary, alternate.provider_name: alternate},
        )
        work_order = _draft_work_order(service, maintenance_fixture)
        create_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=primary.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        update_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        with pytest.raises(ValueError, match="already bound to a different CMMS provider"):
            service.sync_work_order_to_cmms(
                maintenance_fixture["organization_id"],
                work_order.id,
                CmmsSyncRequest(
                    operation="update",
                    provider_name=alternate.provider_name,
                    initiated_by_user_id=maintenance_fixture["technician_id"],
                ),
            )
        session.commit()

        assert create_sync.status == "succeeded"
        assert create_sync.external_id == "A-123"
        assert update_sync.status == "succeeded"
        assert update_sync.external_id == "A-123"
        assert work_order.cmms_provider == primary.provider_name
        assert work_order.cmms_external_id == "A-123"
        assert [call["operation"] for call in primary.calls] == ["create", "update"]
        assert len(alternate.calls) == 0


def test_cmms_create_success_without_external_id_is_failed_adapter_result(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = MissingCreateExternalIdCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert sync.status == "failed"
        assert sync.error_category == "invalid_adapter_result"
        assert sync.external_id is None
        assert sync.attempt_metadata["adapter_reported_status"] == "succeeded"
        assert work_order.cmms_provider is None
        assert work_order.cmms_external_id is None
        assert len(adapter.calls) == 1


def test_cmms_non_create_success_with_different_external_id_fails_closed(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = MismatchedExternalIdCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        create_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        update_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert create_sync.status == "succeeded"
        assert update_sync.status == "failed"
        assert update_sync.error_category == "invalid_adapter_result"
        assert update_sync.external_id == "B-999"
        assert work_order.cmms_provider == adapter.provider_name
        assert work_order.cmms_external_id == "A-123"
        assert [call["operation"] for call in adapter.calls] == ["create", "update"]


def test_cmms_create_timeout_persists_timeout_record_without_external_id(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = TimeoutCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert sync.status == "timeout"
        assert sync.provider_name == adapter.provider_name
        assert sync.external_id is None
        assert sync.error_category == "timeout"
        assert "secret" not in sync.error_message.lower()
        assert work_order.cmms_provider is None
        assert work_order.cmms_external_id is None
        assert len(adapter.calls) == 1


def test_cmms_bound_provider_update_timeout_preserves_existing_external_id(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = CreateThenTimeoutCmmsAdapter()
        alternate = AlternateDeterministicCmmsAdapter()
        service = MaintenanceOperationsService(
            session,
            cmms_adapters={adapter.provider_name: adapter, alternate.provider_name: alternate},
        )
        work_order = _draft_work_order(service, maintenance_fixture)
        create_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        update_sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="update",
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        with pytest.raises(ValueError, match="already bound to a different CMMS provider"):
            service.sync_work_order_to_cmms(
                maintenance_fixture["organization_id"],
                work_order.id,
                CmmsSyncRequest(
                    operation="update",
                    provider_name=alternate.provider_name,
                    initiated_by_user_id=maintenance_fixture["technician_id"],
                ),
            )
        session.commit()

        assert create_sync.status == "succeeded"
        assert create_sync.external_id == "A-123"
        assert update_sync.status == "timeout"
        assert update_sync.provider_name == adapter.provider_name
        assert update_sync.external_id == "A-123"
        assert update_sync.error_category == "timeout"
        assert work_order.cmms_provider == adapter.provider_name
        assert work_order.cmms_external_id == "A-123"
        assert [call["operation"] for call in adapter.calls] == ["create", "update"]
        assert len(alternate.calls) == 0


def test_cmms_generic_adapter_exception_persists_failed_record_without_leaking_details(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = FailingCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert sync.status == "failed"
        assert sync.error_category == "adapter_error"
        assert "secret" not in sync.error_message.lower()
        assert "password" not in sync.error_message.lower()
        assert sync.external_id is None
        assert work_order.cmms_provider is None
        assert work_order.cmms_external_id is None
        assert len(adapter.calls) == 1


def test_cmms_unsupported_adapter_status_becomes_failed_invalid_adapter_result(
    migrated_db,
    maintenance_fixture,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = UnsupportedStatusCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        sync = service.sync_work_order_to_cmms(
            maintenance_fixture["organization_id"],
            work_order.id,
            CmmsSyncRequest(
                operation="create",
                provider_name=adapter.provider_name,
                initiated_by_user_id=maintenance_fixture["technician_id"],
            ),
        )
        session.commit()

        assert sync.status == "failed"
        assert sync.error_category == "invalid_adapter_result"
        assert sync.error_message == "CMMS adapter returned an unsupported sync status"
        assert sync.external_id is None
        assert sync.attempt_metadata["adapter_reported_status"] == "made_up_status"
        assert work_order.cmms_provider is None
        assert work_order.cmms_external_id is None
        assert len(adapter.calls) == 1


def test_cmms_timeout_retry_reuses_provider_aware_idempotency_key(migrated_db, maintenance_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        adapter = TimeoutCmmsAdapter()
        service = MaintenanceOperationsService(session, cmms_adapters={adapter.provider_name: adapter})
        work_order = _draft_work_order(service, maintenance_fixture)
        request = CmmsSyncRequest(
            operation="create",
            provider_name=adapter.provider_name,
            initiated_by_user_id=maintenance_fixture["technician_id"],
        )

        first = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        second = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        session.commit()

        assert first.status == "timeout"
        assert second.status == "timeout"
        assert first.idempotency_key == second.idempotency_key
        assert adapter.calls[0]["idempotency_key"] == adapter.calls[1]["idempotency_key"]
        assert first.provider_name == adapter.provider_name
        assert second.provider_name == adapter.provider_name
        assert work_order.cmms_provider is None
        assert work_order.cmms_external_id is None


def test_api_full_maintenance_workflow(migrated_db, maintenance_fixture):
    _engine, _session_factory = migrated_db
    app_main = importlib.reload(importlib.import_module("app.main"))
    client = TestClient(app_main.app)
    org_id = maintenance_fixture["organization_id"]
    technician = maintenance_fixture["technician_id"]
    manager = maintenance_fixture["manager_id"]

    alert = client.post(
        f"/api/maintenance/{org_id}/alerts/evaluate-prediction",
        json={
            "prediction_id": maintenance_fixture["supported_low_id"],
            "rule_id": "rul-under-24h",
            "rule_name": "RUL under 24 hours",
            "rul_threshold_hours": 24.0,
            "priority": "high",
            "recommended_action": "Human review before any maintenance action.",
        },
    )
    assert alert.status_code == 200
    alert_payload = alert.json()["alert"]
    assert alert_payload["evidence_snapshot"]["predicted_rul_hours"] == 12.0

    acknowledged = client.post(
        f"/api/maintenance/{org_id}/alerts/{alert_payload['id']}/acknowledge",
        json={"acknowledged_by_user_id": technician, "note": "Review accepted."},
    )
    case = client.post(
        f"/api/maintenance/{org_id}/alerts/{alert_payload['id']}/case",
        json={"opened_by_user_id": technician, "priority": "high"},
    )
    assert case.status_code == 200
    case_payload = case.json()
    note = client.post(
        f"/api/maintenance/{org_id}/cases/{case_payload['id']}/notes",
        json={"author_user_id": technician, "body": "Technician will inspect during next stop."},
    )
    inspection = client.post(
        f"/api/maintenance/{org_id}/cases/{case_payload['id']}/inspections",
        json={"requested_by_user_id": technician, "requested_reason": "Confirm model evidence."},
    )
    assert inspection.status_code == 200
    inspection_id = inspection.json()["id"]
    inspection_start = client.post(
        f"/api/maintenance/{org_id}/inspections/{inspection_id}/start",
        json={"started_by_user_id": technician},
    )
    inspection_complete = client.post(
        f"/api/maintenance/{org_id}/inspections/{inspection_id}/complete",
        json={
            "performed_by_user_id": technician,
            "condition": "degraded",
            "findings": "Elevated vibration confirmed by handheld meter.",
            "recommended_follow_up": "Replace bearing.",
        },
    )
    work_order = client.post(
        f"/api/maintenance/{org_id}/cases/{case_payload['id']}/work-orders",
        json={
            "requested_by_user_id": technician,
            "title": "Replace drive-end bearing",
            "requested_work": "Replace drive-end bearing and verify baseline.",
            "priority": "high",
        },
    )
    assert work_order.status_code == 200
    work_order_id = work_order.json()["id"]
    approved = client.post(
        f"/api/maintenance/{org_id}/work-orders/{work_order_id}/approve",
        json={"approved_by_user_id": manager, "note": "Approved after inspection."},
    )
    sync = client.post(
        f"/api/maintenance/{org_id}/work-orders/{work_order_id}/cmms-sync",
        json={"operation": "create", "initiated_by_user_id": technician},
    )
    work_start = client.post(
        f"/api/maintenance/{org_id}/work-orders/{work_order_id}/start",
        json={"started_by_user_id": technician},
    )
    work_complete = client.post(
        f"/api/maintenance/{org_id}/work-orders/{work_order_id}/complete",
        json={
            "completed_by_user_id": technician,
            "completion_notes": "Bearing replaced and post-work reading captured.",
            "work_performed": "Replaced drive-end bearing.",
        },
    )
    resolution = client.post(
        f"/api/maintenance/{org_id}/cases/{case_payload['id']}/resolve",
        json={
            "resolved_by_user_id": manager,
            "outcome": "replaced",
            "summary": "Work completed by technician and reviewed by manager.",
        },
    )
    case_history = client.get(f"/api/maintenance/{org_id}/cases/{case_payload['id']}")
    health = client.get(f"/api/maintenance/{org_id}/health")

    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "acknowledged"
    assert note.status_code == 200
    assert note.json()["metadata"]["human_authored"] is True
    assert inspection_start.status_code == 200
    assert inspection_complete.status_code == 200
    assert inspection_complete.json()["evidence_metadata"]["maintenance_fact"] is True
    assert work_order.json()["status"] == "draft"
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert sync.status_code == 200
    assert sync.json()["status"] == "not_configured"
    assert sync.json()["initiated_by_user_id"] == technician
    assert sync.json()["external_id"] is None
    assert work_start.status_code == 200
    assert work_complete.status_code == 200
    assert work_complete.json()["status"] == "completed"
    assert resolution.status_code == 200
    assert resolution.json()["evidence"]["resolution_kind"] == "human_decision"
    assert case_history.status_code == 200
    assert len(case_history.json()["notes"]) == 1
    assert len(case_history.json()["inspections"]) == 1
    assert len(case_history.json()["work_orders"]) == 1
    assert health.status_code == 200
    assert health.json()["alerts"] == 1
    assert health.json()["cases"] == 1
    assert health.json()["work_orders"] == 1
