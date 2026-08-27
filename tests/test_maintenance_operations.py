import importlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from maintenance_operations.contracts import (
    AlertAcknowledgeRequest,
    AlertResolveRequest,
    CaseCreateFromAlertRequest,
    CmmsSyncRequest,
    InspectionCancelRequest,
    InspectionCompleteRequest,
    InspectionRequestCreate,
    InspectionStartRequest,
    NoteCreate,
    PredictionAlertEvaluationRequest,
    WorkOrderApproveRequest,
    WorkOrderCompleteRequest,
    WorkOrderCreate,
    WorkOrderStartRequest,
)
from maintenance_operations.service import DeterministicCmmsAdapter, MaintenanceOperationsService
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.database import make_engine
from platform_core.models import Base, CmmsSyncRecord, MaintenanceAlert, MaintenanceCase, PredictionRecord
from platform_core.repositories import PlatformRepository

ROOT = Path(__file__).resolve().parent.parent


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
            "model_resolution": {
                "binding_id": "binding-1" if status == "supported" else None,
                "artifact_sha256": resolution.artifact_sha256,
                "feature_schema": resolution.feature_schema,
                "source_model_version": "fixture-model-v1" if status == "supported" else None,
                "source_dataset_version": "fixture-dataset-v1" if status == "supported" else None,
            }
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
        provenance = result.alert["evidence_snapshot"]["model_dataset_provenance"]
        assert provenance["source_model_version"] == "fixture-model-v1"
        assert provenance["source_dataset_version"] == "fixture-dataset-v1"
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
            CmmsSyncRequest(operation="create"),
        )
        session.commit()

        assert sync.status == "not_configured"
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
        request = CmmsSyncRequest(operation="create", provider_name=adapter.provider_name)

        first = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        second = service.sync_work_order_to_cmms(maintenance_fixture["organization_id"], work_order.id, request)
        session.commit()

        assert first.status == "succeeded"
        assert first.external_id is not None
        assert second.status == "skipped"
        assert second.external_id == first.external_id
        assert len(adapter.calls) == 1
        assert session.scalar(select(func.count()).select_from(CmmsSyncRecord)) == 2


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
        json={"operation": "create"},
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
