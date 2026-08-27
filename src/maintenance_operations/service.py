"""Maintenance operations workflow over model evidence and human action records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from maintenance_operations.contracts import (
    AlertAcknowledgeRequest,
    AlertEvaluationResult,
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
from platform_core.models import (
    CmmsSyncRecord,
    MaintenanceAlert,
    MaintenanceCase,
    MaintenanceInspection,
    MaintenanceNote,
    MaintenanceResolution,
    MaintenanceWorkOrder,
)
from platform_core.repositories import PlatformRepository

ACTIVE_ALERT_STATUSES = {"open", "acknowledged"}
ACTIVE_CASE_STATUSES = {"open", "in_progress"}
CASE_TRANSITIONS = {
    "open": {"in_progress"},
    "in_progress": set(),
    "resolved": {"closed"},
    "closed": set(),
}
INSPECTION_TRANSITIONS = {
    "requested": {"in_progress", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
WORK_ORDER_TRANSITIONS = {
    "draft": {"approved", "cancelled"},
    "approved": {"in_progress", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


class MaintenanceOperationsError(ValueError):
    pass


@dataclass(frozen=True)
class CmmsAdapterResult:
    status: str
    external_id: str | None = None
    external_status: str | None = None
    error_category: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class CmmsAdapter(Protocol):
    provider_name: str

    def sync(self, operation: str, work_order: MaintenanceWorkOrder, *, idempotency_key: str) -> CmmsAdapterResult:
        ...


class DisabledCmmsAdapter:
    provider_name = "disabled"

    def sync(self, operation: str, work_order: MaintenanceWorkOrder, *, idempotency_key: str) -> CmmsAdapterResult:
        return CmmsAdapterResult(
            status="not_configured",
            error_category="not_configured",
            error_message="no CMMS adapter is configured for this organization",
            metadata={"operation": operation, "work_order_id": work_order.id, "idempotency_key": idempotency_key},
        )


class DeterministicCmmsAdapter:
    provider_name = "deterministic-test"

    def __init__(self):
        self.created: dict[str, str] = {}
        self.calls: list[dict[str, str]] = []

    def sync(self, operation: str, work_order: MaintenanceWorkOrder, *, idempotency_key: str) -> CmmsAdapterResult:
        self.calls.append({"operation": operation, "idempotency_key": idempotency_key})
        if operation == "create":
            external_id = self.created.setdefault(
                idempotency_key,
                f"TEST-WO-{hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]}",
            )
            return CmmsAdapterResult(
                status="succeeded",
                external_id=external_id,
                external_status="created",
                metadata={"idempotency_key": idempotency_key},
            )
        if not work_order.cmms_external_id:
            return CmmsAdapterResult(
                status="failed",
                error_category="missing_external_id",
                error_message="work order has no external CMMS identifier",
            )
        return CmmsAdapterResult(
            status="succeeded",
            external_id=work_order.cmms_external_id,
            external_status=operation,
            metadata={"idempotency_key": idempotency_key},
        )


class MaintenanceOperationsService:
    def __init__(self, session: Session, cmms_adapters: dict[str, CmmsAdapter] | None = None):
        self.session = session
        self.repo = PlatformRepository(session)
        self.cmms_adapters = cmms_adapters or {"disabled": DisabledCmmsAdapter()}

    def evaluate_prediction_alert(
        self,
        organization_id: str,
        request: PredictionAlertEvaluationRequest,
    ) -> AlertEvaluationResult:
        prediction = self.repo.get_prediction_record(organization_id, request.prediction_id)
        if prediction is None:
            raise MaintenanceOperationsError("prediction does not exist inside this organization")
        rule = _rule_payload(request)
        resolution = self.repo.get_production_model_resolution(organization_id, prediction.model_resolution_id)
        snapshot = _prediction_evidence_snapshot(prediction, resolution)
        if prediction.prediction_status == "supported":
            if request.rul_threshold_hours is None:
                raise MaintenanceOperationsError("supported RUL alert evaluation requires an explicit RUL threshold")
            if prediction.predicted_rul_hours is None or prediction.predicted_rul_hours > request.rul_threshold_hours:
                return AlertEvaluationResult(
                    alert=None,
                    created=False,
                    reason="supported prediction did not cross the supplied RUL threshold",
                    rule=rule,
                )
            alert_kind = "rul_threshold"
            title = "RUL threshold crossed"
            summary = (
                f"Supported RUL prediction {prediction.predicted_rul_hours:.2f}h crossed "
                f"threshold {request.rul_threshold_hours:.2f}h."
            )
            severity = request.severity or _severity_from_rul(prediction.predicted_rul_hours)
        elif request.create_evidence_review_for_abstention:
            alert_kind = "evidence_review"
            title = "Prediction evidence requires human review"
            summary = _abstention_summary(prediction)
            severity = request.severity or "watch"
        else:
            return AlertEvaluationResult(
                alert=None,
                created=False,
                reason="prediction did not produce a supported RUL claim and evidence-review alerts are disabled",
                rule=rule,
            )

        context = self._sensor_hierarchy(organization_id, prediction.sensor_id)
        dedupe_key = _alert_dedupe_key(prediction.id, request.rule_id, alert_kind, request.rul_threshold_hours)
        existing = self.repo.get_active_maintenance_alert_by_dedupe_key(organization_id, dedupe_key)
        alert = self.repo.create_maintenance_alert(
            organization_id,
            site_id=context["site_id"],
            asset_id=context["asset_id"],
            component_id=context["component_id"],
            sensor_id=prediction.sensor_id,
            prediction_id=prediction.id,
            model_resolution_id=prediction.model_resolution_id,
            source_type="prediction",
            source_id=prediction.id,
            alert_kind=alert_kind,
            title=title,
            summary=summary,
            severity=severity,
            priority=request.priority,
            source_kind="prediction",
            source_reason_code=prediction.abstention_code or prediction.prediction_status,
            recommended_action=request.recommended_action,
            dedupe_key=dedupe_key,
            evidence_snapshot=snapshot,
            evidence={"rule": rule, "source_snapshot": snapshot, "maintenance_fact": False},
        )
        return AlertEvaluationResult(
            alert=alert_payload(alert),
            created=existing is None,
            reason="alert created" if existing is None else "active alert already exists for prediction and rule",
            rule=rule,
        )

    def acknowledge_alert(
        self,
        organization_id: str,
        alert_id: str,
        request: AlertAcknowledgeRequest,
    ) -> MaintenanceAlert:
        self._require_member(organization_id, request.acknowledged_by_user_id)
        alert = self._alert(organization_id, alert_id)
        if alert.status != "open":
            raise MaintenanceOperationsError("only open alerts can be acknowledged")
        return self.repo.update_maintenance_alert(
            alert,
            status="acknowledged",
            acknowledged_by_user_id=request.acknowledged_by_user_id,
            acknowledged_at=datetime.now(UTC),
            acknowledgement_note=request.note,
        )

    def resolve_alert(self, organization_id: str, alert_id: str, request: AlertResolveRequest) -> MaintenanceAlert:
        self._require_member(organization_id, request.resolved_by_user_id)
        alert = self._alert(organization_id, alert_id)
        if alert.status not in ACTIVE_ALERT_STATUSES:
            raise MaintenanceOperationsError("only active alerts can be resolved or dismissed")
        return self.repo.update_maintenance_alert(
            alert,
            status=request.disposition,
            resolved_by_user_id=request.resolved_by_user_id,
            resolved_at=datetime.now(UTC),
            disposition=request.disposition,
            disposition_reason=request.reason,
        )

    def open_case(self, organization_id: str, request: CaseCreate) -> MaintenanceCase:
        self._require_member(organization_id, request.opened_by_user_id)
        for user_id in [request.owner_user_id, request.assignee_user_id]:
            if user_id:
                self._require_member(organization_id, user_id)
        data = request.model_dump()
        if request.alert_id:
            alert = self._alert(organization_id, request.alert_id)
            context = self._sensor_hierarchy(organization_id, alert.sensor_id)
            _reject_hierarchy_override(
                provided={
                    "asset_id": data["asset_id"],
                    "component_id": data["component_id"],
                    "sensor_id": data["sensor_id"],
                },
                source={
                    "asset_id": context["asset_id"],
                    "component_id": context["component_id"],
                    "sensor_id": alert.sensor_id,
                },
            )
            data["sensor_id"] = alert.sensor_id
            data["component_id"] = context["component_id"]
            data["asset_id"] = context["asset_id"]
            data["evidence"] = {
                **data["evidence"],
                "source_alert_id": alert.id,
                "source_evidence_snapshot": alert.evidence_snapshot,
            }
        else:
            hierarchy = self._validate_hierarchy(
                organization_id,
                asset_id=data["asset_id"],
                component_id=data["component_id"],
                sensor_id=data["sensor_id"],
            )
            data["asset_id"] = hierarchy["asset_id"]
            data["component_id"] = hierarchy["component_id"]
            data["sensor_id"] = hierarchy["sensor_id"]
        return self.repo.create_maintenance_case(
            organization_id,
            alert_id=data["alert_id"],
            title=data["title"],
            summary=data["summary"],
            priority=data["priority"],
            asset_id=data["asset_id"],
            component_id=data["component_id"],
            sensor_id=data["sensor_id"],
            opened_by_user_id=data["opened_by_user_id"],
            owner_user_id=data["owner_user_id"],
            assignee_user_id=data["assignee_user_id"],
            recommended_action=data["recommended_action"],
            history=[_history_event("opened", data["opened_by_user_id"], data["summary"])],
            evidence={**data["evidence"], "human_workflow_fact": True},
        )

    def open_case_from_alert(
        self,
        organization_id: str,
        alert_id: str,
        request: CaseCreateFromAlertRequest,
    ) -> MaintenanceCase:
        alert = self._alert(organization_id, alert_id)
        return self.open_case(
            organization_id,
            CaseCreate(
                alert_id=alert.id,
                title=request.title or alert.title,
                summary=request.summary or alert.summary,
                priority=request.priority or alert.priority,
                opened_by_user_id=request.opened_by_user_id,
                owner_user_id=request.owner_user_id,
                assignee_user_id=request.assignee_user_id,
                recommended_action=alert.recommended_action,
            ),
        )

    def transition_case(
        self,
        organization_id: str,
        case_id: str,
        request: CaseTransitionRequest,
    ) -> MaintenanceCase:
        self._require_member(organization_id, request.actor_user_id)
        case = self._case(organization_id, case_id)
        if request.target_status == "resolved":
            raise MaintenanceOperationsError("resolve cases through the dedicated case resolution endpoint")
        if request.target_status not in CASE_TRANSITIONS[case.status]:
            raise MaintenanceOperationsError(f"invalid case transition {case.status} -> {request.target_status}")
        values: dict[str, Any] = {"status": request.target_status}
        if request.target_status == "closed":
            values["closed_at"] = datetime.now(UTC)
        values["history"] = [
            *(case.history or []),
            _history_event(f"transition:{request.target_status}", request.actor_user_id, request.note),
        ]
        return self.repo.update_maintenance_case(case, **values)

    def add_note(self, organization_id: str, case_id: str, request: NoteCreate) -> MaintenanceNote:
        self._require_member(organization_id, request.author_user_id)
        if request.note_kind in {"model_prediction", "ai_generated"} or request.metadata.get("ai_generated") is True:
            raise MaintenanceOperationsError("technician notes must be human-authored")
        return self.repo.create_maintenance_note(
            organization_id,
            case_id=case_id,
            author_user_id=request.author_user_id,
            body=request.body,
            note_kind=request.note_kind,
            metadata_json={**request.metadata, "human_authored": True},
        )

    def request_inspection(
        self,
        organization_id: str,
        case_id: str,
        request: InspectionRequestCreate,
    ) -> MaintenanceInspection:
        self._require_member(organization_id, request.requested_by_user_id)
        if request.assigned_to_user_id:
            self._require_member(organization_id, request.assigned_to_user_id)
        case = self._case(organization_id, case_id)
        case_source = {"asset_id": case.asset_id, "component_id": case.component_id, "sensor_id": case.sensor_id}
        if any(case_source.values()):
            _reject_hierarchy_override(
                provided={
                    "asset_id": request.asset_id,
                    "component_id": request.component_id,
                    "sensor_id": request.sensor_id,
                },
                source=case_source,
            )
            hierarchy = case_source
        else:
            hierarchy = self._validate_hierarchy(
                organization_id,
                asset_id=request.asset_id,
                component_id=request.component_id,
                sensor_id=request.sensor_id,
            )
        return self.repo.create_maintenance_inspection(
            organization_id,
            case_id=case_id,
            asset_id=hierarchy["asset_id"],
            component_id=hierarchy["component_id"],
            sensor_id=hierarchy["sensor_id"],
            requested_reason=request.requested_reason,
            requested_by_user_id=request.requested_by_user_id,
            assigned_to_user_id=request.assigned_to_user_id,
            evidence_metadata={
                **request.evidence_metadata,
                "inspection_result_from_prediction": False,
            },
        )

    def start_inspection(
        self,
        organization_id: str,
        inspection_id: str,
        request: InspectionStartRequest,
    ) -> MaintenanceInspection:
        self._require_member(organization_id, request.started_by_user_id)
        inspection = self._inspection(organization_id, inspection_id)
        if "in_progress" not in INSPECTION_TRANSITIONS[inspection.status]:
            raise MaintenanceOperationsError(f"invalid inspection transition {inspection.status} -> in_progress")
        return self.repo.update_maintenance_inspection(
            inspection,
            status="in_progress",
            performed_by_user_id=request.started_by_user_id,
            started_at=_aware_utc(request.started_at or datetime.now(UTC)),
        )

    def complete_inspection(
        self,
        organization_id: str,
        inspection_id: str,
        request: InspectionCompleteRequest,
    ) -> MaintenanceInspection:
        self._require_member(organization_id, request.performed_by_user_id)
        inspection = self._inspection(organization_id, inspection_id)
        if "completed" not in INSPECTION_TRANSITIONS[inspection.status]:
            raise MaintenanceOperationsError(f"invalid inspection transition {inspection.status} -> completed")
        return self.repo.update_maintenance_inspection(
            inspection,
            status="completed",
            performed_by_user_id=request.performed_by_user_id,
            inspected_by_user_id=request.performed_by_user_id,
            completed_at=_aware_utc(request.completed_at or datetime.now(UTC)),
            inspected_at=_aware_utc(request.completed_at or datetime.now(UTC)),
            condition=request.condition,
            findings=request.findings,
            observation=request.findings,
            recommended_follow_up=request.recommended_follow_up,
            evidence_metadata={
                **request.evidence_metadata,
                "maintenance_fact": True,
                "observation_kind": "technician_observation",
            },
            measurements=request.evidence_metadata,
        )

    def cancel_inspection(
        self,
        organization_id: str,
        inspection_id: str,
        request: InspectionCancelRequest,
    ) -> MaintenanceInspection:
        self._require_member(organization_id, request.cancelled_by_user_id)
        inspection = self._inspection(organization_id, inspection_id)
        if "cancelled" not in INSPECTION_TRANSITIONS[inspection.status]:
            raise MaintenanceOperationsError(f"invalid inspection transition {inspection.status} -> cancelled")
        return self.repo.update_maintenance_inspection(
            inspection,
            status="cancelled",
            cancelled_at=datetime.now(UTC),
            evidence_metadata={**(inspection.evidence_metadata or {}), "cancellation_reason": request.reason},
        )

    def create_work_order(
        self,
        organization_id: str,
        case_id: str,
        request: WorkOrderCreate,
    ) -> MaintenanceWorkOrder:
        self._require_member(organization_id, request.requested_by_user_id)
        if request.assignee_user_id:
            self._require_member(organization_id, request.assignee_user_id)
        return self.repo.create_maintenance_work_order(
            organization_id,
            case_id=case_id,
            title=request.title,
            description=request.description,
            priority=request.priority,
            requested_work=request.requested_work,
            requested_by_user_id=request.requested_by_user_id,
            assignee_user_id=request.assignee_user_id,
            planned_start_at=_aware_utc(request.planned_start_at) if request.planned_start_at else None,
            evidence={**request.evidence, "created_from_machine_evidence_as_draft_only": True},
        )

    def approve_work_order(
        self,
        organization_id: str,
        work_order_id: str,
        request: WorkOrderApproveRequest,
    ) -> MaintenanceWorkOrder:
        self._require_member(organization_id, request.approved_by_user_id)
        work_order = self._work_order(organization_id, work_order_id)
        if "approved" not in WORK_ORDER_TRANSITIONS[work_order.status]:
            raise MaintenanceOperationsError(f"invalid work-order transition {work_order.status} -> approved")
        return self.repo.update_maintenance_work_order(
            work_order,
            status="approved",
            approved_by_user_id=request.approved_by_user_id,
            evidence={**(work_order.evidence or {}), "approval_note": request.note, "human_approved": True},
        )

    def start_work_order(
        self,
        organization_id: str,
        work_order_id: str,
        request: WorkOrderStartRequest,
    ) -> MaintenanceWorkOrder:
        self._require_member(organization_id, request.started_by_user_id)
        work_order = self._work_order(organization_id, work_order_id)
        if "in_progress" not in WORK_ORDER_TRANSITIONS[work_order.status]:
            raise MaintenanceOperationsError(f"invalid work-order transition {work_order.status} -> in_progress")
        return self.repo.update_maintenance_work_order(
            work_order,
            status="in_progress",
            started_at=_aware_utc(request.started_at or datetime.now(UTC)),
        )

    def complete_work_order(
        self,
        organization_id: str,
        work_order_id: str,
        request: WorkOrderCompleteRequest,
    ) -> MaintenanceWorkOrder:
        self._require_member(organization_id, request.completed_by_user_id)
        work_order = self._work_order(organization_id, work_order_id)
        if "completed" not in WORK_ORDER_TRANSITIONS[work_order.status]:
            raise MaintenanceOperationsError(f"invalid work-order transition {work_order.status} -> completed")
        return self.repo.update_maintenance_work_order(
            work_order,
            status="completed",
            completed_at=_aware_utc(request.completed_at or datetime.now(UTC)),
            completion_notes=request.completion_notes,
            work_performed=request.work_performed or request.completion_notes,
            evidence={**(work_order.evidence or {}), "completed_by_user_id": request.completed_by_user_id},
        )

    def cancel_work_order(
        self,
        organization_id: str,
        work_order_id: str,
        request: WorkOrderCancelRequest,
    ) -> MaintenanceWorkOrder:
        self._require_member(organization_id, request.cancelled_by_user_id)
        work_order = self._work_order(organization_id, work_order_id)
        if "cancelled" not in WORK_ORDER_TRANSITIONS[work_order.status]:
            raise MaintenanceOperationsError(f"invalid work-order transition {work_order.status} -> cancelled")
        return self.repo.update_maintenance_work_order(
            work_order,
            status="cancelled",
            cancelled_at=datetime.now(UTC),
            evidence={**(work_order.evidence or {}), "cancellation_reason": request.reason},
        )

    def sync_work_order_to_cmms(
        self,
        organization_id: str,
        work_order_id: str,
        request: CmmsSyncRequest,
    ) -> CmmsSyncRecord:
        self._require_member(organization_id, request.initiated_by_user_id)
        work_order = self._work_order(organization_id, work_order_id)
        requested_provider = request.provider_name or request.adapter_name
        if work_order.cmms_external_id and work_order.cmms_provider:
            if requested_provider and requested_provider != work_order.cmms_provider:
                raise MaintenanceOperationsError("work order is already bound to a different CMMS provider")
            provider_name = work_order.cmms_provider
        else:
            provider_name = requested_provider or "disabled"
        adapter = self.cmms_adapters.get(provider_name) or DisabledCmmsAdapter()
        operation = request.operation
        idempotency_key = _cmms_idempotency_key(organization_id, work_order.id, provider_name, operation)
        if operation == "create":
            existing_success = self.repo.get_successful_cmms_sync_record(
                organization_id,
                work_order_id=work_order.id,
                provider_name=provider_name,
                operation=operation,
                idempotency_key=idempotency_key,
            )
            if existing_success:
                return self.repo.create_cmms_sync_record(
                    organization_id,
                    work_order_id=work_order.id,
                    provider_name=existing_success.provider_name,
                    initiator_type="user",
                    initiated_by_user_id=request.initiated_by_user_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    status="skipped",
                    external_id=existing_success.external_id,
                    error_category=None,
                    error_message=None,
                    completed_at=datetime.now(UTC),
                    attempt_metadata={
                        "idempotent_reuse_of_successful_create": True,
                        "source_sync_record_id": existing_success.id,
                    },
                )
        result = adapter.sync(operation, work_order, idempotency_key=idempotency_key)
        sync = self.repo.create_cmms_sync_record(
            organization_id,
            work_order_id=work_order.id,
            provider_name=provider_name,
            initiator_type="user",
            initiated_by_user_id=request.initiated_by_user_id,
            operation=operation,
            idempotency_key=idempotency_key,
            status=result.status,
            external_id=result.external_id,
            error_category=result.error_category,
            error_message=result.error_message,
            completed_at=datetime.now(UTC),
            attempt_metadata={**request.attempt_metadata, **(result.metadata or {})},
        )
        if result.status == "succeeded":
            self.repo.update_maintenance_work_order(
                work_order,
                cmms_provider=provider_name,
                cmms_external_id=result.external_id,
                cmms_state=result.external_status or result.status,
            )
        elif result.status == "not_configured":
            self.repo.update_maintenance_work_order(
                work_order,
                cmms_provider=provider_name,
                cmms_state="not_configured",
            )
        return sync

    def resolve_case(
        self,
        organization_id: str,
        case_id: str,
        request: ResolutionCreate,
    ) -> MaintenanceResolution:
        self._require_member(organization_id, request.resolved_by_user_id)
        case = self._case(organization_id, case_id)
        if case.status not in {"open", "in_progress"}:
            raise MaintenanceOperationsError("only open or in-progress cases can be resolved")
        resolution = self.repo.resolve_maintenance_case(
            organization_id,
            case_id=case_id,
            resolved_by_user_id=request.resolved_by_user_id,
            outcome=request.outcome,
            summary=request.summary,
            evidence={**request.evidence, "resolution_kind": "human_decision", "maintenance_fact": True},
        )
        self.repo.update_maintenance_case(
            case,
            status="resolved",
            resolved_at=datetime.now(UTC),
            history=[*(case.history or []), _history_event("resolved", request.resolved_by_user_id, request.summary)],
        )
        return resolution

    def list_alerts(self, organization_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        return [alert_payload(alert) for alert in self.repo.list_maintenance_alerts(organization_id, status=status)]

    def get_alert(self, organization_id: str, alert_id: str) -> dict[str, Any]:
        return alert_payload(self._alert(organization_id, alert_id))

    def list_cases(self, organization_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        return [case_payload(case) for case in self.repo.list_maintenance_cases(organization_id, status=status)]

    def get_case(self, organization_id: str, case_id: str) -> dict[str, Any]:
        case = self._case(organization_id, case_id)
        notes = self.repo.list_maintenance_notes(organization_id, case_id=case.id)
        inspections = self.repo.list_maintenance_inspections(organization_id, case_id=case.id)
        work_orders = self.repo.list_maintenance_work_orders(organization_id, case_id=case.id)
        return {
            **case_payload(case),
            "notes": [note_payload(note) for note in notes],
            "inspections": [inspection_payload(row) for row in inspections],
            "work_orders": [work_order_payload(row) for row in work_orders],
        }

    def list_notes(self, organization_id: str, case_id: str) -> list[dict[str, Any]]:
        self._case(organization_id, case_id)
        return [note_payload(note) for note in self.repo.list_maintenance_notes(organization_id, case_id=case_id)]

    def list_cmms_sync_records(self, organization_id: str, work_order_id: str) -> list[dict[str, Any]]:
        self._work_order(organization_id, work_order_id)
        return [
            cmms_sync_payload(sync)
            for sync in self.repo.list_cmms_sync_records(organization_id, work_order_id=work_order_id)
        ]

    def health(self, organization_id: str) -> dict[str, Any]:
        return {
            "maintenance_operations": "ok",
            "alerts": self.repo.count_for_organization(MaintenanceAlert, organization_id),
            "open_alerts": self._count_by_status(MaintenanceAlert, organization_id, ["open", "acknowledged"]),
            "cases": self.repo.count_for_organization(MaintenanceCase, organization_id),
            "open_cases": self._count_by_status(MaintenanceCase, organization_id, ["open", "in_progress"]),
            "work_orders": self.repo.count_for_organization(MaintenanceWorkOrder, organization_id),
            "cmms_sync_records": self.repo.count_for_organization(CmmsSyncRecord, organization_id),
        }

    def _require_member(self, organization_id: str, user_id: str) -> None:
        if self.repo.get_active_membership(organization_id, user_id) is None:
            raise MaintenanceOperationsError("maintenance user must be an active member of this organization")

    def _alert(self, organization_id: str, alert_id: str) -> MaintenanceAlert:
        alert = self.repo.get_maintenance_alert(organization_id, alert_id)
        if alert is None:
            raise MaintenanceOperationsError("maintenance alert does not exist inside this organization")
        return alert

    def _case(self, organization_id: str, case_id: str) -> MaintenanceCase:
        case = self.repo.get_maintenance_case(organization_id, case_id)
        if case is None:
            raise MaintenanceOperationsError("maintenance case must belong to the same organization")
        return case

    def _inspection(self, organization_id: str, inspection_id: str) -> MaintenanceInspection:
        inspection = self.repo.get_maintenance_inspection(organization_id, inspection_id)
        if inspection is None:
            raise MaintenanceOperationsError("maintenance inspection must belong to the same organization")
        return inspection

    def _work_order(self, organization_id: str, work_order_id: str) -> MaintenanceWorkOrder:
        work_order = self.repo.get_maintenance_work_order(organization_id, work_order_id)
        if work_order is None:
            raise MaintenanceOperationsError("maintenance work order must belong to the same organization")
        return work_order

    def _sensor_hierarchy(self, organization_id: str, sensor_id: str) -> dict[str, str]:
        sensor = self.repo.get_sensor_by_id(organization_id, sensor_id)
        if sensor is None:
            raise MaintenanceOperationsError("sensor does not exist inside this organization")
        component = self.repo.get_component_by_id(organization_id, sensor.component_id)
        if component is None:
            raise MaintenanceOperationsError("sensor component does not exist inside this organization")
        asset = self.repo.get_asset_by_id(organization_id, component.asset_id)
        if asset is None:
            raise MaintenanceOperationsError("component asset does not exist inside this organization")
        return {"site_id": asset.site_id, "asset_id": asset.id, "component_id": component.id}

    def _validate_hierarchy(
        self,
        organization_id: str,
        *,
        asset_id: str | None,
        component_id: str | None,
        sensor_id: str | None,
    ) -> dict[str, str | None]:
        if sensor_id:
            context = self._sensor_hierarchy(organization_id, sensor_id)
            if component_id and component_id != context["component_id"]:
                raise MaintenanceOperationsError("sensor/component/asset hierarchy does not match")
            if asset_id and asset_id != context["asset_id"]:
                raise MaintenanceOperationsError("sensor/component/asset hierarchy does not match")
            return {
                "asset_id": context["asset_id"],
                "component_id": context["component_id"],
                "sensor_id": sensor_id,
            }
        if component_id:
            component = self.repo.get_component_by_id(organization_id, component_id)
            if component is None:
                raise MaintenanceOperationsError("component does not exist inside this organization")
            if asset_id and asset_id != component.asset_id:
                raise MaintenanceOperationsError("sensor/component/asset hierarchy does not match")
            return {"asset_id": component.asset_id, "component_id": component_id, "sensor_id": None}
        if asset_id:
            if self.repo.get_asset_by_id(organization_id, asset_id) is None:
                raise MaintenanceOperationsError("asset does not exist inside this organization")
            return {"asset_id": asset_id, "component_id": None, "sensor_id": None}
        return {"asset_id": None, "component_id": None, "sensor_id": None}

    def _count_by_status(self, model, organization_id: str, statuses: list[str]) -> int:
        return int(
            self.session.scalar(
                select(func.count()).select_from(model).where(
                    model.organization_id == organization_id,
                    model.status.in_(statuses),
                )
            )
            or 0
        )


def _rule_payload(request: PredictionAlertEvaluationRequest) -> dict[str, Any]:
    return {
        "rule_id": request.rule_id,
        "rule_name": request.rule_name,
        "rul_threshold_hours": request.rul_threshold_hours,
        "create_evidence_review_for_abstention": request.create_evidence_review_for_abstention,
    }


def _prediction_evidence_snapshot(prediction, resolution) -> dict[str, Any]:
    snapshot = {
        "prediction_record_id": prediction.id,
        "prediction_status": prediction.prediction_status,
        "uncertainty": prediction.uncertainty,
        "abstention_code": prediction.abstention_code,
        "abstention_reason": prediction.abstention_reason,
        "model_version_id": prediction.model_version_id,
        "dataset_version_id": prediction.dataset_version_id,
        "model_resolution_id": prediction.model_resolution_id,
        "feature_record_ids": prediction.feature_record_ids,
        "serving_reference_time": prediction.observed_at.isoformat(),
        "prediction_provenance": prediction.provenance or {},
        "model_resolution": _model_resolution_snapshot(resolution),
    }
    if prediction.prediction_status == "supported":
        snapshot["predicted_rul_hours"] = prediction.predicted_rul_hours
    return snapshot


def _severity_from_rul(predicted_rul_hours: float) -> str:
    if predicted_rul_hours <= 24:
        return "critical"
    if predicted_rul_hours <= 72:
        return "warning"
    return "watch"


def _abstention_summary(prediction) -> str:
    reason = prediction.abstention_code or prediction.prediction_status
    return f"Prediction is {prediction.prediction_status}; human evidence review requested for {reason}."


def _alert_dedupe_key(prediction_id: str, rule_id: str, alert_kind: str, threshold: float | None) -> str:
    raw = f"{prediction_id}:{rule_id}:{alert_kind}:{threshold}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _model_resolution_snapshot(resolution) -> dict[str, Any] | None:
    if resolution is None:
        return None
    return {
        "id": resolution.id,
        "status": resolution.status,
        "reason_code": resolution.reason_code,
        "reason": resolution.reason,
        "artifact_sha256": resolution.artifact_sha256,
        "feature_schema": resolution.feature_schema,
        "abstention_policy": resolution.abstention_policy,
        "evidence": resolution.evidence,
    }


def _reject_hierarchy_override(provided: dict[str, str | None], source: dict[str, str | None]) -> None:
    for key, value in provided.items():
        if value is not None and value != source.get(key):
            raise MaintenanceOperationsError("maintenance hierarchy must match source evidence")


def _cmms_idempotency_key(organization_id: str, work_order_id: str, provider_name: str, operation: str) -> str:
    return hashlib.sha256(f"{organization_id}:{work_order_id}:{provider_name}:{operation}".encode()).hexdigest()


def _history_event(kind: str, actor_user_id: str | None, note: str | None) -> dict[str, Any]:
    return {"kind": kind, "actor_user_id": actor_user_id, "note": note, "at": datetime.now(UTC).isoformat()}


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def alert_payload(alert: MaintenanceAlert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "organization_id": alert.organization_id,
        "site_id": alert.site_id,
        "asset_id": alert.asset_id,
        "component_id": alert.component_id,
        "sensor_id": alert.sensor_id,
        "prediction_id": alert.prediction_id,
        "model_resolution_id": alert.model_resolution_id,
        "source_type": alert.source_type,
        "source_id": alert.source_id,
        "alert_kind": alert.alert_kind,
        "title": alert.title,
        "summary": alert.summary,
        "severity": alert.severity,
        "priority": alert.priority,
        "status": alert.status,
        "source_reason_code": alert.source_reason_code,
        "recommended_action": alert.recommended_action,
        "dedupe_key": alert.dedupe_key,
        "evidence_snapshot": alert.evidence_snapshot,
        "evidence": alert.evidence,
        "acknowledged_by_user_id": alert.acknowledged_by_user_id,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "acknowledgement_note": alert.acknowledgement_note,
        "resolved_by_user_id": alert.resolved_by_user_id,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
        "disposition": alert.disposition,
        "disposition_reason": alert.disposition_reason,
    }


def case_payload(case: MaintenanceCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "organization_id": case.organization_id,
        "alert_id": case.alert_id,
        "case_number": case.case_number,
        "title": case.title,
        "summary": case.summary,
        "priority": case.priority,
        "status": case.status,
        "asset_id": case.asset_id,
        "component_id": case.component_id,
        "sensor_id": case.sensor_id,
        "opened_by_user_id": case.opened_by_user_id,
        "owner_user_id": case.owner_user_id,
        "assignee_user_id": case.assignee_user_id,
        "recommended_action": case.recommended_action,
        "history": case.history,
        "evidence": case.evidence,
        "resolved_at": case.resolved_at.isoformat() if case.resolved_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
    }


def note_payload(note: MaintenanceNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "case_id": note.case_id,
        "author_user_id": note.author_user_id,
        "body": note.body,
        "note_kind": note.note_kind,
        "metadata": note.metadata_json,
        "created_at": note.created_at.isoformat() if note.created_at else None,
    }


def inspection_payload(inspection: MaintenanceInspection) -> dict[str, Any]:
    return {
        "id": inspection.id,
        "case_id": inspection.case_id,
        "asset_id": inspection.asset_id,
        "component_id": inspection.component_id,
        "sensor_id": inspection.sensor_id,
        "status": inspection.status,
        "requested_reason": inspection.requested_reason,
        "requested_by_user_id": inspection.requested_by_user_id,
        "assigned_to_user_id": inspection.assigned_to_user_id,
        "performed_by_user_id": inspection.performed_by_user_id,
        "started_at": inspection.started_at.isoformat() if inspection.started_at else None,
        "completed_at": inspection.completed_at.isoformat() if inspection.completed_at else None,
        "condition": inspection.condition,
        "findings": inspection.findings,
        "recommended_follow_up": inspection.recommended_follow_up,
        "evidence_metadata": inspection.evidence_metadata,
    }


def work_order_payload(work_order: MaintenanceWorkOrder) -> dict[str, Any]:
    return {
        "id": work_order.id,
        "case_id": work_order.case_id,
        "asset_id": work_order.asset_id,
        "component_id": work_order.component_id,
        "work_order_number": work_order.work_order_number,
        "status": work_order.status,
        "title": work_order.title,
        "description": work_order.description,
        "priority": work_order.priority,
        "requested_work": work_order.requested_work,
        "requested_by_user_id": work_order.requested_by_user_id,
        "approved_by_user_id": work_order.approved_by_user_id,
        "assignee_user_id": work_order.assignee_user_id,
        "planned_start_at": work_order.planned_start_at.isoformat() if work_order.planned_start_at else None,
        "started_at": work_order.started_at.isoformat() if work_order.started_at else None,
        "completed_at": work_order.completed_at.isoformat() if work_order.completed_at else None,
        "completion_notes": work_order.completion_notes,
        "cmms_provider": work_order.cmms_provider,
        "cmms_external_id": work_order.cmms_external_id,
        "cmms_state": work_order.cmms_state,
        "work_performed": work_order.work_performed,
        "evidence": work_order.evidence,
    }


def cmms_sync_payload(sync: CmmsSyncRecord) -> dict[str, Any]:
    return {
        "id": sync.id,
        "work_order_id": sync.work_order_id,
        "provider_name": sync.provider_name,
        "initiator_type": sync.initiator_type,
        "initiated_by_user_id": sync.initiated_by_user_id,
        "operation": sync.operation,
        "idempotency_key": sync.idempotency_key,
        "status": sync.status,
        "external_id": sync.external_id,
        "error_category": sync.error_category,
        "error_message": sync.error_message,
        "attempted_at": sync.attempted_at.isoformat() if sync.attempted_at else None,
        "completed_at": sync.completed_at.isoformat() if sync.completed_at else None,
        "attempt_metadata": sync.attempt_metadata,
    }


def resolution_payload(resolution: MaintenanceResolution) -> dict[str, Any]:
    return {
        "id": resolution.id,
        "case_id": resolution.case_id,
        "resolved_by_user_id": resolution.resolved_by_user_id,
        "outcome": resolution.outcome,
        "summary": resolution.summary,
        "resolved_at": resolution.resolved_at.isoformat() if resolution.resolved_at else None,
        "evidence": resolution.evidence,
    }
