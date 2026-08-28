"""API contracts for Production Slice 11 maintenance operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AlertSeverity = Literal["info", "watch", "warning", "critical"]
Priority = Literal["low", "medium", "high", "critical"]
AlertStatus = Literal["open", "acknowledged", "resolved", "dismissed"]
CaseStatus = Literal["open", "in_progress", "resolved", "closed"]
InspectionStatus = Literal["requested", "in_progress", "completed", "cancelled"]
WorkOrderStatus = Literal["draft", "approved", "in_progress", "completed", "cancelled"]
InspectionCondition = Literal["normal", "watch", "degraded", "failed", "unknown"]
ResolutionOutcome = Literal["confirmed", "not_found", "monitor", "repaired", "replaced"]
CmmsOperation = Literal["create", "update", "cancel", "close"]


class PredictionAlertEvaluationRequest(BaseModel):
    prediction_id: str
    rule_id: str = Field(min_length=1, max_length=120)
    rule_name: str = Field(default="RUL threshold or evidence review", max_length=255)
    rul_threshold_hours: float | None = Field(default=None, gt=0)
    create_evidence_review_for_abstention: bool = True
    severity: AlertSeverity | None = None
    priority: Priority = "medium"
    recommended_action: str | None = Field(default=None, max_length=1024)


class AlertAcknowledgeRequest(BaseModel):
    acknowledged_by_user_id: str
    note: str | None = Field(default=None, max_length=1024)


class AlertResolveRequest(BaseModel):
    resolved_by_user_id: str
    disposition: Literal["resolved", "dismissed"] = "resolved"
    reason: str = Field(min_length=1, max_length=1024)


class CaseCreate(BaseModel):
    alert_id: str | None = None
    title: str = Field(min_length=1, max_length=255)
    summary: str | None = Field(default=None, max_length=1024)
    priority: Priority = "medium"
    asset_id: str | None = None
    component_id: str | None = None
    sensor_id: str | None = None
    opened_by_user_id: str
    owner_user_id: str | None = None
    assignee_user_id: str | None = None
    recommended_action: str | None = Field(default=None, max_length=1024)
    evidence: dict[str, Any] = Field(default_factory=dict)


class CaseTransitionRequest(BaseModel):
    actor_user_id: str
    target_status: CaseStatus
    note: str | None = Field(default=None, max_length=1024)


class NoteCreate(BaseModel):
    author_user_id: str
    body: str = Field(min_length=1, max_length=2048)
    note_kind: str = Field(default="technician_note", max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InspectionRequestCreate(BaseModel):
    requested_by_user_id: str
    requested_reason: str = Field(min_length=1, max_length=1024)
    asset_id: str | None = None
    component_id: str | None = None
    sensor_id: str | None = None
    assigned_to_user_id: str | None = None
    evidence_metadata: dict[str, Any] = Field(default_factory=dict)


class InspectionStartRequest(BaseModel):
    started_by_user_id: str
    started_at: datetime | None = None


class InspectionCompleteRequest(BaseModel):
    performed_by_user_id: str
    condition: InspectionCondition
    findings: str = Field(min_length=1, max_length=2048)
    recommended_follow_up: str | None = Field(default=None, max_length=1024)
    completed_at: datetime | None = None
    evidence_metadata: dict[str, Any] = Field(default_factory=dict)


class InspectionCancelRequest(BaseModel):
    cancelled_by_user_id: str
    reason: str = Field(min_length=1, max_length=1024)


class WorkOrderCreate(BaseModel):
    requested_by_user_id: str
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2048)
    priority: Priority = "medium"
    requested_work: str = Field(min_length=1, max_length=2048)
    assignee_user_id: str | None = None
    planned_start_at: datetime | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class WorkOrderApproveRequest(BaseModel):
    approved_by_user_id: str
    note: str | None = Field(default=None, max_length=1024)


class WorkOrderStartRequest(BaseModel):
    started_by_user_id: str
    started_at: datetime | None = None


class WorkOrderCompleteRequest(BaseModel):
    completed_by_user_id: str
    completion_notes: str = Field(min_length=1, max_length=2048)
    completed_at: datetime | None = None
    work_performed: str | None = Field(default=None, max_length=2048)


class WorkOrderCancelRequest(BaseModel):
    cancelled_by_user_id: str
    reason: str = Field(min_length=1, max_length=1024)


class CmmsSyncRequest(BaseModel):
    operation: CmmsOperation = "create"
    initiated_by_user_id: str
    provider_name: str | None = Field(default=None, max_length=120)
    adapter_name: str | None = Field(default=None, max_length=120)
    attempt_metadata: dict[str, Any] = Field(default_factory=dict)


class ResolutionCreate(BaseModel):
    resolved_by_user_id: str
    outcome: ResolutionOutcome
    summary: str = Field(min_length=1, max_length=2048)
    evidence: dict[str, Any] = Field(default_factory=dict)


class AlertEvaluationResult(BaseModel):
    alert: dict[str, Any] | None
    created: bool
    reason: str
    rule: dict[str, Any]


class CaseCreateFromAlertRequest(BaseModel):
    opened_by_user_id: str
    title: str | None = Field(default=None, max_length=255)
    summary: str | None = Field(default=None, max_length=1024)
    priority: Priority | None = None
    owner_user_id: str | None = None
    assignee_user_id: str | None = None

    @model_validator(mode="after")
    def validate_assignment(self) -> CaseCreateFromAlertRequest:
        if self.owner_user_id == "":
            self.owner_user_id = None
        if self.assignee_user_id == "":
            self.assignee_user_id = None
        return self
