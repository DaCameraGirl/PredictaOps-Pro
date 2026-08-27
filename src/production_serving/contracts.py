"""API contracts for Production Slice 10 live serving."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PredictionStatus = Literal["supported", "unsupported", "insufficient_evidence"]
ServingScopeType = Literal["organization", "site", "asset", "component", "sensor"]
RequestKind = Literal["live", "historical"]


class ServingBindingCreate(BaseModel):
    registry_id: str
    model_version_id: str
    scope_type: ServingScopeType
    scope_id: str | None = None
    approved_by_user_id: str
    reason: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def validate_scope(self) -> ServingBindingCreate:
        if self.scope_type == "organization" and self.scope_id is not None:
            raise ValueError("organization serving scope must not include scope_id")
        if self.scope_type != "organization" and not self.scope_id:
            raise ValueError("site, asset, component, and sensor serving scopes require scope_id")
        return self


class PredictionRequest(BaseModel):
    sensor_id: str
    observed_at: datetime | None = None
    registry_id: str | None = None
    registry_name: str | None = Field(default=None, max_length=255)
    max_feature_age_minutes: int = Field(default=1440, ge=1, le=525600)

    @model_validator(mode="after")
    def validate_registry_filter(self) -> PredictionRequest:
        if self.registry_id and self.registry_name:
            raise ValueError("provide registry_id or registry_name, not both")
        return self


class PredictionResponse(BaseModel):
    id: str
    model_resolution_id: str
    organization_id: str
    sensor_id: str
    observed_at: datetime
    prediction_status: PredictionStatus
    predicted_rul_hours: float | None
    abstention_code: str | None
    abstention_reason: str | None
    request_kind: RequestKind
    registry_id: str | None
    model_version_id: str | None
    dataset_version_id: str | None
    feature_vector: dict[str, float] | None
    feature_record_ids: list[str] | None
    uncertainty: dict[str, Any] | None
    model_resolution: dict[str, Any]
    provenance: dict[str, Any]
