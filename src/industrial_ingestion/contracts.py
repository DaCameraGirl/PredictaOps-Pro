"""Canonical ingestion contracts shared by every source adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SourceType = Literal["csv", "parquet", "rest", "mqtt", "opcua", "abb", "replay"]
PayloadKind = Literal["scalar", "waveform"]
QualityState = Literal["good", "suspect", "bad", "missing"]


class SensorReference(BaseModel):
    sensor_id: str | None = None
    sensor_external_ref: str | None = None
    site_slug: str | None = None
    asset_slug: str | None = None
    component_slug: str | None = None
    sensor_slug: str | None = None

    @model_validator(mode="after")
    def require_resolvable_identity(self) -> SensorReference:
        has_path = all([self.site_slug, self.asset_slug, self.component_slug, self.sensor_slug])
        if not self.sensor_id and not self.sensor_external_ref and not has_path:
            raise ValueError("sensor_id, sensor_external_ref, or full site/asset/component/sensor path is required")
        return self


class CanonicalIngestionRecord(BaseModel):
    kind: PayloadKind
    observed_at: datetime | str
    sensor: SensorReference
    source_record_id: str | None = Field(default=None, max_length=255)
    source_timezone: str | None = Field(default=None, max_length=120)
    unit: str = Field(min_length=1, max_length=64)
    quality: QualityState = "good"
    metric: str | None = Field(default=None, max_length=120)
    value: float | None = None
    sampling_rate_hz: float | None = None
    samples: list[float] | None = None
    sample_count: int | None = None
    storage_uri: str | None = Field(default=None, max_length=1024)
    sha256: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload_shape(self) -> CanonicalIngestionRecord:
        if self.kind == "scalar":
            if not self.metric:
                raise ValueError("scalar records require metric")
            if self.value is None:
                raise ValueError("scalar records require value")
        if self.kind == "waveform":
            if self.sampling_rate_hz is None or self.sampling_rate_hz <= 0:
                raise ValueError("waveform records require positive sampling_rate_hz")
            if self.samples is None and self.sample_count is None:
                raise ValueError("waveform records require samples or sample_count")
        return self


class AdapterBatch(BaseModel):
    source_type: SourceType
    source_name: str
    records: list[CanonicalIngestionRecord]
    source_uri: str | None = None
    batch_idempotency_key: str | None = Field(default=None, max_length=120)
    provenance: dict[str, Any] = Field(default_factory=dict)


class IngestionFailureReceipt(BaseModel):
    source_record_id: str | None
    reason: str
    quality: QualityState


class IngestionReceipt(BaseModel):
    batch_id: str
    source_id: str
    status: Literal["accepted", "partial", "failed"]
    accepted_count: int
    duplicate_count: int
    failed_count: int
    scalar_count: int
    waveform_count: int
    failures: list[IngestionFailureReceipt] = Field(default_factory=list)


class SourceRegistration(BaseModel):
    organization_id: str
    source_type: SourceType
    name: str = Field(min_length=1, max_length=255)
    external_ref: str | None = Field(default=None, max_length=255)
    config: dict[str, Any] | None = None
