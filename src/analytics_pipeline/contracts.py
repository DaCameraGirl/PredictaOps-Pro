"""Contracts for deterministic analytics receipts and health evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

HealthState = Literal["insufficient_evidence", "healthy", "watch", "warning", "critical", "unknown"]


class FeatureValue(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    value: float
    unit: str | None = Field(default=None, max_length=64)


class AnalyticsFailureReceipt(BaseModel):
    source_kind: Literal["scalar", "waveform"]
    source_record_id: str | None
    reason: str


class AnalyticsReceipt(BaseModel):
    run_id: str
    status: Literal["completed", "partial", "failed"]
    algorithm_version: str
    processed_count: int
    feature_count: int
    duplicate_feature_count: int
    failure_count: int
    health_state_count: int
    failures: list[AnalyticsFailureReceipt] = Field(default_factory=list)


class HealthStateSummary(BaseModel):
    sensor_id: str
    observed_at: str
    health_state: HealthState
    anomaly_score: float | None
    trend_slope: float | None
    confidence: float | None
    evidence: dict[str, Any] | None

