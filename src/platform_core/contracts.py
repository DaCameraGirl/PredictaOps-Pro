"""Canonical industrial data contracts used by APIs and ingestion adapters."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LifecycleState = Literal["active", "inactive", "archived"]
MembershipRole = Literal["owner", "admin", "engineer", "technician", "viewer"]
ReadingQuality = Literal["good", "suspect", "bad", "missing"]


class OrganizationCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=255)


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    full_name: str | None = Field(default=None, max_length=255)
    external_subject: str | None = Field(default=None, max_length=255)


class SiteCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=255)
    timezone: str = Field(default="UTC", max_length=120)


class AssetCreate(BaseModel):
    site_id: str
    slug: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=255)
    asset_type: str = Field(min_length=1, max_length=120)
    external_ref: str | None = Field(default=None, max_length=255)


class ComponentCreate(BaseModel):
    asset_id: str
    slug: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=255)
    component_type: str = Field(min_length=1, max_length=120)
    external_ref: str | None = Field(default=None, max_length=255)


class SensorCreate(BaseModel):
    component_id: str
    slug: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=255)
    sensor_type: str = Field(min_length=1, max_length=120)
    unit: str = Field(min_length=1, max_length=64)
    sampling_rate_hz: float | None = None
    channel_name: str | None = Field(default=None, max_length=120)
    axis: str | None = Field(default=None, max_length=32)
    manufacturer: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    serial_number: str | None = Field(default=None, max_length=120)
    external_ref: str | None = Field(default=None, max_length=255)


class MachineReadingCreate(BaseModel):
    sensor_id: str
    observed_at: datetime
    metric: str = Field(min_length=1, max_length=120)
    value: float
    unit: str = Field(min_length=1, max_length=64)
    source: str = Field(min_length=1, max_length=120)
    quality: ReadingQuality = "good"
    payload: dict[str, Any] | None = None


class EntityRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    name: str
    lifecycle_state: LifecycleState


class PlatformBootstrapSummary(BaseModel):
    organization_id: str
    site_id: str
    asset_count: int
    component_count: int
    sensor_count: int
