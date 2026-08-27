"""SQLAlchemy models for the tenant-owned industrial asset registry."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_uuid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LifecycleMixin:
    lifecycle_state: Mapped[str] = mapped_column(String(32), nullable=False, default="active")


class Organization(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_org_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    sites: Mapped[list["Site"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    assets: Mapped[list["Asset"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
        overlaps="site,assets",
    )
    components: Mapped[list["Component"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
        overlaps="asset,components",
    )
    sensors: Mapped[list["Sensor"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
        overlaps="component,sensors",
    )
    ingestion_sources: Mapped[list["IngestionSource"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    memberships: Mapped[list["OrganizationMembership"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    analytics_runs: Mapped[list["AnalyticsRun"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    ml_dataset_versions: Mapped[list["MLDatasetVersion"]] = relationship(
        back_populates="organization",
        cascade="all, delete-orphan",
    )


class User(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_user_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    external_subject: Mapped[str | None] = mapped_column(String(255), unique=True)

    memberships: Mapped[list["OrganizationMembership"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class OrganizationMembership(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_membership_org_user"),
        CheckConstraint("role in ('owner', 'admin', 'engineer', 'technician', 'viewer')", name="ck_membership_role"),
        CheckConstraint(
            "lifecycle_state in ('active', 'inactive', 'archived')",
            name="ck_membership_lifecycle",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)

    organization: Mapped[Organization] = relationship(back_populates="memberships")
    user: Mapped[User] = relationship(back_populates="memberships")


class Site(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "sites"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_sites_org_id"),
        UniqueConstraint("organization_id", "slug", name="uq_sites_org_slug"),
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_site_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    timezone: Mapped[str] = mapped_column(String(120), nullable=False, default="UTC")

    organization: Mapped[Organization] = relationship(back_populates="sites")
    assets: Mapped[list["Asset"]] = relationship(
        back_populates="site",
        cascade="all, delete-orphan",
        overlaps="organization,assets",
    )


class Asset(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_assets_org_id"),
        UniqueConstraint("organization_id", "site_id", "slug", name="uq_assets_org_site_slug"),
        ForeignKeyConstraint(["organization_id", "site_id"], ["sites.organization_id", "sites.id"]),
        ForeignKeyConstraint(
            ["organization_id", "parent_asset_id"],
            ["assets.organization_id", "assets.id"],
        ),
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_asset_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    site_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    parent_asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(120), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(255))

    organization: Mapped[Organization] = relationship(back_populates="assets", overlaps="assets,site")
    site: Mapped[Site] = relationship(back_populates="assets", overlaps="assets,organization")
    components: Mapped[list["Component"]] = relationship(
        back_populates="asset",
        cascade="all, delete-orphan",
        overlaps="organization,components",
    )


class Component(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "components"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_components_org_id"),
        UniqueConstraint("organization_id", "asset_id", "slug", name="uq_components_org_asset_slug"),
        ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        ForeignKeyConstraint(
            ["organization_id", "parent_component_id"],
            ["components.organization_id", "components.id"],
        ),
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_component_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    asset_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    parent_component_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    component_type: Mapped[str] = mapped_column(String(120), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(255))

    organization: Mapped[Organization] = relationship(back_populates="components", overlaps="asset,components")
    asset: Mapped[Asset] = relationship(back_populates="components", overlaps="components,organization")
    sensors: Mapped[list["Sensor"]] = relationship(
        back_populates="component",
        cascade="all, delete-orphan",
        overlaps="organization,sensors",
    )


class Sensor(Base, TimestampMixin, LifecycleMixin):
    __tablename__ = "sensors"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_sensors_org_id"),
        UniqueConstraint("organization_id", "component_id", "slug", name="uq_sensors_org_component_slug"),
        ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        CheckConstraint("lifecycle_state in ('active', 'inactive', 'archived')", name="ck_sensor_lifecycle"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    component_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sensor_type: Mapped[str] = mapped_column(String(120), nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    sampling_rate_hz: Mapped[float | None] = mapped_column(Float)
    channel_name: Mapped[str | None] = mapped_column(String(120))
    axis: Mapped[str | None] = mapped_column(String(32))
    manufacturer: Mapped[str | None] = mapped_column(String(120))
    model: Mapped[str | None] = mapped_column(String(120))
    serial_number: Mapped[str | None] = mapped_column(String(120))
    external_ref: Mapped[str | None] = mapped_column(String(255))

    organization: Mapped[Organization] = relationship(back_populates="sensors", overlaps="component,sensors")
    component: Mapped[Component] = relationship(back_populates="sensors", overlaps="organization,sensors")
    readings: Mapped[list["MachineReading"]] = relationship(back_populates="sensor", cascade="all, delete-orphan")
    waveforms: Mapped[list["WaveformRecord"]] = relationship(back_populates="sensor", cascade="all, delete-orphan")


class MachineReading(Base, TimestampMixin):
    __tablename__ = "machine_readings"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_readings_org_id"),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint("quality in ('good', 'suspect', 'bad', 'missing')", name="ck_reading_quality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String(120), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    quality: Mapped[str] = mapped_column(String(32), nullable=False, default="good")
    payload: Mapped[dict | None] = mapped_column(JSON)

    sensor: Mapped[Sensor] = relationship(back_populates="readings")


class IngestionSource(Base, TimestampMixin):
    __tablename__ = "ingestion_sources"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ingestion_sources_org_id"),
        UniqueConstraint("organization_id", "name", name="uq_ingestion_sources_org_name"),
        UniqueConstraint("organization_id", "source_type", "external_ref", name="uq_ingestion_sources_org_external"),
        CheckConstraint(
            "source_type in ('csv', 'parquet', 'rest', 'mqtt', 'opcua', 'abb', 'replay')",
            name="ck_ingestion_source_type",
        ),
        CheckConstraint("status in ('active', 'paused', 'unhealthy')", name="ck_ingestion_source_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    config: Mapped[dict | None] = mapped_column(JSON)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    organization: Mapped[Organization] = relationship(back_populates="ingestion_sources")
    batches: Mapped[list["IngestionBatch"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class IngestionBatch(Base, TimestampMixin):
    __tablename__ = "ingestion_batches"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ingestion_batches_org_id"),
        UniqueConstraint("organization_id", "source_id", "idempotency_key", name="uq_ingestion_batches_source_key"),
        ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "replay_of_batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint(
            "source_type in ('csv', 'parquet', 'rest', 'mqtt', 'opcua', 'abb', 'replay')",
            name="ck_batch_source_type",
        ),
        CheckConstraint("status in ('accepted', 'partial', 'failed')", name="ck_ingestion_batch_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="accepted")
    idempotency_key: Mapped[str | None] = mapped_column(String(120))
    source_uri: Mapped[str | None] = mapped_column(String(1024))
    replay_of_batch_id: Mapped[str | None] = mapped_column(String(36))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    accepted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scalar_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    waveform_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provenance: Mapped[dict | None] = mapped_column(JSON)

    source: Mapped[IngestionSource] = relationship(back_populates="batches")


class IngestedRecord(Base, TimestampMixin):
    __tablename__ = "ingested_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "source_id", "idempotency_key", name="uq_ingested_records_source_key"),
        ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint("target_type in ('scalar_reading', 'waveform')", name="ck_ingested_record_target_type"),
        CheckConstraint("quality in ('good', 'suspect', 'bad', 'missing')", name="ck_ingested_record_quality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(120), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(36), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    metric: Mapped[str | None] = mapped_column(String(120))
    quality: Mapped[str] = mapped_column(String(32), nullable=False)
    provenance: Mapped[dict | None] = mapped_column(JSON)


class IngestionFailure(Base, TimestampMixin):
    __tablename__ = "ingestion_failures"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint("quality in ('bad', 'missing', 'suspect')", name="ck_ingestion_failure_quality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    source_record_id: Mapped[str | None] = mapped_column(String(255))
    quality: Mapped[str] = mapped_column(String(32), nullable=False, default="bad")
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSON)
    payload: Mapped[dict | None] = mapped_column(JSON)
    dead_letter: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class WaveformRecord(Base, TimestampMixin):
    __tablename__ = "waveform_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_waveform_records_org_id"),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint("quality in ('good', 'suspect', 'bad', 'missing')", name="ck_waveform_quality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    sampling_rate_hz: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    quality: Mapped[str] = mapped_column(String(32), nullable=False, default="good")
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict | None] = mapped_column(JSON)

    sensor: Mapped[Sensor] = relationship(back_populates="waveforms")


class AnalyticsRun(Base, TimestampMixin):
    __tablename__ = "analytics_runs"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_analytics_runs_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "input_batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint("run_kind in ('batch', 'sensor')", name="ck_analytics_run_kind"),
        CheckConstraint("status in ('running', 'completed', 'partial', 'failed')", name="ck_analytics_run_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    input_batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    run_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    health_state_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provenance: Mapped[dict | None] = mapped_column(JSON)

    organization: Mapped[Organization] = relationship(back_populates="analytics_runs")


class AnalyticsFeatureRecord(Base, TimestampMixin):
    __tablename__ = "analytics_feature_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_analytics_features_org_id"),
        UniqueConstraint(
            "organization_id",
            "algorithm_version",
            "source_kind",
            "source_record_id",
            "feature_name",
            name="uq_analytics_feature_source",
        ),
        ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint("source_kind in ('scalar', 'waveform')", name="ck_analytics_feature_source_kind"),
        CheckConstraint("quality in ('good', 'suspect', 'bad', 'missing')", name="ck_analytics_feature_quality"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    feature_name: Mapped[str] = mapped_column(String(120), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(64))
    quality: Mapped[str] = mapped_column(String(32), nullable=False, default="good")
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    provenance: Mapped[dict | None] = mapped_column(JSON)


class AnalyticsHealthState(Base, TimestampMixin):
    __tablename__ = "analytics_health_states"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_analytics_health_states_org_id"),
        UniqueConstraint(
            "organization_id",
            "algorithm_version",
            "sensor_id",
            "observed_at",
            name="uq_analytics_health_state_sensor_time",
        ),
        ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint(
            "health_state in ('insufficient_evidence', 'healthy', 'watch', 'warning', 'critical', 'unknown')",
            name="ck_analytics_health_state",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    health_state: Mapped[str] = mapped_column(String(32), nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    trend_slope: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSON)


class AnalyticsFailure(Base, TimestampMixin):
    __tablename__ = "analytics_failures"
    __table_args__ = (
        ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        CheckConstraint("source_kind in ('scalar', 'waveform')", name="ck_analytics_failure_source_kind"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    sensor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_record_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSON)
    dead_letter: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class MLDatasetVersion(Base, TimestampMixin):
    __tablename__ = "ml_dataset_versions"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ml_dataset_versions_org_id"),
        UniqueConstraint("organization_id", "name", "version", name="uq_ml_dataset_name_version"),
        CheckConstraint("status in ('created', 'archived')", name="ck_ml_dataset_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    source_algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    target_name: Mapped[str] = mapped_column(String(120), nullable=False)
    target_unit: Mapped[str | None] = mapped_column(String(64))
    feature_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_group_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    filters: Mapped[dict | None] = mapped_column(JSON)
    provenance: Mapped[dict | None] = mapped_column(JSON)

    organization: Mapped[Organization] = relationship(back_populates="ml_dataset_versions")


class MLExperimentRun(Base, TimestampMixin):
    __tablename__ = "ml_experiment_runs"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ml_experiments_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        CheckConstraint("status in ('running', 'completed', 'failed')", name="ck_ml_experiment_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    algorithm: Mapped[str] = mapped_column(String(120), nullable=False)
    validation_method: Mapped[str] = mapped_column(String(120), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_version: Mapped[str] = mapped_column(String(64), nullable=False)
    training_config: Mapped[dict | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    baseline_metrics: Mapped[dict | None] = mapped_column(JSON)
    uncertainty: Mapped[dict | None] = mapped_column(JSON)
    abstention_policy: Mapped[dict | None] = mapped_column(JSON)
    artifact_uri: Mapped[str | None] = mapped_column(String(1024))
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    provenance: Mapped[dict | None] = mapped_column(JSON)


class MLModelRegistry(Base, TimestampMixin):
    __tablename__ = "ml_model_registries"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ml_model_registries_org_id"),
        UniqueConstraint("organization_id", "name", name="uq_ml_model_registry_name"),
        CheckConstraint("status in ('active', 'archived')", name="ck_ml_model_registry_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    description: Mapped[str | None] = mapped_column(String(1024))


class MLModelVersion(Base, TimestampMixin):
    __tablename__ = "ml_model_versions"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_ml_model_versions_org_id"),
        UniqueConstraint("organization_id", "registry_id", "version", name="uq_ml_model_registry_version"),
        ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "experiment_run_id"],
            ["ml_experiment_runs.organization_id", "ml_experiment_runs.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        CheckConstraint(
            "stage in ('candidate', 'validated', 'production', 'archived', 'rejected')",
            name="ck_ml_model_version_stage",
        ),
        CheckConstraint(
            "approval_status in ('not_required', 'pending', 'approved', 'rejected')",
            name="ck_ml_model_version_approval",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    registry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    experiment_run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate")
    approval_status: Mapped[str] = mapped_column(String(32), nullable=False, default="not_required")
    artifact_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    baseline_metrics: Mapped[dict | None] = mapped_column(JSON)
    uncertainty: Mapped[dict | None] = mapped_column(JSON)
    abstention_policy: Mapped[dict | None] = mapped_column(JSON)
    provenance: Mapped[dict | None] = mapped_column(JSON)
    approved_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MLModelPromotionEvent(Base, TimestampMixin):
    __tablename__ = "ml_model_promotion_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        CheckConstraint("action in ('promote', 'rollback')", name="ck_ml_promotion_action"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    registry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    model_version_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    from_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    to_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    reason: Mapped[str | None] = mapped_column(String(1024))
    event_metadata: Mapped[dict | None] = mapped_column(JSON)
