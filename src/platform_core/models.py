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


class OrganizationIdentityProvider(Base, TimestampMixin):
    __tablename__ = "organization_identity_providers"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_org_idp_org_id"),
        UniqueConstraint("organization_id", "name", name="uq_org_idp_org_name"),
        UniqueConstraint("organization_id", "issuer", "audience", name="uq_org_idp_org_issuer_audience"),
        CheckConstraint("status in ('active', 'inactive')", name="ck_org_idp_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    issuer: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    audience: Mapped[str] = mapped_column(String(255), nullable=False)
    discovery_url: Mapped[str | None] = mapped_column(String(1024))
    jwks_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    allowed_algorithms: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    claim_mapping: Mapped[dict | None] = mapped_column(JSON)


class UserIdentity(Base, TimestampMixin):
    __tablename__ = "user_identities"
    __table_args__ = (
        UniqueConstraint("identity_provider_id", "subject", name="uq_user_identity_provider_subject"),
        UniqueConstraint("issuer", "subject", name="uq_user_identity_issuer_subject"),
        ForeignKeyConstraint(
            ["organization_id", "identity_provider_id"],
            ["organization_identity_providers.organization_id", "organization_identity_providers.id"],
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    identity_provider_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    issuer: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    profile: Mapped[dict | None] = mapped_column(JSON)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ServicePrincipal(Base, TimestampMixin):
    __tablename__ = "service_principals"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_service_principal_org_name"),
        UniqueConstraint("organization_id", "external_subject", name="uq_service_principal_org_subject"),
        CheckConstraint("status in ('active', 'inactive', 'archived')", name="ck_service_principal_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    external_subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    issuer: Mapped[str | None] = mapped_column(String(512), index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    permissions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)


class SecretReference(Base, TimestampMixin):
    __tablename__ = "secret_references"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_secret_reference_org_name"),
        CheckConstraint("status in ('active', 'inactive', 'rotating', 'archived')", name="ck_secret_reference_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(120), nullable=False)
    locator: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    rotation_metadata: Mapped[dict | None] = mapped_column(JSON)
    last_rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SecurityAuditEvent(Base):
    __tablename__ = "security_audit_events"
    __table_args__ = (
        CheckConstraint("principal_type in ('user', 'service', 'system', 'anonymous')", name="ck_audit_principal_type"),
        CheckConstraint("outcome in ('allowed', 'denied', 'failed')", name="ck_audit_outcome"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str | None] = mapped_column(ForeignKey("organizations.id"), nullable=True, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    principal_type: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    service_principal_id: Mapped[str | None] = mapped_column(
        ForeignKey("service_principals.id"), nullable=True, index=True
    )
    issuer: Mapped[str | None] = mapped_column(String(512))
    subject_hash: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    required_permission: Mapped[str | None] = mapped_column(String(120))
    resource_type: Mapped[str | None] = mapped_column(String(120))
    resource_id: Mapped[str | None] = mapped_column(String(255))
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(120), nullable=False)
    http_method: Mapped[str | None] = mapped_column(String(16))
    http_path: Mapped[str | None] = mapped_column(String(1024))
    request_metadata: Mapped[dict | None] = mapped_column(JSON)
    event_metadata: Mapped[dict | None] = mapped_column(JSON)


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


class ModelServingBinding(Base, TimestampMixin):
    __tablename__ = "model_serving_bindings"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_model_serving_bindings_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        CheckConstraint(
            "scope_type in ('organization', 'site', 'asset', 'component', 'sensor')",
            name="ck_model_serving_binding_scope",
        ),
        CheckConstraint("status in ('active', 'disabled')", name="ck_model_serving_binding_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    registry_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    model_version_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    approved_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    reason: Mapped[str | None] = mapped_column(String(1024))
    provenance: Mapped[dict | None] = mapped_column(JSON)


class ProductionModelResolution(Base, TimestampMixin):
    __tablename__ = "production_model_resolutions"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_production_model_resolutions_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "binding_id"],
            ["model_serving_bindings.organization_id", "model_serving_bindings.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint("status in ('resolved', 'abstained', 'failed')", name="ck_production_model_resolution_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    binding_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    registry_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(120), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    feature_schema: Mapped[list[str] | None] = mapped_column(JSON)
    abstention_policy: Mapped[dict | None] = mapped_column(JSON)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evidence: Mapped[dict | None] = mapped_column(JSON)


class PredictionRecord(Base, TimestampMixin):
    __tablename__ = "prediction_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_prediction_records_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "model_resolution_id"],
            ["production_model_resolutions.organization_id", "production_model_resolutions.id"],
        ),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        CheckConstraint(
            "prediction_status in ('supported', 'unsupported', 'insufficient_evidence')",
            name="ck_prediction_record_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    model_resolution_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    registry_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prediction_status: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_rul_hours: Mapped[float | None] = mapped_column(Float)
    abstention_code: Mapped[str | None] = mapped_column(String(120))
    uncertainty: Mapped[dict | None] = mapped_column(JSON)
    feature_vector: Mapped[dict | None] = mapped_column(JSON)
    feature_record_ids: Mapped[list[str] | None] = mapped_column(JSON)
    abstention_reason: Mapped[str | None] = mapped_column(String(1024))
    provenance: Mapped[dict | None] = mapped_column(JSON)


class ModelServingMonitor(Base, TimestampMixin):
    __tablename__ = "model_serving_monitors"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint(
            "status in ('ok', 'drifted', 'insufficient_evidence', 'failed')",
            name="ck_model_serving_monitor_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    drift_score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[dict | None] = mapped_column(JSON)


class RetrainingTrigger(Base, TimestampMixin):
    __tablename__ = "retraining_triggers"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint("status in ('open', 'acknowledged', 'resolved')", name="ck_retraining_trigger_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    model_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    trigger_kind: Mapped[str] = mapped_column(String(120), nullable=False)
    reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evidence: Mapped[dict | None] = mapped_column(JSON)


class MaintenanceAlert(Base, TimestampMixin):
    __tablename__ = "maintenance_alerts"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_maintenance_alerts_org_id"),
        ForeignKeyConstraint(["organization_id", "site_id"], ["sites.organization_id", "sites.id"]),
        ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        ForeignKeyConstraint(
            ["organization_id", "prediction_id"],
            ["prediction_records.organization_id", "prediction_records.id"],
        ),
        ForeignKeyConstraint(
            ["organization_id", "model_resolution_id"],
            ["production_model_resolutions.organization_id", "production_model_resolutions.id"],
        ),
        CheckConstraint("severity in ('info', 'watch', 'warning', 'critical')", name="ck_maintenance_alert_severity"),
        CheckConstraint("priority in ('low', 'medium', 'high', 'critical')", name="ck_maintenance_alert_priority"),
        CheckConstraint(
            "status in ('open', 'acknowledged', 'resolved', 'dismissed')",
            name="ck_maintenance_alert_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    site_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    component_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    prediction_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    model_resolution_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(120), nullable=False)
    alert_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(String(1024), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    priority: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    source_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    source_reason_code: Mapped[str | None] = mapped_column(String(120))
    recommended_action: Mapped[str | None] = mapped_column(String(1024))
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence_snapshot: Mapped[dict | None] = mapped_column(JSON)
    evidence: Mapped[dict | None] = mapped_column(JSON)
    acknowledged_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledgement_note: Mapped[str | None] = mapped_column(String(1024))
    resolved_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disposition: Mapped[str | None] = mapped_column(String(64))
    disposition_reason: Mapped[str | None] = mapped_column(String(1024))


class MaintenanceCase(Base, TimestampMixin):
    __tablename__ = "maintenance_cases"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_maintenance_cases_org_id"),
        UniqueConstraint("organization_id", "case_number", name="uq_maintenance_cases_org_case_number"),
        ForeignKeyConstraint(
            ["organization_id", "alert_id"],
            ["maintenance_alerts.organization_id", "maintenance_alerts.id"],
        ),
        ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint("priority in ('low', 'medium', 'high', 'critical')", name="ck_maintenance_case_priority"),
        CheckConstraint(
            "status in ('open', 'in_progress', 'resolved', 'closed')",
            name="ck_maintenance_case_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    alert_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    case_number: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    priority: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    component_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    opened_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    owner_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    assignee_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str | None] = mapped_column(String(1024))
    recommended_action: Mapped[str | None] = mapped_column(String(1024))
    history: Mapped[list[dict] | None] = mapped_column(JSON)
    evidence: Mapped[dict | None] = mapped_column(JSON)


class MaintenanceAcknowledgement(Base, TimestampMixin):
    __tablename__ = "maintenance_acknowledgements"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        CheckConstraint("decision in ('accepted', 'deferred', 'dismissed')", name="ck_maintenance_ack_decision"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    acknowledged_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[str | None] = mapped_column(String(1024))
    acknowledged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class MaintenanceInspection(Base, TimestampMixin):
    __tablename__ = "maintenance_inspections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        CheckConstraint(
            "status in ('requested', 'in_progress', 'completed', 'cancelled')",
            name="ck_inspection_status",
        ),
        CheckConstraint(
            "condition in ('normal', 'watch', 'degraded', 'failed', 'unknown')",
            name="ck_inspection_condition",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    component_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    sensor_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="requested")
    requested_reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    requested_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    assigned_to_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    performed_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    condition: Mapped[str | None] = mapped_column(String(32))
    findings: Mapped[str | None] = mapped_column(String(2048))
    recommended_follow_up: Mapped[str | None] = mapped_column(String(1024))
    evidence_metadata: Mapped[dict | None] = mapped_column(JSON)
    inspected_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    inspected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observation: Mapped[str | None] = mapped_column(String(2048))
    measurements: Mapped[dict | None] = mapped_column(JSON)


class MaintenanceNote(Base, TimestampMixin):
    __tablename__ = "maintenance_notes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    author_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    body: Mapped[str] = mapped_column(String(2048), nullable=False)
    note_kind: Mapped[str] = mapped_column(String(64), nullable=False, default="technician_note")
    metadata_json: Mapped[dict | None] = mapped_column(JSON)


class MaintenanceWorkOrder(Base, TimestampMixin):
    __tablename__ = "maintenance_work_orders"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_maintenance_work_orders_org_id"),
        UniqueConstraint("organization_id", "work_order_number", name="uq_work_orders_org_number"),
        ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        CheckConstraint(
            "status in ('draft', 'approved', 'in_progress', 'completed', 'cancelled')",
            name="ck_work_order_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    component_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    work_order_number: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2048))
    priority: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_work: Mapped[str] = mapped_column(String(2048), nullable=False)
    summary: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    approved_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    assignee_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    planned_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completion_notes: Mapped[str | None] = mapped_column(String(2048))
    cmms_provider: Mapped[str | None] = mapped_column(String(120))
    cmms_external_id: Mapped[str | None] = mapped_column(String(255))
    cmms_state: Mapped[str | None] = mapped_column(String(120))
    work_performed: Mapped[str | None] = mapped_column(String(2048))
    evidence: Mapped[dict | None] = mapped_column(JSON)


class MaintenanceResolution(Base, TimestampMixin):
    __tablename__ = "maintenance_resolutions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        CheckConstraint(
            "outcome in ('confirmed', 'not_found', 'monitor', 'repaired', 'replaced')",
            name="ck_resolution_outcome",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    resolved_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[str] = mapped_column(String(2048), nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evidence: Mapped[dict | None] = mapped_column(JSON)


class CmmsSyncRecord(Base, TimestampMixin):
    __tablename__ = "cmms_sync_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "id", name="uq_cmms_sync_records_org_id"),
        ForeignKeyConstraint(
            ["organization_id", "work_order_id"],
            ["maintenance_work_orders.organization_id", "maintenance_work_orders.id"],
        ),
        CheckConstraint(
            "operation in ('create', 'update', 'cancel', 'close')",
            name="ck_cmms_sync_operation",
        ),
        CheckConstraint(
            "status in ('not_configured', 'succeeded', 'failed', 'timeout', 'skipped')",
            name="ck_cmms_sync_status",
        ),
        CheckConstraint("initiator_type in ('user', 'system')", name="ck_cmms_sync_initiator_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False, index=True)
    work_order_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    provider_name: Mapped[str] = mapped_column(String(120), nullable=False)
    initiator_type: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    initiated_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255))
    error_category: Mapped[str | None] = mapped_column(String(120))
    error_message: Mapped[str | None] = mapped_column(String(1024))
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_metadata: Mapped[dict | None] = mapped_column(JSON)
