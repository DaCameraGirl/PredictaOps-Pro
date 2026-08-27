"""Repository layer for Platform Core persistence."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    MachineReadingCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.models import (
    Asset,
    Component,
    IngestedRecord,
    IngestionBatch,
    IngestionFailure,
    IngestionSource,
    MachineReading,
    Organization,
    OrganizationMembership,
    Sensor,
    Site,
    User,
    WaveformRecord,
)


class TenantBoundaryError(ValueError):
    """Raised when a write attempts to cross organization ownership boundaries."""


class PlatformRepository:
    def __init__(self, session: Session):
        self.session = session

    def count(self, model: type) -> int:
        return int(self.session.scalar(select(func.count()).select_from(model)) or 0)

    def count_for_organization(self, model: type, organization_id: str) -> int:
        return int(
            self.session.scalar(
                select(func.count()).select_from(model).where(model.organization_id == organization_id)
            )
            or 0
        )

    def health_counts(self) -> dict[str, int]:
        return {
            "organizations": self.count(Organization),
            "users": self.count(User),
            "sites": self.count(Site),
            "assets": self.count(Asset),
            "components": self.count(Component),
            "sensors": self.count(Sensor),
            "machine_readings": self.count(MachineReading),
            "ingestion_sources": self.count(IngestionSource),
            "ingestion_batches": self.count(IngestionBatch),
            "ingestion_failures": self.count(IngestionFailure),
            "waveform_records": self.count(WaveformRecord),
        }

    def get_organization_by_slug(self, slug: str) -> Organization | None:
        return self.session.scalar(select(Organization).where(Organization.slug == slug))

    def get_site_by_slug(self, organization_id: str, slug: str) -> Site | None:
        return self.session.scalar(
            select(Site).where(Site.organization_id == organization_id, Site.slug == slug)
        )

    def get_asset_by_slug(self, organization_id: str, site_id: str, slug: str) -> Asset | None:
        return self.session.scalar(
            select(Asset).where(
                Asset.organization_id == organization_id,
                Asset.site_id == site_id,
                Asset.slug == slug,
            )
        )

    def get_component_by_slug(self, organization_id: str, asset_id: str, slug: str) -> Component | None:
        return self.session.scalar(
            select(Component).where(
                Component.organization_id == organization_id,
                Component.asset_id == asset_id,
                Component.slug == slug,
            )
        )

    def get_sensor_by_slug(self, organization_id: str, component_id: str, slug: str) -> Sensor | None:
        return self.session.scalar(
            select(Sensor).where(
                Sensor.organization_id == organization_id,
                Sensor.component_id == component_id,
                Sensor.slug == slug,
            )
        )

    def get_sensor_by_id(self, organization_id: str, sensor_id: str) -> Sensor | None:
        return self.session.scalar(
            select(Sensor).where(Sensor.organization_id == organization_id, Sensor.id == sensor_id)
        )

    def get_sensor_by_external_ref(self, organization_id: str, external_ref: str) -> Sensor | None:
        return self.session.scalar(
            select(Sensor).where(Sensor.organization_id == organization_id, Sensor.external_ref == external_ref)
        )

    def get_sensor_by_path(
        self,
        organization_id: str,
        *,
        site_slug: str,
        asset_slug: str,
        component_slug: str,
        sensor_slug: str,
    ) -> Sensor | None:
        return self.session.scalar(
            select(Sensor)
            .join(Component, Sensor.component_id == Component.id)
            .join(Asset, Component.asset_id == Asset.id)
            .join(Site, Asset.site_id == Site.id)
            .where(
                Sensor.organization_id == organization_id,
                Component.organization_id == organization_id,
                Asset.organization_id == organization_id,
                Site.organization_id == organization_id,
                Site.slug == site_slug,
                Asset.slug == asset_slug,
                Component.slug == component_slug,
                Sensor.slug == sensor_slug,
            )
        )

    def create_organization(self, data: OrganizationCreate) -> Organization:
        org = Organization(slug=data.slug, name=data.name)
        self.session.add(org)
        self.session.flush()
        return org

    def get_or_create_organization(self, data: OrganizationCreate) -> Organization:
        existing = self.get_organization_by_slug(data.slug)
        if existing:
            return existing
        return self.create_organization(data)

    def create_user(self, data: UserCreate) -> User:
        user = User(
            email=data.email.lower(),
            full_name=data.full_name,
            external_subject=data.external_subject,
        )
        self.session.add(user)
        self.session.flush()
        return user

    def add_membership(self, organization_id: str, user_id: str, role: str) -> OrganizationMembership:
        membership = OrganizationMembership(organization_id=organization_id, user_id=user_id, role=role)
        self.session.add(membership)
        self.session.flush()
        return membership

    def create_site(self, organization_id: str, data: SiteCreate) -> Site:
        site = Site(
            organization_id=organization_id,
            slug=data.slug,
            name=data.name,
            timezone=data.timezone,
        )
        self.session.add(site)
        self.session.flush()
        return site

    def get_or_create_site(self, organization_id: str, data: SiteCreate) -> Site:
        existing = self.get_site_by_slug(organization_id, data.slug)
        if existing:
            return existing
        return self.create_site(organization_id, data)

    def create_asset(self, organization_id: str, data: AssetCreate) -> Asset:
        site = self.session.get(Site, data.site_id)
        if site is None or site.organization_id != organization_id:
            raise TenantBoundaryError("asset site must belong to the same organization")
        asset = Asset(
            organization_id=organization_id,
            site_id=data.site_id,
            slug=data.slug,
            name=data.name,
            asset_type=data.asset_type,
            external_ref=data.external_ref,
        )
        self.session.add(asset)
        self.session.flush()
        return asset

    def get_or_create_asset(self, organization_id: str, data: AssetCreate) -> Asset:
        existing = self.get_asset_by_slug(organization_id, data.site_id, data.slug)
        if existing:
            return existing
        return self.create_asset(organization_id, data)

    def create_component(self, organization_id: str, data: ComponentCreate) -> Component:
        asset = self.session.get(Asset, data.asset_id)
        if asset is None or asset.organization_id != organization_id:
            raise TenantBoundaryError("component asset must belong to the same organization")
        component = Component(
            organization_id=organization_id,
            asset_id=data.asset_id,
            slug=data.slug,
            name=data.name,
            component_type=data.component_type,
            external_ref=data.external_ref,
        )
        self.session.add(component)
        self.session.flush()
        return component

    def get_or_create_component(self, organization_id: str, data: ComponentCreate) -> Component:
        existing = self.get_component_by_slug(organization_id, data.asset_id, data.slug)
        if existing:
            return existing
        return self.create_component(organization_id, data)

    def create_sensor(self, organization_id: str, data: SensorCreate) -> Sensor:
        component = self.session.get(Component, data.component_id)
        if component is None or component.organization_id != organization_id:
            raise TenantBoundaryError("sensor component must belong to the same organization")
        sensor = Sensor(
            organization_id=organization_id,
            component_id=data.component_id,
            slug=data.slug,
            name=data.name,
            sensor_type=data.sensor_type,
            unit=data.unit,
            sampling_rate_hz=data.sampling_rate_hz,
            channel_name=data.channel_name,
            axis=data.axis,
            manufacturer=data.manufacturer,
            model=data.model,
            serial_number=data.serial_number,
            external_ref=data.external_ref,
        )
        self.session.add(sensor)
        self.session.flush()
        return sensor

    def get_or_create_sensor(self, organization_id: str, data: SensorCreate) -> Sensor:
        existing = self.get_sensor_by_slug(organization_id, data.component_id, data.slug)
        if existing:
            return existing
        return self.create_sensor(organization_id, data)

    def create_machine_reading(self, organization_id: str, data: MachineReadingCreate) -> MachineReading:
        sensor = self.session.get(Sensor, data.sensor_id)
        if sensor is None or sensor.organization_id != organization_id:
            raise TenantBoundaryError("reading sensor must belong to the same organization")
        reading = MachineReading(
            organization_id=organization_id,
            sensor_id=data.sensor_id,
            observed_at=data.observed_at,
            metric=data.metric,
            value=data.value,
            unit=data.unit,
            source=data.source,
            quality=data.quality,
            payload=data.payload,
        )
        self.session.add(reading)
        self.session.flush()
        return reading

    def get_or_create_ingestion_source(
        self,
        organization_id: str,
        *,
        name: str,
        source_type: str,
        external_ref: str | None = None,
        config: dict | None = None,
    ) -> IngestionSource:
        if self.session.get(Organization, organization_id) is None:
            raise TenantBoundaryError("ingestion source organization does not exist")
        existing = self.session.scalar(
            select(IngestionSource).where(
                IngestionSource.organization_id == organization_id,
                IngestionSource.name == name,
            )
        )
        if existing:
            if existing.source_type != source_type:
                raise TenantBoundaryError("ingestion source name is already registered with a different source type")
            return existing
        source = IngestionSource(
            organization_id=organization_id,
            name=name,
            source_type=source_type,
            external_ref=external_ref,
            status="active",
            config=config,
        )
        self.session.add(source)
        self.session.flush()
        return source

    def create_ingestion_batch(
        self,
        organization_id: str,
        *,
        source_id: str,
        source_type: str,
        idempotency_key: str | None,
        source_uri: str | None,
        provenance: dict | None,
        replay_of_batch_id: str | None = None,
    ) -> IngestionBatch:
        source = self.session.get(IngestionSource, source_id)
        if source is None or source.organization_id != organization_id:
            raise TenantBoundaryError("ingestion source must belong to the same organization")
        if replay_of_batch_id is not None:
            replay_batch = self.session.get(IngestionBatch, replay_of_batch_id)
            if replay_batch is None or replay_batch.organization_id != organization_id:
                raise TenantBoundaryError("replay batch must belong to the same organization")
        batch = IngestionBatch(
            organization_id=organization_id,
            source_id=source_id,
            source_type=source_type,
            idempotency_key=idempotency_key,
            source_uri=source_uri,
            replay_of_batch_id=replay_of_batch_id,
            provenance=provenance,
            status="accepted",
            accepted_count=0,
            duplicate_count=0,
            failed_count=0,
            scalar_count=0,
            waveform_count=0,
        )
        self.session.add(batch)
        self.session.flush()
        return batch

    def get_ingested_record_by_key(
        self,
        organization_id: str,
        *,
        source_id: str,
        idempotency_key: str,
    ) -> IngestedRecord | None:
        return self.session.scalar(
            select(IngestedRecord).where(
                IngestedRecord.organization_id == organization_id,
                IngestedRecord.source_id == source_id,
                IngestedRecord.idempotency_key == idempotency_key,
            )
        )

    def create_ingested_record(
        self,
        organization_id: str,
        *,
        source_id: str,
        batch_id: str,
        idempotency_key: str,
        target_type: str,
        target_id: str,
        observed_at,
        metric: str | None,
        quality: str,
        provenance: dict | None,
    ) -> IngestedRecord:
        record = IngestedRecord(
            organization_id=organization_id,
            source_id=source_id,
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            target_type=target_type,
            target_id=target_id,
            observed_at=observed_at,
            metric=metric,
            quality=quality,
            provenance=provenance,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def create_ingestion_failure(
        self,
        organization_id: str,
        *,
        source_id: str,
        batch_id: str,
        source_record_id: str | None,
        quality: str,
        reason: str,
        detail: dict | None,
        payload: dict | None,
    ) -> IngestionFailure:
        failure = IngestionFailure(
            organization_id=organization_id,
            source_id=source_id,
            batch_id=batch_id,
            source_record_id=source_record_id,
            quality=quality,
            reason=reason,
            detail=detail,
            payload=payload,
            dead_letter=True,
        )
        self.session.add(failure)
        self.session.flush()
        return failure

    def create_waveform_record(
        self,
        organization_id: str,
        *,
        sensor_id: str,
        batch_id: str,
        observed_at,
        unit: str,
        sampling_rate_hz: float,
        sample_count: int,
        source: str,
        quality: str,
        storage_uri: str,
        sha256: str | None,
        metadata_json: dict | None,
    ) -> WaveformRecord:
        sensor = self.session.get(Sensor, sensor_id)
        if sensor is None or sensor.organization_id != organization_id:
            raise TenantBoundaryError("waveform sensor must belong to the same organization")
        batch = self.session.get(IngestionBatch, batch_id)
        if batch is None or batch.organization_id != organization_id:
            raise TenantBoundaryError("waveform batch must belong to the same organization")
        waveform = WaveformRecord(
            organization_id=organization_id,
            sensor_id=sensor_id,
            batch_id=batch_id,
            observed_at=observed_at,
            unit=unit,
            sampling_rate_hz=sampling_rate_hz,
            sample_count=sample_count,
            source=source,
            quality=quality,
            storage_uri=storage_uri,
            sha256=sha256,
            metadata_json=metadata_json,
        )
        self.session.add(waveform)
        self.session.flush()
        return waveform

    def list_ingested_records_for_batch(self, organization_id: str, batch_id: str) -> list[IngestedRecord]:
        statement = (
            select(IngestedRecord)
            .where(IngestedRecord.organization_id == organization_id, IngestedRecord.batch_id == batch_id)
            .order_by(IngestedRecord.observed_at, IngestedRecord.id)
        )
        return list(self.session.scalars(statement))

    def get_machine_reading(self, organization_id: str, reading_id: str) -> MachineReading | None:
        return self.session.scalar(
            select(MachineReading).where(
                MachineReading.organization_id == organization_id,
                MachineReading.id == reading_id,
            )
        )

    def get_waveform_record(self, organization_id: str, waveform_id: str) -> WaveformRecord | None:
        return self.session.scalar(
            select(WaveformRecord).where(
                WaveformRecord.organization_id == organization_id,
                WaveformRecord.id == waveform_id,
            )
        )

    def ingestion_health(self, organization_id: str) -> dict:
        source_rows = self.session.execute(
            select(
                IngestionSource.id,
                IngestionSource.name,
                IngestionSource.source_type,
                IngestionSource.status,
                func.count(IngestionBatch.id),
            )
            .outerjoin(
                IngestionBatch,
                (IngestionBatch.organization_id == IngestionSource.organization_id)
                & (IngestionBatch.source_id == IngestionSource.id),
            )
            .where(IngestionSource.organization_id == organization_id)
            .group_by(IngestionSource.id)
            .order_by(IngestionSource.name)
        )
        return {
            "sources": [
                {
                    "id": row.id,
                    "name": row.name,
                    "source_type": row.source_type,
                    "status": row.status,
                    "batch_count": int(row[4]),
                }
                for row in source_rows
            ],
            "batches": self.count_for_organization(IngestionBatch, organization_id),
            "failures": self.count_for_organization(IngestionFailure, organization_id),
            "waveforms": self.count_for_organization(WaveformRecord, organization_id),
        }

    def list_assets(self, organization_id: str) -> list[Asset]:
        statement: Select[tuple[Asset]] = select(Asset).where(Asset.organization_id == organization_id)
        return list(self.session.scalars(statement.order_by(Asset.slug)))

    def list_sensors(self, organization_id: str) -> list[Sensor]:
        statement: Select[tuple[Sensor]] = select(Sensor).where(Sensor.organization_id == organization_id)
        return list(self.session.scalars(statement.order_by(Sensor.slug)))

    def add_all(self, rows: Iterable[object]) -> None:
        self.session.add_all(rows)
        self.session.flush()
