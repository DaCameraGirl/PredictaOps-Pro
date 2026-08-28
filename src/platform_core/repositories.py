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
    AnalyticsFailure,
    AnalyticsFeatureRecord,
    AnalyticsHealthState,
    AnalyticsRun,
    Asset,
    CmmsSyncRecord,
    Component,
    IngestedRecord,
    IngestionBatch,
    IngestionFailure,
    IngestionSource,
    MachineReading,
    MaintenanceAcknowledgement,
    MaintenanceAlert,
    MaintenanceCase,
    MaintenanceInspection,
    MaintenanceNote,
    MaintenanceResolution,
    MaintenanceWorkOrder,
    MLDatasetVersion,
    MLExperimentRun,
    MLModelPromotionEvent,
    MLModelRegistry,
    MLModelVersion,
    ModelServingBinding,
    ModelServingMonitor,
    Organization,
    OrganizationMembership,
    PredictionRecord,
    ProductionModelResolution,
    RetrainingTrigger,
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
            "analytics_runs": self.count(AnalyticsRun),
            "analytics_feature_records": self.count(AnalyticsFeatureRecord),
            "analytics_health_states": self.count(AnalyticsHealthState),
            "analytics_failures": self.count(AnalyticsFailure),
            "ml_dataset_versions": self.count(MLDatasetVersion),
            "ml_experiment_runs": self.count(MLExperimentRun),
            "ml_model_registries": self.count(MLModelRegistry),
            "ml_model_versions": self.count(MLModelVersion),
            "ml_model_promotion_events": self.count(MLModelPromotionEvent),
            "model_serving_bindings": self.count(ModelServingBinding),
            "production_model_resolutions": self.count(ProductionModelResolution),
            "prediction_records": self.count(PredictionRecord),
            "model_serving_monitors": self.count(ModelServingMonitor),
            "retraining_triggers": self.count(RetrainingTrigger),
            "maintenance_alerts": self.count(MaintenanceAlert),
            "maintenance_cases": self.count(MaintenanceCase),
            "maintenance_work_orders": self.count(MaintenanceWorkOrder),
            "cmms_sync_records": self.count(CmmsSyncRecord),
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

    def get_component_by_id(self, organization_id: str, component_id: str) -> Component | None:
        return self.session.scalar(
            select(Component).where(Component.organization_id == organization_id, Component.id == component_id)
        )

    def get_asset_by_id(self, organization_id: str, asset_id: str) -> Asset | None:
        return self.session.scalar(select(Asset).where(Asset.organization_id == organization_id, Asset.id == asset_id))

    def get_site_by_id(self, organization_id: str, site_id: str) -> Site | None:
        return self.session.scalar(select(Site).where(Site.organization_id == organization_id, Site.id == site_id))

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

    def get_ingestion_batch(self, organization_id: str, batch_id: str) -> IngestionBatch | None:
        return self.session.scalar(
            select(IngestionBatch).where(
                IngestionBatch.organization_id == organization_id,
                IngestionBatch.id == batch_id,
            )
        )

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

    def list_machine_readings_for_sensor(self, organization_id: str, sensor_id: str) -> list[MachineReading]:
        statement = (
            select(MachineReading)
            .where(MachineReading.organization_id == organization_id, MachineReading.sensor_id == sensor_id)
            .order_by(MachineReading.observed_at, MachineReading.id)
        )
        return list(self.session.scalars(statement))

    def list_waveform_records_for_sensor(self, organization_id: str, sensor_id: str) -> list[WaveformRecord]:
        statement = (
            select(WaveformRecord)
            .where(WaveformRecord.organization_id == organization_id, WaveformRecord.sensor_id == sensor_id)
            .order_by(WaveformRecord.observed_at, WaveformRecord.id)
        )
        return list(self.session.scalars(statement))

    def create_analytics_run(
        self,
        organization_id: str,
        *,
        run_kind: str,
        algorithm_version: str,
        input_batch_id: str | None = None,
        sensor_id: str | None = None,
        provenance: dict | None = None,
    ) -> AnalyticsRun:
        if self.session.get(Organization, organization_id) is None:
            raise TenantBoundaryError("analytics run organization does not exist")
        if input_batch_id is not None:
            batch = self.get_ingestion_batch(organization_id, input_batch_id)
            if batch is None:
                raise TenantBoundaryError("analytics batch must belong to the same organization")
        if sensor_id is not None and self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("analytics sensor must belong to the same organization")
        run = AnalyticsRun(
            organization_id=organization_id,
            input_batch_id=input_batch_id,
            sensor_id=sensor_id,
            run_kind=run_kind,
            status="running",
            algorithm_version=algorithm_version,
            provenance=provenance,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def get_analytics_feature(
        self,
        organization_id: str,
        *,
        algorithm_version: str,
        source_kind: str,
        source_record_id: str,
        feature_name: str,
    ) -> AnalyticsFeatureRecord | None:
        return self.session.scalar(
            select(AnalyticsFeatureRecord).where(
                AnalyticsFeatureRecord.organization_id == organization_id,
                AnalyticsFeatureRecord.algorithm_version == algorithm_version,
                AnalyticsFeatureRecord.source_kind == source_kind,
                AnalyticsFeatureRecord.source_record_id == source_record_id,
                AnalyticsFeatureRecord.feature_name == feature_name,
            )
        )

    def create_analytics_feature(
        self,
        organization_id: str,
        *,
        run_id: str,
        sensor_id: str,
        batch_id: str | None,
        source_kind: str,
        source_record_id: str,
        observed_at,
        feature_name: str,
        value: float,
        unit: str | None,
        quality: str,
        algorithm_version: str,
        provenance: dict | None,
    ) -> AnalyticsFeatureRecord:
        run = self.session.scalar(
            select(AnalyticsRun).where(AnalyticsRun.organization_id == organization_id, AnalyticsRun.id == run_id)
        )
        if run is None:
            raise TenantBoundaryError("analytics feature run must exist")
        sensor = self.get_sensor_by_id(organization_id, sensor_id)
        if sensor is None:
            raise TenantBoundaryError("analytics feature sensor must belong to the same organization")
        feature = AnalyticsFeatureRecord(
            organization_id=organization_id,
            run_id=run_id,
            sensor_id=sensor_id,
            batch_id=batch_id,
            source_kind=source_kind,
            source_record_id=source_record_id,
            observed_at=observed_at,
            feature_name=feature_name,
            value=value,
            unit=unit,
            quality=quality,
            algorithm_version=algorithm_version,
            provenance=provenance,
        )
        self.session.add(feature)
        self.session.flush()
        return feature

    def list_analytics_features_for_sensor(
        self,
        organization_id: str,
        sensor_id: str,
        *,
        algorithm_version: str,
        feature_name: str | None = None,
    ) -> list[AnalyticsFeatureRecord]:
        statement = select(AnalyticsFeatureRecord).where(
            AnalyticsFeatureRecord.organization_id == organization_id,
            AnalyticsFeatureRecord.sensor_id == sensor_id,
            AnalyticsFeatureRecord.algorithm_version == algorithm_version,
        )
        if feature_name:
            statement = statement.where(AnalyticsFeatureRecord.feature_name == feature_name)
        return list(
            self.session.scalars(statement.order_by(AnalyticsFeatureRecord.observed_at, AnalyticsFeatureRecord.id))
        )

    def create_analytics_health_state(
        self,
        organization_id: str,
        *,
        run_id: str,
        sensor_id: str,
        observed_at,
        health_state: str,
        anomaly_score: float | None,
        trend_slope: float | None,
        confidence: float | None,
        algorithm_version: str,
        evidence: dict | None,
    ) -> AnalyticsHealthState:
        run = self.session.scalar(
            select(AnalyticsRun).where(AnalyticsRun.organization_id == organization_id, AnalyticsRun.id == run_id)
        )
        if run is None:
            raise TenantBoundaryError("analytics health-state run must exist")
        if self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("analytics health-state sensor must belong to the same organization")
        existing = self.session.scalar(
            select(AnalyticsHealthState).where(
                AnalyticsHealthState.organization_id == organization_id,
                AnalyticsHealthState.algorithm_version == algorithm_version,
                AnalyticsHealthState.sensor_id == sensor_id,
                AnalyticsHealthState.observed_at == observed_at,
            )
        )
        if existing:
            existing.run_id = run_id
            existing.health_state = health_state
            existing.anomaly_score = anomaly_score
            existing.trend_slope = trend_slope
            existing.confidence = confidence
            existing.evidence = evidence
            self.session.flush()
            return existing
        health_state_row = AnalyticsHealthState(
            organization_id=organization_id,
            run_id=run_id,
            sensor_id=sensor_id,
            observed_at=observed_at,
            health_state=health_state,
            anomaly_score=anomaly_score,
            trend_slope=trend_slope,
            confidence=confidence,
            algorithm_version=algorithm_version,
            evidence=evidence,
        )
        self.session.add(health_state_row)
        self.session.flush()
        return health_state_row

    def create_analytics_failure(
        self,
        organization_id: str,
        *,
        run_id: str,
        sensor_id: str | None,
        batch_id: str | None,
        source_kind: str,
        source_record_id: str | None,
        reason: str,
        detail: dict | None,
    ) -> AnalyticsFailure:
        run = self.session.scalar(
            select(AnalyticsRun).where(AnalyticsRun.organization_id == organization_id, AnalyticsRun.id == run_id)
        )
        if run is None:
            raise TenantBoundaryError("analytics failure run must exist")
        if sensor_id is not None and self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("analytics failure sensor must belong to the same organization")
        failure = AnalyticsFailure(
            organization_id=organization_id,
            run_id=run_id,
            sensor_id=sensor_id,
            batch_id=batch_id,
            source_kind=source_kind,
            source_record_id=source_record_id,
            reason=reason,
            detail=detail,
            dead_letter=True,
        )
        self.session.add(failure)
        self.session.flush()
        return failure

    def latest_analytics_health(self, organization_id: str) -> list[AnalyticsHealthState]:
        row_numbers = (
            select(
                AnalyticsHealthState.id,
                func.row_number()
                .over(
                    partition_by=AnalyticsHealthState.sensor_id,
                    order_by=(AnalyticsHealthState.observed_at.desc(), AnalyticsHealthState.id.desc()),
                )
                .label("row_number"),
            )
            .where(AnalyticsHealthState.organization_id == organization_id)
            .subquery()
        )
        statement = (
            select(AnalyticsHealthState)
            .join(row_numbers, AnalyticsHealthState.id == row_numbers.c.id)
            .where(row_numbers.c.row_number == 1)
            .order_by(AnalyticsHealthState.observed_at.desc(), AnalyticsHealthState.sensor_id)
        )
        return list(self.session.scalars(statement))

    def list_analytics_features_for_dataset(
        self,
        organization_id: str,
        *,
        algorithm_version: str | None = None,
        sensor_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
    ) -> list[AnalyticsFeatureRecord]:
        statement = select(AnalyticsFeatureRecord).where(AnalyticsFeatureRecord.organization_id == organization_id)
        if algorithm_version:
            statement = statement.where(AnalyticsFeatureRecord.algorithm_version == algorithm_version)
        if sensor_ids:
            statement = statement.where(AnalyticsFeatureRecord.sensor_id.in_(sensor_ids))
        if feature_names:
            statement = statement.where(AnalyticsFeatureRecord.feature_name.in_(feature_names))
        return list(
            self.session.scalars(
                statement.order_by(
                    AnalyticsFeatureRecord.sensor_id,
                    AnalyticsFeatureRecord.observed_at,
                    AnalyticsFeatureRecord.feature_name,
                    AnalyticsFeatureRecord.id,
                )
            )
        )

    def create_ml_dataset_version(
        self,
        organization_id: str,
        *,
        name: str,
        version: str,
        source_algorithm_version: str,
        target_name: str,
        target_unit: str | None,
        feature_names: list[str],
        row_count: int,
        validation_group_count: int,
        fingerprint: str,
        filters: dict | None,
        provenance: dict | None,
    ) -> MLDatasetVersion:
        if self.session.get(Organization, organization_id) is None:
            raise TenantBoundaryError("ML dataset organization does not exist")
        dataset = MLDatasetVersion(
            organization_id=organization_id,
            name=name,
            version=version,
            status="created",
            source_algorithm_version=source_algorithm_version,
            target_name=target_name,
            target_unit=target_unit,
            feature_names=feature_names,
            row_count=row_count,
            validation_group_count=validation_group_count,
            fingerprint=fingerprint,
            filters=filters,
            provenance=provenance,
        )
        self.session.add(dataset)
        self.session.flush()
        return dataset

    def get_ml_dataset_version(self, organization_id: str, dataset_version_id: str) -> MLDatasetVersion | None:
        return self.session.scalar(
            select(MLDatasetVersion).where(
                MLDatasetVersion.organization_id == organization_id,
                MLDatasetVersion.id == dataset_version_id,
            )
        )

    def list_ml_dataset_versions(self, organization_id: str) -> list[MLDatasetVersion]:
        statement = (
            select(MLDatasetVersion)
            .where(MLDatasetVersion.organization_id == organization_id)
            .order_by(MLDatasetVersion.name, MLDatasetVersion.version)
        )
        return list(self.session.scalars(statement))

    def create_ml_experiment_run(
        self,
        organization_id: str,
        *,
        dataset_version_id: str,
        name: str,
        algorithm: str,
        validation_method: str,
        code_version: str,
        training_config: dict | None,
        abstention_policy: dict | None,
        provenance: dict | None,
    ) -> MLExperimentRun:
        if self.get_ml_dataset_version(organization_id, dataset_version_id) is None:
            raise TenantBoundaryError("experiment dataset version must belong to the same organization")
        experiment = MLExperimentRun(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            name=name,
            status="running",
            algorithm=algorithm,
            validation_method=validation_method,
            code_version=code_version,
            training_config=training_config,
            abstention_policy=abstention_policy,
            provenance=provenance,
        )
        self.session.add(experiment)
        self.session.flush()
        return experiment

    def get_ml_experiment_run(self, organization_id: str, experiment_run_id: str) -> MLExperimentRun | None:
        return self.session.scalar(
            select(MLExperimentRun).where(
                MLExperimentRun.organization_id == organization_id,
                MLExperimentRun.id == experiment_run_id,
            )
        )

    def list_ml_experiment_runs(self, organization_id: str) -> list[MLExperimentRun]:
        statement = (
            select(MLExperimentRun)
            .where(MLExperimentRun.organization_id == organization_id)
            .order_by(MLExperimentRun.created_at.desc(), MLExperimentRun.id)
        )
        return list(self.session.scalars(statement))

    def get_or_create_ml_model_registry(
        self,
        organization_id: str,
        *,
        name: str,
        task: str,
        description: str | None = None,
    ) -> MLModelRegistry:
        if self.session.get(Organization, organization_id) is None:
            raise TenantBoundaryError("model registry organization does not exist")
        existing = self.session.scalar(
            select(MLModelRegistry).where(
                MLModelRegistry.organization_id == organization_id,
                MLModelRegistry.name == name,
            )
        )
        if existing:
            return existing
        registry = MLModelRegistry(
            organization_id=organization_id,
            name=name,
            task=task,
            status="active",
            description=description,
        )
        self.session.add(registry)
        self.session.flush()
        return registry

    def get_ml_model_registry(self, organization_id: str, registry_id: str) -> MLModelRegistry | None:
        return self.session.scalar(
            select(MLModelRegistry).where(
                MLModelRegistry.organization_id == organization_id,
                MLModelRegistry.id == registry_id,
            )
        )

    def list_ml_model_registries(self, organization_id: str) -> list[MLModelRegistry]:
        statement = (
            select(MLModelRegistry)
            .where(MLModelRegistry.organization_id == organization_id)
            .order_by(MLModelRegistry.name)
        )
        return list(self.session.scalars(statement))

    def get_ml_model_registry_by_name(self, organization_id: str, name: str) -> MLModelRegistry | None:
        return self.session.scalar(
            select(MLModelRegistry).where(
                MLModelRegistry.organization_id == organization_id,
                MLModelRegistry.name == name,
            )
        )

    def create_ml_model_version(
        self,
        organization_id: str,
        *,
        registry_id: str,
        experiment_run_id: str,
        dataset_version_id: str,
        version: str,
        artifact_uri: str,
        artifact_sha256: str,
        metrics: dict | None,
        baseline_metrics: dict | None,
        uncertainty: dict | None,
        abstention_policy: dict | None,
        provenance: dict | None,
    ) -> MLModelVersion:
        if self.get_ml_model_registry(organization_id, registry_id) is None:
            raise TenantBoundaryError("model version registry must belong to the same organization")
        if self.get_ml_experiment_run(organization_id, experiment_run_id) is None:
            raise TenantBoundaryError("model version experiment must belong to the same organization")
        if self.get_ml_dataset_version(organization_id, dataset_version_id) is None:
            raise TenantBoundaryError("model version dataset must belong to the same organization")
        model_version = MLModelVersion(
            organization_id=organization_id,
            registry_id=registry_id,
            experiment_run_id=experiment_run_id,
            dataset_version_id=dataset_version_id,
            version=version,
            stage="candidate",
            approval_status="not_required",
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            uncertainty=uncertainty,
            abstention_policy=abstention_policy,
            provenance=provenance,
        )
        self.session.add(model_version)
        self.session.flush()
        return model_version

    def get_ml_model_version(self, organization_id: str, model_version_id: str) -> MLModelVersion | None:
        return self.session.scalar(
            select(MLModelVersion).where(
                MLModelVersion.organization_id == organization_id,
                MLModelVersion.id == model_version_id,
            )
        )

    def list_ml_model_versions(self, organization_id: str, registry_id: str) -> list[MLModelVersion]:
        statement = (
            select(MLModelVersion)
            .where(MLModelVersion.organization_id == organization_id, MLModelVersion.registry_id == registry_id)
            .order_by(MLModelVersion.created_at, MLModelVersion.version)
        )
        return list(self.session.scalars(statement))

    def get_production_model_version(self, organization_id: str, registry_id: str) -> MLModelVersion | None:
        return self.session.scalar(
            select(MLModelVersion).where(
                MLModelVersion.organization_id == organization_id,
                MLModelVersion.registry_id == registry_id,
                MLModelVersion.stage == "production",
            )
        )

    def model_version_has_reached_production(self, organization_id: str, model_version_id: str) -> bool:
        count = self.session.scalar(
            select(func.count())
            .select_from(MLModelPromotionEvent)
            .where(
                MLModelPromotionEvent.organization_id == organization_id,
                MLModelPromotionEvent.model_version_id == model_version_id,
                MLModelPromotionEvent.to_stage == "production",
            )
        )
        return bool(count)

    def create_ml_promotion_event(
        self,
        organization_id: str,
        *,
        registry_id: str,
        model_version_id: str,
        from_stage: str,
        to_stage: str,
        action: str,
        approved_by_user_id: str | None,
        reason: str | None,
        event_metadata: dict | None,
    ) -> MLModelPromotionEvent:
        if self.get_ml_model_registry(organization_id, registry_id) is None:
            raise TenantBoundaryError("promotion registry must belong to the same organization")
        if self.get_ml_model_version(organization_id, model_version_id) is None:
            raise TenantBoundaryError("promotion model version must belong to the same organization")
        event = MLModelPromotionEvent(
            organization_id=organization_id,
            registry_id=registry_id,
            model_version_id=model_version_id,
            from_stage=from_stage,
            to_stage=to_stage,
            action=action,
            approved_by_user_id=approved_by_user_id,
            reason=reason,
            event_metadata=event_metadata,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def create_model_serving_binding(
        self,
        organization_id: str,
        *,
        registry_id: str,
        model_version_id: str,
        scope_type: str,
        scope_id: str | None,
        approved_by_user_id: str,
        reason: str | None,
        provenance: dict | None,
    ) -> ModelServingBinding:
        if self.get_ml_model_registry(organization_id, registry_id) is None:
            raise TenantBoundaryError("serving binding registry must belong to the same organization")
        model_version = self.get_ml_model_version(organization_id, model_version_id)
        if model_version is None or model_version.registry_id != registry_id:
            raise TenantBoundaryError("serving binding model version must belong to the same registry")
        user = self.session.get(User, approved_by_user_id)
        if user is None:
            raise TenantBoundaryError("serving binding approval user does not exist")
        membership = self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == approved_by_user_id,
                OrganizationMembership.lifecycle_state == "active",
            )
        )
        if membership is None:
            raise TenantBoundaryError("serving binding approval user must belong to this organization")

        self._assert_serving_scope(organization_id, scope_type, scope_id)
        for existing in self.list_model_serving_bindings(
            organization_id,
            status="active",
            registry_id=registry_id,
            scope_type=scope_type,
            scope_id=scope_id,
        ):
            existing.status = "disabled"
        binding = ModelServingBinding(
            organization_id=organization_id,
            registry_id=registry_id,
            model_version_id=model_version_id,
            scope_type=scope_type,
            scope_id=scope_id,
            status="active",
            approved_by_user_id=approved_by_user_id,
            reason=reason,
            provenance=provenance,
        )
        self.session.add(binding)
        self.session.flush()
        return binding

    def _assert_serving_scope(self, organization_id: str, scope_type: str, scope_id: str | None) -> None:
        if scope_type == "organization":
            if scope_id is not None:
                raise TenantBoundaryError("organization serving scope must not provide scope_id")
            if self.session.get(Organization, organization_id) is None:
                raise TenantBoundaryError("serving binding organization does not exist")
            return
        if scope_id is None:
            raise TenantBoundaryError("serving binding scope_id is required")
        checks = {
            "site": self.get_site_by_id,
            "asset": self.get_asset_by_id,
            "component": self.get_component_by_id,
            "sensor": self.get_sensor_by_id,
        }
        if scope_type not in checks or checks[scope_type](organization_id, scope_id) is None:
            raise TenantBoundaryError("serving binding scope must belong to the same organization")

    def list_model_serving_bindings(
        self,
        organization_id: str,
        *,
        status: str | None = None,
        registry_id: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> list[ModelServingBinding]:
        statement = select(ModelServingBinding).where(ModelServingBinding.organization_id == organization_id)
        if status:
            statement = statement.where(ModelServingBinding.status == status)
        if registry_id:
            statement = statement.where(ModelServingBinding.registry_id == registry_id)
        if scope_type:
            statement = statement.where(ModelServingBinding.scope_type == scope_type)
        if scope_id is not None:
            statement = statement.where(ModelServingBinding.scope_id == scope_id)
        return list(
            self.session.scalars(statement.order_by(ModelServingBinding.created_at.desc(), ModelServingBinding.id))
        )

    def list_active_model_serving_bindings_for_sensor(
        self,
        organization_id: str,
        *,
        registry_id: str | None,
        sensor_id: str,
        component_id: str,
        asset_id: str,
        site_id: str,
    ) -> list[ModelServingBinding]:
        scope_filters = [
            (ModelServingBinding.scope_type == "organization") & (ModelServingBinding.scope_id.is_(None)),
            (ModelServingBinding.scope_type == "site") & (ModelServingBinding.scope_id == site_id),
            (ModelServingBinding.scope_type == "asset") & (ModelServingBinding.scope_id == asset_id),
            (ModelServingBinding.scope_type == "component") & (ModelServingBinding.scope_id == component_id),
            (ModelServingBinding.scope_type == "sensor") & (ModelServingBinding.scope_id == sensor_id),
        ]
        statement = select(ModelServingBinding).where(
            ModelServingBinding.organization_id == organization_id,
            ModelServingBinding.status == "active",
            scope_filters[0] | scope_filters[1] | scope_filters[2] | scope_filters[3] | scope_filters[4],
        )
        if registry_id:
            statement = statement.where(ModelServingBinding.registry_id == registry_id)
        return list(
            self.session.scalars(statement.order_by(ModelServingBinding.created_at.desc(), ModelServingBinding.id))
        )

    def list_latest_analytics_features_for_sensor(
        self,
        organization_id: str,
        *,
        sensor_id: str,
        algorithm_version: str,
        feature_names: list[str],
        observed_at,
    ) -> dict[str, AnalyticsFeatureRecord]:
        statement = (
            select(AnalyticsFeatureRecord)
            .where(
                AnalyticsFeatureRecord.organization_id == organization_id,
                AnalyticsFeatureRecord.sensor_id == sensor_id,
                AnalyticsFeatureRecord.algorithm_version == algorithm_version,
                AnalyticsFeatureRecord.feature_name.in_(feature_names),
                AnalyticsFeatureRecord.observed_at <= observed_at,
            )
            .order_by(
                AnalyticsFeatureRecord.observed_at.desc(),
                AnalyticsFeatureRecord.feature_name,
                AnalyticsFeatureRecord.id.desc(),
            )
        )
        snapshots: dict[object, dict[str, AnalyticsFeatureRecord]] = {}
        for row in self.session.scalars(statement):
            snapshot = snapshots.setdefault(row.observed_at, {})
            snapshot.setdefault(row.feature_name, row)
        for snapshot in snapshots.values():
            if all(name in snapshot for name in feature_names):
                return snapshot
        if snapshots:
            return next(iter(snapshots.values()))
        return {}

    def latest_feature_observed_at(
        self,
        organization_id: str,
        *,
        sensor_id: str,
        algorithm_version: str,
    ):
        return self.session.scalar(
            select(func.max(AnalyticsFeatureRecord.observed_at)).where(
                AnalyticsFeatureRecord.organization_id == organization_id,
                AnalyticsFeatureRecord.sensor_id == sensor_id,
                AnalyticsFeatureRecord.algorithm_version == algorithm_version,
            )
        )

    def create_production_model_resolution(
        self,
        organization_id: str,
        *,
        binding_id: str | None,
        registry_id: str | None,
        model_version_id: str | None,
        dataset_version_id: str | None,
        sensor_id: str,
        status: str,
        reason_code: str,
        reason: str,
        artifact_sha256: str | None,
        feature_schema: list[str] | None,
        abstention_policy: dict | None,
        evidence: dict | None,
    ) -> ProductionModelResolution:
        if self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("production resolution sensor must belong to the same organization")
        resolution = ProductionModelResolution(
            organization_id=organization_id,
            binding_id=binding_id,
            registry_id=registry_id,
            model_version_id=model_version_id,
            dataset_version_id=dataset_version_id,
            sensor_id=sensor_id,
            status=status,
            reason_code=reason_code,
            reason=reason,
            artifact_sha256=artifact_sha256,
            feature_schema=feature_schema,
            abstention_policy=abstention_policy,
            evidence=evidence,
        )
        self.session.add(resolution)
        self.session.flush()
        return resolution

    def get_production_model_resolution(
        self,
        organization_id: str,
        resolution_id: str,
    ) -> ProductionModelResolution | None:
        return self.session.scalar(
            select(ProductionModelResolution).where(
                ProductionModelResolution.organization_id == organization_id,
                ProductionModelResolution.id == resolution_id,
            )
        )

    def create_prediction_record(
        self,
        organization_id: str,
        *,
        model_resolution_id: str,
        registry_id: str | None,
        model_version_id: str | None,
        dataset_version_id: str | None,
        sensor_id: str,
        observed_at,
        prediction_status: str,
        predicted_rul_hours: float | None,
        abstention_code: str | None,
        uncertainty: dict | None,
        feature_vector: dict | None,
        feature_record_ids: list[str] | None,
        abstention_reason: str | None,
        provenance: dict | None,
    ) -> PredictionRecord:
        prediction = PredictionRecord(
            organization_id=organization_id,
            model_resolution_id=model_resolution_id,
            registry_id=registry_id,
            model_version_id=model_version_id,
            dataset_version_id=dataset_version_id,
            sensor_id=sensor_id,
            observed_at=observed_at,
            prediction_status=prediction_status,
            predicted_rul_hours=predicted_rul_hours,
            abstention_code=abstention_code,
            uncertainty=uncertainty,
            feature_vector=feature_vector,
            feature_record_ids=feature_record_ids,
            abstention_reason=abstention_reason,
            provenance=provenance,
        )
        self.session.add(prediction)
        self.session.flush()
        return prediction

    def list_prediction_records(self, organization_id: str, *, sensor_id: str | None = None) -> list[PredictionRecord]:
        statement = select(PredictionRecord).where(PredictionRecord.organization_id == organization_id)
        if sensor_id:
            statement = statement.where(PredictionRecord.sensor_id == sensor_id)
        return list(self.session.scalars(statement.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id)))

    def create_model_serving_monitor(
        self,
        organization_id: str,
        *,
        model_version_id: str | None,
        sensor_id: str,
        observed_at,
        metric_name: str,
        status: str,
        drift_score: float | None,
        threshold: float | None,
        evidence: dict | None,
    ) -> ModelServingMonitor:
        monitor = ModelServingMonitor(
            organization_id=organization_id,
            model_version_id=model_version_id,
            sensor_id=sensor_id,
            observed_at=observed_at,
            metric_name=metric_name,
            status=status,
            drift_score=drift_score,
            threshold=threshold,
            evidence=evidence,
        )
        self.session.add(monitor)
        self.session.flush()
        return monitor

    def create_retraining_trigger(
        self,
        organization_id: str,
        *,
        model_version_id: str | None,
        sensor_id: str | None,
        trigger_kind: str,
        reason: str,
        evidence: dict | None,
    ) -> RetrainingTrigger:
        existing = self.session.scalar(
            select(RetrainingTrigger).where(
                RetrainingTrigger.organization_id == organization_id,
                RetrainingTrigger.model_version_id == model_version_id,
                RetrainingTrigger.sensor_id == sensor_id,
                RetrainingTrigger.trigger_kind == trigger_kind,
                RetrainingTrigger.status == "open",
            )
        )
        if existing:
            existing.reason = reason
            existing.evidence = evidence
            self.session.flush()
            return existing
        trigger = RetrainingTrigger(
            organization_id=organization_id,
            model_version_id=model_version_id,
            sensor_id=sensor_id,
            trigger_kind=trigger_kind,
            reason=reason,
            status="open",
            evidence=evidence,
        )
        self.session.add(trigger)
        self.session.flush()
        return trigger

    def list_model_serving_monitors(self, organization_id: str) -> list[ModelServingMonitor]:
        statement = (
            select(ModelServingMonitor)
            .where(ModelServingMonitor.organization_id == organization_id)
            .order_by(ModelServingMonitor.created_at.desc(), ModelServingMonitor.id)
        )
        return list(self.session.scalars(statement))

    def list_retraining_triggers(self, organization_id: str) -> list[RetrainingTrigger]:
        statement = (
            select(RetrainingTrigger)
            .where(RetrainingTrigger.organization_id == organization_id)
            .order_by(RetrainingTrigger.created_at.desc(), RetrainingTrigger.id)
        )
        return list(self.session.scalars(statement))

    def get_prediction_record(self, organization_id: str, prediction_id: str) -> PredictionRecord | None:
        return self.session.scalar(
            select(PredictionRecord).where(
                PredictionRecord.organization_id == organization_id,
                PredictionRecord.id == prediction_id,
            )
        )

    def get_active_membership(self, organization_id: str, user_id: str) -> OrganizationMembership | None:
        return self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == user_id,
                OrganizationMembership.lifecycle_state == "active",
            )
        )

    def create_maintenance_alert(
        self,
        organization_id: str,
        *,
        site_id: str | None,
        asset_id: str | None,
        component_id: str | None,
        sensor_id: str,
        prediction_id: str | None,
        model_resolution_id: str | None,
        source_type: str,
        source_id: str,
        alert_kind: str,
        title: str,
        summary: str,
        severity: str,
        priority: str,
        source_kind: str,
        source_reason_code: str | None,
        recommended_action: str | None,
        dedupe_key: str,
        evidence_snapshot: dict | None,
        evidence: dict | None,
    ) -> MaintenanceAlert:
        existing = self.get_active_maintenance_alert_by_dedupe_key(organization_id, dedupe_key)
        if existing:
            return existing
        if site_id and self.get_site_by_id(organization_id, site_id) is None:
            raise TenantBoundaryError("maintenance alert site must belong to the same organization")
        if asset_id and self.get_asset_by_id(organization_id, asset_id) is None:
            raise TenantBoundaryError("maintenance alert asset must belong to the same organization")
        if component_id and self.get_component_by_id(organization_id, component_id) is None:
            raise TenantBoundaryError("maintenance alert component must belong to the same organization")
        if self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("maintenance alert sensor must belong to the same organization")
        if prediction_id and self.get_prediction_record(organization_id, prediction_id) is None:
            raise TenantBoundaryError("maintenance alert prediction must belong to the same organization")
        alert = MaintenanceAlert(
            organization_id=organization_id,
            site_id=site_id,
            asset_id=asset_id,
            component_id=component_id,
            sensor_id=sensor_id,
            prediction_id=prediction_id,
            model_resolution_id=model_resolution_id,
            source_type=source_type,
            source_id=source_id,
            alert_kind=alert_kind,
            title=title,
            summary=summary,
            severity=severity,
            priority=priority,
            status="open",
            source_kind=source_kind,
            source_reason_code=source_reason_code,
            recommended_action=recommended_action,
            dedupe_key=dedupe_key,
            evidence_snapshot=evidence_snapshot,
            evidence=evidence,
        )
        self.session.add(alert)
        self.session.flush()
        return alert

    def get_active_maintenance_alert_by_dedupe_key(
        self,
        organization_id: str,
        dedupe_key: str,
    ) -> MaintenanceAlert | None:
        return self.session.scalar(
            select(MaintenanceAlert).where(
                MaintenanceAlert.organization_id == organization_id,
                MaintenanceAlert.dedupe_key == dedupe_key,
                MaintenanceAlert.status.in_(["open", "acknowledged"]),
            )
        )

    def get_maintenance_alert(self, organization_id: str, alert_id: str) -> MaintenanceAlert | None:
        return self.session.scalar(
            select(MaintenanceAlert).where(
                MaintenanceAlert.organization_id == organization_id,
                MaintenanceAlert.id == alert_id,
            )
        )

    def list_maintenance_alerts(self, organization_id: str, *, status: str | None = None) -> list[MaintenanceAlert]:
        statement = select(MaintenanceAlert).where(MaintenanceAlert.organization_id == organization_id)
        if status:
            statement = statement.where(MaintenanceAlert.status == status)
        return list(self.session.scalars(statement.order_by(MaintenanceAlert.created_at.desc(), MaintenanceAlert.id)))

    def update_maintenance_alert(self, alert: MaintenanceAlert, **values) -> MaintenanceAlert:
        for key, value in values.items():
            setattr(alert, key, value)
        self.session.flush()
        return alert

    def next_maintenance_case_number(self, organization_id: str) -> str:
        total = self.count_for_organization(MaintenanceCase, organization_id) + 1
        return f"CASE-{total:06d}"

    def next_work_order_number(self, organization_id: str) -> str:
        total = self.count_for_organization(MaintenanceWorkOrder, organization_id) + 1
        return f"WO-{total:06d}"

    def create_maintenance_case(
        self,
        organization_id: str,
        *,
        alert_id: str | None,
        title: str,
        summary: str | None,
        priority: str,
        asset_id: str | None,
        component_id: str | None,
        sensor_id: str | None,
        opened_by_user_id: str | None,
        owner_user_id: str | None,
        assignee_user_id: str | None,
        recommended_action: str | None,
        history: list[dict] | None,
        evidence: dict | None,
    ) -> MaintenanceCase:
        existing = self.get_active_maintenance_case_for_alert(organization_id, alert_id) if alert_id else None
        if existing:
            return existing
        if alert_id and self.get_maintenance_alert(organization_id, alert_id) is None:
            raise TenantBoundaryError("maintenance case alert must belong to the same organization")
        if asset_id and self.get_asset_by_id(organization_id, asset_id) is None:
            raise TenantBoundaryError("maintenance case asset must belong to the same organization")
        if component_id and self.get_component_by_id(organization_id, component_id) is None:
            raise TenantBoundaryError("maintenance case component must belong to the same organization")
        if sensor_id and self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("maintenance case sensor must belong to the same organization")
        case = MaintenanceCase(
            organization_id=organization_id,
            alert_id=alert_id,
            case_number=self.next_maintenance_case_number(organization_id),
            title=title,
            summary=summary,
            priority=priority,
            status="open",
            asset_id=asset_id,
            component_id=component_id,
            sensor_id=sensor_id,
            opened_by_user_id=opened_by_user_id,
            owner_user_id=owner_user_id,
            assignee_user_id=assignee_user_id,
            recommended_action=recommended_action,
            history=history,
            evidence=evidence,
        )
        self.session.add(case)
        self.session.flush()
        return case

    def get_active_maintenance_case_for_alert(
        self,
        organization_id: str,
        alert_id: str | None,
    ) -> MaintenanceCase | None:
        if not alert_id:
            return None
        return self.session.scalar(
            select(MaintenanceCase).where(
                MaintenanceCase.organization_id == organization_id,
                MaintenanceCase.alert_id == alert_id,
                MaintenanceCase.status.in_(["open", "in_progress"]),
            )
        )

    def get_maintenance_case(self, organization_id: str, case_id: str) -> MaintenanceCase | None:
        return self.session.scalar(
            select(MaintenanceCase).where(
                MaintenanceCase.organization_id == organization_id,
                MaintenanceCase.id == case_id,
            )
        )

    def list_maintenance_cases(self, organization_id: str, *, status: str | None = None) -> list[MaintenanceCase]:
        statement = select(MaintenanceCase).where(MaintenanceCase.organization_id == organization_id)
        if status:
            statement = statement.where(MaintenanceCase.status == status)
        return list(self.session.scalars(statement.order_by(MaintenanceCase.created_at.desc(), MaintenanceCase.id)))

    def update_maintenance_case(self, case: MaintenanceCase, **values) -> MaintenanceCase:
        for key, value in values.items():
            setattr(case, key, value)
        self.session.flush()
        return case

    def create_maintenance_acknowledgement(
        self,
        organization_id: str,
        *,
        case_id: str,
        acknowledged_by_user_id: str,
        decision: str,
        comment: str | None,
    ) -> MaintenanceAcknowledgement:
        case = self.get_maintenance_case(organization_id, case_id)
        if case is None:
            raise TenantBoundaryError("maintenance acknowledgement case must belong to the same organization")
        acknowledgement = MaintenanceAcknowledgement(
            organization_id=organization_id,
            case_id=case_id,
            acknowledged_by_user_id=acknowledged_by_user_id,
            decision=decision,
            comment=comment,
        )
        self.session.add(acknowledgement)
        self.session.flush()
        return acknowledgement

    def create_maintenance_inspection(
        self,
        organization_id: str,
        *,
        case_id: str,
        asset_id: str | None,
        component_id: str | None,
        sensor_id: str | None,
        requested_reason: str,
        requested_by_user_id: str,
        assigned_to_user_id: str | None,
        evidence_metadata: dict | None,
    ) -> MaintenanceInspection:
        case = self.get_maintenance_case(organization_id, case_id)
        if case is None:
            raise TenantBoundaryError("maintenance inspection case must belong to the same organization")
        if asset_id and self.get_asset_by_id(organization_id, asset_id) is None:
            raise TenantBoundaryError("maintenance inspection asset must belong to the same organization")
        if component_id and self.get_component_by_id(organization_id, component_id) is None:
            raise TenantBoundaryError("maintenance inspection component must belong to the same organization")
        if sensor_id and self.get_sensor_by_id(organization_id, sensor_id) is None:
            raise TenantBoundaryError("maintenance inspection sensor must belong to the same organization")
        inspection = MaintenanceInspection(
            organization_id=organization_id,
            case_id=case_id,
            asset_id=asset_id,
            component_id=component_id,
            sensor_id=sensor_id,
            status="requested",
            requested_reason=requested_reason,
            requested_by_user_id=requested_by_user_id,
            assigned_to_user_id=assigned_to_user_id,
            evidence_metadata=evidence_metadata,
        )
        self.session.add(inspection)
        self.session.flush()
        return inspection

    def get_maintenance_inspection(self, organization_id: str, inspection_id: str) -> MaintenanceInspection | None:
        return self.session.scalar(
            select(MaintenanceInspection).where(
                MaintenanceInspection.organization_id == organization_id,
                MaintenanceInspection.id == inspection_id,
            )
        )

    def list_maintenance_inspections(self, organization_id: str, *, case_id: str) -> list[MaintenanceInspection]:
        statement = select(MaintenanceInspection).where(
            MaintenanceInspection.organization_id == organization_id,
            MaintenanceInspection.case_id == case_id,
        )
        return list(
            self.session.scalars(statement.order_by(MaintenanceInspection.created_at, MaintenanceInspection.id))
        )

    def update_maintenance_inspection(
        self,
        inspection: MaintenanceInspection,
        **values,
    ) -> MaintenanceInspection:
        for key, value in values.items():
            setattr(inspection, key, value)
        self.session.flush()
        return inspection

    def create_maintenance_note(
        self,
        organization_id: str,
        *,
        case_id: str,
        author_user_id: str,
        body: str,
        note_kind: str,
        metadata_json: dict | None,
    ) -> MaintenanceNote:
        if self.get_maintenance_case(organization_id, case_id) is None:
            raise TenantBoundaryError("maintenance note case must belong to the same organization")
        note = MaintenanceNote(
            organization_id=organization_id,
            case_id=case_id,
            author_user_id=author_user_id,
            body=body,
            note_kind=note_kind,
            metadata_json=metadata_json,
        )
        self.session.add(note)
        self.session.flush()
        return note

    def list_maintenance_notes(self, organization_id: str, *, case_id: str) -> list[MaintenanceNote]:
        statement = select(MaintenanceNote).where(
            MaintenanceNote.organization_id == organization_id,
            MaintenanceNote.case_id == case_id,
        )
        return list(self.session.scalars(statement.order_by(MaintenanceNote.created_at, MaintenanceNote.id)))

    def create_maintenance_work_order(
        self,
        organization_id: str,
        *,
        case_id: str,
        title: str,
        description: str | None,
        priority: str,
        requested_work: str,
        requested_by_user_id: str,
        assignee_user_id: str | None,
        planned_start_at,
        evidence: dict | None,
    ) -> MaintenanceWorkOrder:
        case = self.get_maintenance_case(organization_id, case_id)
        if case is None:
            raise TenantBoundaryError("maintenance work order case must belong to the same organization")
        work_order = MaintenanceWorkOrder(
            organization_id=organization_id,
            case_id=case_id,
            asset_id=case.asset_id,
            component_id=case.component_id,
            work_order_number=self.next_work_order_number(organization_id),
            status="draft",
            title=title,
            description=description,
            priority=priority,
            requested_work=requested_work,
            summary=title,
            requested_by_user_id=requested_by_user_id,
            assignee_user_id=assignee_user_id,
            planned_start_at=planned_start_at,
            evidence=evidence,
        )
        self.session.add(work_order)
        self.session.flush()
        return work_order

    def get_maintenance_work_order(self, organization_id: str, work_order_id: str) -> MaintenanceWorkOrder | None:
        return self.session.scalar(
            select(MaintenanceWorkOrder).where(
                MaintenanceWorkOrder.organization_id == organization_id,
                MaintenanceWorkOrder.id == work_order_id,
            )
        )

    def list_maintenance_work_orders(
        self,
        organization_id: str,
        *,
        case_id: str | None = None,
    ) -> list[MaintenanceWorkOrder]:
        statement = select(MaintenanceWorkOrder).where(MaintenanceWorkOrder.organization_id == organization_id)
        if case_id:
            statement = statement.where(MaintenanceWorkOrder.case_id == case_id)
        return list(self.session.scalars(statement.order_by(MaintenanceWorkOrder.created_at, MaintenanceWorkOrder.id)))

    def update_maintenance_work_order(self, work_order: MaintenanceWorkOrder, **values) -> MaintenanceWorkOrder:
        for key, value in values.items():
            setattr(work_order, key, value)
        self.session.flush()
        return work_order

    def resolve_maintenance_case(
        self,
        organization_id: str,
        *,
        case_id: str,
        resolved_by_user_id: str,
        outcome: str,
        summary: str,
        evidence: dict | None,
    ) -> MaintenanceResolution:
        case = self.get_maintenance_case(organization_id, case_id)
        if case is None:
            raise TenantBoundaryError("maintenance resolution case must belong to the same organization")
        resolution = MaintenanceResolution(
            organization_id=organization_id,
            case_id=case_id,
            resolved_by_user_id=resolved_by_user_id,
            outcome=outcome,
            summary=summary,
            evidence=evidence,
        )
        self.session.add(resolution)
        case.status = "resolved"
        self.session.flush()
        return resolution

    def create_cmms_sync_record(
        self,
        organization_id: str,
        *,
        work_order_id: str,
        provider_name: str,
        initiator_type: str,
        initiated_by_user_id: str | None,
        operation: str,
        idempotency_key: str,
        status: str,
        external_id: str | None,
        error_category: str | None,
        error_message: str | None,
        completed_at,
        attempt_metadata: dict | None,
    ) -> CmmsSyncRecord:
        if self.get_maintenance_work_order(organization_id, work_order_id) is None:
            raise TenantBoundaryError("CMMS sync work order must belong to the same organization")
        sync = CmmsSyncRecord(
            organization_id=organization_id,
            work_order_id=work_order_id,
            provider_name=provider_name,
            initiator_type=initiator_type,
            initiated_by_user_id=initiated_by_user_id,
            operation=operation,
            idempotency_key=idempotency_key,
            status=status,
            external_id=external_id,
            error_category=error_category,
            error_message=error_message,
            completed_at=completed_at,
            attempt_metadata=attempt_metadata,
        )
        self.session.add(sync)
        self.session.flush()
        return sync

    def get_successful_cmms_sync_record(
        self,
        organization_id: str,
        *,
        work_order_id: str,
        provider_name: str,
        operation: str,
        idempotency_key: str,
    ) -> CmmsSyncRecord | None:
        return self.session.scalar(
            select(CmmsSyncRecord).where(
                CmmsSyncRecord.organization_id == organization_id,
                CmmsSyncRecord.work_order_id == work_order_id,
                CmmsSyncRecord.provider_name == provider_name,
                CmmsSyncRecord.operation == operation,
                CmmsSyncRecord.idempotency_key == idempotency_key,
                CmmsSyncRecord.status == "succeeded",
            )
        )

    def list_cmms_sync_records(self, organization_id: str, *, work_order_id: str | None = None) -> list[CmmsSyncRecord]:
        statement = select(CmmsSyncRecord).where(CmmsSyncRecord.organization_id == organization_id)
        if work_order_id:
            statement = statement.where(CmmsSyncRecord.work_order_id == work_order_id)
        return list(self.session.scalars(statement.order_by(CmmsSyncRecord.created_at, CmmsSyncRecord.id)))

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
