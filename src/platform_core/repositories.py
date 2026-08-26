"""Repository layer for Platform Core persistence."""

from __future__ import annotations

import json
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
    MachineReading,
    Organization,
    OrganizationMembership,
    Sensor,
    Site,
    User,
)


class TenantBoundaryError(ValueError):
    """Raised when a write attempts to cross organization ownership boundaries."""


class PlatformRepository:
    def __init__(self, session: Session):
        self.session = session

    def count(self, model: type) -> int:
        return int(self.session.scalar(select(func.count()).select_from(model)) or 0)

    def health_counts(self) -> dict[str, int]:
        return {
            "organizations": self.count(Organization),
            "users": self.count(User),
            "sites": self.count(Site),
            "assets": self.count(Asset),
            "components": self.count(Component),
            "sensors": self.count(Sensor),
            "machine_readings": self.count(MachineReading),
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
            payload_json=json.dumps(data.payload, sort_keys=True) if data.payload is not None else None,
        )
        self.session.add(reading)
        self.session.flush()
        return reading

    def list_assets(self, organization_id: str) -> list[Asset]:
        statement: Select[tuple[Asset]] = select(Asset).where(Asset.organization_id == organization_id)
        return list(self.session.scalars(statement.order_by(Asset.slug)))

    def list_sensors(self, organization_id: str) -> list[Sensor]:
        statement: Select[tuple[Sensor]] = select(Sensor).where(Sensor.organization_id == organization_id)
        return list(self.session.scalars(statement.order_by(Sensor.slug)))

    def add_all(self, rows: Iterable[object]) -> None:
        self.session.add_all(rows)
        self.session.flush()

