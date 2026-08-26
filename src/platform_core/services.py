"""Platform Core service layer and NASA/IMS bootstrap import."""

from __future__ import annotations

from sqlalchemy.orm import Session

from bearing_data import ALL_RUN_SPECS, RunSpec
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    PlatformBootstrapSummary,
    SensorCreate,
    SiteCreate,
)
from platform_core.models import Asset, Component, Sensor
from platform_core.repositories import PlatformRepository

NASA_IMS_ORG_SLUG = "nasa-ims"
NASA_IMS_SITE_SLUG = "ims-bearing-test-rigs"


class PlatformService:
    def __init__(self, session: Session):
        self.repo = PlatformRepository(session)

    def bootstrap_ims_registry(self, run_specs: dict[str, RunSpec] | None = None) -> PlatformBootstrapSummary:
        run_specs = run_specs or ALL_RUN_SPECS
        org = self.repo.get_or_create_organization(
            OrganizationCreate(slug=NASA_IMS_ORG_SLUG, name="NASA/IMS Bearing Data Set")
        )
        site = self.repo.get_or_create_site(
            org.id,
            SiteCreate(slug=NASA_IMS_SITE_SLUG, name="IMS Bearing Test Rigs", timezone="UTC"),
        )

        for run_spec in run_specs.values():
            asset = self.repo.get_or_create_asset(
                org.id,
                AssetCreate(
                    site_id=site.id,
                    slug=run_spec.run_id.replace("_", "-"),
                    name=f"{run_spec.dataset_name} Machine",
                    asset_type="bearing_test_rig",
                    external_ref=run_spec.run_id,
                ),
            )
            components_by_bearing = self._bootstrap_components(org.id, asset, run_spec)
            self._bootstrap_sensors(org.id, components_by_bearing, run_spec)

        return PlatformBootstrapSummary(
            organization_id=org.id,
            site_id=site.id,
            asset_count=self.repo.count_for_organization(Asset, org.id),
            component_count=self.repo.count_for_organization(Component, org.id),
            sensor_count=self.repo.count_for_organization(Sensor, org.id),
        )

    def _bootstrap_components(
        self,
        organization_id: str,
        asset: Asset,
        run_spec: RunSpec,
    ) -> dict[str, Component]:
        components = {}
        failed = {failure.bearing: failure.failure_mode for failure in run_spec.failures}
        for bearing in run_spec.bearing_cols:
            failure_note = f" ({failed[bearing]})" if bearing in failed else ""
            components[bearing] = self.repo.get_or_create_component(
                organization_id,
                ComponentCreate(
                    asset_id=asset.id,
                    slug=bearing.replace("_", "-"),
                    name=f"{bearing.replace('_', ' ').title()}{failure_note}",
                    component_type="bearing",
                    external_ref=f"{run_spec.run_id}:{bearing}",
                ),
            )
        return components

    def _bootstrap_sensors(
        self,
        organization_id: str,
        components_by_bearing: dict[str, Component],
        run_spec: RunSpec,
    ) -> None:
        for channel in run_spec.channel_map:
            component = components_by_bearing[channel.bearing]
            sensor_slug = f"{channel.bearing}-{channel.sensor_id}".replace("_", "-")
            self.repo.get_or_create_sensor(
                organization_id,
                SensorCreate(
                    component_id=component.id,
                    slug=sensor_slug,
                    name=f"{channel.bearing.replace('_', ' ').title()} {channel.sensor_id}",
                    sensor_type="accelerometer",
                    unit="g",
                    sampling_rate_hz=float(run_spec.sampling_rate_hz),
                    channel_name=f"channel_{channel.channel_index}",
                    external_ref=f"{run_spec.run_id}:{channel.bearing}:{channel.sensor_id}",
                ),
            )

    def platform_summary(self) -> dict[str, int]:
        return self.repo.health_counts()


def bootstrap_ims_registry(session: Session) -> PlatformBootstrapSummary:
    return PlatformService(session).bootstrap_ims_registry()


def get_platform_inventory(session: Session) -> dict:
    repo = PlatformRepository(session)
    org = repo.get_organization_by_slug(NASA_IMS_ORG_SLUG)
    if org is None:
        return {"organizations": [], "assets": [], "sensors": []}

    site = repo.get_site_by_slug(org.id, NASA_IMS_SITE_SLUG)
    return {
        "organizations": [{"id": org.id, "slug": org.slug, "name": org.name}],
        "sites": [] if site is None else [{"id": site.id, "slug": site.slug, "name": site.name}],
        "assets": [
            {
                "id": asset.id,
                "site_id": asset.site_id,
                "slug": asset.slug,
                "name": asset.name,
                "asset_type": asset.asset_type,
            }
            for asset in repo.list_assets(org.id)
        ],
        "sensors": [
            {
                "id": sensor.id,
                "component_id": sensor.component_id,
                "slug": sensor.slug,
                "name": sensor.name,
                "sensor_type": sensor.sensor_type,
                "unit": sensor.unit,
                "sampling_rate_hz": sensor.sampling_rate_hz,
                "channel_name": sensor.channel_name,
                "axis": sensor.axis,
                "manufacturer": sensor.manufacturer,
                "model": sensor.model,
            }
            for sensor in repo.list_sensors(org.id)
        ],
    }
