from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from alembic import command
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    MachineReadingCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.database import make_engine
from platform_core.models import Base, MachineReading, Sensor
from platform_core.repositories import PlatformRepository, TenantBoundaryError
from platform_core.services import NASA_IMS_ORG_SLUG, PlatformService

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "platform.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("PMS_DATABASE_URL", url)
    cfg = Config(str(ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr("platform_core.database.engine", engine)
    monkeypatch.setattr("platform_core.database.SessionLocal", session_factory)
    try:
        yield engine, session_factory
    finally:
        engine.dispose()


def test_alembic_migration_creates_platform_core_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables).issubset(tables)


def test_alembic_migration_downgrades_platform_core_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "platform.db"
    monkeypatch.setenv("PMS_DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    cfg = Config(str(ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = make_engine(f"sqlite:///{db_path.as_posix()}")
    try:
        remaining = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert not set(Base.metadata.tables).intersection(remaining)


def test_bootstrap_registers_ims_as_normal_platform_entities(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        summary = PlatformService(session).bootstrap_ims_registry()
        session.commit()

        org = PlatformRepository(session).get_organization_by_slug(NASA_IMS_ORG_SLUG)
        assert org is not None
        UUID(org.id)
        assert summary.asset_count == 3
        assert summary.component_count == 12
        assert summary.sensor_count == 16
        sensors = session.query(Sensor).filter_by(organization_id=org.id).all()
        assert {sensor.unit for sensor in sensors} == {"g"}
        assert {sensor.sampling_rate_hz for sensor in sensors} == {20000.0}


def test_ims_bootstrap_is_idempotent(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = PlatformService(session)
        first = service.bootstrap_ims_registry()
        second = service.bootstrap_ims_registry()
        session.commit()

    assert first == second


def test_repository_blocks_cross_tenant_relationships(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org_a = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
        org_b = repo.create_organization(OrganizationCreate(slug="globex", name="Globex"))
        site_b = repo.create_site(org_b.id, SiteCreate(slug="atlanta", name="Atlanta Plant"))

        with pytest.raises(TenantBoundaryError):
            repo.create_asset(
                org_a.id,
                AssetCreate(
                    site_id=site_b.id,
                    slug="pump-p-104",
                    name="Pump P-104",
                    asset_type="pump",
                ),
            )


def test_users_can_be_attached_to_organizations(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
        user = repo.create_user(UserCreate(email="TECHNICIAN@example.com", full_name="Plant Technician"))
        membership = repo.add_membership(org.id, user.id, "technician")
        session.commit()

        assert user.email == "technician@example.com"
        assert membership.organization_id == org.id
        assert membership.user_id == user.id
        assert membership.role == "technician"


def test_canonical_machine_reading_is_tenant_owned(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
        site = repo.create_site(org.id, SiteCreate(slug="atlanta", name="Atlanta Plant"))
        asset = repo.create_asset(
            org.id,
            AssetCreate(site_id=site.id, slug="pump-p-104", name="Pump P-104", asset_type="pump"),
        )
        component = repo.create_component(
            org.id,
            ComponentCreate(
                asset_id=asset.id,
                slug="drive-end-bearing",
                name="Drive-End Bearing",
                component_type="bearing",
            ),
        )
        sensor = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-017",
                name="Accelerometer VS-017",
                sensor_type="accelerometer",
                unit="g",
                sampling_rate_hz=20000.0,
                channel_name="vibration_x",
                axis="x",
            ),
        )
        reading = repo.create_machine_reading(
            org.id,
            MachineReadingCreate(
                sensor_id=sensor.id,
                observed_at=datetime(2026, 8, 26, 12, 0, tzinfo=UTC),
                metric="rms",
                value=0.18,
                unit="g",
                source="rest",
                payload={"schema": "canonical.machine_reading.v1"},
            ),
        )
        session.commit()

        saved = session.get(MachineReading, reading.id)
        assert saved is not None
        assert saved.organization_id == org.id
        assert saved.sensor_id == sensor.id
        assert saved.payload_json == '{"schema": "canonical.machine_reading.v1"}'


def test_platform_api_health_bootstrap_and_inventory(migrated_db, model_dir, feature_table):
    import main

    reloaded_main = importlib.reload(main)
    client = TestClient(reloaded_main.app)

    health = client.get("/api/platform/health").json()
    assert health["status"] == "ok"
    assert health["migrated"] is True
    assert "missing_tables" in health

    bootstrap = client.post("/api/platform/bootstrap/ims")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["asset_count"] == 3

    inventory = client.get("/api/platform/inventory").json()
    assert inventory["organizations"][0]["slug"] == NASA_IMS_ORG_SLUG
    assert len(inventory["assets"]) == 3
    assert len(inventory["sensors"]) == 16


def test_existing_ims_dashboard_api_still_serves_after_platform_core(migrated_db, model_dir, feature_table):
    import main

    reloaded_main = importlib.reload(main)
    client = TestClient(reloaded_main.app)

    assert client.get("/api/health").json()["status"] == "ok"
    supported = client.get("/api/snapshot/930/bearing/bearing_1").json()
    unsupported = client.get("/api/snapshot/930/bearing/bearing_2").json()
    assert supported["prediction_status"] == "supported"
    assert unsupported["prediction_status"] == "unsupported"
