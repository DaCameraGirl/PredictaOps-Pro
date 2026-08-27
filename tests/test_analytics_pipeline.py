from __future__ import annotations

import importlib
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from analytics_pipeline.service import AnalyticsService
from industrial_ingestion.service import IngestionService
from industrial_ingestion.waveform_store import LocalWaveformStore
from platform_core.contracts import AssetCreate, ComponentCreate, OrganizationCreate, SensorCreate, SiteCreate
from platform_core.database import make_engine
from platform_core.models import (
    AnalyticsFailure,
    AnalyticsFeatureRecord,
    AnalyticsHealthState,
    Base,
    WaveformRecord,
)
from platform_core.repositories import PlatformRepository

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    external_url = os.environ.get("PMS_PLATFORM_CORE_TEST_DATABASE_URL")
    if external_url:
        url = external_url
    else:
        db_path = tmp_path / "platform.db"
        url = f"sqlite:///{db_path.as_posix()}"
        monkeypatch.setenv("PMS_DATABASE_URL", url)

    cfg = Config(str(ROOT / "alembic.ini"))
    if external_url:
        clean_engine = make_engine(url)
        try:
            Base.metadata.drop_all(clean_engine)
            with clean_engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        finally:
            clean_engine.dispose()
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr("platform_core.database.engine", engine)
    monkeypatch.setattr("platform_core.database.SessionLocal", session_factory)
    try:
        yield engine, session_factory
    finally:
        if external_url:
            Base.metadata.drop_all(engine)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        engine.dispose()


@pytest.fixture
def tenant_sensor(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
        site = repo.create_site(org.id, SiteCreate(slug="atlanta", name="Atlanta Plant", timezone="America/New_York"))
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
        accel = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-017",
                name="Accelerometer VS-017",
                sensor_type="accelerometer",
                unit="g",
                sampling_rate_hz=1024.0,
                external_ref="acme:atlanta:pump-p-104:drive-end-bearing:vs-017",
            ),
        )
        temp = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="temp-1",
                name="Temperature 1",
                sensor_type="temperature",
                unit="c",
                external_ref="acme:temp-1",
            ),
        )
        other_org = repo.create_organization(OrganizationCreate(slug="other", name="Other Manufacturing"))
        session.commit()
        return {
            "organization_id": org.id,
            "other_organization_id": other_org.id,
            "sensor_id": accel.id,
            "sensor_external_ref": accel.external_ref,
            "temp_sensor_id": temp.id,
        }


def test_migration_creates_analytics_pipeline_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert {
        "analytics_runs",
        "analytics_feature_records",
        "analytics_health_states",
        "analytics_failures",
    }.issubset(tables)


def test_waveform_features_and_fft_are_persisted_with_provenance(migrated_db, tenant_sensor, tmp_path):
    _engine, session_factory = migrated_db
    sampling_rate_hz = 1024.0
    sample_count = 1024
    t = np.arange(sample_count) / sampling_rate_hz
    samples = np.sin(2 * np.pi * 128.0 * t).tolist()

    with session_factory() as session:
        ingest_receipt = IngestionService(session, LocalWaveformStore(tmp_path / "waveforms")).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": sampling_rate_hz,
                        "samples": samples,
                        "source_record_id": "wf-128hz",
                    }
                ]
            },
        )
        receipt = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        session.commit()

        features = {
            row.feature_name: row
            for row in session.scalars(select(AnalyticsFeatureRecord).order_by(AnalyticsFeatureRecord.feature_name))
        }
        waveform = session.scalar(select(WaveformRecord))
        assert receipt.status == "completed"
        assert receipt.feature_count == 10
        assert receipt.failure_count == 0
        assert features["waveform.rms"].value == pytest.approx(1 / math.sqrt(2), rel=1e-3)
        assert features["waveform.peak_to_peak"].value == pytest.approx(2.0, rel=1e-3)
        assert features["waveform.crest_factor"].value == pytest.approx(math.sqrt(2), rel=1e-3)
        assert features["fft.dominant_frequency_hz"].value == pytest.approx(128.0)
        assert features["waveform.rms"].provenance["content_sha256"] == waveform.sha256
        assert features["waveform.rms"].provenance["checksum_verified"] is True


def test_scalar_baseline_anomaly_and_health_state_are_evidence_based(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    records = []
    base_time = datetime(2026, 8, 27, 12, tzinfo=UTC)
    for index, value in enumerate([70.0, 70.1, 69.9, 70.0, 95.0]):
        records.append(
            {
                "kind": "scalar",
                "sensor_id": tenant_sensor["temp_sensor_id"],
                "observed_at": (base_time + timedelta(minutes=index)).isoformat(),
                "metric": "temperature",
                "value": value,
                "unit": "c",
                "source_record_id": f"temp-{index}",
            }
        )

    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={"records": records},
        )
        receipt = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        session.commit()

        latest = session.scalars(
            select(AnalyticsHealthState).order_by(AnalyticsHealthState.observed_at.desc())
        ).first()
        assert receipt.health_state_count == 5
        assert latest.health_state == "critical"
        assert latest.anomaly_score is not None and latest.anomaly_score >= 6.0
        assert latest.trend_slope is not None and latest.trend_slope > 0
        assert latest.evidence["feature_name"] == "scalar.temperature"
        assert latest.evidence["baseline_samples"] == 4


def test_recompute_is_deterministic_and_does_not_duplicate_features(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["temp_sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "metric": "temperature",
                        "value": 70.0,
                        "unit": "c",
                        "source_record_id": "temp-one",
                    }
                ]
            },
        )
        first = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        second = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        session.commit()

        feature_count = session.scalar(select(func.count()).select_from(AnalyticsFeatureRecord))
        assert first.feature_count == 1
        assert second.feature_count == 0
        assert second.duplicate_feature_count == 1
        assert feature_count == 1


def test_sensor_recompute_rebuilds_health_from_existing_features_without_duplicates(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    records = []
    base_time = datetime(2026, 8, 27, 12, tzinfo=UTC)
    for index, value in enumerate([70.0, 70.1, 69.9, 70.0, 95.0]):
        records.append(
            {
                "kind": "scalar",
                "sensor_id": tenant_sensor["temp_sensor_id"],
                "observed_at": (base_time + timedelta(minutes=index)).isoformat(),
                "metric": "temperature",
                "value": value,
                "unit": "c",
                "source_record_id": f"sensor-recompute-{index}",
            }
        )

    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={"records": records},
        )
        batch_receipt = AnalyticsService(session).compute_batch(
            tenant_sensor["organization_id"],
            ingest_receipt.batch_id,
        )
        recompute_receipt = AnalyticsService(session).recompute_sensor(
            tenant_sensor["organization_id"],
            tenant_sensor["temp_sensor_id"],
        )
        session.commit()

        feature_count = session.scalar(select(func.count()).select_from(AnalyticsFeatureRecord))
        health_count = session.scalar(select(func.count()).select_from(AnalyticsHealthState))
        latest = session.scalars(
            select(AnalyticsHealthState).order_by(AnalyticsHealthState.observed_at.desc())
        ).first()
        assert batch_receipt.feature_count == 5
        assert recompute_receipt.feature_count == 0
        assert recompute_receipt.duplicate_feature_count == 5
        assert recompute_receipt.health_state_count == 5
        assert feature_count == 5
        assert health_count == 5
        assert latest.run_id == recompute_receipt.run_id
        assert latest.health_state == "critical"
        assert latest.evidence["baseline_samples"] == 4


def test_analytics_rejects_cross_tenant_batch_access(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["temp_sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "metric": "temperature",
                        "value": 70.0,
                        "unit": "c",
                        "source_record_id": "tenant-check",
                    }
                ]
            },
        )
        with pytest.raises(ValueError, match="ingestion batch does not exist inside this organization"):
            AnalyticsService(session).compute_batch(tenant_sensor["other_organization_id"], ingest_receipt.batch_id)


def test_corrupted_waveform_records_analytics_failure_without_features(migrated_db, tenant_sensor, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        ingest_receipt = IngestionService(session, LocalWaveformStore(tmp_path / "waveforms")).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": 1024.0,
                        "samples": [0.0, 1.0, 0.0, -1.0],
                        "source_record_id": "wf-corrupt",
                    }
                ]
            },
        )
        waveform = session.scalar(select(WaveformRecord))
        Path(waveform.storage_uri).write_text(json.dumps({"samples": [0.0, 1.0]}), encoding="utf-8")
        receipt = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        session.commit()

        assert receipt.status == "failed"
        assert receipt.failure_count == 1
        assert receipt.failures[0].reason == "waveform content checksum does not match provenance"
        assert session.scalar(select(func.count()).select_from(AnalyticsFeatureRecord)) == 0
        assert session.scalar(select(func.count()).select_from(AnalyticsFailure)) == 1


def test_external_missing_waveform_content_becomes_failure(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": 1024.0,
                        "storage_uri": "s3://bucket/vibration.bin",
                        "sample_count": 4096,
                        "source_record_id": "wf-external",
                    }
                ]
            },
        )
        waveform = session.scalar(select(WaveformRecord))
        assert waveform.sha256 is None
        receipt = AnalyticsService(session).compute_batch(tenant_sensor["organization_id"], ingest_receipt.batch_id)
        session.commit()

        assert receipt.status == "failed"
        assert receipt.failures[0].reason == "waveform content is external and not locally available for analytics"
        assert session.scalar(select(func.count()).select_from(AnalyticsFailure)) == 1


def test_analytics_api_computes_batch_and_returns_latest_health(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        ingest_receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["temp_sensor_id"],
                        "observed_at": "2026-08-27T12:00:00Z",
                        "metric": "temperature",
                        "value": 70.0,
                        "unit": "c",
                        "source_record_id": "api-temp",
                    }
                ]
            },
        )
        session.commit()

    app_main = importlib.reload(importlib.import_module("app.main"))
    client = TestClient(app_main.app)
    compute = client.post(
        f"/api/analytics/{tenant_sensor['organization_id']}/batches/{ingest_receipt.batch_id}/compute"
    )
    health = client.get(f"/api/analytics/{tenant_sensor['organization_id']}/health")

    assert compute.status_code == 200
    assert compute.json()["feature_count"] == 1
    assert health.status_code == 200
    assert health.json()["states"][0]["health_state"] == "insufficient_evidence"
