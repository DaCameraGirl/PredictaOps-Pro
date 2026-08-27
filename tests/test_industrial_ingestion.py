from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from industrial_ingestion.connectors import AbbApiConnector, MqttIngestionConnector, OpcUaIngestionConnector, run_async
from industrial_ingestion.service import IngestionService
from industrial_ingestion.waveform_store import LocalWaveformStore
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
)
from platform_core.database import make_engine
from platform_core.models import (
    Base,
    IngestionBatch,
    IngestionFailure,
    MachineReading,
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
        sensor = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-017",
                name="Accelerometer VS-017",
                sensor_type="accelerometer",
                unit="g",
                sampling_rate_hz=20000.0,
                external_ref="acme:atlanta:pump-p-104:drive-end-bearing:vs-017",
            ),
        )
        temp_sensor = repo.create_sensor(
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
        session.commit()
        return {
            "organization_id": org.id,
            "site_slug": site.slug,
            "asset_slug": asset.slug,
            "component_slug": component.slug,
            "sensor_id": sensor.id,
            "sensor_external_ref": sensor.external_ref,
            "temp_sensor_id": temp_sensor.id,
        }


def test_migration_creates_industrial_ingestion_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert {
        "ingestion_sources",
        "ingestion_batches",
        "ingested_records",
        "ingestion_failures",
        "waveform_records",
    }.issubset(tables)


def test_rest_scalar_ingestion_normalizes_units_and_timestamps(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["temp_sensor_id"],
                        "observed_at": "2026-08-26T08:00:00",
                        "source_timezone": "America/New_York",
                        "metric": "temperature",
                        "value": 212.0,
                        "unit": "f",
                        "source_record_id": "rest-1",
                    }
                ]
            },
        )
        session.commit()

        reading = session.scalar(select(MachineReading))
        assert receipt.accepted_count == 1
        assert receipt.failed_count == 0
        assert reading is not None
        assert reading.unit == "c"
        assert reading.value == pytest.approx(100.0)
        assert reading.observed_at.isoformat().startswith("2026-08-26T12:00:00")


def test_ingestion_idempotency_skips_duplicate_source_records(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    payload = {
        "records": [
            {
                "kind": "scalar",
                "sensor_id": tenant_sensor["sensor_id"],
                "observed_at": "2026-08-26T12:00:00Z",
                "metric": "rms",
                "value": 9.80665,
                "unit": "m/s^2",
                "source_record_id": "dupe-1",
            }
        ]
    }
    with session_factory() as session:
        service = IngestionService(session)
        first = service.ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload=payload,
        )
        second = service.ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload=payload,
        )
        session.commit()

        reading_count = session.scalar(select(func.count()).select_from(MachineReading))
        assert first.accepted_count == 1
        assert second.duplicate_count == 1
        assert reading_count == 1


def test_csv_ingestion_resolves_sensor_path_and_quality(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    csv_payload = "\n".join(
        [
            "kind,site_slug,asset_slug,component_slug,sensor_slug,observed_at,metric,value,unit,quality,source_record_id",
            "scalar,atlanta,pump-p-104,drive-end-bearing,vs-017,2026-08-26T12:00:00Z,rms,0.22,g,suspect,csv-1",
        ]
    )
    with session_factory() as session:
        receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="csv",
            source_name="CSV Upload",
            payload=csv_payload,
            source_uri="upload://batch.csv",
        )
        session.commit()

        reading = session.scalar(select(MachineReading))
        assert receipt.accepted_count == 1
        assert reading.quality == "suspect"
        assert reading.payload["source_record_id"] == "csv-1"


def test_parquet_ingestion_uses_same_canonical_contract(migrated_db, tenant_sensor):
    pytest.importorskip("pyarrow")
    _engine, session_factory = migrated_db
    frame = pd.DataFrame(
        [
            {
                "kind": "scalar",
                "sensor_external_ref": tenant_sensor["sensor_external_ref"],
                "observed_at": "2026-08-26T12:00:00Z",
                "metric": "rms",
                "value": 0.31,
                "unit": "g",
                "quality": "good",
                "source_record_id": "parquet-1",
            }
        ]
    )
    payload = io.BytesIO()
    frame.to_parquet(payload)

    with session_factory() as session:
        receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="parquet",
            source_name="Parquet Upload",
            payload=payload.getvalue(),
        )
        session.commit()

        assert receipt.accepted_count == 1
        assert session.scalar(select(MachineReading)).metric == "rms"


def test_mqtt_opcua_and_abb_adapters_emit_canonical_scalars(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = IngestionService(session)
        mqtt = service.ingest(
            tenant_sensor["organization_id"],
            source_type="mqtt",
            source_name="MQTT Bridge",
            topic="plant/acme/vs-017",
            payload=(
                '{"records":[{"kind":"scalar","sensor_external_ref":"'
                + tenant_sensor["sensor_external_ref"]
                + '","observed_at":"2026-08-26T12:00:00Z","metric":"rms","value":0.2,'
                '"unit":"g","source_record_id":"mqtt-1"}]}'
            ),
        )
        opcua = service.ingest(
            tenant_sensor["organization_id"],
            source_type="opcua",
            source_name="OPC-UA Bridge",
            payload={
                "nodes": [
                    {
                        "node_id": tenant_sensor["sensor_external_ref"],
                        "server_timestamp": "2026-08-26T12:01:00Z",
                        "browse_name": "rms",
                        "value": 0.21,
                        "unit": "g",
                        "sequence": "opcua-1",
                    }
                ]
            },
        )
        abb = service.ingest(
            tenant_sensor["organization_id"],
            source_type="abb",
            source_name="ABB Adapter",
            payload={
                "measurements": [
                    {
                        "sensorExternalRef": tenant_sensor["sensor_external_ref"],
                        "observedAt": "2026-08-26T12:02:00Z",
                        "metric": "rms",
                        "value": 0.22,
                        "unit": "g",
                        "id": "abb-1",
                    }
                ]
            },
        )
        session.commit()

        assert mqtt.accepted_count == opcua.accepted_count == abb.accepted_count == 1
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 3


def test_native_mqtt_connector_subscribes_and_ingests_messages(migrated_db, tenant_sensor):
    class FakeMqttClient:
        def __init__(self):
            self.subscriptions = []
            self.connected = None
            self.loop_started = False
            self.on_message = None

        def username_pw_set(self, username, password):
            self.credentials = (username, password)

        def connect(self, host, port, keepalive):
            self.connected = (host, port, keepalive)

        def subscribe(self, topic):
            self.subscriptions.append(topic)

        def loop_start(self):
            self.loop_started = True

        def loop_stop(self):
            self.loop_started = False

        def disconnect(self):
            self.disconnected = True

    fake_client = FakeMqttClient()
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = IngestionService(session)
        handle = MqttIngestionConnector(service, client_factory=lambda: fake_client).start_subscription(
            tenant_sensor["organization_id"],
            broker_host="mqtt.example.test",
            topics=["plant/acme/vs-017"],
            username="service",
            password="secret",
        )
        message = SimpleNamespace(
            topic="plant/acme/vs-017",
            payload=(
                '{"records":[{"kind":"scalar","sensor_external_ref":"'
                + tenant_sensor["sensor_external_ref"]
                + '","observed_at":"2026-08-26T12:00:00Z","metric":"rms","value":0.2,'
                '"unit":"g","source_record_id":"native-mqtt-1"}]}'
            ).encode(),
        )
        fake_client.on_message(fake_client, None, message)
        handle.stop()
        session.commit()

        assert fake_client.connected == ("mqtt.example.test", 1883, 60)
        assert fake_client.credentials == ("service", "secret")
        assert fake_client.subscriptions == ["plant/acme/vs-017"]
        assert fake_client.disconnected is True
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 1


def test_native_opcua_connector_connects_subscribes_and_polls(migrated_db, tenant_sensor):
    class FakeOpcNode:
        def __init__(self, node_id):
            self.node_id = node_id

        async def read_value(self):
            return 0.33

        def __str__(self):
            return self.node_id

    class FakeSubscription:
        def __init__(self):
            self.nodes = []

        async def subscribe_data_change(self, nodes):
            self.nodes = nodes

    class FakeOpcClient:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.connected = False
            self.disconnected = False
            self.subscription = FakeSubscription()

        async def connect(self):
            self.connected = True

        async def disconnect(self):
            self.disconnected = True

        def get_node(self, node_id):
            return FakeOpcNode(node_id)

        async def create_subscription(self, publishing_interval_ms, handler):
            self.publishing_interval_ms = publishing_interval_ms
            self.handler = handler
            return self.subscription

    clients = []

    def client_factory(endpoint):
        client = FakeOpcClient(endpoint)
        clients.append(client)
        return client

    _engine, session_factory = migrated_db
    with session_factory() as session:
        connector = OpcUaIngestionConnector(IngestionService(session), client_factory=client_factory)
        subscription = run_async(
            connector.subscribe_data_changes(
                tenant_sensor["organization_id"],
                endpoint="opc.tcp://localhost:4840",
                node_ids=[tenant_sensor["sensor_external_ref"]],
            )
        )
        receipts = run_async(
            connector.poll_once(
                tenant_sensor["organization_id"],
                endpoint="opc.tcp://localhost:4840",
                node_ids=[tenant_sensor["sensor_external_ref"]],
            )
        )
        session.commit()

        assert subscription.nodes[0].node_id == tenant_sensor["sensor_external_ref"]
        assert clients[0].connected is True
        assert clients[0].publishing_interval_ms == 1000
        assert clients[1].disconnected is True
        assert receipts[0].accepted_count == 1
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 1


def test_abb_api_connector_fetches_with_credentials_and_ingests(migrated_db, tenant_sensor):
    class FakeResponse:
        def raise_for_status(self):
            self.raised = True

        def json(self):
            return {
                "measurements": [
                    {
                        "sensorExternalRef": tenant_sensor["sensor_external_ref"],
                        "observedAt": "2026-08-26T12:00:00Z",
                        "metric": "rms",
                        "value": 0.28,
                        "unit": "g",
                        "id": "abb-native-1",
                    }
                ]
            }

    class FakeHttpClient:
        def get(self, path, params=None, headers=None):
            self.request = {"path": path, "params": params, "headers": headers}
            return FakeResponse()

    fake_client = FakeHttpClient()

    def client_factory(**_kwargs):
        return fake_client

    _engine, session_factory = migrated_db
    with session_factory() as session:
        receipt = AbbApiConnector(IngestionService(session), client_factory=client_factory).fetch_measurements(
            tenant_sensor["organization_id"],
            base_url="https://abb.example.test/api",
            token="token-123",
            asset_id="pump-p-104",
        )
        session.commit()

        assert receipt.accepted_count == 1
        assert fake_client.request["path"] == "/measurements"
        assert fake_client.request["params"] == {"asset_id": "pump-p-104"}
        assert fake_client.request["headers"] == {"Authorization": "Bearer token-123"}
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 1


def test_waveform_ingestion_uses_first_class_records_not_machine_reading_payload(migrated_db, tenant_sensor, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = IngestionService(session, waveform_store=LocalWaveformStore(tmp_path / "waveforms"))
        receipt = service.ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="Waveform REST",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-26T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": 20000.0,
                        "samples": [0.1, 0.2, 0.1, -0.1],
                        "source_record_id": "wave-1",
                    }
                ]
            },
        )
        session.commit()

        waveform = session.scalar(select(WaveformRecord))
        assert receipt.waveform_count == 1
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 0
        assert waveform.sample_count == 4
        assert Path(waveform.storage_uri).exists()
        assert ".." not in Path(waveform.storage_uri).name


def test_waveform_store_hashes_external_record_ids_for_object_names(migrated_db, tenant_sensor, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = IngestionService(session, waveform_store=LocalWaveformStore(tmp_path / "waveforms"))
        service.ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="Waveform REST",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-26T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": 20000.0,
                        "samples": [0.1, 0.2],
                        "source_record_id": "../escape",
                    }
                ]
            },
        )
        session.commit()

        waveform = session.scalar(select(WaveformRecord))
        path = Path(waveform.storage_uri)
        assert path.exists()
        assert path.parent.parent == tmp_path / "waveforms" / tenant_sensor["organization_id"]
        assert "escape" not in path.name
        assert "/" not in path.name
        assert "\\" not in path.name


def test_external_waveform_without_content_checksum_remains_unverified(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        receipt = IngestionService(session).ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="Waveform REST",
            payload={
                "records": [
                    {
                        "kind": "waveform",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-26T12:00:00Z",
                        "unit": "g",
                        "sampling_rate_hz": 20000.0,
                        "storage_uri": "s3://bucket/vibration.bin",
                        "sample_count": 20480,
                        "source_record_id": "external-wave-1",
                    }
                ]
            },
        )
        session.commit()

        waveform = session.scalar(select(WaveformRecord))
        assert receipt.waveform_count == 1
        assert waveform.storage_uri == "s3://bucket/vibration.bin"
        assert waveform.sha256 is None


def test_replay_reingests_previous_batch_through_replay_adapter(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        service = IngestionService(session)
        first = service.ingest(
            tenant_sensor["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-26T12:00:00Z",
                        "metric": "rms",
                        "value": 0.2,
                        "unit": "g",
                        "source_record_id": "replay-source-1",
                    }
                ]
            },
        )
        replay = service.replay_batch(tenant_sensor["organization_id"], first.batch_id)
        session.commit()

        assert replay.accepted_count == 1
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 2
        assert session.scalar(select(func.count()).select_from(IngestionBatch)) == 2


def test_cross_tenant_sensor_injection_is_dead_lettered(migrated_db, tenant_sensor):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        other_org = repo.create_organization(OrganizationCreate(slug="globex", name="Globex"))
        session.commit()

        receipt = IngestionService(session).ingest(
            other_org.id,
            source_type="rest",
            source_name="REST Push",
            payload={
                "records": [
                    {
                        "kind": "scalar",
                        "sensor_id": tenant_sensor["sensor_id"],
                        "observed_at": "2026-08-26T12:00:00Z",
                        "metric": "rms",
                        "value": 0.2,
                        "unit": "g",
                        "source_record_id": "attack-1",
                    }
                ]
            },
        )
        session.commit()

        failure = session.scalar(select(IngestionFailure))
        assert receipt.status == "failed"
        assert receipt.failed_count == 1
        assert failure.dead_letter is True
        assert session.scalar(select(func.count()).select_from(MachineReading)) == 0


def test_ingestion_api_rest_replay_and_health(migrated_db, tenant_sensor, model_dir, feature_table):
    import main

    reloaded_main = importlib.reload(main)
    client = TestClient(reloaded_main.app)

    payload = {
        "records": [
            {
                "kind": "scalar",
                "sensor_id": tenant_sensor["sensor_id"],
                "observed_at": "2026-08-26T12:00:00Z",
                "metric": "rms",
                "value": 0.2,
                "unit": "g",
                "source_record_id": "api-1",
            }
        ]
    }
    ingested = client.post(
        f"/api/ingestion/{tenant_sensor['organization_id']}/rest?source_name=REST%20API",
        json=payload,
    )
    assert ingested.status_code == 200
    assert ingested.json()["accepted_count"] == 1

    health = client.get(f"/api/ingestion/{tenant_sensor['organization_id']}/health")
    assert health.status_code == 200
    assert health.json()["batches"] == 1

    replay = client.post(f"/api/ingestion/{tenant_sensor['organization_id']}/replay/{ingested.json()['batch_id']}")
    assert replay.status_code == 200
    assert replay.json()["accepted_count"] == 1

    malformed = client.post(
        f"/api/ingestion/{tenant_sensor['organization_id']}/rest",
        content="{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 400
