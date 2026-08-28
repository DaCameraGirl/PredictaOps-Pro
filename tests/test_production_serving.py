from __future__ import annotations

import hashlib
import importlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from alembic import command
from ml_platform.artifact_store import ModelArtifactStore
from ml_platform.contracts import (
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    PromoteModelVersion,
    RegistryCreate,
)
from ml_platform.service import MLPlatformService
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.database import make_engine
from platform_core.models import (
    Base,
    ModelServingBinding,
    ModelServingMonitor,
    PredictionRecord,
    ProductionModelResolution,
    RetrainingTrigger,
)
from platform_core.repositories import PlatformRepository
from production_serving.contracts import PredictionRequest, ServingBindingCreate
from production_serving.service import ProductionServingService

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
def serving_fixture(migrated_db):
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
            ComponentCreate(asset_id=asset.id, slug="bearing", name="Drive-End Bearing", component_type="bearing"),
        )
        sensor_a = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-017",
                name="VS-017",
                sensor_type="accelerometer",
                unit="g",
            ),
        )
        sensor_b = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-018",
                name="VS-018",
                sensor_type="accelerometer",
                unit="g",
            ),
        )
        sensor_sparse = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-019",
                name="VS-019",
                sensor_type="accelerometer",
                unit="g",
            ),
        )
        other_org = repo.create_organization(OrganizationCreate(slug="other", name="Other Manufacturing"))
        approver = repo.create_user(
            UserCreate(email="approver@example.com", full_name="Approver", external_subject="oidc:approver")
        )
        repo.add_membership(org.id, approver.id, "engineer")
        session.commit()
        return {
            "organization_id": org.id,
            "other_organization_id": other_org.id,
            "site_id": site.id,
            "asset_id": asset.id,
            "component_id": component.id,
            "sensor_a_id": sensor_a.id,
            "sensor_b_id": sensor_b.id,
            "sensor_sparse_id": sensor_sparse.id,
            "approver_id": approver.id,
        }


def _seed_feature_rows(
    session,
    fixture,
    *,
    feature_names: list[str] | None = None,
    base_time: datetime | None = None,
    feature_units: dict[str, str | None] | None = None,
) -> None:
    feature_names = feature_names or ["scalar.rms"]
    feature_units = feature_units or {}
    repo = PlatformRepository(session)
    run = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_a_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "production-serving"},
    )
    run_b = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_b_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "production-serving"},
    )
    base_time = base_time or datetime.now(UTC)
    for sensor_id, run_id, group, offset in [
        (fixture["sensor_a_id"], run.id, "bearing-a", 0.0),
        (fixture["sensor_b_id"], run_b.id, "bearing-b", 10.0),
    ]:
        for index in range(4):
            for feature_name in feature_names:
                value = float(index + offset)
                if feature_name == "scalar.std":
                    value += 0.5
                repo.create_analytics_feature(
                    fixture["organization_id"],
                    run_id=run_id,
                    sensor_id=sensor_id,
                    batch_id=None,
                    source_kind="scalar",
                    source_record_id=f"{group}-{index}",
                    observed_at=base_time + timedelta(minutes=index),
                    feature_name=feature_name,
                    value=value,
                    unit=feature_units.get(feature_name, "g"),
                    quality="good",
                    algorithm_version="analytics-v1",
                    provenance={
                        "target_rul_hours": float(8 - index - offset / 10),
                        "validation_group": group,
                    },
                )


def _production_model(
    session,
    fixture,
    tmp_path,
    *,
    feature_names: list[str] | None = None,
    base_time: datetime | None = None,
    feature_units: dict[str, str | None] | None = None,
    abstention_policy: dict | None = None,
    registry_name: str = "bearing-rul",
    registry_task: str = "rul_regression",
    target_name: str = "RUL_hours",
    target_unit: str | None = "h",
    dataset_version: str = "v1",
    model_version_label: str = "1.0.0",
):
    feature_names = feature_names or ["scalar.rms"]
    _seed_feature_rows(session, fixture, feature_names=feature_names, base_time=base_time, feature_units=feature_units)
    ml_service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
    dataset = ml_service.create_dataset_version(
        fixture["organization_id"],
        DatasetVersionCreate(
            name=f"{registry_name}-features",
            version=dataset_version,
            feature_names=feature_names,
            target_name=target_name,
            target_unit=target_unit,
        ),
    )
    experiment = ml_service.run_experiment(
        fixture["organization_id"],
        ExperimentCreate(
            dataset_version_id=dataset.id,
            name="serving experiment",
            training_config={"n_estimators": 5, "random_state": 17},
            abstention_policy=abstention_policy or {},
        ),
    )
    registry = ml_service.create_registry(
        fixture["organization_id"],
        RegistryCreate(name=registry_name, task=registry_task),
    )
    model_version = ml_service.register_model_version(
        fixture["organization_id"],
        ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version=model_version_label),
    )
    ml_service.promote_model_version(
        fixture["organization_id"],
        model_version.id,
        PromoteModelVersion(target_stage="validated"),
    )
    ml_service.promote_model_version(
        fixture["organization_id"],
        model_version.id,
        PromoteModelVersion(
            target_stage="production",
            approved_by_user_id=fixture["approver_id"],
            reason="approved for live serving",
        ),
    )
    return registry, model_version, dataset


def _bind_sensor(session, fixture, registry, model_version):
    return ProductionServingService(session).bind_model(
        fixture["organization_id"],
        ServingBindingCreate(
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="sensor",
            scope_id=fixture["sensor_a_id"],
            approved_by_user_id=fixture["approver_id"],
            reason="serve drive-end bearing",
        ),
    )


def _bind_organization(session, fixture, registry, model_version):
    return ProductionServingService(session).bind_model(
        fixture["organization_id"],
        ServingBindingCreate(
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="organization",
            approved_by_user_id=fixture["approver_id"],
            reason="serve organization",
        ),
    )


def test_migration_creates_production_serving_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert {
        "model_serving_bindings",
        "production_model_resolutions",
        "prediction_records",
        "model_serving_monitors",
        "retraining_triggers",
    }.issubset(tables)
    indexes = {index["name"] for index in inspect(engine).get_indexes("model_serving_bindings")}
    assert {
        "uq_active_model_serving_binding_scope_id",
        "uq_active_model_serving_binding_org_scope",
    }.issubset(indexes)


def test_supported_prediction_verifies_artifact_schema_and_persists_full_provenance(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, dataset = _production_model(session, serving_fixture, tmp_path)
        binding = _bind_sensor(session, serving_fixture, registry, model_version)
        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert binding.status == "active"
        assert prediction.prediction_status == "supported"
        assert prediction.predicted_rul_hours is not None
        assert prediction.model_version_id == model_version.id
        assert prediction.dataset_version_id == dataset.id
        assert prediction.model_resolution["artifact_sha256"] == model_version.artifact_sha256
        assert prediction.model_resolution["feature_schema"] == ["scalar.rms"]
        assert prediction.model_resolution["reason_code"] == "SUPPORTED"
        assert prediction.abstention_code is None
        assert prediction.request_kind == "live"
        assert prediction.provenance["artifact_sha256"] == model_version.artifact_sha256
        assert prediction.provenance["feature_record_ids"]
        assert prediction.uncertainty["prediction_interval_80"]["basis"].startswith("cross-group")
        assert model_version.provenance["model_domain"]["ordered_feature_names"] == ["scalar.rms"]
        assert session.scalar(select(func.count()).select_from(PredictionRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ProductionModelResolution)) == 1


def test_dimensionless_waveform_features_preserve_none_units_for_serving(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        feature_names = ["waveform.kurtosis", "waveform.crest_factor"]
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            feature_names=feature_names,
            feature_units={name: None for name in feature_names},
        )
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert model_version.provenance["model_domain"]["feature_units"] == {
            "waveform.kurtosis": None,
            "waveform.crest_factor": None,
        }
        assert prediction.prediction_status == "supported"
        assert prediction.abstention_code is None


def test_sensor_binding_wins_over_organization_binding(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_organization(session, serving_fixture, registry, model_version)
        sensor_binding = _bind_sensor(session, serving_fixture, registry, model_version)
        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "supported"
        assert prediction.provenance["binding_id"] == sensor_binding.id


def test_prediction_abstains_when_no_binding_or_features_are_available(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        service = ProductionServingService(session, ModelArtifactStore(tmp_path / "models"))
        no_binding = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        service.bind_model(
            serving_fixture["organization_id"],
            ServingBindingCreate(
                registry_id=registry.id,
                model_version_id=model_version.id,
                scope_type="sensor",
                scope_id=serving_fixture["sensor_sparse_id"],
                approved_by_user_id=serving_fixture["approver_id"],
            ),
        )
        no_features = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_sparse_id"], registry_id=registry.id),
        )
        session.commit()

        assert no_binding.prediction_status == "unsupported"
        assert no_binding.abstention_code == "NO_MODEL_BINDING"
        assert no_binding.abstention_reason == "no active production model binding resolves for this sensor"
        assert no_features.prediction_status == "insufficient_evidence"
        assert no_features.abstention_code == "MISSING_FEATURES"
        assert no_features.abstention_reason == "no analytics features exist for this sensor"


def test_prediction_rejects_cross_tenant_sensor_access(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _production_model(session, serving_fixture, tmp_path)
        with pytest.raises(ValueError, match="sensor does not exist inside this organization"):
            ProductionServingService(session).predict_rul(
                serving_fixture["other_organization_id"],
                PredictionRequest(sensor_id=serving_fixture["sensor_a_id"]),
            )


def test_artifact_checksum_mismatch_abstains_before_deserialization(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        Path(model_version.artifact_uri).write_text("tampered artifact", encoding="utf-8")

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.predicted_rul_hours is None
        assert prediction.abstention_code == "ARTIFACT_CHECKSUM_FAILED"
        assert "SHA-256" in prediction.abstention_reason


def test_missing_artifact_abstains_without_prediction(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        Path(model_version.artifact_uri).unlink()

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.predicted_rul_hours is None
        assert prediction.abstention_code == "ARTIFACT_MISSING"
        assert "model artifact does not exist" in prediction.abstention_reason


def test_verified_artifact_deserializes_the_same_bytes_that_were_hashed(tmp_path, monkeypatch):
    store = ModelArtifactStore(tmp_path / "models")
    org_id = "org-1"
    directory = tmp_path / "models" / org_id
    directory.mkdir(parents=True)
    path = directory / "model.joblib"
    original = b"verified model bytes"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    def fake_load(handle):
        path.write_bytes(b"unverified replacement")
        return handle.read()

    monkeypatch.setattr("ml_platform.artifact_store.joblib.load", fake_load)

    assert store.load_verified_model(
        organization_id=org_id,
        artifact_uri=path.as_posix(),
        expected_sha256=digest,
    ) == original


def test_checksum_valid_but_undeserializable_artifact_abstains(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        artifact = Path(model_version.artifact_uri)
        artifact.write_bytes(b"not a joblib payload")
        model_version.artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "ARTIFACT_LOAD_FAILED"
        assert "could not be deserialized" in prediction.abstention_reason


def test_artifact_outside_registry_root_abstains_even_with_valid_checksum(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        outside = tmp_path / "outside-root.joblib"
        outside.write_bytes(Path(model_version.artifact_uri).read_bytes())
        model_version.artifact_uri = outside.as_posix()

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "ARTIFACT_OUTSIDE_TRUST_ROOT"


def test_artifact_symlink_escape_from_registry_root_abstains(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        outside = tmp_path / "external-artifact.joblib"
        outside.write_bytes(Path(model_version.artifact_uri).read_bytes())
        link = tmp_path / "models" / serving_fixture["organization_id"] / "escape.joblib"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"symlink creation is unavailable on this platform: {exc}")
        model_version.artifact_uri = link.as_posix()

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "ARTIFACT_OUTSIDE_TRUST_ROOT"


def test_missing_stale_or_non_good_features_abstain_as_insufficient_evidence(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        base_time = datetime(2026, 8, 27, 13, tzinfo=UTC)
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            feature_names=["scalar.rms", "scalar.std"],
            base_time=base_time,
        )
        _bind_sensor(session, serving_fixture, registry, model_version)
        ProductionServingService(session).bind_model(
            serving_fixture["organization_id"],
            ServingBindingCreate(
                registry_id=registry.id,
                model_version_id=model_version.id,
                scope_type="sensor",
                scope_id=serving_fixture["sensor_sparse_id"],
                approved_by_user_id=serving_fixture["approver_id"],
            ),
        )
        repo = PlatformRepository(session)
        run = repo.create_analytics_run(
            serving_fixture["organization_id"],
            run_kind="sensor",
            sensor_id=serving_fixture["sensor_sparse_id"],
            algorithm_version="analytics-v1",
        )
        repo.create_analytics_feature(
            serving_fixture["organization_id"],
            run_id=run.id,
            sensor_id=serving_fixture["sensor_sparse_id"],
            batch_id=None,
            source_kind="scalar",
            source_record_id="sparse-live",
            observed_at=datetime(2026, 8, 27, 13, tzinfo=UTC),
            feature_name="scalar.rms",
            value=1.0,
            unit="g",
            quality="good",
            algorithm_version="analytics-v1",
            provenance={},
        )
        missing = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_sparse_id"], registry_id=registry.id),
        )
        stale = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(
                sensor_id=serving_fixture["sensor_a_id"],
                registry_id=registry.id,
                observed_at=datetime(2026, 8, 28, 12, tzinfo=UTC),
                max_feature_age_minutes=1,
            ),
        )
        quality_run = repo.create_analytics_run(
            serving_fixture["organization_id"],
            run_kind="sensor",
            sensor_id=serving_fixture["sensor_a_id"],
            algorithm_version="analytics-v1",
        )
        quality_observed_at = datetime(2026, 8, 28, 13, tzinfo=UTC)
        for feature_name, quality in [("scalar.rms", "suspect"), ("scalar.std", "good")]:
            repo.create_analytics_feature(
                serving_fixture["organization_id"],
                run_id=quality_run.id,
                sensor_id=serving_fixture["sensor_a_id"],
                batch_id=None,
                source_kind="scalar",
                source_record_id="suspect-live",
                observed_at=quality_observed_at,
                feature_name=feature_name,
                value=1.0,
                unit="g",
                quality=quality,
                algorithm_version="analytics-v1",
                provenance={},
            )
        suspect = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert missing.prediction_status == "insufficient_evidence"
        assert missing.model_resolution["evidence"]["missing_features"] == ["scalar.std"]
        assert stale.prediction_status == "insufficient_evidence"
        assert stale.abstention_reason == "live analytics features are stale"
        assert suspect.prediction_status == "insufficient_evidence"
        assert suspect.model_resolution["evidence"]["non_good_features"] == ["scalar.rms"]


def test_live_prediction_uses_request_time_for_freshness_when_observed_at_is_omitted(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        base_time = datetime.now(UTC) - timedelta(days=21)
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path, base_time=base_time)
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "insufficient_evidence"
        assert prediction.abstention_code == "STALE_FEATURES"
        assert prediction.request_kind == "live"


def test_prediction_uses_latest_complete_feature_snapshot(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        base_time = datetime.now(UTC) - timedelta(minutes=30)
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            feature_names=["scalar.rms", "scalar.std"],
            base_time=base_time,
        )
        _bind_sensor(session, serving_fixture, registry, model_version)
        repo = PlatformRepository(session)
        partial_run = repo.create_analytics_run(
            serving_fixture["organization_id"],
            run_kind="sensor",
            sensor_id=serving_fixture["sensor_a_id"],
            algorithm_version="analytics-v1",
        )
        repo.create_analytics_feature(
            serving_fixture["organization_id"],
            run_id=partial_run.id,
            sensor_id=serving_fixture["sensor_a_id"],
            batch_id=None,
            source_kind="scalar",
            source_record_id="newer-partial",
            observed_at=base_time + timedelta(minutes=20),
            feature_name="scalar.rms",
            value=999.0,
            unit="g",
            quality="good",
            algorithm_version="analytics-v1",
            provenance={},
        )

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(
                sensor_id=serving_fixture["sensor_a_id"],
                registry_id=registry.id,
                max_feature_age_minutes=60,
            ),
        )
        session.commit()

        assert prediction.prediction_status == "supported"
        assert prediction.feature_vector == {"scalar.rms": 3.0, "scalar.std": 3.5}


def test_request_max_feature_age_can_tighten_but_not_weaken_policy(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        base_time = datetime.now(UTC) - timedelta(hours=2)
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            base_time=base_time,
            abstention_policy={"max_feature_age_minutes": 60},
        )
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(
                sensor_id=serving_fixture["sensor_a_id"],
                registry_id=registry.id,
                max_feature_age_minutes=525600,
            ),
        )
        session.commit()

        assert prediction.prediction_status == "insufficient_evidence"
        assert prediction.abstention_code == "STALE_FEATURES"
        assert prediction.model_resolution["evidence"]["max_feature_age_minutes"] == 60
        assert prediction.model_resolution["evidence"]["request_max_feature_age_minutes"] == 525600
        assert prediction.model_resolution["evidence"]["policy_max_feature_age_minutes"] == 60


def test_prediction_rejects_non_rul_dataset_target(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            target_name="temperature",
            target_unit="C",
        )
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "FEATURE_SCHEMA_MISMATCH"
        assert "RUL_hours" in prediction.abstention_reason


def test_prediction_rejects_unmet_min_validation_groups_policy(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            abstention_policy={"min_validation_groups": 3},
        )
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "UNMET_ABSTENTION_POLICY"
        assert prediction.model_resolution["evidence"]["validation_group_count"] == 2


def test_historical_prediction_uses_explicit_observed_at_as_as_of_time(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        base_time = datetime.now(UTC) - timedelta(days=21)
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path, base_time=base_time)
        _bind_sensor(session, serving_fixture, registry, model_version)

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(
                sensor_id=serving_fixture["sensor_a_id"],
                registry_id=registry.id,
                observed_at=base_time + timedelta(minutes=3),
            ),
        )
        session.commit()

        assert prediction.prediction_status == "supported"
        assert prediction.request_kind == "historical"


def test_non_finite_live_features_abstain_before_inference(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        repo = PlatformRepository(session)
        run = repo.create_analytics_run(
            serving_fixture["organization_id"],
            run_kind="sensor",
            sensor_id=serving_fixture["sensor_a_id"],
            algorithm_version="analytics-v1",
        )
        repo.create_analytics_feature(
            serving_fixture["organization_id"],
            run_id=run.id,
            sensor_id=serving_fixture["sensor_a_id"],
            batch_id=None,
            source_kind="scalar",
            source_record_id="live-infinite",
            observed_at=datetime.now(UTC),
            feature_name="scalar.rms",
            value=float("inf"),
            unit="g",
            quality="good",
            algorithm_version="analytics-v1",
            provenance={},
        )

        prediction = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "insufficient_evidence"
        assert prediction.abstention_code == "BAD_FEATURE_QUALITY"
        assert prediction.model_resolution["evidence"]["non_finite_features"] == ["scalar.rms"]


def test_predict_rul_rejects_non_rul_registry_bindings(migrated_db, serving_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(
            session,
            serving_fixture,
            tmp_path,
            registry_name="bearing-classifier",
            registry_task="classification",
        )
        service = ProductionServingService(session, ModelArtifactStore(tmp_path / "models"))
        with pytest.raises(ValueError, match="rul_regression"):
            _bind_sensor(session, serving_fixture, registry, model_version)
        repo = PlatformRepository(session)
        repo.create_model_serving_binding(
            serving_fixture["organization_id"],
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="sensor",
            scope_id=serving_fixture["sensor_a_id"],
            approved_by_user_id=serving_fixture["approver_id"],
            reason="simulate legacy non-RUL binding",
            provenance={},
        )

        prediction = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"]),
        )
        explicit = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert prediction.abstention_code == "NON_RUL_MODEL_BINDING"
        assert explicit.prediction_status == "unsupported"
        assert explicit.abstention_code == "NON_RUL_MODEL_BINDING"


def test_ambiguous_rul_bindings_fail_closed_without_registry_selector(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        ml_service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        second_registry = ml_service.create_registry(
            serving_fixture["organization_id"],
            RegistryCreate(name="bearing-rul-backup"),
        )
        second_model = ml_service.register_model_version(
            serving_fixture["organization_id"],
            ModelVersionCreate(
                registry_id=second_registry.id,
                experiment_run_id=model_version.experiment_run_id,
                version="1.0.0",
            ),
        )
        ml_service.promote_model_version(
            serving_fixture["organization_id"],
            second_model.id,
            PromoteModelVersion(target_stage="validated"),
        )
        ml_service.promote_model_version(
            serving_fixture["organization_id"],
            second_model.id,
            PromoteModelVersion(
                target_stage="production",
                approved_by_user_id=serving_fixture["approver_id"],
                reason="approved alternate registry",
            ),
        )
        _bind_sensor(session, serving_fixture, registry, model_version)
        ProductionServingService(session).bind_model(
            serving_fixture["organization_id"],
            ServingBindingCreate(
                registry_id=second_registry.id,
                model_version_id=second_model.id,
                scope_type="sensor",
                scope_id=serving_fixture["sensor_a_id"],
                approved_by_user_id=serving_fixture["approver_id"],
            ),
        )

        ambiguous = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"]),
        )
        explicit = ProductionServingService(session, ModelArtifactStore(tmp_path / "models")).predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert ambiguous.prediction_status == "unsupported"
        assert ambiguous.abstention_code == "AMBIGUOUS_BINDING"
        assert explicit.prediction_status == "supported"


def test_database_enforces_one_active_binding_per_registry_scope(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        session.commit()

        first = ModelServingBinding(
            organization_id=serving_fixture["organization_id"],
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="sensor",
            scope_id=serving_fixture["sensor_a_id"],
            status="active",
            approved_by_user_id=serving_fixture["approver_id"],
        )
        second = ModelServingBinding(
            organization_id=serving_fixture["organization_id"],
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="sensor",
            scope_id=serving_fixture["sensor_a_id"],
            status="active",
            approved_by_user_id=serving_fixture["approver_id"],
        )
        session.add(first)
        session.flush()
        session.add(second)
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()

        first_org = ModelServingBinding(
            organization_id=serving_fixture["organization_id"],
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="organization",
            scope_id=None,
            status="active",
            approved_by_user_id=serving_fixture["approver_id"],
        )
        second_org = ModelServingBinding(
            organization_id=serving_fixture["organization_id"],
            registry_id=registry.id,
            model_version_id=model_version.id,
            scope_type="organization",
            scope_id=None,
            status="active",
            approved_by_user_id=serving_fixture["approver_id"],
        )
        session.add(first_org)
        session.flush()
        session.add(second_org)
        with pytest.raises(IntegrityError):
            session.flush()


def test_drift_monitoring_creates_retraining_trigger_without_auto_promotion(
    migrated_db,
    serving_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        _bind_sensor(session, serving_fixture, registry, model_version)
        repo = PlatformRepository(session)
        run = repo.create_analytics_run(
            serving_fixture["organization_id"],
            run_kind="sensor",
            sensor_id=serving_fixture["sensor_a_id"],
            algorithm_version="analytics-v1",
        )
        repo.create_analytics_feature(
            serving_fixture["organization_id"],
            run_id=run.id,
            sensor_id=serving_fixture["sensor_a_id"],
            batch_id=None,
            source_kind="scalar",
            source_record_id="live-drift",
            observed_at=datetime.now(UTC),
            feature_name="scalar.rms",
            value=999.0,
            unit="g",
            quality="good",
            algorithm_version="analytics-v1",
            provenance={},
        )

        service = ProductionServingService(session, ModelArtifactStore(tmp_path / "models"))
        prediction = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        repeated = service.predict_rul(
            serving_fixture["organization_id"],
            PredictionRequest(sensor_id=serving_fixture["sensor_a_id"], registry_id=registry.id),
        )
        session.commit()

        assert prediction.prediction_status == "unsupported"
        assert repeated.prediction_status == "unsupported"
        assert prediction.abstention_code == "OUT_OF_TRAINING_DOMAIN"
        assert prediction.abstention_reason == "live feature vector is outside the validated training domain"
        assert session.scalar(select(func.count()).select_from(ModelServingMonitor)) == 2
        assert session.scalar(select(func.count()).select_from(RetrainingTrigger)) == 1
        trigger = session.scalar(select(RetrainingTrigger))
        assert trigger is not None
        assert trigger.status == "open"
        assert trigger.evidence["silent_auto_promotion"] is False
        assert model_version.stage == "production"


def test_production_serving_api_binding_prediction_history_and_health(
    migrated_db,
    serving_fixture,
    tmp_path,
    monkeypatch,
):
    _engine, session_factory = migrated_db
    monkeypatch.setenv("PMS_MODEL_REGISTRY_ROOT", str(tmp_path / "models"))
    with session_factory() as session:
        registry, model_version, _dataset = _production_model(session, serving_fixture, tmp_path)
        session.commit()

    app_main = importlib.reload(importlib.import_module("app.main"))
    client = TestClient(app_main.app)
    org_id = serving_fixture["organization_id"]

    binding = client.post(
        f"/api/serving/{org_id}/bindings",
        json={
            "registry_id": registry.id,
            "model_version_id": model_version.id,
            "scope_type": "sensor",
            "scope_id": serving_fixture["sensor_a_id"],
            "approved_by_user_id": serving_fixture["approver_id"],
            "reason": "serve sensor through API",
        },
    )
    prediction = client.post(
        f"/api/serving/{org_id}/predict/rul",
        json={"sensor_id": serving_fixture["sensor_a_id"], "registry_id": registry.id},
    )
    history = client.get(f"/api/serving/{org_id}/predictions", params={"sensor_id": serving_fixture["sensor_a_id"]})
    health = client.get(f"/api/serving/{org_id}/health")

    assert binding.status_code == 200
    assert binding.json()["status"] == "active"
    assert prediction.status_code == 200
    assert prediction.json()["prediction_status"] == "supported"
    assert prediction.json()["predicted_rul_hours"] is not None
    assert prediction.json()["provenance"]["feature_record_ids"]
    assert history.status_code == 200
    assert len(history.json()["predictions"]) == 1
    assert health.status_code == 200
    assert len(health.json()["bindings"]) == 1
