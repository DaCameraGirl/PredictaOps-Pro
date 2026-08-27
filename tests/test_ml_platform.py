from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from analytics_pipeline.service import AnalyticsService
from industrial_ingestion.service import IngestionService
from ml_platform.artifact_store import ModelArtifactStore
from ml_platform.contracts import (
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    PromoteModelVersion,
    RegistryCreate,
    RollbackModelVersion,
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
    AnalyticsFeatureRecord,
    Base,
    MLDatasetVersion,
    MLExperimentRun,
    MLModelPromotionEvent,
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
def ml_fixture(migrated_db):
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
                name="Accelerometer VS-017",
                sensor_type="accelerometer",
                unit="g",
            ),
        )
        sensor_b = repo.create_sensor(
            org.id,
            SensorCreate(
                component_id=component.id,
                slug="vs-018",
                name="Accelerometer VS-018",
                sensor_type="accelerometer",
                unit="g",
            ),
        )
        other_org = repo.create_organization(OrganizationCreate(slug="other", name="Other Manufacturing"))
        user = repo.create_user(
            UserCreate(email="approver@example.com", full_name="Approver", external_subject="oidc:approver")
        )
        repo.add_membership(org.id, user.id, "engineer")
        session.commit()
        return {
            "organization_id": org.id,
            "other_organization_id": other_org.id,
            "sensor_a_id": sensor_a.id,
            "sensor_b_id": sensor_b.id,
            "approver_id": user.id,
        }


def _seed_analytics_features(session, fixture, *, feature_name: str = "scalar.rms") -> None:
    repo = PlatformRepository(session)
    run_a = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_a_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "ml-platform"},
    )
    run_b = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_b_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "ml-platform"},
    )
    base_time = datetime(2026, 8, 27, 12, tzinfo=UTC)
    for sensor_id, run_id, group, offset in [
        (fixture["sensor_a_id"], run_a.id, "bearing-a", 0.0),
        (fixture["sensor_b_id"], run_b.id, "bearing-b", 10.0),
    ]:
        for index in range(4):
            repo.create_analytics_feature(
                fixture["organization_id"],
                run_id=run_id,
                sensor_id=sensor_id,
                batch_id=None,
                source_kind="scalar",
                source_record_id=f"{group}-{index}",
                observed_at=base_time + timedelta(minutes=index),
                feature_name=feature_name,
                value=float(index + offset),
                unit="g",
                quality="good",
                algorithm_version="analytics-v1",
                provenance={
                    "target_rul_hours": float(8 - index - offset / 10),
                    "validation_group": group,
                    "run_id": group,
                    "failure_mode": "synthetic bearing degradation",
                },
            )
    session.commit()


def _create_dataset(service: MLPlatformService, fixture, *, version: str = "v1") -> MLDatasetVersion:
    return service.create_dataset_version(
        fixture["organization_id"],
        DatasetVersionCreate(
            name="bearing-rul-features",
            version=version,
            feature_names=["scalar.rms"],
            target_provenance_key="target_rul_hours",
            validation_group_provenance_key="validation_group",
        ),
    )


def _run_experiment(service: MLPlatformService, fixture, dataset_id: str) -> MLExperimentRun:
    return service.run_experiment(
        fixture["organization_id"],
        ExperimentCreate(
            dataset_version_id=dataset_id,
            name="cross-bearing validation",
            training_config={"n_estimators": 5, "random_state": 7},
        ),
    )


def test_migration_creates_ml_platform_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    assert {
        "ml_dataset_versions",
        "ml_experiment_runs",
        "ml_model_registries",
        "ml_model_versions",
        "ml_model_promotion_events",
    }.issubset(tables)


def test_dataset_versions_are_immutable_snapshots_of_canonical_analytics_features(migrated_db, ml_fixture):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        dataset = _create_dataset(MLPlatformService(session), ml_fixture)
        session.commit()

        source_feature_count = session.scalar(select(func.count()).select_from(AnalyticsFeatureRecord))
        assert source_feature_count == 8
        assert dataset.row_count == 8
        assert dataset.validation_group_count == 2
        assert dataset.source_algorithm_version == "analytics-v1"
        assert len(dataset.fingerprint) == 64
        assert dataset.provenance["source"] == "analytics_feature_records"
        assert dataset.provenance["rows"][0]["source_feature_ids"]
        assert dataset.provenance["rows"][0]["features"] == {"scalar.rms": 0.0}


def test_dataset_version_uses_labels_preserved_from_real_ingestion_metadata(migrated_db, ml_fixture):
    _engine, session_factory = migrated_db
    base_time = datetime(2026, 8, 27, 12, tzinfo=UTC)
    records = []
    for sensor_id, group, offset in [
        (ml_fixture["sensor_a_id"], "bearing-a", 0.0),
        (ml_fixture["sensor_b_id"], "bearing-b", 10.0),
    ]:
        for index in range(4):
            records.append(
                {
                    "kind": "scalar",
                    "sensor_id": sensor_id,
                    "observed_at": (base_time + timedelta(minutes=index)).isoformat(),
                    "metric": "rms",
                    "value": float(index + offset),
                    "unit": "g",
                    "source_record_id": f"labeled-{group}-{index}",
                    "metadata": {
                        "target_rul_hours": float(8 - index - offset / 10),
                        "validation_group": group,
                        "failure_mode": "synthetic bearing degradation",
                    },
                }
            )

    with session_factory() as session:
        receipt = IngestionService(session).ingest(
            ml_fixture["organization_id"],
            source_type="rest",
            source_name="REST Push",
            payload={"records": records},
        )
        analytics_receipt = AnalyticsService(session).compute_batch(
            ml_fixture["organization_id"],
            receipt.batch_id,
        )
        dataset = _create_dataset(MLPlatformService(session), ml_fixture, version="from-ingestion")
        feature = session.scalar(
            select(AnalyticsFeatureRecord)
            .where(
                AnalyticsFeatureRecord.sensor_id == ml_fixture["sensor_a_id"],
                AnalyticsFeatureRecord.feature_name == "scalar.rms",
            )
            .order_by(AnalyticsFeatureRecord.observed_at)
        )
        session.commit()

        assert receipt.accepted_count == 8
        assert analytics_receipt.feature_count == 8
        assert dataset.row_count == 8
        assert dataset.validation_group_count == 2
        assert feature is not None
        assert feature.provenance["source_metadata"]["target_rul_hours"] == 8.0
        assert feature.provenance["ingestion_provenance"]["metadata"]["validation_group"] == "bearing-a"
        assert dataset.provenance["rows"][0]["target"] == 8.0
        assert dataset.provenance["rows"][0]["validation_group"] == "bearing-a"


def test_experiment_records_reproducible_metadata_metrics_uncertainty_and_artifact(migrated_db, ml_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = _create_dataset(service, ml_fixture)
        experiment = _run_experiment(service, ml_fixture, dataset.id)
        session.commit()

        assert experiment.status == "completed"
        assert experiment.metrics["validation_method"] == "leave-one-validation-group-out"
        assert experiment.metrics["n_validation_groups"] == 2
        assert "baseline" in experiment.baseline_metrics
        assert experiment.uncertainty["method"] == "empirical_cross_group_residual_quantiles"
        assert experiment.abstention_policy["require_registered_dataset_version"] is True
        assert experiment.provenance["model_serving"] == "out_of_scope_for_slice_9"
        assert Path(experiment.artifact_uri).exists()
        assert len(experiment.artifact_sha256) == 64


def test_experiment_execution_rejects_unsupported_algorithm_and_validation_metadata(
    migrated_db,
    ml_fixture,
    tmp_path,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = _create_dataset(service, ml_fixture)

        unsupported_algorithm = ExperimentCreate.model_construct(
            dataset_version_id=dataset.id,
            name="unsupported algorithm",
            algorithm="xgboost.XGBRegressor",
            validation_method="leave-one-validation-group-out",
            training_config={},
            abstention_policy={},
        )
        unsupported_validation = ExperimentCreate.model_construct(
            dataset_version_id=dataset.id,
            name="unsupported validation",
            algorithm="sklearn.RandomForestRegressor",
            validation_method="k-fold",
            training_config={},
            abstention_policy={},
        )

        with pytest.raises(ValueError, match="unsupported experiment algorithm"):
            service.run_experiment(ml_fixture["organization_id"], unsupported_algorithm)
        with pytest.raises(ValueError, match="unsupported validation method"):
            service.run_experiment(ml_fixture["organization_id"], unsupported_validation)


def test_model_registry_promotion_requires_human_approval_and_supports_rollback(migrated_db, ml_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = _create_dataset(service, ml_fixture)
        experiment = _run_experiment(service, ml_fixture, dataset.id)
        registry = service.create_registry(
            ml_fixture["organization_id"],
            RegistryCreate(name="bearing-rul", task="rul_regression"),
        )
        v1 = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.0"),
        )
        v2 = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.1"),
        )

        service.promote_model_version(
            ml_fixture["organization_id"],
            v1.id,
            PromoteModelVersion(target_stage="validated"),
        )
        with pytest.raises(ValueError, match="explicit human approval"):
            service.promote_model_version(
                ml_fixture["organization_id"],
                v1.id,
                PromoteModelVersion(target_stage="production"),
            )
        service.promote_model_version(
            ml_fixture["organization_id"],
            v1.id,
            PromoteModelVersion(
                target_stage="production",
                approved_by_user_id=ml_fixture["approver_id"],
                reason="validated metrics accepted",
            ),
        )
        service.promote_model_version(
            ml_fixture["organization_id"],
            v2.id,
            PromoteModelVersion(target_stage="validated"),
        )
        service.promote_model_version(
            ml_fixture["organization_id"],
            v2.id,
            PromoteModelVersion(
                target_stage="production",
                approved_by_user_id=ml_fixture["approver_id"],
                reason="candidate improves baseline",
            ),
        )
        service.rollback_model_version(
            ml_fixture["organization_id"],
            registry.id,
            RollbackModelVersion(
                target_model_version_id=v1.id,
                approved_by_user_id=ml_fixture["approver_id"],
                reason="rollback after review",
            ),
        )
        session.commit()

        events = session.scalar(select(func.count()).select_from(MLModelPromotionEvent))
        assert v1.stage == "production"
        assert v1.approval_status == "approved"
        assert v2.stage == "archived"
        assert events == 5
        assert v1.provenance["dataset_version_id"] == dataset.id
        assert v1.provenance["feature_names"] == ["scalar.rms"]


def test_rollback_rejects_versions_without_prior_production_history(migrated_db, ml_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = _create_dataset(service, ml_fixture)
        experiment = _run_experiment(service, ml_fixture, dataset.id)
        registry = service.create_registry(
            ml_fixture["organization_id"],
            RegistryCreate(name="bearing-rul", task="rul_regression"),
        )
        candidate = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.0"),
        )
        validated = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.1"),
        )
        rejected = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.2"),
        )

        service.promote_model_version(
            ml_fixture["organization_id"],
            validated.id,
            PromoteModelVersion(target_stage="validated"),
        )
        service.promote_model_version(
            ml_fixture["organization_id"],
            rejected.id,
            PromoteModelVersion(target_stage="rejected", reason="failed review"),
        )

        for target in [candidate, validated, rejected]:
            with pytest.raises(ValueError, match="previously reached production"):
                service.rollback_model_version(
                    ml_fixture["organization_id"],
                    registry.id,
                    RollbackModelVersion(
                        target_model_version_id=target.id,
                        approved_by_user_id=ml_fixture["approver_id"],
                        reason="invalid rollback",
                    ),
                )


def test_ml_platform_enforces_tenant_boundaries(migrated_db, ml_fixture, tmp_path):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = _create_dataset(service, ml_fixture)
        experiment = _run_experiment(service, ml_fixture, dataset.id)
        registry = service.create_registry(ml_fixture["organization_id"], RegistryCreate(name="bearing-rul"))
        model_version = service.register_model_version(
            ml_fixture["organization_id"],
            ModelVersionCreate(registry_id=registry.id, experiment_run_id=experiment.id, version="1.0.0"),
        )

        with pytest.raises(ValueError, match="dataset version does not exist inside this organization"):
            service.run_experiment(
                ml_fixture["other_organization_id"],
                ExperimentCreate(dataset_version_id=dataset.id, name="wrong tenant"),
            )
        with pytest.raises(ValueError, match="model version does not exist inside this organization"):
            service.promote_model_version(
                ml_fixture["other_organization_id"],
                model_version.id,
                PromoteModelVersion(target_stage="validated"),
            )


def test_ml_platform_api_dataset_experiment_registry_promotion_and_rollback(
    migrated_db,
    ml_fixture,
    tmp_path,
    monkeypatch,
):
    _engine, session_factory = migrated_db
    monkeypatch.setenv("PMS_MODEL_REGISTRY_ROOT", str(tmp_path / "models"))
    with session_factory() as session:
        _seed_analytics_features(session, ml_fixture)

    app_main = importlib.reload(importlib.import_module("app.main"))
    client = TestClient(app_main.app)
    org_id = ml_fixture["organization_id"]

    dataset = client.post(
        f"/api/ml/{org_id}/dataset-versions",
        json={
            "name": "bearing-rul-features",
            "version": "api-v1",
            "feature_names": ["scalar.rms"],
        },
    )
    experiment = client.post(
        f"/api/ml/{org_id}/experiments",
        json={
            "dataset_version_id": dataset.json()["id"],
            "name": "api experiment",
            "training_config": {"n_estimators": 5, "random_state": 13},
        },
    )
    bad_algorithm = client.post(
        f"/api/ml/{org_id}/experiments",
        json={
            "dataset_version_id": dataset.json()["id"],
            "name": "bad algorithm",
            "algorithm": "xgboost.XGBRegressor",
        },
    )
    bad_validation = client.post(
        f"/api/ml/{org_id}/experiments",
        json={
            "dataset_version_id": dataset.json()["id"],
            "name": "bad validation",
            "validation_method": "k-fold",
        },
    )
    registry = client.post(f"/api/ml/{org_id}/registries", json={"name": "bearing-rul"})
    v1 = client.post(
        f"/api/ml/{org_id}/model-versions",
        json={
            "registry_id": registry.json()["id"],
            "experiment_run_id": experiment.json()["id"],
            "version": "api-1.0.0",
        },
    )
    v2 = client.post(
        f"/api/ml/{org_id}/model-versions",
        json={
            "registry_id": registry.json()["id"],
            "experiment_run_id": experiment.json()["id"],
            "version": "api-1.0.1",
        },
    )
    validated = client.post(
        f"/api/ml/{org_id}/model-versions/{v1.json()['id']}/promote",
        json={"target_stage": "validated"},
    )
    missing_approval = client.post(
        f"/api/ml/{org_id}/model-versions/{v1.json()['id']}/promote",
        json={"target_stage": "production"},
    )
    production = client.post(
        f"/api/ml/{org_id}/model-versions/{v1.json()['id']}/promote",
        json={
            "target_stage": "production",
            "approved_by_user_id": ml_fixture["approver_id"],
            "reason": "human approved",
        },
    )
    client.post(
        f"/api/ml/{org_id}/model-versions/{v2.json()['id']}/promote",
        json={"target_stage": "validated"},
    )
    client.post(
        f"/api/ml/{org_id}/model-versions/{v2.json()['id']}/promote",
        json={
            "target_stage": "production",
            "approved_by_user_id": ml_fixture["approver_id"],
            "reason": "human approved v2",
        },
    )
    rollback = client.post(
        f"/api/ml/{org_id}/registries/{registry.json()['id']}/rollback",
        json={
            "target_model_version_id": v1.json()["id"],
            "approved_by_user_id": ml_fixture["approver_id"],
            "reason": "return to prior model",
        },
    )
    registries = client.get(f"/api/ml/{org_id}/registries")

    assert dataset.status_code == 200
    assert dataset.json()["row_count"] == 8
    assert experiment.status_code == 200
    assert experiment.json()["status"] == "completed"
    assert bad_algorithm.status_code == 422
    assert bad_validation.status_code == 422
    assert registry.status_code == 200
    assert v1.status_code == 200
    assert v1.json()["stage"] == "candidate"
    assert validated.status_code == 200
    assert validated.json()["stage"] == "validated"
    assert missing_approval.status_code == 400
    assert production.status_code == 200
    assert production.json()["approval_status"] == "approved"
    assert rollback.status_code == 200
    assert rollback.json()["id"] == v1.json()["id"]
    assert rollback.json()["stage"] == "production"
    assert registries.status_code == 200
    assert len(registries.json()["registries"][0]["versions"]) == 2
