"""ML platform service for Slice 9 model lifecycle metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session

from ml_platform.artifact_store import ModelArtifactStore
from ml_platform.contracts import (
    SUPPORTED_ALGORITHM,
    SUPPORTED_VALIDATION_METHOD,
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    PromoteModelVersion,
    RegistryCreate,
    RollbackModelVersion,
)
from platform_core.models import (
    AnalyticsFeatureRecord,
    MLDatasetVersion,
    MLExperimentRun,
    MLModelRegistry,
    MLModelVersion,
    OrganizationMembership,
    User,
)
from platform_core.repositories import PlatformRepository


class MLPlatformError(ValueError):
    pass


class MLPlatformService:
    def __init__(self, session: Session, artifact_store: ModelArtifactStore | None = None):
        self.repo = PlatformRepository(session)
        self.artifact_store = artifact_store or ModelArtifactStore()

    def create_dataset_version(self, organization_id: str, request: DatasetVersionCreate) -> MLDatasetVersion:
        features = self.repo.list_analytics_features_for_dataset(
            organization_id,
            algorithm_version=request.source_algorithm_version,
            sensor_ids=request.sensor_ids,
            feature_names=request.feature_names,
        )
        rows = _dataset_rows_from_features(
            features,
            feature_names=request.feature_names,
            target_key=request.target_provenance_key,
            validation_group_key=request.validation_group_provenance_key,
        )
        if not rows:
            raise MLPlatformError("dataset version has no labeled analytics feature rows")
        validation_groups = sorted({row["validation_group"] for row in rows})
        fingerprint = _fingerprint({"feature_names": request.feature_names, "rows": rows})
        return self.repo.create_ml_dataset_version(
            organization_id,
            name=request.name,
            version=request.version,
            source_algorithm_version=request.source_algorithm_version,
            target_name=request.target_name,
            target_unit=request.target_unit,
            feature_names=request.feature_names,
            row_count=len(rows),
            validation_group_count=len(validation_groups),
            fingerprint=fingerprint,
            filters={
                "sensor_ids": request.sensor_ids,
                "target_provenance_key": request.target_provenance_key,
                "validation_group_provenance_key": request.validation_group_provenance_key,
            },
            provenance={
                "source": "analytics_feature_records",
                "rows": rows,
                "validation_groups": validation_groups,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def list_dataset_versions(self, organization_id: str) -> list[MLDatasetVersion]:
        return self.repo.list_ml_dataset_versions(organization_id)

    def run_experiment(self, organization_id: str, request: ExperimentCreate) -> MLExperimentRun:
        _validate_supported_experiment_contract(request)
        dataset = self.repo.get_ml_dataset_version(organization_id, request.dataset_version_id)
        if dataset is None:
            raise MLPlatformError("dataset version does not exist inside this organization")
        rows = _dataset_rows(dataset)
        if dataset.validation_group_count < 2:
            raise MLPlatformError("cross-group validation requires at least two validation groups")
        abstention_policy = _abstention_policy(request.abstention_policy)
        experiment = self.repo.create_ml_experiment_run(
            organization_id,
            dataset_version_id=dataset.id,
            name=request.name,
            algorithm=request.algorithm,
            validation_method=request.validation_method,
            code_version=_git_commit(),
            training_config=request.training_config,
            abstention_policy=abstention_policy,
            provenance={
                "dataset_fingerprint": dataset.fingerprint,
                "dataset_row_count": dataset.row_count,
                "model_serving": "out_of_scope_for_slice_9",
            },
        )
        try:
            metrics, baseline_metrics, uncertainty, model = _train_and_validate(
                rows,
                feature_names=dataset.feature_names,
                config=request.training_config,
            )
            artifact_uri, artifact_sha256 = self.artifact_store.write_model(
                organization_id=organization_id,
                experiment_run_id=experiment.id,
                model=model,
            )
            experiment.status = "completed"
            experiment.finished_at = datetime.now(UTC)
            experiment.metrics = metrics
            experiment.baseline_metrics = baseline_metrics
            experiment.uncertainty = uncertainty
            experiment.artifact_uri = artifact_uri
            experiment.artifact_sha256 = artifact_sha256
            self.repo.session.flush()
        except Exception:
            experiment.status = "failed"
            experiment.finished_at = datetime.now(UTC)
            self.repo.session.flush()
            raise
        return experiment

    def get_experiment(self, organization_id: str, experiment_run_id: str) -> MLExperimentRun:
        experiment = self.repo.get_ml_experiment_run(organization_id, experiment_run_id)
        if experiment is None:
            raise MLPlatformError("experiment run does not exist inside this organization")
        return experiment

    def list_experiments(self, organization_id: str) -> list[MLExperimentRun]:
        return self.repo.list_ml_experiment_runs(organization_id)

    def create_registry(self, organization_id: str, request: RegistryCreate) -> MLModelRegistry:
        return self.repo.get_or_create_ml_model_registry(
            organization_id,
            name=request.name,
            task=request.task,
            description=request.description,
        )

    def list_registries(self, organization_id: str) -> list[dict[str, Any]]:
        registries = self.repo.list_ml_model_registries(organization_id)
        return [
            {
                "id": registry.id,
                "name": registry.name,
                "task": registry.task,
                "status": registry.status,
                "description": registry.description,
                "versions": [
                    _model_version_payload(version)
                    for version in self.repo.list_ml_model_versions(organization_id, registry.id)
                ],
            }
            for registry in registries
        ]

    def register_model_version(self, organization_id: str, request: ModelVersionCreate) -> MLModelVersion:
        registry = self.repo.get_ml_model_registry(organization_id, request.registry_id)
        if registry is None:
            raise MLPlatformError("model registry does not exist inside this organization")
        experiment = self.repo.get_ml_experiment_run(organization_id, request.experiment_run_id)
        if experiment is None:
            raise MLPlatformError("experiment run does not exist inside this organization")
        if experiment.status != "completed" or not experiment.artifact_uri or not experiment.artifact_sha256:
            raise MLPlatformError("only completed experiments with artifacts can be registered")
        dataset = self.repo.get_ml_dataset_version(organization_id, experiment.dataset_version_id)
        if dataset is None:
            raise MLPlatformError("experiment dataset is missing")
        return self.repo.create_ml_model_version(
            organization_id,
            registry_id=registry.id,
            experiment_run_id=experiment.id,
            dataset_version_id=dataset.id,
            version=request.version,
            artifact_uri=experiment.artifact_uri,
            artifact_sha256=experiment.artifact_sha256,
            metrics=experiment.metrics,
            baseline_metrics=experiment.baseline_metrics,
            uncertainty=experiment.uncertainty,
            abstention_policy=experiment.abstention_policy,
            provenance={
                "dataset_version_id": dataset.id,
                "dataset_fingerprint": dataset.fingerprint,
                "feature_names": dataset.feature_names,
                "model_domain": _model_domain_contract(dataset),
                "code_version": experiment.code_version,
                "experiment_run_id": experiment.id,
                "created_at": datetime.now(UTC).isoformat(),
                "model_serving": "out_of_scope_for_slice_9",
            },
        )

    def promote_model_version(self, organization_id: str, model_version_id: str, request: PromoteModelVersion):
        model_version = self._model_version(organization_id, model_version_id)
        target_stage = request.target_stage
        if target_stage == "production":
            if model_version.stage != "validated":
                raise MLPlatformError("production promotion requires a validated model version")
            self._validate_human_approval(organization_id, request.approved_by_user_id)
        elif target_stage == "validated" and model_version.stage != "candidate":
            raise MLPlatformError("validated promotion requires a candidate model version")
        elif target_stage not in {"validated", "rejected"}:
            raise MLPlatformError("only validated, production, or rejected promotions are supported in Slice 9")

        prior_stage = model_version.stage
        if target_stage == "production":
            current = self.repo.get_production_model_version(organization_id, model_version.registry_id)
            if current is not None and current.id != model_version.id:
                current.stage = "archived"
        model_version.stage = target_stage
        if target_stage == "production":
            model_version.approval_status = "approved"
            model_version.approved_by_user_id = request.approved_by_user_id
            model_version.approved_at = datetime.now(UTC)
        self.repo.create_ml_promotion_event(
            organization_id,
            registry_id=model_version.registry_id,
            model_version_id=model_version.id,
            from_stage=prior_stage,
            to_stage=target_stage,
            action="promote",
            approved_by_user_id=request.approved_by_user_id,
            reason=request.reason,
            event_metadata={"human_approval_required": target_stage == "production"},
        )
        self.repo.session.flush()
        return model_version

    def rollback_model_version(self, organization_id: str, registry_id: str, request: RollbackModelVersion):
        registry = self.repo.get_ml_model_registry(organization_id, registry_id)
        if registry is None:
            raise MLPlatformError("model registry does not exist inside this organization")
        target = self._model_version(organization_id, request.target_model_version_id)
        if target.registry_id != registry_id:
            raise MLPlatformError("rollback target belongs to a different registry")
        if target.stage != "archived" or not self.repo.model_version_has_reached_production(
            organization_id,
            target.id,
        ):
            raise MLPlatformError(
                "rollback target must be an archived model version that previously reached production"
            )
        self._validate_human_approval(organization_id, request.approved_by_user_id)
        current = self.repo.get_production_model_version(organization_id, registry_id)
        if current is not None and current.id != target.id:
            current.stage = "archived"
        prior_stage = target.stage
        target.stage = "production"
        target.approval_status = "approved"
        target.approved_by_user_id = request.approved_by_user_id
        target.approved_at = datetime.now(UTC)
        self.repo.create_ml_promotion_event(
            organization_id,
            registry_id=registry_id,
            model_version_id=target.id,
            from_stage=prior_stage,
            to_stage="production",
            action="rollback",
            approved_by_user_id=request.approved_by_user_id,
            reason=request.reason,
            event_metadata={"previous_production_model_version_id": current.id if current else None},
        )
        self.repo.session.flush()
        return target

    def _model_version(self, organization_id: str, model_version_id: str) -> MLModelVersion:
        model_version = self.repo.get_ml_model_version(organization_id, model_version_id)
        if model_version is None:
            raise MLPlatformError("model version does not exist inside this organization")
        return model_version

    def _validate_human_approval(self, organization_id: str, user_id: str | None) -> None:
        if not user_id:
            raise MLPlatformError("production promotion requires explicit human approval")
        user = self.repo.session.get(User, user_id)
        if user is None:
            raise MLPlatformError("approval user does not exist")
        membership = self.repo.session.query(OrganizationMembership).filter_by(
            organization_id=organization_id,
            user_id=user_id,
            lifecycle_state="active",
        ).first()
        if membership is None:
            raise MLPlatformError("approval user must belong to this organization")


def _dataset_rows_from_features(
    features: list[AnalyticsFeatureRecord],
    *,
    feature_names: list[str],
    target_key: str,
    validation_group_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"features": {}, "feature_units": {}, "source_feature_ids": [], "source_feature_provenance": []}
    )
    for feature in features:
        key = (feature.sensor_id, feature.observed_at.isoformat())
        row = grouped[key]
        row["sensor_id"] = feature.sensor_id
        row["observed_at"] = feature.observed_at.isoformat()
        row["features"][feature.feature_name] = feature.value
        row["feature_units"][feature.feature_name] = feature.unit
        row["source_feature_ids"].append(feature.id)
        row["source_feature_provenance"].append(feature.provenance or {})
        provenance = feature.provenance or {}
        target_value = _provenance_value(provenance, target_key)
        if target_value is not None:
            row["target"] = float(target_value)
        validation_group_value = _provenance_value(provenance, validation_group_key)
        if validation_group_value is not None:
            row["validation_group"] = str(validation_group_value)

    rows = []
    for row in grouped.values():
        if any(name not in row["features"] for name in feature_names):
            continue
        if "target" not in row:
            continue
        row.setdefault("validation_group", row["sensor_id"])
        row["features"] = {name: float(row["features"][name]) for name in feature_names}
        row["feature_units"] = {name: row["feature_units"].get(name) for name in feature_names}
        row["source_feature_ids"] = sorted(row["source_feature_ids"])
        row.pop("source_feature_provenance", None)
        rows.append(row)
    return sorted(rows, key=lambda item: (item["validation_group"], item["observed_at"], item["sensor_id"]))


def _provenance_value(provenance: dict[str, Any], key: str) -> Any | None:
    ingestion_provenance = provenance.get("ingestion_provenance")
    source_metadata = provenance.get("source_metadata")
    ingestion_metadata = None
    if isinstance(ingestion_provenance, dict):
        ingestion_metadata = ingestion_provenance.get("metadata")
    for scope in [provenance, source_metadata, ingestion_provenance, ingestion_metadata]:
        if isinstance(scope, dict):
            value = _nested_value(scope, key)
            if value is not None:
                return value
    return None


def _nested_value(payload: dict[str, Any], key: str) -> Any | None:
    if key in payload:
        return payload[key]
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _dataset_rows(dataset: MLDatasetVersion) -> list[dict[str, Any]]:
    rows = (dataset.provenance or {}).get("rows") or []
    if not rows:
        raise MLPlatformError("dataset version has no materialized row snapshot")
    return rows


def _model_domain_contract(dataset: MLDatasetVersion) -> dict[str, Any]:
    rows = _dataset_rows(dataset)
    feature_stats = {}
    feature_units = {}
    for feature_name in dataset.feature_names:
        values = [float(row["features"][feature_name]) for row in rows if feature_name in row.get("features", {})]
        unit_values = []
        for row in rows:
            row_units = row.get("feature_units") or {}
            if feature_name in row_units:
                unit_values.append(row_units[feature_name])
        non_null_units = sorted({unit for unit in unit_values if unit is not None})
        if unit_values and all(unit is None for unit in unit_values):
            feature_units[feature_name] = None
        elif len(non_null_units) == 1 and all(unit is not None for unit in unit_values):
            feature_units[feature_name] = non_null_units[0]
        else:
            feature_units[feature_name] = [None, *non_null_units]
        feature_stats[feature_name] = {
            "min": min(values),
            "max": max(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "missing_count": dataset.row_count - len(values),
            "training_rows": len(values),
        }
    return {
        "schema_version": "model-domain-v1",
        "ordered_feature_names": list(dataset.feature_names),
        "feature_units": feature_units,
        "analytics_algorithm_version": dataset.source_algorithm_version,
        "preprocessing_version": dataset.source_algorithm_version,
        "numeric_type": "float64",
        "row_count": dataset.row_count,
        "validation_group_count": dataset.validation_group_count,
        "feature_stats": feature_stats,
        "dataset_fingerprint": dataset.fingerprint,
    }


def _validate_supported_experiment_contract(request: ExperimentCreate) -> None:
    if request.algorithm != SUPPORTED_ALGORITHM:
        raise MLPlatformError(f"unsupported experiment algorithm {request.algorithm!r}")
    if request.validation_method != SUPPORTED_VALIDATION_METHOD:
        raise MLPlatformError(f"unsupported validation method {request.validation_method!r}")


def _train_and_validate(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], RandomForestRegressor]:
    groups = sorted({row["validation_group"] for row in rows})
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    baseline_all: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []
    for group in groups:
        train_rows = [row for row in rows if row["validation_group"] != group]
        test_rows = [row for row in rows if row["validation_group"] == group]
        if not train_rows or not test_rows:
            raise MLPlatformError("cross-group validation produced an empty train/test fold")
        model = _make_model(config)
        x_train, y_train = _matrix(train_rows, feature_names)
        x_test, y_test = _matrix(test_rows, feature_names)
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        baseline = np.full(len(y_test), float(np.mean(y_train)))
        y_true_all.append(y_test)
        y_pred_all.append(pred)
        baseline_all.append(baseline)
        folds.append(
            {
                "held_out_group": group,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "mae": float(mean_absolute_error(y_test, pred)),
                "baseline_mae": float(mean_absolute_error(y_test, baseline)),
            }
        )

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_baseline = np.concatenate(baseline_all)
    residual = y_pred - y_true
    final_model = _make_model(config)
    x_all, y_all = _matrix(rows, feature_names)
    final_model.fit(x_all, y_all)

    metrics = {
        "validation_method": "leave-one-validation-group-out",
        "n_validation_groups": len(groups),
        "n_rows": len(rows),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "dangerous_overprediction_pct": float(np.mean(residual > 0) * 100.0),
        "folds": folds,
    }
    baseline_metrics = {
        "baseline": "training-fold-mean-target",
        "mae": float(mean_absolute_error(y_true, y_baseline)),
        "rmse": float(mean_squared_error(y_true, y_baseline) ** 0.5),
        "model_beats_baseline_mae": metrics["mae"] < float(mean_absolute_error(y_true, y_baseline)),
    }
    uncertainty = {
        "method": "empirical_cross_group_residual_quantiles",
        "residual_p10": float(np.percentile(residual, 10)),
        "residual_p90": float(np.percentile(residual, 90)),
        "residual_std": float(np.std(residual)),
        "calibration_scope": "cross-validation folds only; live serving calibration is Slice 10",
    }
    return metrics, baseline_metrics, uncertainty, final_model


def _make_model(config: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(config.get("n_estimators", 25)),
        max_depth=config.get("max_depth"),
        random_state=int(config.get("random_state", 42)),
    )


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[row["features"][name] for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([row["target"] for row in rows], dtype=float)
    return x, y


def _abstention_policy(policy: dict[str, Any]) -> dict[str, Any]:
    supported_fields = {
        "contract",
        "min_validation_groups",
        "require_registered_dataset_version",
        "require_model_stage_for_serving",
        "live_serving_enforcement",
        "max_feature_age_minutes",
        "require_feature_quality",
        "training_domain_behavior",
        "allow_historical_predictions",
    }
    unknown = sorted(set(policy).difference(supported_fields))
    if unknown:
        raise MLPlatformError(f"unsupported abstention policy fields: {', '.join(unknown)}")
    merged = {
        "contract": "abstain when validation evidence is outside the model version domain",
        "min_validation_groups": 2,
        "require_registered_dataset_version": True,
        "require_model_stage_for_serving": "production",
        "live_serving_enforcement": "production-slice-10",
        "max_feature_age_minutes": 1440,
        "require_feature_quality": "good",
        "training_domain_behavior": "abstain",
        "allow_historical_predictions": True,
    }
    merged.update(policy)
    return merged


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _model_version_payload(model_version: MLModelVersion) -> dict[str, Any]:
    return {
        "id": model_version.id,
        "version": model_version.version,
        "stage": model_version.stage,
        "approval_status": model_version.approval_status,
        "artifact_sha256": model_version.artifact_sha256,
        "metrics": model_version.metrics,
        "baseline_metrics": model_version.baseline_metrics,
        "uncertainty": model_version.uncertainty,
        "abstention_policy": model_version.abstention_policy,
        "provenance": model_version.provenance,
    }
