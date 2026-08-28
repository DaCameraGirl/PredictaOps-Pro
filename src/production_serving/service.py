"""Live production inference over approved ML Platform model versions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ml_platform.artifact_store import ModelArtifactStore
from platform_core.models import (
    AnalyticsFeatureRecord,
    Asset,
    Component,
    MLDatasetVersion,
    MLModelRegistry,
    MLModelVersion,
    ModelServingBinding,
    Sensor,
)
from platform_core.repositories import PlatformRepository
from production_serving.contracts import PredictionRequest, PredictionResponse, ServingBindingCreate

SCOPE_SPECIFICITY = {"organization": 0, "site": 1, "asset": 2, "component": 3, "sensor": 4}
RUL_TASK = "rul_regression"
RUL_TARGET_NAME = "RUL_hours"
RUL_TARGET_UNIT = "h"
SUPPORTED_REASON_CODE = "SUPPORTED"
DEFAULT_SERVING_POLICY = {
    "max_feature_age_minutes": 1440,
    "require_feature_quality": "good",
    "training_domain_behavior": "abstain",
    "allow_historical_predictions": True,
}
SUPPORTED_SERVING_POLICY_FIELDS = set(DEFAULT_SERVING_POLICY) | {
    "contract",
    "min_validation_groups",
    "require_registered_dataset_version",
    "require_model_stage_for_serving",
    "live_serving_enforcement",
}


class ProductionServingError(ValueError):
    pass


@dataclass(frozen=True)
class SensorContext:
    sensor: Sensor
    component: Component
    asset: Asset
    site_id: str


@dataclass(frozen=True)
class BindingSelection:
    binding: ModelServingBinding | None
    code: str | None
    reason: str | None
    evidence: dict[str, Any]


class ProductionServingService:
    def __init__(self, session: Session, artifact_store: ModelArtifactStore | None = None):
        self.repo = PlatformRepository(session)
        self.artifact_store = artifact_store or ModelArtifactStore()

    def bind_model(self, organization_id: str, request: ServingBindingCreate) -> ModelServingBinding:
        model_version = self.repo.get_ml_model_version(organization_id, request.model_version_id)
        if model_version is None:
            raise ProductionServingError("model version does not exist inside this organization")
        if model_version.registry_id != request.registry_id:
            raise ProductionServingError("model version belongs to a different registry")
        registry = self.repo.get_ml_model_registry(organization_id, request.registry_id)
        if registry is None:
            raise ProductionServingError("model registry does not exist inside this organization")
        if registry.task != RUL_TASK:
            raise ProductionServingError("RUL serving bindings require a rul_regression registry")
        if model_version.stage != "production" or model_version.approval_status != "approved":
            raise ProductionServingError("only approved production model versions can be bound for serving")
        return self.repo.create_model_serving_binding(
            organization_id,
            registry_id=request.registry_id,
            model_version_id=request.model_version_id,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            approved_by_user_id=request.approved_by_user_id,
            reason=request.reason,
            provenance={
                "human_approval_preserved": True,
                "model_stage_at_binding": model_version.stage,
                "artifact_sha256": model_version.artifact_sha256,
            },
        )

    def predict_rul(self, organization_id: str, request: PredictionRequest) -> PredictionResponse:
        context = self._sensor_context(organization_id, request.sensor_id)
        registry = self._registry_filter(organization_id, request)
        serving_reference_time = _aware_utc(request.observed_at or datetime.now(UTC))
        request_kind = "historical" if request.observed_at is not None else "live"
        if registry is not None and registry.task != RUL_TASK:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code="NON_RUL_MODEL_BINDING",
                reason="RUL prediction requires a rul_regression registry",
                binding=None,
                model_version=None,
                dataset=None,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={"registry_id": registry.id, "registry_task": registry.task},
                request_kind=request_kind,
            )
        registry_id = registry.id if registry else None

        selection = self._resolve_binding(organization_id, context, registry_id=registry_id)
        if selection.binding is None:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code=selection.code or "NO_MODEL_BINDING",
                reason=selection.reason or "no active production model binding resolves for this sensor",
                binding=None,
                model_version=None,
                dataset=None,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence=selection.evidence,
                request_kind=request_kind,
            )
        binding = selection.binding

        model_version = self.repo.get_ml_model_version(organization_id, binding.model_version_id)
        if model_version is None or model_version.registry_id != binding.registry_id:
            return self._unsupported_binding(
                organization_id,
                context,
                binding,
                serving_reference_time,
                request_kind,
                "serving binding points to a missing or mismatched model version",
                "MODEL_VERSION_MISMATCH",
            )
        if model_version.stage != "production" or model_version.approval_status != "approved":
            return self._unsupported_binding(
                organization_id,
                context,
                binding,
                serving_reference_time,
                request_kind,
                "serving binding model version is not approved for production",
                "MODEL_NOT_APPROVED",
                model_version=model_version,
            )

        policy_error = _policy_error(model_version.abstention_policy or {})
        if policy_error:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code="UNSUPPORTED_ABSTENTION_POLICY",
                reason=policy_error,
                binding=binding,
                model_version=model_version,
                dataset=None,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={"abstention_policy": model_version.abstention_policy},
                request_kind=request_kind,
            )
        policy = _serving_policy(model_version.abstention_policy or {})
        if request_kind == "historical" and not policy["allow_historical_predictions"]:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code="HISTORICAL_INFERENCE_DISABLED",
                reason="model abstention policy does not allow historical inference",
                binding=binding,
                model_version=model_version,
                dataset=None,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={"abstention_policy": policy},
                request_kind=request_kind,
            )

        dataset = self.repo.get_ml_dataset_version(organization_id, model_version.dataset_version_id)
        if dataset is None:
            return self._unsupported_binding(
                organization_id,
                context,
                binding,
                serving_reference_time,
                request_kind,
                "production model dataset snapshot is missing",
                "MISSING_DATASET_SNAPSHOT",
                model_version=model_version,
            )
        schema_error = _feature_schema_error(model_version, dataset)
        if schema_error:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code="FEATURE_SCHEMA_MISMATCH",
                reason=schema_error,
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={"feature_schema": dataset.feature_names},
                request_kind=request_kind,
            )
        policy_contract_error = _policy_contract_error(model_version.abstention_policy or {}, dataset)
        if policy_contract_error:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=serving_reference_time,
                status="unsupported",
                code="UNMET_ABSTENTION_POLICY",
                reason=policy_contract_error,
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={
                    "abstention_policy": model_version.abstention_policy,
                    "validation_group_count": dataset.validation_group_count,
                },
                request_kind=request_kind,
            )

        prediction_time = serving_reference_time
        latest_feature_time = self.repo.latest_feature_observed_at(
            organization_id,
            sensor_id=context.sensor.id,
            algorithm_version=dataset.source_algorithm_version,
        )
        if latest_feature_time is None:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="insufficient_evidence",
                code="MISSING_FEATURES",
                reason="no analytics features exist for this sensor",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=None,
                feature_vector=None,
                uncertainty=None,
                evidence={
                    "required_features": dataset.feature_names,
                    "serving_reference_time": prediction_time.isoformat(),
                },
                request_kind=request_kind,
            )

        feature_rows = self.repo.list_latest_analytics_features_for_sensor(
            organization_id,
            sensor_id=context.sensor.id,
            algorithm_version=dataset.source_algorithm_version,
            feature_names=dataset.feature_names,
            observed_at=prediction_time,
        )
        missing = [name for name in dataset.feature_names if name not in feature_rows]
        if missing:
            self._persist_feature_quality_monitors(
                organization_id,
                context,
                model_version,
                prediction_time,
                "insufficient_evidence",
                {"missing_features": missing, "required_features": dataset.feature_names},
            )
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="insufficient_evidence",
                code="MISSING_FEATURES",
                reason="live analytics feature vector is incomplete",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=None,
                uncertainty=None,
                evidence={"missing_features": missing, "required_features": dataset.feature_names},
                request_kind=request_kind,
            )

        effective_max_feature_age_minutes = min(
            int(policy["max_feature_age_minutes"]),
            request.max_feature_age_minutes,
        )
        stale = _stale_features(feature_rows, prediction_time, effective_max_feature_age_minutes)
        if stale:
            self._persist_feature_quality_monitors(
                organization_id,
                context,
                model_version,
                prediction_time,
                "insufficient_evidence",
                {"stale_features": stale, "max_feature_age_minutes": effective_max_feature_age_minutes},
            )
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="insufficient_evidence",
                code="STALE_FEATURES",
                reason="live analytics features are stale",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=None,
                uncertainty=None,
                evidence={
                    "stale_features": stale,
                    "max_feature_age_minutes": effective_max_feature_age_minutes,
                    "request_max_feature_age_minutes": request.max_feature_age_minutes,
                    "policy_max_feature_age_minutes": policy["max_feature_age_minutes"],
                },
                request_kind=request_kind,
            )

        live_schema_error = _live_feature_schema_error(model_version, dataset, feature_rows)
        if live_schema_error:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="unsupported",
                code="FEATURE_SCHEMA_MISMATCH",
                reason=live_schema_error,
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=None,
                uncertainty=None,
                evidence={"feature_units": {name: row.unit for name, row in feature_rows.items()}},
                request_kind=request_kind,
            )

        required_quality = str(policy["require_feature_quality"])
        bad_quality = [name for name, row in feature_rows.items() if row.quality != required_quality]
        if bad_quality:
            self._persist_feature_quality_monitors(
                organization_id,
                context,
                model_version,
                prediction_time,
                "insufficient_evidence",
                {"non_good_features": bad_quality},
            )
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="insufficient_evidence",
                code="BAD_FEATURE_QUALITY",
                reason="live analytics features have non-good quality states",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=None,
                uncertainty=None,
                evidence={"non_good_features": bad_quality, "required_quality": required_quality},
                request_kind=request_kind,
            )

        feature_vector = {name: float(feature_rows[name].value) for name in dataset.feature_names}
        non_finite_features = [name for name, value in feature_vector.items() if not np.isfinite(value)]
        if non_finite_features:
            self._persist_feature_quality_monitors(
                organization_id,
                context,
                model_version,
                prediction_time,
                "insufficient_evidence",
                {"non_finite_features": non_finite_features},
            )
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="insufficient_evidence",
                code="BAD_FEATURE_QUALITY",
                reason="live analytics features contain non-finite values",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=None,
                uncertainty=None,
                evidence={"non_finite_features": non_finite_features},
                request_kind=request_kind,
            )

        domain = _training_domain(model_version)
        if not domain:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="unsupported",
                code="FEATURE_SCHEMA_MISMATCH",
                reason="model version does not contain immutable training-domain feature evidence",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=feature_vector,
                uncertainty=None,
                evidence={"model_domain": (model_version.provenance or {}).get("model_domain")},
                request_kind=request_kind,
            )
        drifted = self._persist_domain_monitors(
            organization_id,
            context,
            model_version,
            prediction_time,
            feature_vector,
            domain,
        )
        if drifted:
            self.repo.create_retraining_trigger(
                organization_id,
                model_version_id=model_version.id,
                sensor_id=context.sensor.id,
                trigger_kind="feature_domain_drift",
                reason="live feature vector is outside the validated training domain",
                evidence={"drifted_features": drifted, "silent_auto_promotion": False},
            )
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="unsupported",
                code="OUT_OF_TRAINING_DOMAIN",
                reason="live feature vector is outside the validated training domain",
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=feature_vector,
                uncertainty=None,
                evidence={"drifted_features": drifted},
                request_kind=request_kind,
            )

        try:
            model = self.artifact_store.load_verified_model(
                organization_id=organization_id,
                artifact_uri=model_version.artifact_uri,
                expected_sha256=model_version.artifact_sha256,
            )
        except (FileNotFoundError, ValueError) as exc:
            code = _artifact_abstention_code(exc)
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="unsupported",
                code=code,
                reason=str(exc),
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=feature_vector,
                uncertainty=None,
                evidence={"artifact_uri": model_version.artifact_uri, "expected_sha256": model_version.artifact_sha256},
                request_kind=request_kind,
            )

        model_schema_error = _model_schema_error(model, dataset.feature_names)
        if model_schema_error:
            return self._persist_prediction(
                organization_id,
                context,
                observed_at=prediction_time,
                status="unsupported",
                code="FEATURE_SCHEMA_MISMATCH",
                reason=model_schema_error,
                binding=binding,
                model_version=model_version,
                dataset=dataset,
                feature_rows=feature_rows,
                feature_vector=feature_vector,
                uncertainty=None,
                evidence={"model_features": getattr(model, "n_features_in_", None)},
                request_kind=request_kind,
            )

        x = np.asarray([[feature_vector[name] for name in dataset.feature_names]], dtype=float)
        predicted_rul_hours = max(0.0, float(model.predict(x)[0]))
        return self._persist_prediction(
            organization_id,
            context,
            observed_at=prediction_time,
            status="supported",
            code=SUPPORTED_REASON_CODE,
            reason="approved production model, verified artifact, compatible schema, and current in-domain features",
            binding=binding,
            model_version=model_version,
            dataset=dataset,
            feature_rows=feature_rows,
            feature_vector=feature_vector,
            uncertainty=_prediction_uncertainty(model_version, predicted_rul_hours),
            predicted_rul_hours=predicted_rul_hours,
            evidence={
                "training_domain": domain,
                "artifact_verified": True,
                "feature_schema_verified": True,
            },
            request_kind=request_kind,
        )

    def prediction_history(self, organization_id: str, sensor_id: str | None = None) -> list[dict[str, Any]]:
        records = self.repo.list_prediction_records(organization_id, sensor_id=sensor_id)
        return [_prediction_payload(row) for row in records]

    def health(self, organization_id: str) -> dict[str, Any]:
        return {
            "bindings": [_binding_payload(row) for row in self.repo.list_model_serving_bindings(organization_id)],
            "monitors": [_monitor_payload(row) for row in self.repo.list_model_serving_monitors(organization_id)],
            "retraining_triggers": [
                _trigger_payload(row) for row in self.repo.list_retraining_triggers(organization_id)
            ],
        }

    def _sensor_context(self, organization_id: str, sensor_id: str) -> SensorContext:
        sensor = self.repo.get_sensor_by_id(organization_id, sensor_id)
        if sensor is None:
            raise ProductionServingError("sensor does not exist inside this organization")
        component = self.repo.get_component_by_id(organization_id, sensor.component_id)
        if component is None:
            raise ProductionServingError("sensor component is missing")
        asset = self.repo.get_asset_by_id(organization_id, component.asset_id)
        if asset is None:
            raise ProductionServingError("sensor asset is missing")
        return SensorContext(sensor=sensor, component=component, asset=asset, site_id=asset.site_id)

    def _registry_filter(self, organization_id: str, request: PredictionRequest) -> MLModelRegistry | None:
        if request.registry_id:
            registry = self.repo.get_ml_model_registry(organization_id, request.registry_id)
            if registry is None:
                raise ProductionServingError("model registry does not exist inside this organization")
            return registry
        if request.registry_name:
            registry = self.repo.get_ml_model_registry_by_name(organization_id, request.registry_name)
            if registry is None:
                raise ProductionServingError("model registry does not exist inside this organization")
            return registry
        return None

    def _resolve_binding(
        self,
        organization_id: str,
        context: SensorContext,
        *,
        registry_id: str | None,
    ) -> BindingSelection:
        bindings = self.repo.list_active_model_serving_bindings_for_sensor(
            organization_id,
            registry_id=registry_id,
            sensor_id=context.sensor.id,
            component_id=context.component.id,
            asset_id=context.asset.id,
            site_id=context.site_id,
        )
        if not bindings:
            return BindingSelection(
                binding=None,
                code="NO_MODEL_BINDING",
                reason="no active production model binding resolves for this sensor",
                evidence={"registry_filter": registry_id, "resolution_scope": "sensor_hierarchy"},
            )
        rul_bindings = [
            binding
            for binding in bindings
            if (registry := self.repo.get_ml_model_registry(organization_id, binding.registry_id)) is not None
            and registry.task == RUL_TASK
        ]
        if not rul_bindings:
            return BindingSelection(
                binding=None,
                code="NON_RUL_MODEL_BINDING",
                reason="active model bindings for this sensor are not RUL regression registries",
                evidence={"candidate_binding_ids": [binding.id for binding in bindings]},
            )
        winning_specificity = max(SCOPE_SPECIFICITY[binding.scope_type] for binding in rul_bindings)
        winners = [
            binding
            for binding in rul_bindings
            if SCOPE_SPECIFICITY[binding.scope_type] == winning_specificity
        ]
        if registry_id is None and len(winners) > 1:
            return BindingSelection(
                binding=None,
                code="AMBIGUOUS_BINDING",
                reason="multiple equally specific RUL model bindings resolve for this sensor",
                evidence={
                    "candidate_binding_ids": [binding.id for binding in winners],
                    "specificity": winning_specificity,
                },
            )
        return BindingSelection(
            binding=sorted(winners, key=lambda row: (row.created_at, row.id))[-1],
            code=None,
            reason=None,
            evidence={"registry_filter": registry_id, "resolution_scope": "sensor_hierarchy"},
        )

    def _unsupported_binding(
        self,
        organization_id: str,
        context: SensorContext,
        binding: ModelServingBinding,
        observed_at: datetime | None,
        request_kind: str,
        reason: str,
        code: str,
        *,
        model_version: MLModelVersion | None = None,
    ) -> PredictionResponse:
        return self._persist_prediction(
            organization_id,
            context,
            observed_at=observed_at or datetime.now(UTC),
            status="unsupported",
            code=code,
            reason=reason,
            binding=binding,
            model_version=model_version,
            dataset=None,
            feature_rows=None,
            feature_vector=None,
            uncertainty=None,
            evidence={"binding_id": binding.id},
            request_kind=request_kind,
        )

    def _persist_domain_monitors(
        self,
        organization_id: str,
        context: SensorContext,
        model_version: MLModelVersion,
        observed_at: datetime,
        feature_vector: dict[str, float],
        domain: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        drifted = []
        for name, value in feature_vector.items():
            stats = domain[name]
            drift_score = _range_drift_score(value, stats["min"], stats["max"])
            status = "drifted" if drift_score > 0 else "ok"
            self.repo.create_model_serving_monitor(
                organization_id,
                model_version_id=model_version.id,
                sensor_id=context.sensor.id,
                observed_at=observed_at,
                metric_name=f"feature_domain.{name}",
                status=status,
                drift_score=drift_score,
                threshold=0.0,
                evidence={"value": value, **stats},
            )
            if status == "drifted":
                drifted.append({"feature_name": name, "value": value, **stats, "drift_score": drift_score})
        return drifted

    def _persist_feature_quality_monitors(
        self,
        organization_id: str,
        context: SensorContext,
        model_version: MLModelVersion,
        observed_at: datetime,
        status: str,
        evidence: dict[str, Any],
    ) -> None:
        self.repo.create_model_serving_monitor(
            organization_id,
            model_version_id=model_version.id,
            sensor_id=context.sensor.id,
            observed_at=observed_at,
            metric_name="data_quality.feature_vector",
            status=status,
            drift_score=None,
            threshold=None,
            evidence=evidence,
        )

    def _persist_prediction(
        self,
        organization_id: str,
        context: SensorContext,
        *,
        observed_at: datetime,
        status: str,
        code: str,
        reason: str,
        binding: ModelServingBinding | None,
        model_version: MLModelVersion | None,
        dataset: MLDatasetVersion | None,
        feature_rows: dict[str, AnalyticsFeatureRecord] | None,
        feature_vector: dict[str, float] | None,
        uncertainty: dict[str, Any] | None,
        evidence: dict[str, Any],
        request_kind: str,
        predicted_rul_hours: float | None = None,
    ) -> PredictionResponse:
        feature_names = list(dataset.feature_names) if dataset is not None else None
        resolution = self.repo.create_production_model_resolution(
            organization_id,
            binding_id=binding.id if binding else None,
            registry_id=binding.registry_id if binding else None,
            model_version_id=model_version.id if model_version else None,
            dataset_version_id=dataset.id if dataset else None,
            sensor_id=context.sensor.id,
            status="resolved" if status == "supported" else "abstained",
            reason_code=code,
            reason=reason[:255],
            artifact_sha256=model_version.artifact_sha256 if model_version else None,
            feature_schema=feature_names,
            abstention_policy=model_version.abstention_policy if model_version else None,
            evidence=evidence,
        )
        feature_record_ids = None
        if feature_rows:
            feature_record_ids = sorted(row.id for row in feature_rows.values())
        provenance = {
            "serving_slice": "production-slice-10",
            "sensor_id": context.sensor.id,
            "component_id": context.component.id,
            "asset_id": context.asset.id,
            "site_id": context.site_id,
            "binding_id": binding.id if binding else None,
            "model_version_id": model_version.id if model_version else None,
            "dataset_version_id": dataset.id if dataset else None,
            "artifact_sha256": model_version.artifact_sha256 if model_version else None,
            "feature_record_ids": feature_record_ids,
            "abstention_policy": model_version.abstention_policy if model_version else None,
            "request_kind": request_kind,
            "serving_reference_time": observed_at.isoformat(),
            "evidence": evidence,
        }
        prediction = self.repo.create_prediction_record(
            organization_id,
            model_resolution_id=resolution.id,
            registry_id=binding.registry_id if binding else None,
            model_version_id=model_version.id if model_version else None,
            dataset_version_id=dataset.id if dataset else None,
            sensor_id=context.sensor.id,
            observed_at=observed_at,
            prediction_status=status,
            predicted_rul_hours=predicted_rul_hours if status == "supported" else None,
            abstention_code=None if status == "supported" else code,
            uncertainty=uncertainty,
            feature_vector=feature_vector,
            feature_record_ids=feature_record_ids,
            abstention_reason=None if status == "supported" else reason,
            provenance=provenance,
        )
        return PredictionResponse(
            id=prediction.id,
            model_resolution_id=resolution.id,
            organization_id=organization_id,
            sensor_id=context.sensor.id,
            observed_at=observed_at,
            prediction_status=status,
            predicted_rul_hours=prediction.predicted_rul_hours,
            abstention_code=prediction.abstention_code,
            abstention_reason=prediction.abstention_reason,
            request_kind=request_kind,
            registry_id=prediction.registry_id,
            model_version_id=prediction.model_version_id,
            dataset_version_id=prediction.dataset_version_id,
            feature_vector=feature_vector,
            feature_record_ids=feature_record_ids,
            uncertainty=uncertainty,
            model_resolution={
                "id": resolution.id,
                "status": resolution.status,
                "reason_code": resolution.reason_code,
                "reason": resolution.reason,
                "artifact_sha256": resolution.artifact_sha256,
                "feature_schema": resolution.feature_schema,
                "abstention_policy": resolution.abstention_policy,
                "evidence": resolution.evidence,
            },
            provenance=provenance,
        )


def _feature_schema_error(model_version: MLModelVersion, dataset: MLDatasetVersion) -> str | None:
    dataset_features = list(dataset.feature_names or [])
    model_features = (model_version.provenance or {}).get("feature_names")
    model_domain = (model_version.provenance or {}).get("model_domain") or {}
    domain_features = model_domain.get("ordered_feature_names")
    if dataset.target_name != RUL_TARGET_NAME or dataset.target_unit != RUL_TARGET_UNIT:
        return "RUL serving requires a dataset target of RUL_hours with unit h"
    if not dataset_features:
        return "dataset version has no feature schema"
    if model_features is not None and list(model_features) != dataset_features:
        return "model version feature schema does not match dataset version"
    if list(domain_features or []) != dataset_features:
        return "model-domain feature schema does not match dataset version"
    if model_domain.get("analytics_algorithm_version") != dataset.source_algorithm_version:
        return "model-domain analytics algorithm does not match dataset version"
    if model_domain.get("numeric_type") != "float64":
        return "model-domain numeric type is unsupported for serving"
    feature_stats = model_domain.get("feature_stats")
    if not isinstance(feature_stats, dict) or any(name not in feature_stats for name in dataset_features):
        return "model-domain feature statistics do not cover the dataset feature schema"
    return None


def _live_feature_schema_error(
    model_version: MLModelVersion,
    dataset: MLDatasetVersion,
    feature_rows: dict[str, AnalyticsFeatureRecord],
) -> str | None:
    model_domain = (model_version.provenance or {}).get("model_domain") or {}
    feature_units = model_domain.get("feature_units") or {}
    for name in dataset.feature_names:
        row = feature_rows[name]
        if row.algorithm_version != model_domain.get("analytics_algorithm_version"):
            return "live analytics algorithm version does not match model-domain contract"
        if name not in feature_units:
            return "model-domain feature units do not cover the dataset feature schema"
        expected_unit = feature_units[name]
        if expected_unit is None and row.unit is not None:
            return "live feature unit does not match model-domain contract"
        if isinstance(expected_unit, str) and row.unit != expected_unit:
            return "live feature unit does not match model-domain contract"
        if isinstance(expected_unit, list) and row.unit not in expected_unit:
            return "live feature unit does not match model-domain contract"
    return None


def _model_schema_error(model: Any, feature_names: list[str]) -> str | None:
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(feature_names):
        return "model artifact feature count does not match dataset feature schema"
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and list(model_feature_names) != feature_names:
        return "model artifact feature names do not match dataset feature schema"
    return None


def _training_domain(model_version: MLModelVersion) -> dict[str, dict[str, float]]:
    model_domain = (model_version.provenance or {}).get("model_domain") or {}
    feature_stats = model_domain.get("feature_stats") or {}
    if not isinstance(feature_stats, dict):
        return {}
    return feature_stats


def _policy_error(policy: dict[str, Any]) -> str | None:
    unknown = sorted(set(policy).difference(SUPPORTED_SERVING_POLICY_FIELDS))
    if unknown:
        return f"unsupported abstention policy fields: {', '.join(unknown)}"
    if "max_feature_age_minutes" in policy:
        try:
            max_feature_age_minutes = int(policy["max_feature_age_minutes"])
        except (TypeError, ValueError):
            return "abstention policy max_feature_age_minutes must be an integer"
        if max_feature_age_minutes < 1:
            return "abstention policy max_feature_age_minutes must be positive"
    if policy.get("require_feature_quality", "good") != "good":
        return "only require_feature_quality='good' is supported in Slice 10"
    if policy.get("training_domain_behavior", "abstain") != "abstain":
        return "only training_domain_behavior='abstain' is supported in Slice 10"
    if "allow_historical_predictions" in policy and not isinstance(policy["allow_historical_predictions"], bool):
        return "abstention policy allow_historical_predictions must be boolean"
    return None


def _serving_policy(policy: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_SERVING_POLICY)
    for key in DEFAULT_SERVING_POLICY:
        if key in policy:
            merged[key] = policy[key]
    merged["max_feature_age_minutes"] = int(merged["max_feature_age_minutes"])
    return merged


def _policy_contract_error(policy: dict[str, Any], dataset: MLDatasetVersion) -> str | None:
    min_validation_groups = policy.get("min_validation_groups")
    if min_validation_groups is None:
        return None
    required_groups = _strict_integer_policy_value(min_validation_groups)
    if required_groups is None:
        return "abstention policy min_validation_groups must be an integer"
    if required_groups < 1:
        return "abstention policy min_validation_groups must be positive"
    if dataset.validation_group_count < required_groups:
        return "model validation evidence does not satisfy min_validation_groups abstention policy"
    return None


def _strict_integer_policy_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value[1:] if value.startswith("-") else value
        if digits and all("0" <= char <= "9" for char in digits):
            return int(value)
    return None


def _artifact_abstention_code(exc: Exception) -> str:
    message = str(exc)
    if "outside the trusted registry root" in message:
        return "ARTIFACT_OUTSIDE_TRUST_ROOT"
    if "SHA-256" in message:
        return "ARTIFACT_CHECKSUM_FAILED"
    if isinstance(exc, FileNotFoundError):
        return "ARTIFACT_MISSING"
    return "ARTIFACT_LOAD_FAILED"


def _range_drift_score(value: float, minimum: float, maximum: float) -> float:
    if minimum <= value <= maximum:
        return 0.0
    width = max(maximum - minimum, 1.0)
    if value < minimum:
        return float((minimum - value) / width)
    return float((value - maximum) / width)


def _stale_features(
    features: dict[str, AnalyticsFeatureRecord],
    observed_at: datetime,
    max_feature_age_minutes: int,
) -> list[dict[str, Any]]:
    normalized_observed_at = _aware_utc(observed_at)
    stale = []
    for name, row in features.items():
        age_minutes = (_aware_utc(normalized_observed_at) - _aware_utc(row.observed_at)).total_seconds() / 60.0
        if age_minutes > max_feature_age_minutes:
            stale.append(
                {
                    "feature_name": name,
                    "age_minutes": age_minutes,
                    "feature_observed_at": row.observed_at.isoformat(),
                }
            )
    return stale


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _prediction_uncertainty(model_version: MLModelVersion, predicted_rul_hours: float) -> dict[str, Any]:
    uncertainty = dict(model_version.uncertainty or {})
    p10 = uncertainty.get("residual_p10")
    p90 = uncertainty.get("residual_p90")
    if p10 is not None and p90 is not None:
        uncertainty["prediction_interval_80"] = {
            "low": max(0.0, float(predicted_rul_hours) - float(p90)),
            "high": max(0.0, float(predicted_rul_hours) - float(p10)),
            "basis": "cross-group residual quantiles from registered model version",
        }
    uncertainty["calibration_scope"] = uncertainty.get(
        "calibration_scope",
        "registered model uncertainty evidence",
    )
    return uncertainty


def _binding_payload(binding: ModelServingBinding) -> dict[str, Any]:
    return {
        "id": binding.id,
        "registry_id": binding.registry_id,
        "model_version_id": binding.model_version_id,
        "scope_type": binding.scope_type,
        "scope_id": binding.scope_id,
        "status": binding.status,
        "approved_by_user_id": binding.approved_by_user_id,
        "activated_at": binding.activated_at.isoformat() if binding.activated_at else None,
        "reason": binding.reason,
        "provenance": binding.provenance,
    }


def _prediction_payload(prediction) -> dict[str, Any]:
    return {
        "id": prediction.id,
        "model_resolution_id": prediction.model_resolution_id,
        "sensor_id": prediction.sensor_id,
        "observed_at": prediction.observed_at.isoformat(),
        "prediction_status": prediction.prediction_status,
        "predicted_rul_hours": prediction.predicted_rul_hours,
        "abstention_code": prediction.abstention_code,
        "abstention_reason": prediction.abstention_reason,
        "registry_id": prediction.registry_id,
        "model_version_id": prediction.model_version_id,
        "dataset_version_id": prediction.dataset_version_id,
        "feature_record_ids": prediction.feature_record_ids,
        "provenance": prediction.provenance,
    }


def _monitor_payload(monitor) -> dict[str, Any]:
    return {
        "id": monitor.id,
        "model_version_id": monitor.model_version_id,
        "sensor_id": monitor.sensor_id,
        "observed_at": monitor.observed_at.isoformat(),
        "metric_name": monitor.metric_name,
        "status": monitor.status,
        "drift_score": monitor.drift_score,
        "threshold": monitor.threshold,
        "evidence": monitor.evidence,
    }


def _trigger_payload(trigger) -> dict[str, Any]:
    return {
        "id": trigger.id,
        "model_version_id": trigger.model_version_id,
        "sensor_id": trigger.sensor_id,
        "trigger_kind": trigger.trigger_kind,
        "reason": trigger.reason,
        "status": trigger.status,
        "triggered_at": trigger.triggered_at.isoformat() if trigger.triggered_at else None,
        "evidence": trigger.evidence,
    }
