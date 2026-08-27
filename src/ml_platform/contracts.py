"""API and service contracts for Production Slice 9."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelStage = Literal["candidate", "validated", "production", "archived", "rejected"]
SUPPORTED_ALGORITHM = "sklearn.RandomForestRegressor"
SUPPORTED_VALIDATION_METHOD = "leave-one-validation-group-out"
SupportedAlgorithm = Literal["sklearn.RandomForestRegressor"]
SupportedValidationMethod = Literal["leave-one-validation-group-out"]


class DatasetVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=64)
    feature_names: list[str] = Field(min_length=1)
    source_algorithm_version: str = Field(default="analytics-v1", max_length=64)
    target_provenance_key: str = Field(default="target_rul_hours", max_length=120)
    validation_group_provenance_key: str = Field(default="validation_group", max_length=120)
    target_name: str = Field(default="RUL_hours", max_length=120)
    target_unit: str | None = Field(default="h", max_length=64)
    sensor_ids: list[str] | None = None


class ExperimentCreate(BaseModel):
    dataset_version_id: str
    name: str = Field(min_length=1, max_length=255)
    algorithm: SupportedAlgorithm = SUPPORTED_ALGORITHM
    validation_method: SupportedValidationMethod = SUPPORTED_VALIDATION_METHOD
    training_config: dict[str, Any] = Field(default_factory=dict)
    abstention_policy: dict[str, Any] = Field(default_factory=dict)


class RegistryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    task: str = Field(default="rul_regression", max_length=120)
    description: str | None = Field(default=None, max_length=1024)


class ModelVersionCreate(BaseModel):
    registry_id: str
    experiment_run_id: str
    version: str = Field(min_length=1, max_length=64)


class PromoteModelVersion(BaseModel):
    target_stage: ModelStage
    approved_by_user_id: str | None = None
    reason: str | None = Field(default=None, max_length=1024)


class RollbackModelVersion(BaseModel):
    target_model_version_id: str
    approved_by_user_id: str
    reason: str = Field(min_length=1, max_length=1024)
