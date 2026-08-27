"""add production serving tables

Revision ID: 0005_production_serving
Revises: 0004_ml_platform
Create Date: 2026-08-27
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_production_serving"
down_revision: str | None = "0004_ml_platform"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "model_serving_bindings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("registry_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("scope_type", sa.String(length=32), nullable=False),
        sa.Column("scope_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("approved_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "scope_type in ('organization', 'site', 'asset', 'component', 'sensor')",
            name="ck_model_serving_binding_scope",
        ),
        sa.CheckConstraint("status in ('active', 'disabled')", name="ck_model_serving_binding_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["approved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_model_serving_bindings_org_id"),
    )
    op.create_index("ix_model_serving_bindings_approved_by_user_id", "model_serving_bindings", ["approved_by_user_id"])
    op.create_index("ix_model_serving_bindings_model_version_id", "model_serving_bindings", ["model_version_id"])
    op.create_index("ix_model_serving_bindings_organization_id", "model_serving_bindings", ["organization_id"])
    op.create_index("ix_model_serving_bindings_registry_id", "model_serving_bindings", ["registry_id"])
    op.create_index("ix_model_serving_bindings_scope_id", "model_serving_bindings", ["scope_id"])

    op.create_table(
        "production_model_resolutions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("binding_id", sa.String(length=36), nullable=True),
        sa.Column("registry_id", sa.String(length=36), nullable=True),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("feature_schema", sa.JSON(), nullable=True),
        sa.Column("abstention_policy", sa.JSON(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "status in ('resolved', 'abstained', 'failed')",
            name="ck_production_model_resolution_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "binding_id"],
            ["model_serving_bindings.organization_id", "model_serving_bindings.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_production_model_resolutions_org_id"),
    )
    op.create_index("ix_production_model_resolutions_binding_id", "production_model_resolutions", ["binding_id"])
    op.create_index("ix_production_model_resolutions_dataset_version_id", "production_model_resolutions", ["dataset_version_id"])
    op.create_index("ix_production_model_resolutions_model_version_id", "production_model_resolutions", ["model_version_id"])
    op.create_index("ix_production_model_resolutions_organization_id", "production_model_resolutions", ["organization_id"])
    op.create_index("ix_production_model_resolutions_registry_id", "production_model_resolutions", ["registry_id"])
    op.create_index("ix_production_model_resolutions_sensor_id", "production_model_resolutions", ["sensor_id"])

    op.create_table(
        "prediction_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("model_resolution_id", sa.String(length=36), nullable=False),
        sa.Column("registry_id", sa.String(length=36), nullable=True),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_status", sa.String(length=32), nullable=False),
        sa.Column("predicted_rul_hours", sa.Float(), nullable=True),
        sa.Column("abstention_code", sa.String(length=120), nullable=True),
        sa.Column("uncertainty", sa.JSON(), nullable=True),
        sa.Column("feature_vector", sa.JSON(), nullable=True),
        sa.Column("feature_record_ids", sa.JSON(), nullable=True),
        sa.Column("abstention_reason", sa.String(length=1024), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "prediction_status in ('supported', 'unsupported', 'insufficient_evidence')",
            name="ck_prediction_record_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_resolution_id"],
            ["production_model_resolutions.organization_id", "production_model_resolutions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_prediction_records_org_id"),
    )
    op.create_index("ix_prediction_records_dataset_version_id", "prediction_records", ["dataset_version_id"])
    op.create_index("ix_prediction_records_model_resolution_id", "prediction_records", ["model_resolution_id"])
    op.create_index("ix_prediction_records_model_version_id", "prediction_records", ["model_version_id"])
    op.create_index("ix_prediction_records_observed_at", "prediction_records", ["observed_at"])
    op.create_index("ix_prediction_records_organization_id", "prediction_records", ["organization_id"])
    op.create_index("ix_prediction_records_registry_id", "prediction_records", ["registry_id"])
    op.create_index("ix_prediction_records_sensor_id", "prediction_records", ["sensor_id"])

    op.create_table(
        "model_serving_monitors",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric_name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "status in ('ok', 'drifted', 'insufficient_evidence', 'failed')",
            name="ck_model_serving_monitor_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_serving_monitors_model_version_id", "model_serving_monitors", ["model_version_id"])
    op.create_index("ix_model_serving_monitors_observed_at", "model_serving_monitors", ["observed_at"])
    op.create_index("ix_model_serving_monitors_organization_id", "model_serving_monitors", ["organization_id"])
    op.create_index("ix_model_serving_monitors_sensor_id", "model_serving_monitors", ["sensor_id"])

    op.create_table(
        "retraining_triggers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=True),
        sa.Column("trigger_kind", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('open', 'acknowledged', 'resolved')", name="ck_retraining_trigger_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_retraining_triggers_model_version_id", "retraining_triggers", ["model_version_id"])
    op.create_index("ix_retraining_triggers_organization_id", "retraining_triggers", ["organization_id"])
    op.create_index("ix_retraining_triggers_sensor_id", "retraining_triggers", ["sensor_id"])


def downgrade() -> None:
    op.drop_table("retraining_triggers")
    op.drop_table("model_serving_monitors")
    op.drop_table("prediction_records")
    op.drop_table("production_model_resolutions")
    op.drop_table("model_serving_bindings")
