"""add ml platform tables

Revision ID: 0004_ml_platform
Revises: 0003_analytics_pipeline
Create Date: 2026-08-27
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_ml_platform"
down_revision: str | None = "0003_analytics_pipeline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "ml_dataset_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("target_name", sa.String(length=120), nullable=False),
        sa.Column("target_unit", sa.String(length=64), nullable=True),
        sa.Column("feature_names", sa.JSON(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("validation_group_count", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("filters", sa.JSON(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('created', 'archived')", name="ck_ml_dataset_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ml_dataset_versions_org_id"),
        sa.UniqueConstraint("organization_id", "name", "version", name="uq_ml_dataset_name_version"),
    )
    op.create_index("ix_ml_dataset_versions_organization_id", "ml_dataset_versions", ["organization_id"])

    op.create_table(
        "ml_experiment_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("algorithm", sa.String(length=120), nullable=False),
        sa.Column("validation_method", sa.String(length=120), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("code_version", sa.String(length=64), nullable=False),
        sa.Column("training_config", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("baseline_metrics", sa.JSON(), nullable=True),
        sa.Column("uncertainty", sa.JSON(), nullable=True),
        sa.Column("abstention_policy", sa.JSON(), nullable=True),
        sa.Column("artifact_uri", sa.String(length=1024), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('running', 'completed', 'failed')", name="ck_ml_experiment_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ml_experiments_org_id"),
    )
    op.create_index("ix_ml_experiment_runs_dataset_version_id", "ml_experiment_runs", ["dataset_version_id"])
    op.create_index("ix_ml_experiment_runs_organization_id", "ml_experiment_runs", ["organization_id"])

    op.create_table(
        "ml_model_registries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("task", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('active', 'archived')", name="ck_ml_model_registry_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ml_model_registries_org_id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_ml_model_registry_name"),
    )
    op.create_index("ix_ml_model_registries_organization_id", "ml_model_registries", ["organization_id"])

    op.create_table(
        "ml_model_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("registry_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_run_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("approval_status", sa.String(length=32), nullable=False),
        sa.Column("artifact_uri", sa.String(length=1024), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("baseline_metrics", sa.JSON(), nullable=True),
        sa.Column("uncertainty", sa.JSON(), nullable=True),
        sa.Column("abstention_policy", sa.JSON(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column("approved_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "stage in ('candidate', 'validated', 'production', 'archived', 'rejected')",
            name="ck_ml_model_version_stage",
        ),
        sa.CheckConstraint(
            "approval_status in ('not_required', 'pending', 'approved', 'rejected')",
            name="ck_ml_model_version_approval",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["approved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "experiment_run_id"],
            ["ml_experiment_runs.organization_id", "ml_experiment_runs.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "dataset_version_id"],
            ["ml_dataset_versions.organization_id", "ml_dataset_versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ml_model_versions_org_id"),
        sa.UniqueConstraint("organization_id", "registry_id", "version", name="uq_ml_model_registry_version"),
    )
    op.create_index("ix_ml_model_versions_approved_by_user_id", "ml_model_versions", ["approved_by_user_id"])
    op.create_index("ix_ml_model_versions_dataset_version_id", "ml_model_versions", ["dataset_version_id"])
    op.create_index("ix_ml_model_versions_experiment_run_id", "ml_model_versions", ["experiment_run_id"])
    op.create_index("ix_ml_model_versions_organization_id", "ml_model_versions", ["organization_id"])
    op.create_index("ix_ml_model_versions_registry_id", "ml_model_versions", ["registry_id"])

    op.create_table(
        "ml_model_promotion_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("registry_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("from_stage", sa.String(length=32), nullable=False),
        sa.Column("to_stage", sa.String(length=32), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("approved_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("event_metadata", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("action in ('promote', 'rollback')", name="ck_ml_promotion_action"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["approved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_version_id"],
            ["ml_model_versions.organization_id", "ml_model_versions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "registry_id"],
            ["ml_model_registries.organization_id", "ml_model_registries.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ml_model_promotion_events_approved_by_user_id", "ml_model_promotion_events", ["approved_by_user_id"])
    op.create_index("ix_ml_model_promotion_events_model_version_id", "ml_model_promotion_events", ["model_version_id"])
    op.create_index("ix_ml_model_promotion_events_organization_id", "ml_model_promotion_events", ["organization_id"])
    op.create_index("ix_ml_model_promotion_events_registry_id", "ml_model_promotion_events", ["registry_id"])


def downgrade() -> None:
    op.drop_table("ml_model_promotion_events")
    op.drop_table("ml_model_versions")
    op.drop_table("ml_model_registries")
    op.drop_table("ml_experiment_runs")
    op.drop_table("ml_dataset_versions")
