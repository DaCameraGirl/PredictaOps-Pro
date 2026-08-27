"""add analytics pipeline tables

Revision ID: 0003_analytics_pipeline
Revises: 0002_industrial_ingestion
Create Date: 2026-08-27
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_analytics_pipeline"
down_revision: str | None = "0002_industrial_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


QUALITY_STATES = "'good', 'suspect', 'bad', 'missing'"
HEALTH_STATES = "'insufficient_evidence', 'healthy', 'watch', 'warning', 'critical', 'unknown'"


def upgrade() -> None:
    op.create_table(
        "analytics_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("input_batch_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=True),
        sa.Column("run_kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("health_state_count", sa.Integer(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("run_kind in ('batch', 'sensor')", name="ck_analytics_run_kind"),
        sa.CheckConstraint("status in ('running', 'completed', 'partial', 'failed')", name="ck_analytics_run_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "input_batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_analytics_runs_org_id"),
    )
    op.create_index("ix_analytics_runs_input_batch_id", "analytics_runs", ["input_batch_id"])
    op.create_index("ix_analytics_runs_organization_id", "analytics_runs", ["organization_id"])
    op.create_index("ix_analytics_runs_sensor_id", "analytics_runs", ["sensor_id"])

    op.create_table(
        "analytics_feature_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=True),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_record_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_name", sa.String(length=120), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("source_kind in ('scalar', 'waveform')", name="ck_analytics_feature_source_kind"),
        sa.CheckConstraint(f"quality in ({QUALITY_STATES})", name="ck_analytics_feature_quality"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_analytics_features_org_id"),
        sa.UniqueConstraint(
            "organization_id",
            "algorithm_version",
            "source_kind",
            "source_record_id",
            "feature_name",
            name="uq_analytics_feature_source",
        ),
    )
    op.create_index("ix_analytics_feature_records_batch_id", "analytics_feature_records", ["batch_id"])
    op.create_index("ix_analytics_feature_records_observed_at", "analytics_feature_records", ["observed_at"])
    op.create_index("ix_analytics_feature_records_organization_id", "analytics_feature_records", ["organization_id"])
    op.create_index("ix_analytics_feature_records_run_id", "analytics_feature_records", ["run_id"])
    op.create_index("ix_analytics_feature_records_sensor_id", "analytics_feature_records", ["sensor_id"])
    op.create_index("ix_analytics_feature_records_source_record_id", "analytics_feature_records", ["source_record_id"])

    op.create_table(
        "analytics_health_states",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("health_state", sa.String(length=32), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("trend_slope", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(f"health_state in ({HEALTH_STATES})", name="ck_analytics_health_state"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_analytics_health_states_org_id"),
        sa.UniqueConstraint(
            "organization_id",
            "algorithm_version",
            "sensor_id",
            "observed_at",
            name="uq_analytics_health_state_sensor_time",
        ),
    )
    op.create_index("ix_analytics_health_states_observed_at", "analytics_health_states", ["observed_at"])
    op.create_index("ix_analytics_health_states_organization_id", "analytics_health_states", ["organization_id"])
    op.create_index("ix_analytics_health_states_run_id", "analytics_health_states", ["run_id"])
    op.create_index("ix_analytics_health_states_sensor_id", "analytics_health_states", ["sensor_id"])

    op.create_table(
        "analytics_failures",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=True),
        sa.Column("batch_id", sa.String(length=36), nullable=True),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_record_id", sa.String(length=36), nullable=True),
        sa.Column("reason", sa.String(length=255), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("dead_letter", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.CheckConstraint("source_kind in ('scalar', 'waveform')", name="ck_analytics_failure_source_kind"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "run_id"], ["analytics_runs.organization_id", "analytics_runs.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analytics_failures_batch_id", "analytics_failures", ["batch_id"])
    op.create_index("ix_analytics_failures_organization_id", "analytics_failures", ["organization_id"])
    op.create_index("ix_analytics_failures_run_id", "analytics_failures", ["run_id"])
    op.create_index("ix_analytics_failures_sensor_id", "analytics_failures", ["sensor_id"])
    op.create_index("ix_analytics_failures_source_record_id", "analytics_failures", ["source_record_id"])


def downgrade() -> None:
    op.drop_table("analytics_failures")
    op.drop_table("analytics_health_states")
    op.drop_table("analytics_feature_records")
    op.drop_table("analytics_runs")
