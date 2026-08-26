"""add industrial ingestion tables

Revision ID: 0002_industrial_ingestion
Revises: 0001_platform_core
Create Date: 2026-08-26
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_industrial_ingestion"
down_revision: str | None = "0001_platform_core"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


SOURCE_TYPES = "'csv', 'parquet', 'rest', 'mqtt', 'opcua', 'abb', 'replay'"
QUALITY_STATES = "'good', 'suspect', 'bad', 'missing'"


def upgrade() -> None:
    op.create_table(
        "ingestion_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("external_ref", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.CheckConstraint(f"source_type in ({SOURCE_TYPES})", name="ck_ingestion_source_type"),
        sa.CheckConstraint("status in ('active', 'paused', 'unhealthy')", name="ck_ingestion_source_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ingestion_sources_org_id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_ingestion_sources_org_name"),
        sa.UniqueConstraint("organization_id", "source_type", "external_ref", name="uq_ingestion_sources_org_external"),
    )
    op.create_index("ix_ingestion_sources_organization_id", "ingestion_sources", ["organization_id"])

    op.create_table(
        "ingestion_batches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=120), nullable=True),
        sa.Column("source_uri", sa.String(length=1024), nullable=True),
        sa.Column("replay_of_batch_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_count", sa.Integer(), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("scalar_count", sa.Integer(), nullable=False),
        sa.Column("waveform_count", sa.Integer(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(f"source_type in ({SOURCE_TYPES})", name="ck_batch_source_type"),
        sa.CheckConstraint("status in ('accepted', 'partial', 'failed')", name="ck_ingestion_batch_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "replay_of_batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_ingestion_batches_org_id"),
        sa.UniqueConstraint("organization_id", "source_id", "idempotency_key", name="uq_ingestion_batches_source_key"),
    )
    op.create_index("ix_ingestion_batches_organization_id", "ingestion_batches", ["organization_id"])
    op.create_index("ix_ingestion_batches_source_id", "ingestion_batches", ["source_id"])

    op.create_table(
        "ingested_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=120), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric", sa.String(length=120), nullable=True),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("target_type in ('scalar_reading', 'waveform')", name="ck_ingested_record_target_type"),
        sa.CheckConstraint(f"quality in ({QUALITY_STATES})", name="ck_ingested_record_quality"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "source_id", "idempotency_key", name="uq_ingested_records_source_key"),
    )
    op.create_index("ix_ingested_records_batch_id", "ingested_records", ["batch_id"])
    op.create_index("ix_ingested_records_observed_at", "ingested_records", ["observed_at"])
    op.create_index("ix_ingested_records_organization_id", "ingested_records", ["organization_id"])
    op.create_index("ix_ingested_records_source_id", "ingested_records", ["source_id"])

    op.create_table(
        "ingestion_failures",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=True),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("dead_letter", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.CheckConstraint("quality in ('bad', 'missing', 'suspect')", name="ck_ingestion_failure_quality"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "source_id"],
            ["ingestion_sources.organization_id", "ingestion_sources.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ingestion_failures_batch_id", "ingestion_failures", ["batch_id"])
    op.create_index("ix_ingestion_failures_organization_id", "ingestion_failures", ["organization_id"])
    op.create_index("ix_ingestion_failures_source_id", "ingestion_failures", ["source_id"])

    op.create_table(
        "waveform_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("sampling_rate_hz", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=120), nullable=False),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.Column("storage_uri", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(f"quality in ({QUALITY_STATES})", name="ck_waveform_quality"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "batch_id"],
            ["ingestion_batches.organization_id", "ingestion_batches.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_waveform_records_org_id"),
    )
    op.create_index("ix_waveform_records_batch_id", "waveform_records", ["batch_id"])
    op.create_index("ix_waveform_records_observed_at", "waveform_records", ["observed_at"])
    op.create_index("ix_waveform_records_organization_id", "waveform_records", ["organization_id"])
    op.create_index("ix_waveform_records_sensor_id", "waveform_records", ["sensor_id"])


def downgrade() -> None:
    op.drop_table("waveform_records")
    op.drop_table("ingestion_failures")
    op.drop_table("ingested_records")
    op.drop_table("ingestion_batches")
    op.drop_table("ingestion_sources")
