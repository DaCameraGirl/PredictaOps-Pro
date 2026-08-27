"""add maintenance operations tables

Revision ID: 0006_maintenance_operations
Revises: 0005_production_serving
Create Date: 2026-08-27
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_maintenance_operations"
down_revision: str | None = "0005_production_serving"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "maintenance_alerts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("site_id", sa.String(length=36), nullable=True),
        sa.Column("asset_id", sa.String(length=36), nullable=True),
        sa.Column("component_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=True),
        sa.Column("model_resolution_id", sa.String(length=36), nullable=True),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=120), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("source_reason_code", sa.String(length=120), nullable=True),
        sa.Column("alert_kind", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.String(length=1024), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("recommended_action", sa.String(length=1024), nullable=True),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("evidence_snapshot", sa.JSON(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("acknowledged_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledgement_note", sa.String(length=1024), nullable=True),
        sa.Column("resolved_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disposition", sa.String(length=64), nullable=True),
        sa.Column("disposition_reason", sa.String(length=1024), nullable=True),
        *timestamps(),
        sa.CheckConstraint("severity in ('info', 'watch', 'warning', 'critical')", name="ck_maintenance_alert_severity"),
        sa.CheckConstraint("priority in ('low', 'medium', 'high', 'critical')", name="ck_maintenance_alert_priority"),
        sa.CheckConstraint(
            "status in ('open', 'acknowledged', 'resolved', 'dismissed')",
            name="ck_maintenance_alert_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["acknowledged_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["resolved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id", "site_id"], ["sites.organization_id", "sites.id"]),
        sa.ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "prediction_id"],
            ["prediction_records.organization_id", "prediction_records.id"],
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "model_resolution_id"],
            ["production_model_resolutions.organization_id", "production_model_resolutions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_maintenance_alerts_org_id"),
    )
    for column in [
        "acknowledged_by_user_id",
        "asset_id",
        "component_id",
        "model_resolution_id",
        "organization_id",
        "prediction_id",
        "resolved_by_user_id",
        "sensor_id",
        "site_id",
    ]:
        op.create_index(f"ix_maintenance_alerts_{column}", "maintenance_alerts", [column])

    op.create_table(
        "maintenance_cases",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("alert_id", sa.String(length=36), nullable=True),
        sa.Column("case_number", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.String(length=1024), nullable=True),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=True),
        sa.Column("component_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=True),
        sa.Column("opened_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("owner_user_id", sa.String(length=36), nullable=True),
        sa.Column("assignee_user_id", sa.String(length=36), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recommended_action", sa.String(length=1024), nullable=True),
        sa.Column("history", sa.JSON(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("priority in ('low', 'medium', 'high', 'critical')", name="ck_maintenance_case_priority"),
        sa.CheckConstraint(
            "status in ('open', 'in_progress', 'resolved', 'closed')",
            name="ck_maintenance_case_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["opened_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["assignee_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "alert_id"],
            ["maintenance_alerts.organization_id", "maintenance_alerts.id"],
        ),
        sa.ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_maintenance_cases_org_id"),
        sa.UniqueConstraint("organization_id", "case_number", name="uq_maintenance_cases_org_case_number"),
    )
    for column in [
        "alert_id",
        "asset_id",
        "assignee_user_id",
        "component_id",
        "opened_by_user_id",
        "organization_id",
        "owner_user_id",
        "sensor_id",
    ]:
        op.create_index(f"ix_maintenance_cases_{column}", "maintenance_cases", [column])

    op.create_table(
        "maintenance_notes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("author_user_id", sa.String(length=36), nullable=False),
        sa.Column("body", sa.String(length=2048), nullable=False),
        sa.Column("note_kind", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["author_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_maintenance_notes_author_user_id", "maintenance_notes", ["author_user_id"])
    op.create_index("ix_maintenance_notes_case_id", "maintenance_notes", ["case_id"])
    op.create_index("ix_maintenance_notes_organization_id", "maintenance_notes", ["organization_id"])

    op.create_table(
        "maintenance_inspections",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=True),
        sa.Column("component_id", sa.String(length=36), nullable=True),
        sa.Column("sensor_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_reason", sa.String(length=1024), nullable=False),
        sa.Column("requested_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("assigned_to_user_id", sa.String(length=36), nullable=True),
        sa.Column("performed_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("condition", sa.String(length=32), nullable=True),
        sa.Column("findings", sa.String(length=2048), nullable=True),
        sa.Column("recommended_follow_up", sa.String(length=1024), nullable=True),
        sa.Column("evidence_metadata", sa.JSON(), nullable=True),
        sa.Column("inspected_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("inspected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observation", sa.String(length=2048), nullable=True),
        sa.Column("measurements", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "status in ('requested', 'in_progress', 'completed', 'cancelled')",
            name="ck_inspection_status",
        ),
        sa.CheckConstraint(
            "condition in ('normal', 'watch', 'degraded', 'failed', 'unknown')",
            name="ck_inspection_condition",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["assigned_to_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["performed_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["inspected_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        sa.ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in [
        "assigned_to_user_id",
        "case_id",
        "component_id",
        "inspected_by_user_id",
        "organization_id",
        "performed_by_user_id",
        "requested_by_user_id",
        "sensor_id",
    ]:
        op.create_index(f"ix_maintenance_inspections_{column}", "maintenance_inspections", [column])

    op.create_table(
        "maintenance_work_orders",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=True),
        sa.Column("component_id", sa.String(length=36), nullable=True),
        sa.Column("work_order_number", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=2048), nullable=True),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("requested_work", sa.String(length=2048), nullable=False),
        sa.Column("summary", sa.String(length=255), nullable=False),
        sa.Column("requested_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("approved_by_user_id", sa.String(length=36), nullable=True),
        sa.Column("assignee_user_id", sa.String(length=36), nullable=True),
        sa.Column("planned_start_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completion_notes", sa.String(length=2048), nullable=True),
        sa.Column("cmms_provider", sa.String(length=120), nullable=True),
        sa.Column("cmms_external_id", sa.String(length=255), nullable=True),
        sa.Column("cmms_state", sa.String(length=120), nullable=True),
        sa.Column("work_performed", sa.String(length=2048), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "status in ('draft', 'approved', 'in_progress', 'completed', 'cancelled')",
            name="ck_work_order_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["approved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["assignee_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        sa.ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_maintenance_work_orders_org_id"),
        sa.UniqueConstraint("organization_id", "work_order_number", name="uq_work_orders_org_number"),
    )
    for column in [
        "approved_by_user_id",
        "asset_id",
        "assignee_user_id",
        "case_id",
        "component_id",
        "organization_id",
        "requested_by_user_id",
    ]:
        op.create_index(f"ix_maintenance_work_orders_{column}", "maintenance_work_orders", [column])

    op.create_table(
        "cmms_sync_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("work_order_id", sa.String(length=36), nullable=False),
        sa.Column("provider_name", sa.String(length=120), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("error_category", sa.String(length=120), nullable=True),
        sa.Column("error_message", sa.String(length=1024), nullable=True),
        sa.Column("attempted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_metadata", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "operation in ('create', 'update', 'cancel', 'close')",
            name="ck_cmms_sync_operation",
        ),
        sa.CheckConstraint(
            "status in ('not_configured', 'succeeded', 'failed', 'timeout', 'skipped')",
            name="ck_cmms_sync_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "work_order_id"],
            ["maintenance_work_orders.organization_id", "maintenance_work_orders.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_cmms_sync_records_org_id"),
    )
    op.create_index("ix_cmms_sync_records_idempotency_key", "cmms_sync_records", ["idempotency_key"])
    op.create_index("ix_cmms_sync_records_organization_id", "cmms_sync_records", ["organization_id"])
    op.create_index("ix_cmms_sync_records_work_order_id", "cmms_sync_records", ["work_order_id"])

    op.create_table(
        "maintenance_acknowledgements",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("acknowledged_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("comment", sa.String(length=1024), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        *timestamps(),
        sa.CheckConstraint("decision in ('accepted', 'deferred', 'dismissed')", name="ck_maintenance_ack_decision"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["acknowledged_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_maintenance_acknowledgements_acknowledged_by_user_id",
        "maintenance_acknowledgements",
        ["acknowledged_by_user_id"],
    )
    op.create_index("ix_maintenance_acknowledgements_case_id", "maintenance_acknowledgements", ["case_id"])
    op.create_index(
        "ix_maintenance_acknowledgements_organization_id",
        "maintenance_acknowledgements",
        ["organization_id"],
    )

    op.create_table(
        "maintenance_resolutions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("resolved_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("summary", sa.String(length=2048), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "outcome in ('confirmed', 'not_found', 'monitor', 'repaired', 'replaced')",
            name="ck_resolution_outcome",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["resolved_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "case_id"],
            ["maintenance_cases.organization_id", "maintenance_cases.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_maintenance_resolutions_case_id", "maintenance_resolutions", ["case_id"])
    op.create_index("ix_maintenance_resolutions_organization_id", "maintenance_resolutions", ["organization_id"])
    op.create_index(
        "ix_maintenance_resolutions_resolved_by_user_id",
        "maintenance_resolutions",
        ["resolved_by_user_id"],
    )


def downgrade() -> None:
    op.drop_table("maintenance_resolutions")
    op.drop_table("maintenance_acknowledgements")
    op.drop_table("cmms_sync_records")
    op.drop_table("maintenance_work_orders")
    op.drop_table("maintenance_inspections")
    op.drop_table("maintenance_notes")
    op.drop_table("maintenance_cases")
    op.drop_table("maintenance_alerts")
