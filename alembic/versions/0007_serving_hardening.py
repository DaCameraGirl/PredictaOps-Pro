"""add serving hardening indexes

Revision ID: 0007_serving_hardening
Revises: 0006_maintenance_operations
Create Date: 2026-08-28
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_serving_hardening"
down_revision: str | None = "0006_maintenance_operations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_active_model_serving_binding_scope_id",
        "model_serving_bindings",
        ["organization_id", "registry_id", "scope_type", "scope_id"],
        unique=True,
        sqlite_where=sa.text("status = 'active' AND scope_id IS NOT NULL"),
        postgresql_where=sa.text("status = 'active' AND scope_id IS NOT NULL"),
    )
    op.create_index(
        "uq_active_model_serving_binding_org_scope",
        "model_serving_bindings",
        ["organization_id", "registry_id"],
        unique=True,
        sqlite_where=sa.text("status = 'active' AND scope_type = 'organization' AND scope_id IS NULL"),
        postgresql_where=sa.text("status = 'active' AND scope_type = 'organization' AND scope_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_active_model_serving_binding_org_scope", table_name="model_serving_bindings")
    op.drop_index("uq_active_model_serving_binding_scope_id", table_name="model_serving_bindings")
