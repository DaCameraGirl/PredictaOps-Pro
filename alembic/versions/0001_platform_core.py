"""create platform core tables

Revision ID: 0001_platform_core
Revises:
Create Date: 2026-08-26
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_platform_core"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def lifecycle_check(table: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(
        "lifecycle_state in ('active', 'inactive', 'archived')",
        name=f"ck_{table}_lifecycle",
    )


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
        lifecycle_check("org"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("external_subject", sa.String(length=255), nullable=True),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("external_subject"),
        lifecycle_check("user"),
    )
    op.create_table(
        "organization_memberships",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        sa.CheckConstraint(
            "role in ('owner', 'admin', 'engineer', 'technician', 'viewer')",
            name="ck_membership_role",
        ),
        lifecycle_check("membership"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "user_id", name="uq_membership_org_user"),
    )
    op.create_index("ix_organization_memberships_organization_id", "organization_memberships", ["organization_id"])
    op.create_index("ix_organization_memberships_user_id", "organization_memberships", ["user_id"])
    op.create_table(
        "sites",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("timezone", sa.String(length=120), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        lifecycle_check("site"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_sites_org_id"),
        sa.UniqueConstraint("organization_id", "slug", name="uq_sites_org_slug"),
    )
    op.create_index("ix_sites_organization_id", "sites", ["organization_id"])
    op.create_table(
        "assets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("site_id", sa.String(length=36), nullable=False),
        sa.Column("parent_asset_id", sa.String(length=36), nullable=True),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("asset_type", sa.String(length=120), nullable=False),
        sa.Column("external_ref", sa.String(length=255), nullable=True),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        lifecycle_check("asset"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "parent_asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(["organization_id", "site_id"], ["sites.organization_id", "sites.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_assets_org_id"),
        sa.UniqueConstraint("organization_id", "site_id", "slug", name="uq_assets_org_site_slug"),
    )
    op.create_index("ix_assets_organization_id", "assets", ["organization_id"])
    op.create_index("ix_assets_site_id", "assets", ["site_id"])
    op.create_table(
        "components",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("asset_id", sa.String(length=36), nullable=False),
        sa.Column("parent_component_id", sa.String(length=36), nullable=True),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("component_type", sa.String(length=120), nullable=False),
        sa.Column("external_ref", sa.String(length=255), nullable=True),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        lifecycle_check("component"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "asset_id"], ["assets.organization_id", "assets.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "parent_component_id"],
            ["components.organization_id", "components.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_components_org_id"),
        sa.UniqueConstraint("organization_id", "asset_id", "slug", name="uq_components_org_asset_slug"),
    )
    op.create_index("ix_components_asset_id", "components", ["asset_id"])
    op.create_index("ix_components_organization_id", "components", ["organization_id"])
    op.create_table(
        "sensors",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("component_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("sensor_type", sa.String(length=120), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("sampling_rate_hz", sa.Float(), nullable=True),
        sa.Column("channel_name", sa.String(length=120), nullable=True),
        sa.Column("axis", sa.String(length=32), nullable=True),
        sa.Column("manufacturer", sa.String(length=120), nullable=True),
        sa.Column("model", sa.String(length=120), nullable=True),
        sa.Column("serial_number", sa.String(length=120), nullable=True),
        sa.Column("external_ref", sa.String(length=255), nullable=True),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        *timestamps(),
        lifecycle_check("sensor"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "component_id"], ["components.organization_id", "components.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_sensors_org_id"),
        sa.UniqueConstraint("organization_id", "component_id", "slug", name="uq_sensors_org_component_slug"),
    )
    op.create_index("ix_sensors_component_id", "sensors", ["component_id"])
    op.create_index("ix_sensors_organization_id", "sensors", ["organization_id"])
    op.create_table(
        "machine_readings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("sensor_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric", sa.String(length=120), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=120), nullable=False),
        sa.Column("quality", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("quality in ('good', 'suspect', 'bad', 'missing')", name="ck_reading_quality"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["organization_id", "sensor_id"], ["sensors.organization_id", "sensors.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_readings_org_id"),
    )
    op.create_index("ix_machine_readings_observed_at", "machine_readings", ["observed_at"])
    op.create_index("ix_machine_readings_organization_id", "machine_readings", ["organization_id"])
    op.create_index("ix_machine_readings_sensor_id", "machine_readings", ["sensor_id"])


def downgrade() -> None:
    op.drop_table("machine_readings")
    op.drop_table("sensors")
    op.drop_table("components")
    op.drop_table("assets")
    op.drop_table("sites")
    op.drop_table("organization_memberships")
    op.drop_table("users")
    op.drop_table("organizations")
