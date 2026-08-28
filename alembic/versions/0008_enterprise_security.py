"""Add enterprise security identity, secret reference, and audit tables.

Revision ID: 0008_enterprise_security
Revises: 0007_serving_hardening
Create Date: 2026-08-28
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_enterprise_security"
down_revision: str | None = "0007_serving_hardening"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "organization_identity_providers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("issuer", sa.String(length=512), nullable=False),
        sa.Column("audience", sa.String(length=255), nullable=False),
        sa.Column("discovery_url", sa.String(length=1024), nullable=True),
        sa.Column("jwks_uri", sa.String(length=1024), nullable=False),
        sa.Column("allowed_algorithms", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("claim_mapping", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('active', 'inactive')", name="ck_org_idp_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "id", name="uq_org_idp_org_id"),
        sa.UniqueConstraint("organization_id", "issuer", "audience", name="uq_org_idp_org_issuer_audience"),
        sa.UniqueConstraint("organization_id", "name", name="uq_org_idp_org_name"),
    )
    op.create_index("ix_organization_identity_providers_issuer", "organization_identity_providers", ["issuer"])
    op.create_index(
        "ix_organization_identity_providers_organization_id",
        "organization_identity_providers",
        ["organization_id"],
    )

    op.create_table(
        "user_identities",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("identity_provider_id", sa.String(length=36), nullable=False),
        sa.Column("issuer", sa.String(length=512), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("profile", sa.JSON(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(
            ["organization_id", "identity_provider_id"],
            ["organization_identity_providers.organization_id", "organization_identity_providers.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("issuer", "subject", name="uq_user_identity_global_issuer_subject"),
        sa.UniqueConstraint("identity_provider_id", "subject", name="uq_user_identity_provider_subject"),
        sa.UniqueConstraint("organization_id", "issuer", "subject", name="uq_user_identity_org_issuer_subject"),
    )
    for column in ["identity_provider_id", "issuer", "organization_id", "subject", "user_id"]:
        op.create_index(f"ix_user_identities_{column}", "user_identities", [column])

    op.create_table(
        "service_principals",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("external_subject", sa.String(length=255), nullable=False),
        sa.Column("issuer", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("permissions", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        *timestamps(),
        sa.CheckConstraint("status in ('active', 'inactive', 'archived')", name="ck_service_principal_status"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "issuer",
            "external_subject",
            name="uq_service_principal_org_issuer_subject",
        ),
        sa.UniqueConstraint("organization_id", "name", name="uq_service_principal_org_name"),
    )
    for column in ["external_subject", "issuer", "organization_id"]:
        op.create_index(f"ix_service_principals_{column}", "service_principals", [column])

    op.create_table(
        "secret_references",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("purpose", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=120), nullable=False),
        sa.Column("locator", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("rotation_metadata", sa.JSON(), nullable=True),
        sa.Column("last_rotated_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "status in ('active', 'inactive', 'rotating', 'archived')",
            name="ck_secret_reference_status",
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "name", name="uq_secret_reference_org_name"),
    )
    op.create_index("ix_secret_references_created_by_user_id", "secret_references", ["created_by_user_id"])
    op.create_index("ix_secret_references_organization_id", "secret_references", ["organization_id"])

    op.create_table(
        "security_audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("principal_type", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("service_principal_id", sa.String(length=36), nullable=True),
        sa.Column("issuer", sa.String(length=512), nullable=True),
        sa.Column("subject_hash", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=255), nullable=False),
        sa.Column("required_permission", sa.String(length=120), nullable=True),
        sa.Column("resource_type", sa.String(length=120), nullable=True),
        sa.Column("resource_id", sa.String(length=255), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=120), nullable=False),
        sa.Column("http_method", sa.String(length=16), nullable=True),
        sa.Column("http_path", sa.String(length=1024), nullable=True),
        sa.Column("request_metadata", sa.JSON(), nullable=True),
        sa.Column("event_metadata", sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "principal_type in ('user', 'service', 'system', 'anonymous')",
            name="ck_audit_principal_type",
        ),
        sa.CheckConstraint("outcome in ('allowed', 'denied', 'failed')", name="ck_audit_outcome"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["service_principal_id"], ["service_principals.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ["organization_id", "request_id", "service_principal_id", "user_id"]:
        op.create_index(f"ix_security_audit_events_{column}", "security_audit_events", [column])
    op.create_index(
        "ix_security_audit_events_org_occurred_id",
        "security_audit_events",
        ["organization_id", sa.text("occurred_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_security_audit_events_org_occurred_id", table_name="security_audit_events")
    op.drop_table("security_audit_events")
    op.drop_table("secret_references")
    op.drop_table("service_principals")
    op.drop_table("user_identities")
    op.drop_table("organization_identity_providers")
