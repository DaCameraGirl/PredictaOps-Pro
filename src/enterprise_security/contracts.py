"""Typed contracts for Production Slice 12 enterprise security."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

PrincipalType = Literal["user", "service", "system", "anonymous"]
AuditOutcome = Literal["allowed", "denied", "failed"]
ASYMMETRIC_OIDC_ALGORITHMS = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"}


class IdentityProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    issuer: str = Field(min_length=1, max_length=512)
    audience: str = Field(min_length=1, max_length=255)
    discovery_url: str | None = Field(default=None, max_length=1024)
    jwks_uri: str = Field(min_length=1, max_length=1024)
    allowed_algorithms: list[str] = Field(default_factory=lambda: ["RS256"], min_length=1)
    claim_mapping: dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_algorithms")
    @classmethod
    def require_asymmetric_jwks_algorithms(cls, value: list[str]) -> list[str]:
        algorithms = sorted(set(value))
        unsupported = sorted(set(algorithms) - ASYMMETRIC_OIDC_ALGORITHMS)
        if unsupported:
            raise ValueError("OIDC JWKS providers support only asymmetric signing algorithms")
        return algorithms


class IdentityProviderUpdate(BaseModel):
    status: Literal["active", "inactive"]


class UserIdentityOnboard(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    full_name: str | None = Field(default=None, max_length=255)
    identity_provider_id: str
    issuer: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=255)
    role: Literal["owner", "admin", "engineer", "technician", "viewer"]
    profile: dict[str, Any] = Field(default_factory=dict)


class UserIdentityCreate(BaseModel):
    user_id: str
    identity_provider_id: str
    issuer: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=255)
    profile: dict[str, Any] = Field(default_factory=dict)


class ServicePrincipalCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    identity_provider_id: str
    external_subject: str = Field(min_length=1, max_length=255)
    issuer: str = Field(min_length=1, max_length=512)
    permissions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServicePrincipalUpdate(BaseModel):
    status: Literal["active", "inactive", "archived"] | None = None
    permissions: list[str] | None = None


class SecretReferenceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    purpose: str = Field(min_length=1, max_length=255)
    provider: str = Field(min_length=1, max_length=120)
    locator: str = Field(min_length=1, max_length=1024)
    rotation_metadata: dict[str, Any] = Field(default_factory=dict)


class SecretReferenceUpdate(BaseModel):
    status: Literal["active", "inactive", "rotating", "archived"] | None = None
    rotation_metadata: dict[str, Any] | None = None
    last_rotated_at: datetime | None = None


class MembershipChange(BaseModel):
    user_id: str
    role: Literal["owner", "admin", "engineer", "technician", "viewer"]


class MembershipStatusChange(BaseModel):
    lifecycle_state: Literal["active", "inactive", "archived"]


class BootstrapSecurityRequest(BaseModel):
    organization_slug: str = Field(min_length=1, max_length=120)
    organization_name: str = Field(min_length=1, max_length=255)
    owner_email: str = Field(min_length=3, max_length=320)
    owner_full_name: str | None = Field(default=None, max_length=255)
    issuer: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=255)
    audience: str = Field(min_length=1, max_length=255)
    jwks_uri: str = Field(min_length=1, max_length=1024)
    idp_name: str = Field(default="primary-oidc", min_length=1, max_length=255)
    allowed_algorithms: list[str] = Field(default_factory=lambda: ["RS256"], min_length=1)

    @field_validator("allowed_algorithms")
    @classmethod
    def require_asymmetric_jwks_algorithms(cls, value: list[str]) -> list[str]:
        algorithms = sorted(set(value))
        unsupported = sorted(set(algorithms) - ASYMMETRIC_OIDC_ALGORITHMS)
        if unsupported:
            raise ValueError("OIDC JWKS providers support only asymmetric signing algorithms")
        return algorithms


class TokenClaims(BaseModel):
    issuer: str
    subject: str
    audience: str | list[str]
    expires_at: int
    not_before: int | None = None
    algorithm: str
    key_id: str | None = None
    profile: dict[str, Any] = Field(default_factory=dict)


class SecurityContext(BaseModel):
    principal_type: PrincipalType
    organization_id: str
    user_id: str | None = None
    service_principal_id: str | None = None
    role: str | None = None
    permissions: set[str] = Field(default_factory=set)
    issuer: str | None = None
    subject: str | None = None
    request_id: str

    def require_user(self) -> str:
        if self.principal_type != "user" or not self.user_id:
            raise PermissionError("human user principal required")
        return self.user_id

