"""Enterprise authentication, authorization, secret-reference, and audit services."""

from __future__ import annotations

import hashlib
import ipaddress
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt
from jwt import PyJWK
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from enterprise_security.contracts import (
    BootstrapSecurityRequest,
    IdentityProviderCreate,
    MembershipChange,
    MembershipStatusChange,
    SecretReferenceCreate,
    SecretReferenceUpdate,
    SecurityContext,
    ServicePrincipalCreate,
    ServicePrincipalUpdate,
    TokenClaims,
    UserIdentityCreate,
)
from enterprise_security.permissions import (
    OWNERS_MANAGE,
    permissions_for_role,
    validate_service_permissions,
)
from enterprise_security.redaction import redact_value
from platform_core.contracts import OrganizationCreate, UserCreate
from platform_core.models import (
    Organization,
    OrganizationIdentityProvider,
    OrganizationMembership,
    SecretReference,
    SecurityAuditEvent,
    ServicePrincipal,
    User,
    UserIdentity,
)
from platform_core.repositories import PlatformRepository

SAFE_AUTH_ERROR = "invalid or missing authentication"
SAFE_FORBIDDEN_ERROR = "not authorized for this organization"


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


class SecurityConfigurationError(ValueError):
    pass


class SecretResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class SecuritySettings:
    mode: str
    environment: str
    cors_allowed_origins: tuple[str, ...]
    test_auth_enabled: bool
    oidc_http_timeout_seconds: float
    docs_enabled: bool


def security_settings() -> SecuritySettings:
    environment = os.environ.get("PMS_ENVIRONMENT", "development").lower()
    mode = os.environ.get("PMS_SECURITY_MODE", "disabled").lower()
    origins = tuple(
        origin.strip()
        for origin in os.environ.get("PMS_CORS_ALLOWED_ORIGINS", "http://localhost:8000").split(",")
        if origin.strip()
    )
    settings = SecuritySettings(
        mode=mode,
        environment=environment,
        cors_allowed_origins=origins,
        test_auth_enabled=os.environ.get("PMS_TEST_AUTH", "0") == "1",
        oidc_http_timeout_seconds=float(os.environ.get("PMS_OIDC_HTTP_TIMEOUT_SECONDS", "2.0")),
        docs_enabled=os.environ.get("PMS_ENABLE_DOCS", "1") == "1",
    )
    validate_security_settings(settings)
    return settings


def validate_security_settings(settings: SecuritySettings) -> None:
    if settings.environment == "production":
        if "*" in settings.cors_allowed_origins:
            raise SecurityConfigurationError("wildcard CORS is not allowed in production")
        if settings.test_auth_enabled or settings.mode == "test":
            raise SecurityConfigurationError("test authentication mode is not allowed in production")
        if settings.mode == "enterprise" and not settings.cors_allowed_origins:
            raise SecurityConfigurationError("production enterprise security requires explicit allowed origins")
        if settings.docs_enabled:
            raise SecurityConfigurationError("OpenAPI docs must be explicitly disabled or protected in production")


def validate_oidc_endpoint(url: str, *, allow_development_targets: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" and not allow_development_targets:
        raise SecurityConfigurationError("OIDC endpoints must use HTTPS outside development/test mode")
    host = parsed.hostname
    if not host:
        raise SecurityConfigurationError("OIDC endpoint must include a host")
    if host in {"localhost", "127.0.0.1", "::1"} and not allow_development_targets:
        raise SecurityConfigurationError("OIDC endpoint must not target loopback hosts outside development/test mode")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not allow_development_targets and (
        address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_reserved
        or address.is_multicast
    ):
        raise SecurityConfigurationError("OIDC endpoint must not target private or unsafe addresses")


class OidcTokenVerifier:
    def __init__(self, *, http_timeout_seconds: float = 2.0):
        self.http_timeout_seconds = http_timeout_seconds
        self._jwks_cache: dict[str, dict[str, Any]] = {}

    def verify(self, token: str, idp: OrganizationIdentityProvider) -> TokenClaims:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        algorithm = str(header.get("alg") or "")
        if algorithm.lower() == "none" or algorithm not in set(idp.allowed_algorithms or []):
            raise AuthenticationError(SAFE_AUTH_ERROR)
        key_id = header.get("kid")
        key = self._key_for_token(idp.jwks_uri, key_id)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(idp.allowed_algorithms or []),
                audience=idp.audience,
                issuer=idp.issuer,
                options={"require": ["exp", "iss", "sub", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        subject = claims.get("sub")
        if not subject:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        profile = {
            key: value
            for key, value in claims.items()
            if key not in {"iss", "sub", "aud", "exp", "nbf", "iat", "jti"}
        }
        return TokenClaims(
            issuer=claims["iss"],
            subject=subject,
            audience=claims["aud"],
            expires_at=claims["exp"],
            not_before=claims.get("nbf"),
            algorithm=algorithm,
            key_id=key_id,
            profile=redact_value(profile),
        )

    def _key_for_token(self, jwks_uri: str, key_id: str | None):
        jwks = self._load_jwks(jwks_uri, refresh=False)
        key = _select_jwk(jwks, key_id)
        if key is None:
            jwks = self._load_jwks(jwks_uri, refresh=True)
            key = _select_jwk(jwks, key_id)
        if key is None:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        try:
            return PyJWK.from_dict(key).key
        except Exception as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc

    def _load_jwks(self, jwks_uri: str, *, refresh: bool) -> dict[str, Any]:
        if not refresh and jwks_uri in self._jwks_cache:
            return self._jwks_cache[jwks_uri]
        validate_oidc_endpoint(jwks_uri, allow_development_targets=os.environ.get("PMS_ENVIRONMENT") != "production")
        try:
            with httpx.Client(timeout=self.http_timeout_seconds, follow_redirects=False) as client:
                response = client.get(jwks_uri)
                response.raise_for_status()
                jwks = response.json()
        except Exception as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise AuthenticationError(SAFE_AUTH_ERROR)
        self._jwks_cache[jwks_uri] = jwks
        return jwks


class DeterministicTokenVerifier:
    """Test-only verifier keyed by bearer-token strings."""

    def __init__(self, tokens: dict[str, TokenClaims]):
        self.tokens = tokens

    def verify(self, token: str, idp: OrganizationIdentityProvider) -> TokenClaims:
        claims = self.tokens.get(token)
        if claims is None or claims.issuer != idp.issuer:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        audience = claims.audience if isinstance(claims.audience, list) else [claims.audience]
        if idp.audience not in audience:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        if claims.algorithm not in set(idp.allowed_algorithms or []):
            raise AuthenticationError(SAFE_AUTH_ERROR)
        return claims


def _select_jwk(jwks: dict[str, Any], key_id: str | None) -> dict[str, Any] | None:
    keys = jwks.get("keys") or []
    if key_id:
        return next((key for key in keys if key.get("kid") == key_id), None)
    return keys[0] if len(keys) == 1 else None


class SecretResolver:
    def resolve(self, reference: SecretReference) -> str:
        raise SecretResolutionError("secret resolver is not configured")


class EnvironmentSecretResolver(SecretResolver):
    def resolve(self, reference: SecretReference) -> str:
        if reference.provider != "env":
            raise SecretResolutionError("secret provider is not configured")
        value = os.environ.get(reference.locator)
        if value is None:
            raise SecretResolutionError("secret is not available")
        return value


class InMemorySecretResolver(SecretResolver):
    def __init__(self, values: dict[str, str]):
        self.values = values

    def resolve(self, reference: SecretReference) -> str:
        value = self.values.get(reference.locator)
        if value is None:
            raise SecretResolutionError("secret is not available")
        return value


class SecurityService:
    def __init__(self, session: Session, *, verifier: OidcTokenVerifier | DeterministicTokenVerifier | None = None):
        self.session = session
        self.repo = PlatformRepository(session)
        self.verifier = verifier or OidcTokenVerifier(
            http_timeout_seconds=security_settings().oidc_http_timeout_seconds
        )

    def create_identity_provider(
        self,
        organization_id: str,
        request: IdentityProviderCreate,
        *,
        allow_development_targets: bool = False,
    ) -> OrganizationIdentityProvider:
        self._require_org(organization_id)
        validate_oidc_endpoint(request.jwks_uri, allow_development_targets=allow_development_targets)
        if request.discovery_url:
            validate_oidc_endpoint(request.discovery_url, allow_development_targets=allow_development_targets)
        if "none" in {algorithm.lower() for algorithm in request.allowed_algorithms}:
            raise SecurityConfigurationError("OIDC alg=none is not supported")
        idp = OrganizationIdentityProvider(
            organization_id=organization_id,
            name=request.name,
            issuer=request.issuer,
            audience=request.audience,
            discovery_url=request.discovery_url,
            jwks_uri=request.jwks_uri,
            allowed_algorithms=sorted(set(request.allowed_algorithms)),
            status="active",
            claim_mapping=request.claim_mapping,
        )
        self.session.add(idp)
        self.session.flush()
        return idp

    def create_user_identity(self, organization_id: str, request: UserIdentityCreate) -> UserIdentity:
        idp = self.get_identity_provider(organization_id, request.identity_provider_id)
        if idp is None:
            raise AuthorizationError("identity provider does not belong to this organization")
        if idp.issuer != request.issuer:
            raise SecurityConfigurationError("identity issuer must match the identity provider")
        if self.session.get(User, request.user_id) is None:
            raise AuthorizationError("user does not exist")
        identity = UserIdentity(
            organization_id=organization_id,
            user_id=request.user_id,
            identity_provider_id=request.identity_provider_id,
            issuer=request.issuer,
            subject=request.subject,
            profile=redact_value(request.profile),
        )
        self.session.add(identity)
        self.session.flush()
        return identity

    def create_service_principal(self, organization_id: str, request: ServicePrincipalCreate) -> ServicePrincipal:
        self._require_org(organization_id)
        permissions = validate_service_permissions(request.permissions)
        principal = ServicePrincipal(
            organization_id=organization_id,
            name=request.name,
            external_subject=request.external_subject,
            issuer=request.issuer,
            permissions=permissions,
            status="active",
            metadata_json=redact_value(request.metadata),
        )
        self.session.add(principal)
        self.session.flush()
        return principal

    def update_service_principal(
        self,
        organization_id: str,
        principal_id: str,
        request: ServicePrincipalUpdate,
    ) -> ServicePrincipal:
        principal = self.get_service_principal(organization_id, principal_id)
        if principal is None:
            raise AuthorizationError("service principal does not belong to this organization")
        if request.status:
            principal.status = request.status
        if request.permissions is not None:
            principal.permissions = validate_service_permissions(request.permissions)
        self.session.flush()
        return principal

    def create_secret_reference(
        self,
        organization_id: str,
        request: SecretReferenceCreate,
        *,
        created_by_user_id: str,
    ) -> SecretReference:
        if self.repo.get_active_membership(organization_id, created_by_user_id) is None:
            raise AuthorizationError("secret reference creator must be an active organization member")
        secret = SecretReference(
            organization_id=organization_id,
            name=request.name,
            purpose=request.purpose,
            provider=request.provider,
            locator=request.locator,
            status="active",
            created_by_user_id=created_by_user_id,
            rotation_metadata=redact_value(request.rotation_metadata),
        )
        self.session.add(secret)
        self.session.flush()
        return secret

    def update_secret_reference(
        self,
        organization_id: str,
        secret_id: str,
        request: SecretReferenceUpdate,
    ) -> SecretReference:
        secret = self.get_secret_reference(organization_id, secret_id)
        if secret is None:
            raise AuthorizationError("secret reference does not belong to this organization")
        if request.status:
            secret.status = request.status
        if request.rotation_metadata is not None:
            secret.rotation_metadata = redact_value(request.rotation_metadata)
        if request.last_rotated_at is not None:
            secret.last_rotated_at = request.last_rotated_at
        self.session.flush()
        return secret

    def bootstrap_initial_owner(self, request: BootstrapSecurityRequest) -> dict[str, str]:
        org = self.repo.get_or_create_organization(
            OrganizationCreate(slug=request.organization_slug, name=request.organization_name)
        )
        user = self.session.scalar(select(User).where(User.email == request.owner_email.lower()))
        if user is None:
            user = self.repo.create_user(
                UserCreate(
                    email=request.owner_email,
                    full_name=request.owner_full_name,
                    external_subject=f"{request.issuer}#{request.subject}",
                )
            )
        membership = self.repo.get_active_membership(org.id, user.id)
        if membership is None:
            existing = self.session.scalar(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == org.id,
                    OrganizationMembership.user_id == user.id,
                )
            )
            if existing is None:
                membership = self.repo.add_membership(org.id, user.id, "owner")
            else:
                existing.lifecycle_state = "active"
                existing.role = "owner"
                membership = existing
        elif membership.role != "owner":
            membership.role = "owner"

        idp = self.session.scalar(
            select(OrganizationIdentityProvider).where(
                OrganizationIdentityProvider.organization_id == org.id,
                OrganizationIdentityProvider.issuer == request.issuer,
                OrganizationIdentityProvider.audience == request.audience,
            )
        )
        if idp is None:
            idp = self.create_identity_provider(
                org.id,
                IdentityProviderCreate(
                    name=request.idp_name,
                    issuer=request.issuer,
                    audience=request.audience,
                    jwks_uri=request.jwks_uri,
                    allowed_algorithms=request.allowed_algorithms,
                ),
                allow_development_targets=True,
            )
        identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == request.issuer, UserIdentity.subject == request.subject)
        )
        if identity is None:
            identity = self.create_user_identity(
                org.id,
                UserIdentityCreate(
                    user_id=user.id,
                    identity_provider_id=idp.id,
                    issuer=request.issuer,
                    subject=request.subject,
                ),
            )
        self.session.flush()
        return {
            "organization_id": org.id,
            "user_id": user.id,
            "membership_id": membership.id,
            "identity_provider_id": idp.id,
            "user_identity_id": identity.id,
        }

    def authenticate_bearer(self, organization_id: str, token: str, *, request_id: str) -> SecurityContext:
        idps = self.list_active_identity_providers(organization_id)
        last_error: Exception | None = None
        for idp in idps:
            try:
                claims = self.verifier.verify(token, idp)
            except AuthenticationError as exc:
                last_error = exc
                continue
            return self.context_from_claims(organization_id, claims, request_id=request_id)
        raise AuthenticationError(SAFE_AUTH_ERROR) from last_error

    def context_from_claims(self, organization_id: str, claims: TokenClaims, *, request_id: str) -> SecurityContext:
        identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == claims.issuer, UserIdentity.subject == claims.subject)
        )
        if identity is not None:
            if identity.organization_id != organization_id:
                raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
            membership = self.repo.get_active_membership(organization_id, identity.user_id)
            if membership is None:
                raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
            identity.last_seen_at = datetime.now(UTC)
            permissions = permissions_for_role(membership.role)
            return SecurityContext(
                principal_type="user",
                organization_id=organization_id,
                user_id=identity.user_id,
                role=membership.role,
                permissions=permissions,
                issuer=claims.issuer,
                subject=claims.subject,
                request_id=request_id,
            )
        service = self.session.scalar(
            select(ServicePrincipal).where(
                ServicePrincipal.organization_id == organization_id,
                ServicePrincipal.external_subject == claims.subject,
                ServicePrincipal.status == "active",
            )
        )
        if service is None or (service.issuer is not None and service.issuer != claims.issuer):
            raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
        return SecurityContext(
            principal_type="service",
            organization_id=organization_id,
            service_principal_id=service.id,
            permissions=set(service.permissions or []),
            issuer=claims.issuer,
            subject=claims.subject,
            request_id=request_id,
        )

    def require_permission(
        self,
        context: SecurityContext,
        permission: str,
        *,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        http_method: str | None = None,
        http_path: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> None:
        if permission not in context.permissions:
            self.record_audit_event(
                context=context,
                action=action,
                required_permission=permission,
                resource_type=resource_type,
                resource_id=resource_id,
                outcome="denied",
                reason_code="missing_permission",
                http_method=http_method,
                http_path=http_path,
                request_metadata=request_metadata,
            )
            raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
        self.record_audit_event(
            context=context,
            action=action,
            required_permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome="allowed",
            reason_code="permission_granted",
            http_method=http_method,
            http_path=http_path,
            request_metadata=request_metadata,
        )

    def change_membership_role(
        self,
        organization_id: str,
        request: MembershipChange,
        *,
        actor: SecurityContext,
    ) -> OrganizationMembership:
        self.require_permission(
            actor,
            OWNERS_MANAGE if request.role == "owner" else "members.manage",
            action="members.role",
        )
        membership = self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == request.user_id,
            )
        )
        if membership is None:
            membership = self.repo.add_membership(organization_id, request.user_id, request.role)
        elif membership.role == "owner" and request.role != "owner" and OWNERS_MANAGE not in actor.permissions:
            raise AuthorizationError("only owners may demote owners")
        else:
            membership.role = request.role
            membership.lifecycle_state = "active"
        self.session.flush()
        return membership

    def change_membership_status(
        self,
        organization_id: str,
        user_id: str,
        request: MembershipStatusChange,
        *,
        actor: SecurityContext,
    ) -> OrganizationMembership:
        membership = self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == user_id,
            )
        )
        if membership is None:
            raise AuthorizationError("membership does not exist")
        permission = OWNERS_MANAGE if membership.role == "owner" else "members.manage"
        self.require_permission(actor, permission, action="members.status")
        if membership.role == "owner" and request.lifecycle_state != "active":
            active_owner_count = self.session.scalar(
                select(func.count()).select_from(OrganizationMembership).where(
                    OrganizationMembership.organization_id == organization_id,
                    OrganizationMembership.role == "owner",
                    OrganizationMembership.lifecycle_state == "active",
                )
            )
            if int(active_owner_count or 0) <= 1:
                raise AuthorizationError("cannot deactivate the final active owner")
        membership.lifecycle_state = request.lifecycle_state
        self.session.flush()
        return membership

    def record_audit_event(
        self,
        *,
        context: SecurityContext | None,
        action: str,
        required_permission: str | None,
        resource_type: str | None,
        resource_id: str | None,
        outcome: str,
        reason_code: str,
        http_method: str | None = None,
        http_path: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        event_metadata: dict[str, Any] | None = None,
    ) -> SecurityAuditEvent:
        event = SecurityAuditEvent(
            organization_id=context.organization_id if context else None,
            request_id=context.request_id if context else "unknown",
            principal_type=context.principal_type if context else "anonymous",
            user_id=context.user_id if context else None,
            service_principal_id=context.service_principal_id if context else None,
            issuer=context.issuer if context else None,
            subject_hash=_subject_hash(context.subject if context else None),
            action=action,
            required_permission=required_permission,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            reason_code=reason_code,
            http_method=http_method,
            http_path=http_path,
            request_metadata=redact_value(request_metadata or {}),
            event_metadata=redact_value(event_metadata or {}),
        )
        self.session.add(event)
        self.session.flush()
        return event

    def list_audit_events(self, organization_id: str, *, limit: int = 100, offset: int = 0) -> list[SecurityAuditEvent]:
        bounded_limit = max(1, min(limit, 500))
        statement = (
            select(SecurityAuditEvent)
            .where(SecurityAuditEvent.organization_id == organization_id)
            .order_by(SecurityAuditEvent.occurred_at.desc(), SecurityAuditEvent.id.desc())
            .limit(bounded_limit)
            .offset(max(0, offset))
        )
        return list(self.session.scalars(statement))

    def list_active_identity_providers(self, organization_id: str) -> list[OrganizationIdentityProvider]:
        return list(
            self.session.scalars(
                select(OrganizationIdentityProvider).where(
                    OrganizationIdentityProvider.organization_id == organization_id,
                    OrganizationIdentityProvider.status == "active",
                )
            )
        )

    def list_identity_providers(self, organization_id: str) -> list[OrganizationIdentityProvider]:
        return list(
            self.session.scalars(
                select(OrganizationIdentityProvider)
                .where(OrganizationIdentityProvider.organization_id == organization_id)
                .order_by(OrganizationIdentityProvider.name)
            )
        )

    def get_identity_provider(
        self,
        organization_id: str,
        identity_provider_id: str,
    ) -> OrganizationIdentityProvider | None:
        return self.session.scalar(
            select(OrganizationIdentityProvider).where(
                OrganizationIdentityProvider.organization_id == organization_id,
                OrganizationIdentityProvider.id == identity_provider_id,
            )
        )

    def get_service_principal(self, organization_id: str, principal_id: str) -> ServicePrincipal | None:
        return self.session.scalar(
            select(ServicePrincipal).where(
                ServicePrincipal.organization_id == organization_id,
                ServicePrincipal.id == principal_id,
            )
        )

    def list_service_principals(self, organization_id: str) -> list[ServicePrincipal]:
        return list(
            self.session.scalars(
                select(ServicePrincipal)
                .where(ServicePrincipal.organization_id == organization_id)
                .order_by(ServicePrincipal.name)
            )
        )

    def get_secret_reference(self, organization_id: str, secret_id: str) -> SecretReference | None:
        return self.session.scalar(
            select(SecretReference).where(
                SecretReference.organization_id == organization_id,
                SecretReference.id == secret_id,
            )
        )

    def list_secret_references(self, organization_id: str) -> list[SecretReference]:
        return list(
            self.session.scalars(
                select(SecretReference)
                .where(SecretReference.organization_id == organization_id)
                .order_by(SecretReference.name)
            )
        )

    def _require_org(self, organization_id: str) -> Organization:
        org = self.session.get(Organization, organization_id)
        if org is None:
            raise AuthorizationError("organization does not exist")
        return org


def _subject_hash(subject: str | None) -> str | None:
    if not subject:
        return None
    return hashlib.sha256(subject.encode()).hexdigest()


def identity_provider_payload(idp: OrganizationIdentityProvider) -> dict[str, Any]:
    return {
        "id": idp.id,
        "organization_id": idp.organization_id,
        "name": idp.name,
        "issuer": idp.issuer,
        "audience": idp.audience,
        "jwks_uri": idp.jwks_uri,
        "allowed_algorithms": idp.allowed_algorithms,
        "status": idp.status,
        "created_at": idp.created_at,
        "updated_at": idp.updated_at,
    }


def service_principal_payload(principal: ServicePrincipal) -> dict[str, Any]:
    return {
        "id": principal.id,
        "organization_id": principal.organization_id,
        "name": principal.name,
        "external_subject": principal.external_subject,
        "issuer": principal.issuer,
        "status": principal.status,
        "permissions": principal.permissions,
        "metadata": redact_value(principal.metadata_json or {}),
        "created_at": principal.created_at,
        "updated_at": principal.updated_at,
    }


def secret_reference_payload(secret: SecretReference, *, include_locator: bool = False) -> dict[str, Any]:
    payload = {
        "id": secret.id,
        "organization_id": secret.organization_id,
        "name": secret.name,
        "purpose": secret.purpose,
        "provider": secret.provider,
        "status": secret.status,
        "rotation_metadata": redact_value(secret.rotation_metadata or {}),
        "last_rotated_at": secret.last_rotated_at,
        "created_at": secret.created_at,
        "updated_at": secret.updated_at,
    }
    if include_locator:
        payload["locator"] = secret.locator
    return payload


def audit_event_payload(event: SecurityAuditEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "organization_id": event.organization_id,
        "occurred_at": event.occurred_at,
        "request_id": event.request_id,
        "principal_type": event.principal_type,
        "user_id": event.user_id,
        "service_principal_id": event.service_principal_id,
        "issuer": event.issuer,
        "subject_hash": event.subject_hash,
        "action": event.action,
        "required_permission": event.required_permission,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "outcome": event.outcome,
        "reason_code": event.reason_code,
        "http_method": event.http_method,
        "http_path": event.http_path,
        "request_metadata": redact_value(event.request_metadata or {}),
        "event_metadata": redact_value(event.event_metadata or {}),
    }
