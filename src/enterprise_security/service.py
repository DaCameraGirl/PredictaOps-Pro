"""Enterprise authentication, authorization, secret-reference, and audit services."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import socket
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt
from jwt import PyJWK
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from enterprise_security.contracts import (
    ASYMMETRIC_OIDC_ALGORITHMS,
    BootstrapSecurityRequest,
    IdentityProviderCreate,
    IdentityProviderUpdate,
    MembershipChange,
    MembershipStatusChange,
    SecretReferenceCreate,
    SecretReferenceUpdate,
    SecurityContext,
    ServicePrincipalCreate,
    ServicePrincipalUpdate,
    TokenClaims,
    UserIdentityCreate,
    UserIdentityOnboard,
)
from enterprise_security.permissions import (
    OWNERS_MANAGE,
    permissions_for_role,
    validate_service_permissions,
)
from enterprise_security.redaction import redact_value
from platform_core.contracts import OrganizationCreate, UserCreate
from platform_core.models import (
    ExternalPrincipalIdentity,
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
SUPPORTED_SECURITY_MODES = {"disabled", "enterprise", "test"}
SUPPORTED_ENVIRONMENTS = {"development", "test", "production"}
DEFAULT_HTTPS_PORT = 443
DEFAULT_JWKS_CACHE_TTL_SECONDS = 300.0
MAX_JWKS_RESPONSE_BYTES = 65_536
MAX_AUDIT_HTTP_PATH_LENGTH = 1024
MAX_AUDIT_RESOURCE_ID_LENGTH = 255


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


class ConflictError(AuthorizationError):
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
        oidc_http_timeout_seconds=_oidc_http_timeout_setting(),
        docs_enabled=os.environ.get("PMS_ENABLE_DOCS", "1") == "1",
    )
    validate_security_settings(settings)
    return settings


def validate_security_settings(settings: SecuritySettings) -> None:
    if settings.environment not in SUPPORTED_ENVIRONMENTS:
        raise SecurityConfigurationError(f"unsupported environment: {settings.environment}")
    if settings.mode not in SUPPORTED_SECURITY_MODES:
        raise SecurityConfigurationError(f"unsupported security mode: {settings.mode}")
    if not math.isfinite(settings.oidc_http_timeout_seconds) or settings.oidc_http_timeout_seconds <= 0:
        raise SecurityConfigurationError("OIDC HTTP timeout must be a positive finite number")
    if settings.environment == "production":
        if "*" in settings.cors_allowed_origins:
            raise SecurityConfigurationError("wildcard CORS is not allowed in production")
        if settings.test_auth_enabled or settings.mode == "test":
            raise SecurityConfigurationError("test authentication mode is not allowed in production")
        if settings.mode == "enterprise" and not settings.cors_allowed_origins:
            raise SecurityConfigurationError("production enterprise security requires explicit allowed origins")
        if settings.docs_enabled:
            raise SecurityConfigurationError("OpenAPI docs must be explicitly disabled or protected in production")


def _oidc_http_timeout_setting() -> float:
    raw_timeout = os.environ.get("PMS_OIDC_HTTP_TIMEOUT_SECONDS", "2.0")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise SecurityConfigurationError("OIDC HTTP timeout must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise SecurityConfigurationError("OIDC HTTP timeout must be a positive finite number")
    return timeout


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
        if not allow_development_targets:
            _validated_oidc_addresses(host, parsed.port or DEFAULT_HTTPS_PORT)
        return
    if not allow_development_targets and _unsafe_oidc_address(address):
        raise SecurityConfigurationError("OIDC endpoint must not target private or unsafe addresses")


def _validated_oidc_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise SecurityConfigurationError("OIDC endpoint host could not be resolved") from exc
    addresses = sorted({item[4][0] for item in resolved})
    if not addresses:
        raise SecurityConfigurationError("OIDC endpoint host did not resolve to an address")
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise SecurityConfigurationError("OIDC endpoint resolved to an invalid address") from exc
        if _unsafe_oidc_address(address):
            raise SecurityConfigurationError("OIDC endpoint resolved to a private or unsafe address")
    return tuple(addresses)


def _unsafe_oidc_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not address.is_global


class OidcTokenVerifier:
    def __init__(
        self,
        *,
        http_timeout_seconds: float = 2.0,
        jwks_cache_ttl_seconds: float = DEFAULT_JWKS_CACHE_TTL_SECONDS,
    ):
        self.http_timeout_seconds = http_timeout_seconds
        self.jwks_cache_ttl_seconds = jwks_cache_ttl_seconds
        self._jwks_cache: dict[str, tuple[dict[str, Any], datetime]] = {}

    def verify(self, token: str, idp: OrganizationIdentityProvider) -> TokenClaims:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        algorithm = str(header.get("alg") or "")
        if algorithm not in ASYMMETRIC_OIDC_ALGORITHMS or algorithm not in set(idp.allowed_algorithms or []):
            raise AuthenticationError(SAFE_AUTH_ERROR)
        key_id = header.get("kid")
        key = self._key_for_token(idp.jwks_uri, key_id, algorithm)
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
        try:
            expires_at = _numeric_date_to_int(claims["exp"])
            not_before = _numeric_date_to_int(claims["nbf"]) if claims.get("nbf") is not None else None
        except (TypeError, ValueError) as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        profile = {
            key: value
            for key, value in claims.items()
            if key not in {"iss", "sub", "aud", "exp", "nbf", "iat", "jti"}
        }
        try:
            return TokenClaims(
                issuer=claims["iss"],
                subject=subject,
                audience=claims["aud"],
                expires_at=expires_at,
                not_before=not_before,
                algorithm=algorithm,
                key_id=key_id,
                profile=redact_value(profile),
            )
        except ValidationError as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc

    def _key_for_token(self, jwks_uri: str, key_id: str | None, algorithm: str):
        jwks = self._load_jwks(jwks_uri, refresh=False)
        key = _select_jwk(jwks, key_id, algorithm)
        if key is None:
            jwks = self._load_jwks(jwks_uri, refresh=True)
            key = _select_jwk(jwks, key_id, algorithm)
        if key is None:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        try:
            return PyJWK.from_dict(key).key
        except Exception as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc

    def _load_jwks(self, jwks_uri: str, *, refresh: bool) -> dict[str, Any]:
        cached = self._jwks_cache.get(jwks_uri)
        if not refresh and cached is not None:
            jwks, cached_at = cached
            if (datetime.now(UTC) - cached_at).total_seconds() <= self.jwks_cache_ttl_seconds:
                return jwks
            self._jwks_cache.pop(jwks_uri, None)
        allow_development_targets = os.environ.get("PMS_ENVIRONMENT", "development").lower() != "production"
        validate_oidc_endpoint(jwks_uri, allow_development_targets=allow_development_targets)
        try:
            if allow_development_targets:
                with httpx.Client(timeout=self.http_timeout_seconds, follow_redirects=False) as client:
                    response = client.get(jwks_uri)
                    response.raise_for_status()
                    jwks = response.json()
            else:
                jwks = _fetch_jwks_with_pinned_address(jwks_uri, timeout_seconds=self.http_timeout_seconds)
        except Exception as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise AuthenticationError(SAFE_AUTH_ERROR)
        _validated_jwk_entries(jwks["keys"])
        self._jwks_cache[jwks_uri] = (jwks, datetime.now(UTC))
        return jwks


def _fetch_jwks_with_pinned_address(jwks_uri: str, *, timeout_seconds: float) -> dict[str, Any]:
    parsed = urlparse(jwks_uri)
    host = parsed.hostname
    if not host:
        raise AuthenticationError(SAFE_AUTH_ERROR)
    port = parsed.port or DEFAULT_HTTPS_PORT
    addresses = _validated_oidc_addresses(host, port)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    host_header = host if port == DEFAULT_HTTPS_PORT else f"{host}:{port}"
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Accept: application/json\r\n"
        "Connection: close\r\n"
        "User-Agent: abb-predictive-maintenance-studio/enterprise-security\r\n"
        "\r\n"
    ).encode("ascii")
    last_error: Exception | None = None
    for address in addresses:
        try:
            deadline = time.monotonic() + timeout_seconds
            with socket.create_connection((address, port), timeout=timeout_seconds) as raw_socket:
                context = ssl.create_default_context()
                with context.wrap_socket(raw_socket, server_hostname=host) as tls_socket:
                    tls_socket.settimeout(timeout_seconds)
                    tls_socket.sendall(request)
                    chunks = []
                    total_bytes = 0
                    while True:
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            raise AuthenticationError(SAFE_AUTH_ERROR)
                        tls_socket.settimeout(remaining_seconds)
                        chunk = tls_socket.recv(65536)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > MAX_JWKS_RESPONSE_BYTES:
                            raise AuthenticationError(SAFE_AUTH_ERROR)
                        chunks.append(chunk)
            return _parse_jwks_http_response(b"".join(chunks))
        except Exception as exc:
            last_error = exc
    raise AuthenticationError(SAFE_AUTH_ERROR) from last_error


def _parse_jwks_http_response(raw_response: bytes) -> dict[str, Any]:
    header_blob, separator, body = raw_response.partition(b"\r\n\r\n")
    if not separator:
        raise AuthenticationError(SAFE_AUTH_ERROR)
    header_lines = header_blob.decode("iso-8859-1").split("\r\n")
    try:
        _protocol, status_code, _reason = header_lines[0].split(" ", 2)
    except ValueError as exc:
        raise AuthenticationError(SAFE_AUTH_ERROR) from exc
    try:
        status = int(status_code)
    except ValueError as exc:
        raise AuthenticationError(SAFE_AUTH_ERROR) from exc
    if status < 200 or status >= 300:
        raise AuthenticationError(SAFE_AUTH_ERROR)
    headers = {}
    for line in header_lines[1:]:
        name, _, value = line.partition(":")
        if name:
            headers[name.lower()] = value.strip().lower()
    if headers.get("transfer-encoding") == "chunked":
        body = _decode_chunked_body(body)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuthenticationError(SAFE_AUTH_ERROR) from exc
    if not isinstance(payload, dict):
        raise AuthenticationError(SAFE_AUTH_ERROR)
    return payload


def _decode_chunked_body(body: bytes) -> bytes:
    decoded = bytearray()
    remaining = body
    while True:
        size_blob, separator, rest = remaining.partition(b"\r\n")
        if not separator:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        try:
            size = int(size_blob.split(b";", 1)[0], 16)
        except ValueError as exc:
            raise AuthenticationError(SAFE_AUTH_ERROR) from exc
        if size == 0:
            return bytes(decoded)
        chunk = rest[:size]
        if len(chunk) != size or rest[size : size + 2] != b"\r\n":
            raise AuthenticationError(SAFE_AUTH_ERROR)
        decoded.extend(chunk)
        remaining = rest[size + 2 :]


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
        if claims.algorithm not in ASYMMETRIC_OIDC_ALGORITHMS:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        return claims


def _select_jwk(jwks: dict[str, Any], key_id: str | None, algorithm: str) -> dict[str, Any] | None:
    keys = jwks.get("keys") or []
    _validated_jwk_entries(keys)
    if key_id:
        key = next((key for key in keys if key.get("kid") == key_id), None)
        return key if key is not None and _jwk_allows_signing(key, algorithm) else None
    if len(keys) != 1:
        return None
    key = keys[0]
    return key if _jwk_allows_signing(key, algorithm) else None


def _validated_jwk_entries(keys: list[Any]) -> None:
    if any(not isinstance(key, Mapping) for key in keys):
        raise AuthenticationError(SAFE_AUTH_ERROR)


def _jwk_allows_signing(key: Mapping[str, Any], algorithm: str) -> bool:
    use = key.get("use")
    if use is not None and use != "sig":
        return False
    key_ops = key.get("key_ops")
    if key_ops is not None and (not isinstance(key_ops, list) or "verify" not in key_ops):
        return False
    key_algorithm = key.get("alg")
    if key_algorithm is not None and key_algorithm != algorithm:
        return False
    return True


def _numeric_date_to_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("NumericDate must be numeric")
    return int(value)


def _audit_resource_id(resource_id: str | None) -> str | None:
    if resource_id is None or len(resource_id) <= MAX_AUDIT_RESOURCE_ID_LENGTH:
        return resource_id
    return f"sha256:{hashlib.sha256(resource_id.encode('utf-8')).hexdigest()}"


def _require_asymmetric_oidc_algorithms(algorithms: list[str]) -> None:
    unsupported = sorted(set(algorithms) - ASYMMETRIC_OIDC_ALGORITHMS)
    if unsupported:
        raise SecurityConfigurationError("OIDC JWKS providers support only asymmetric signing algorithms")


class SecretResolver:
    def resolve(self, reference: SecretReference) -> str:
        raise SecretResolutionError("secret resolver is not configured")


class EnvironmentSecretResolver(SecretResolver):
    def resolve(self, reference: SecretReference) -> str:
        _require_resolvable_secret(reference)
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
        _require_resolvable_secret(reference)
        value = self.values.get(reference.locator)
        if value is None:
            raise SecretResolutionError("secret is not available")
        return value


def _require_resolvable_secret(reference: SecretReference) -> None:
    if reference.status in {"active", "rotating"}:
        return
    if reference.status in {"inactive", "archived"}:
        raise SecretResolutionError("secret reference is not active")
    raise SecretResolutionError("secret reference has an unsupported lifecycle status")


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
        _require_asymmetric_oidc_algorithms(request.allowed_algorithms)
        validate_oidc_endpoint(request.jwks_uri, allow_development_targets=allow_development_targets)
        if request.discovery_url:
            validate_oidc_endpoint(request.discovery_url, allow_development_targets=allow_development_targets)
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
        try:
            with self.session.begin_nested():
                self.session.add(idp)
                self.session.flush()
        except IntegrityError as exc:
            raise ConflictError("identity provider already exists") from exc
        return idp

    def update_identity_provider(
        self,
        organization_id: str,
        identity_provider_id: str,
        request: IdentityProviderUpdate,
    ) -> OrganizationIdentityProvider:
        self._lock_organization_for_owner_transition(organization_id)
        idp = self.get_identity_provider(organization_id, identity_provider_id)
        if idp is None:
            raise AuthorizationError("identity provider does not belong to this organization")
        if idp.status == "active" and request.status != "active":
            if self._active_identity_provider_count(organization_id) <= 1:
                raise AuthorizationError("cannot deactivate the final active identity provider")
            if self._reachable_active_owner_count_excluding_idp(organization_id, idp.id) <= 0:
                raise AuthorizationError(
                    "cannot deactivate an identity provider while no active owner is reachable elsewhere"
                )
        idp.status = request.status
        self.session.flush()
        return idp

    def create_user_identity(self, organization_id: str, request: UserIdentityCreate) -> UserIdentity:
        idp = self.get_identity_provider(organization_id, request.identity_provider_id)
        if idp is None:
            raise AuthorizationError("identity provider does not belong to this organization")
        if idp.issuer != request.issuer:
            raise SecurityConfigurationError("identity issuer must match the identity provider")
        user = self.session.get(User, request.user_id)
        if user is None:
            raise AuthorizationError("user does not exist")
        if user.lifecycle_state != "active":
            raise AuthorizationError("user is not active")
        if self.repo.get_active_membership(organization_id, user.id) is None:
            raise AuthorizationError("user identity requires an active organization membership")
        existing_identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == request.issuer, UserIdentity.subject == request.subject)
        )
        if existing_identity is not None and existing_identity.user_id != user.id:
            raise AuthorizationError("issuer and subject are already bound to another local user")
        self._reject_external_principal_collision(
            issuer=request.issuer,
            subject=request.subject,
            expected_principal_type="user",
            expected_identity_provider_id=request.identity_provider_id,
            expected_user_id=user.id,
        )
        identity = UserIdentity(
            organization_id=organization_id,
            user_id=request.user_id,
            identity_provider_id=request.identity_provider_id,
            issuer=request.issuer,
            subject=request.subject,
            profile=redact_value(request.profile),
        )
        try:
            with self.session.begin_nested():
                self.session.add(identity)
                self.session.flush()
                self._create_external_principal_identity(
                    organization_id=organization_id,
                    identity_provider_id=request.identity_provider_id,
                    issuer=request.issuer,
                    subject=request.subject,
                    principal_type="user",
                    user_identity_id=identity.id,
                )
        except IntegrityError as exc:
            raise ConflictError("issuer and subject are already bound to another principal") from exc
        return identity

    def onboard_user_identity(
        self,
        organization_id: str,
        request: UserIdentityOnboard,
        *,
        actor: SecurityContext,
    ) -> dict[str, Any]:
        self._require_org(organization_id)
        idp = self.get_identity_provider(organization_id, request.identity_provider_id)
        if idp is None or idp.status != "active":
            raise AuthorizationError("active identity provider does not belong to this organization")
        if idp.issuer != request.issuer:
            raise SecurityConfigurationError("identity issuer must match the identity provider")
        self._authorize_membership_role_assignment(actor, request.role, audit_allowed=False)

        email = request.email.lower()
        existing_identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == request.issuer, UserIdentity.subject == request.subject)
        )
        self._reject_external_principal_collision(
            issuer=request.issuer,
            subject=request.subject,
            expected_principal_type="user",
            expected_identity_provider_id=idp.id,
            expected_user_id=existing_identity.user_id if existing_identity is not None else None,
        )
        user = self.session.scalar(select(User).where(User.email == email))
        if existing_identity is not None:
            if (
                existing_identity.organization_id != organization_id
                or existing_identity.identity_provider_id != idp.id
            ):
                raise AuthorizationError("issuer and subject are already bound to another identity")
            existing_user = self.session.get(User, existing_identity.user_id)
            if user is not None and user.id != existing_identity.user_id:
                raise AuthorizationError("issuer and subject are already bound to another local user")
            user = existing_user
        if user is None:
            user = self.repo.create_user(UserCreate(email=request.email, full_name=request.full_name))
        if user.lifecycle_state != "active":
            raise AuthorizationError("user is not active")

        membership = self.change_membership_role(
            organization_id,
            MembershipChange(user_id=user.id, role=request.role),
            actor=actor,
        )

        identity = self.session.scalar(
            select(UserIdentity).where(
                UserIdentity.organization_id == organization_id,
                UserIdentity.issuer == request.issuer,
                UserIdentity.subject == request.subject,
            )
        )
        if identity is None:
            identity = self.create_user_identity(
                organization_id,
                UserIdentityCreate(
                    user_id=user.id,
                    identity_provider_id=idp.id,
                    issuer=request.issuer,
                    subject=request.subject,
                    profile=request.profile,
                ),
            )
        elif identity.user_id != user.id or identity.identity_provider_id != idp.id:
            raise AuthorizationError("issuer and subject are already bound to another identity")
        self.session.flush()
        return {"user": user, "membership": membership, "identity": identity}

    def create_service_principal(self, organization_id: str, request: ServicePrincipalCreate) -> ServicePrincipal:
        self._require_org(organization_id)
        idp = self.get_identity_provider(organization_id, request.identity_provider_id)
        if idp is None or idp.status != "active":
            raise AuthorizationError("active identity provider does not belong to this organization")
        if idp.issuer != request.issuer:
            raise SecurityConfigurationError("service principal issuer must match the identity provider")
        permissions = validate_service_permissions(request.permissions)
        self._reject_external_principal_collision(
            issuer=request.issuer,
            subject=request.external_subject,
            expected_principal_type="service",
            expected_identity_provider_id=idp.id,
        )
        principal = ServicePrincipal(
            organization_id=organization_id,
            name=request.name,
            identity_provider_id=idp.id,
            external_subject=request.external_subject,
            issuer=request.issuer,
            permissions=permissions,
            status="active",
            metadata_json=redact_value(request.metadata),
        )
        try:
            with self.session.begin_nested():
                self.session.add(principal)
                self.session.flush()
                self._create_external_principal_identity(
                    organization_id=organization_id,
                    identity_provider_id=idp.id,
                    issuer=request.issuer,
                    subject=request.external_subject,
                    principal_type="service",
                    service_principal_id=principal.id,
                )
        except IntegrityError as exc:
            raise ConflictError("issuer and subject are already bound to another principal") from exc
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
        allow_development_targets = os.environ.get("PMS_ENVIRONMENT", "development").lower() != "production"
        validate_oidc_endpoint(request.jwks_uri, allow_development_targets=allow_development_targets)
        org = self.repo.get_or_create_organization(
            OrganizationCreate(slug=request.organization_slug, name=request.organization_name)
        )
        if org.lifecycle_state != "active":
            raise SecurityConfigurationError("bootstrap organization is not active")
        existing_identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == request.issuer, UserIdentity.subject == request.subject)
        )
        if existing_identity is None and self.session.scalar(
            select(ServicePrincipal).where(
                ServicePrincipal.issuer == request.issuer,
                ServicePrincipal.external_subject == request.subject,
            )
        ):
            raise AuthorizationError("issuer and subject are already bound to a service principal")
        user = self.session.scalar(select(User).where(User.email == request.owner_email.lower()))
        if existing_identity is not None:
            if existing_identity.organization_id != org.id:
                raise AuthorizationError("issuer and subject are already bound to another organization")
            existing_user = self.session.get(User, existing_identity.user_id)
            if user is not None and user.id != existing_identity.user_id:
                raise AuthorizationError("issuer and subject are already bound to another local user")
            user = existing_user
        if user is None:
            user = self.repo.create_user(
                UserCreate(
                    email=request.owner_email,
                    full_name=request.owner_full_name,
                )
            )
        elif user.lifecycle_state != "active":
            raise AuthorizationError("bootstrap owner user is not active")
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
                allow_development_targets=allow_development_targets,
            )
        else:
            _require_asymmetric_oidc_algorithms(request.allowed_algorithms)
            idp.name = request.idp_name
            idp.jwks_uri = request.jwks_uri
            idp.allowed_algorithms = sorted(set(request.allowed_algorithms))
            idp.status = "active"
        if existing_identity is not None and existing_identity.identity_provider_id != idp.id:
            raise SecurityConfigurationError(
                "bootstrap owner identity is already bound to a different identity provider"
            )
        identity = self.session.scalar(
            select(UserIdentity).where(
                UserIdentity.organization_id == org.id,
                UserIdentity.issuer == request.issuer,
                UserIdentity.subject == request.subject,
            )
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
        self._require_active_org(organization_id)
        idps = self.list_active_identity_providers(organization_id)
        last_error: Exception | None = None
        for idp in idps:
            try:
                claims = self.verifier.verify(token, idp)
            except AuthenticationError as exc:
                last_error = exc
                continue
            try:
                return self.context_from_claims(
                    organization_id,
                    claims,
                    request_id=request_id,
                    identity_provider_id=idp.id,
                )
            except AuthorizationError as exc:
                last_error = exc
                continue
        if isinstance(last_error, AuthorizationError):
            raise AuthorizationError(SAFE_FORBIDDEN_ERROR) from last_error
        raise AuthenticationError(SAFE_AUTH_ERROR) from last_error

    def context_from_claims(
        self,
        organization_id: str,
        claims: TokenClaims,
        *,
        request_id: str,
        identity_provider_id: str | None = None,
    ) -> SecurityContext:
        self._require_active_org(organization_id)
        identity_filters = [
            UserIdentity.organization_id == organization_id,
            UserIdentity.issuer == claims.issuer,
            UserIdentity.subject == claims.subject,
        ]
        if identity_provider_id is not None:
            identity_filters.append(UserIdentity.identity_provider_id == identity_provider_id)
        identities = list(self.session.scalars(select(UserIdentity).where(*identity_filters)))
        service_filters = [
            ServicePrincipal.organization_id == organization_id,
            ServicePrincipal.external_subject == claims.subject,
            ServicePrincipal.issuer == claims.issuer,
            ServicePrincipal.status == "active",
        ]
        if identity_provider_id is not None:
            service_filters.append(ServicePrincipal.identity_provider_id == identity_provider_id)
        services = list(self.session.scalars(select(ServicePrincipal).where(*service_filters)))
        if not identities and not services:
            raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
        if len(identities) + len(services) > 1:
            raise AuthenticationError(SAFE_AUTH_ERROR)
        identity = identities[0] if identities else None
        if identity is not None:
            user = self.session.get(User, identity.user_id)
            if user is None or user.lifecycle_state != "active":
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
        service = services[0]
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
        audit_allowed: bool = True,
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
        if not audit_allowed:
            return
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
        self._lock_organization_for_owner_transition(organization_id)
        self._authorize_membership_role_assignment(actor, request.role, audit_allowed=False)
        membership = self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == request.user_id,
            )
        )
        if membership is None:
            user = self.session.get(User, request.user_id)
            if user is None:
                raise AuthorizationError("user does not exist")
            membership = self.repo.add_membership(organization_id, request.user_id, request.role)
        elif membership.role == "owner" and request.role != "owner" and OWNERS_MANAGE not in actor.permissions:
            raise AuthorizationError("only owners may demote owners")
        else:
            if membership.role == "owner" and membership.lifecycle_state == "active" and request.role != "owner":
                self.session.flush()
                active_owner_count = self._active_owner_count(organization_id)
                if active_owner_count <= 1:
                    raise AuthorizationError("cannot demote the final active owner")
            membership.role = request.role
            membership.lifecycle_state = "active"
        self.session.flush()
        return membership

    def _active_owner_count(self, organization_id: str) -> int:
        count = self.session.scalar(
            select(func.count()).select_from(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.role == "owner",
                OrganizationMembership.lifecycle_state == "active",
            )
        )
        return int(count or 0)

    def _active_identity_provider_count(self, organization_id: str) -> int:
        count = self.session.scalar(
            select(func.count()).select_from(OrganizationIdentityProvider).where(
                OrganizationIdentityProvider.organization_id == organization_id,
                OrganizationIdentityProvider.status == "active",
            )
        )
        return int(count or 0)

    def _reachable_active_owner_count_excluding_idp(self, organization_id: str, excluded_idp_id: str) -> int:
        count = self.session.scalar(
            select(func.count(func.distinct(OrganizationMembership.user_id)))
            .select_from(OrganizationMembership)
            .join(User, User.id == OrganizationMembership.user_id)
            .join(
                UserIdentity,
                UserIdentity.user_id == OrganizationMembership.user_id,
            )
            .join(
                OrganizationIdentityProvider,
                OrganizationIdentityProvider.id == UserIdentity.identity_provider_id,
            )
            .where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.role == "owner",
                OrganizationMembership.lifecycle_state == "active",
                User.lifecycle_state == "active",
                UserIdentity.organization_id == organization_id,
                OrganizationIdentityProvider.organization_id == organization_id,
                OrganizationIdentityProvider.status == "active",
                OrganizationIdentityProvider.id != excluded_idp_id,
            )
        )
        return int(count or 0)

    def _authorize_membership_role_assignment(
        self,
        actor: SecurityContext,
        role: str,
        *,
        audit_allowed: bool = True,
    ) -> None:
        self.require_permission(
            actor,
            OWNERS_MANAGE if role == "owner" else "members.manage",
            action="members.role",
            audit_allowed=audit_allowed,
        )

    def _lock_organization_for_owner_transition(self, organization_id: str) -> Organization:
        organization = self.session.scalar(
            select(Organization).where(Organization.id == organization_id).with_for_update()
        )
        if organization is None:
            raise AuthorizationError("organization does not exist")
        return organization

    def _reject_external_principal_collision(
        self,
        *,
        issuer: str,
        subject: str,
        expected_principal_type: str,
        expected_identity_provider_id: str,
        expected_user_id: str | None = None,
    ) -> None:
        existing_binding = self.session.scalar(
            select(ExternalPrincipalIdentity).where(
                ExternalPrincipalIdentity.issuer == issuer,
                ExternalPrincipalIdentity.subject == subject,
            )
        )
        if existing_binding is not None:
            if (
                existing_binding.principal_type != expected_principal_type
                or existing_binding.identity_provider_id != expected_identity_provider_id
            ):
                raise AuthorizationError("issuer and subject are already bound to another principal")
        existing_service = self.session.scalar(
            select(ServicePrincipal).where(
                ServicePrincipal.issuer == issuer,
                ServicePrincipal.external_subject == subject,
            )
        )
        if expected_principal_type == "user" and existing_service is not None:
            raise AuthorizationError("issuer and subject are already bound to a service principal")
        existing_identity = self.session.scalar(
            select(UserIdentity).where(UserIdentity.issuer == issuer, UserIdentity.subject == subject)
        )
        if expected_principal_type == "service" and existing_identity is not None:
            raise AuthorizationError("issuer and subject are already bound to a human identity")
        if (
            expected_principal_type == "user"
            and existing_identity is not None
            and existing_identity.user_id != expected_user_id
        ):
            raise AuthorizationError("issuer and subject are already bound to another local user")

    def _create_external_principal_identity(
        self,
        *,
        organization_id: str,
        identity_provider_id: str,
        issuer: str,
        subject: str,
        principal_type: str,
        user_identity_id: str | None = None,
        service_principal_id: str | None = None,
    ) -> None:
        binding = ExternalPrincipalIdentity(
            organization_id=organization_id,
            identity_provider_id=identity_provider_id,
            issuer=issuer,
            subject=subject,
            principal_type=principal_type,
            user_identity_id=user_identity_id,
            service_principal_id=service_principal_id,
        )
        self.session.add(binding)
        self.session.flush()

    def change_membership_status(
        self,
        organization_id: str,
        user_id: str,
        request: MembershipStatusChange,
        *,
        actor: SecurityContext,
    ) -> OrganizationMembership:
        self._lock_organization_for_owner_transition(organization_id)
        membership = self.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == user_id,
            )
        )
        if membership is None:
            raise AuthorizationError("membership does not exist")
        permission = OWNERS_MANAGE if membership.role == "owner" else "members.manage"
        self.require_permission(actor, permission, action="members.status", audit_allowed=False)
        if membership.role == "owner" and request.lifecycle_state != "active":
            self.session.flush()
            if self._active_owner_count(organization_id) <= 1:
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
            resource_id=_audit_resource_id(resource_id),
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

    def _require_active_org(self, organization_id: str) -> Organization:
        org = self._require_org(organization_id)
        if org.lifecycle_state != "active":
            raise AuthorizationError(SAFE_FORBIDDEN_ERROR)
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
        "identity_provider_id": principal.identity_provider_id,
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
