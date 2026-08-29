from __future__ import annotations

import base64
import importlib
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from alembic.config import Config
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from alembic import command
from enterprise_security.contracts import (
    BootstrapSecurityRequest,
    IdentityProviderCreate,
    IdentityProviderUpdate,
    MembershipChange,
    MembershipStatusChange,
    SecretReferenceCreate,
    ServicePrincipalCreate,
    TokenClaims,
    UserIdentityCreate,
    UserIdentityOnboard,
)
from enterprise_security.permissions import (
    INGESTION_WRITE,
    MAINTENANCE_MANAGE,
    MEMBERS_MANAGE,
    ML_MODEL_PROMOTE_PRODUCTION,
    OWNERS_MANAGE,
    SECRETS_MANAGE,
    SECURITY_MANAGE,
    SERVICE_ALLOWED_PERMISSIONS,
    SERVING_PREDICT,
)
from enterprise_security.redaction import assert_no_plaintext_secrets
from enterprise_security.service import (
    MAX_AUDIT_HTTP_PATH_LENGTH,
    MAX_JWKS_RESPONSE_BYTES,
    AuthenticationError,
    DeterministicTokenVerifier,
    EnvironmentSecretResolver,
    InMemorySecretResolver,
    OidcTokenVerifier,
    SecretResolutionError,
    SecurityConfigurationError,
    SecurityService,
    _fetch_jwks_with_pinned_address,
    _validated_oidc_addresses,
    audit_event_payload,
    secret_reference_payload,
    security_settings,
    validate_oidc_endpoint,
)
from industrial_ingestion.contracts import SourceRegistration
from industrial_ingestion.service import IngestionService
from maintenance_operations.contracts import CaseCreate
from maintenance_operations.service import MaintenanceOperationsService
from ml_platform.artifact_store import ModelArtifactStore
from ml_platform.contracts import (
    DatasetVersionCreate,
    ExperimentCreate,
    ModelVersionCreate,
    RegistryCreate,
)
from ml_platform.service import MLPlatformService
from platform_core.contracts import (
    AssetCreate,
    ComponentCreate,
    OrganizationCreate,
    SensorCreate,
    SiteCreate,
    UserCreate,
)
from platform_core.database import make_engine
from platform_core.models import (
    Base,
    ExternalPrincipalIdentity,
    IngestionSource,
    MaintenanceCase,
    MaintenanceNote,
    MLModelPromotionEvent,
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

ROOT = Path(__file__).resolve().parent.parent
ISSUER_A = "https://issuer-a.example/"
ISSUER_B = "https://issuer-b.example/"
AUDIENCE = "predictive-maintenance-api"
JWKS_URI = "https://issuer-a.example/.well-known/jwks.json"


@pytest.fixture
def migrated_db(tmp_path, monkeypatch):
    external_url = os.environ.get("PMS_PLATFORM_CORE_TEST_DATABASE_URL")
    if external_url:
        url = external_url
    else:
        db_path = tmp_path / "platform.db"
        url = f"sqlite:///{db_path.as_posix()}"
        monkeypatch.setenv("PMS_DATABASE_URL", url)

    cfg = Config(str(ROOT / "alembic.ini"))
    if external_url:
        clean_engine = make_engine(url)
        try:
            Base.metadata.drop_all(clean_engine)
            with clean_engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        finally:
            clean_engine.dispose()
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    try:
        yield engine, session_factory
    finally:
        if external_url:
            Base.metadata.drop_all(engine)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        engine.dispose()


@pytest.fixture
def rsa_keys():
    primary = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    alternate = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {"primary": primary, "alternate": alternate}


@pytest.fixture
def jwks(rsa_keys):
    return {"keys": [_jwk_from_public_key(rsa_keys["primary"].public_key(), kid="primary")]}


def _b64url_int(value: int) -> str:
    size = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(size, "big")).rstrip(b"=").decode()


def _jwk_from_public_key(public_key, *, kid: str) -> dict:
    numbers = public_key.public_numbers()
    return {
        "kty": "RSA",
        "use": "sig",
        "kid": kid,
        "alg": "RS256",
        "n": _b64url_int(numbers.n),
        "e": _b64url_int(numbers.e),
    }


def _token(
    key,
    *,
    issuer: str = ISSUER_A,
    audience: str = AUDIENCE,
    subject: str = "alice-sub",
    kid: str = "primary",
    expires_delta: timedelta = timedelta(minutes=10),
) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": subject,
            "iat": now,
            "nbf": now - timedelta(seconds=1),
            "exp": now + expires_delta,
            "email": f"{subject}@example.com",
        },
        key,
        algorithm="RS256",
        headers={"kid": kid},
    )


def _seed_security(session, *, jwks_uri: str = JWKS_URI):
    repo = PlatformRepository(session)
    org = repo.create_organization(OrganizationCreate(slug="acme", name="Acme Manufacturing"))
    other_org = repo.create_organization(OrganizationCreate(slug="globex", name="Globex"))
    site = repo.create_site(org.id, SiteCreate(slug="atlanta", name="Atlanta Plant"))
    asset = repo.create_asset(org.id, AssetCreate(site_id=site.id, slug="pump", name="Pump", asset_type="pump"))
    component = repo.create_component(
        org.id,
        ComponentCreate(asset_id=asset.id, slug="bearing", name="Bearing", component_type="bearing"),
    )
    sensor = repo.create_sensor(
        org.id,
        SensorCreate(component_id=component.id, slug="vib", name="Vibration", sensor_type="accelerometer", unit="g"),
    )
    users = {
        "owner": repo.create_user(UserCreate(email="owner@example.com", external_subject="legacy-owner")),
        "admin": repo.create_user(UserCreate(email="admin@example.com", external_subject="legacy-admin")),
        "engineer": repo.create_user(UserCreate(email="engineer@example.com", external_subject="legacy-engineer")),
        "technician": repo.create_user(UserCreate(email="tech@example.com", external_subject="legacy-tech")),
        "viewer": repo.create_user(UserCreate(email="viewer@example.com", external_subject="legacy-viewer")),
        "bob": repo.create_user(UserCreate(email="bob@example.com", external_subject="legacy-bob")),
        "outsider": repo.create_user(UserCreate(email="outsider@example.com", external_subject="legacy-outsider")),
    }
    for role in ["owner", "admin", "engineer", "technician", "viewer"]:
        repo.add_membership(org.id, users[role].id, role)
    repo.add_membership(org.id, users["bob"].id, "technician")
    repo.add_membership(other_org.id, users["outsider"].id, "owner")
    security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
    idp = security.create_identity_provider(
        org.id,
        IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience=AUDIENCE, jwks_uri=jwks_uri),
        allow_development_targets=True,
    )
    other_idp = security.create_identity_provider(
        other_org.id,
        IdentityProviderCreate(
            name="primary",
            issuer=ISSUER_A,
            audience=AUDIENCE,
            jwks_uri=JWKS_URI,
        ),
        allow_development_targets=True,
    )
    subject_by_role = {
        "owner": "owner-sub",
        "admin": "admin-sub",
        "engineer": "engineer-sub",
        "technician": "tech-sub",
        "viewer": "viewer-sub",
        "bob": "bob-sub",
    }
    for role, subject in subject_by_role.items():
        security.create_user_identity(
            org.id,
            UserIdentityCreate(
                user_id=users[role].id,
                identity_provider_id=idp.id,
                issuer=ISSUER_A,
                subject=subject,
            ),
        )
    security.create_user_identity(
        other_org.id,
        UserIdentityCreate(
            user_id=users["outsider"].id,
            identity_provider_id=other_idp.id,
            issuer=ISSUER_A,
            subject="outsider-sub",
        ),
    )
    service_principal = security.create_service_principal(
        org.id,
        ServicePrincipalCreate(
            name="ingestion-robot",
            identity_provider_id=idp.id,
            external_subject="robot-sub",
            issuer=ISSUER_A,
            permissions=[INGESTION_WRITE],
        ),
    )
    return {
        "organization_id": org.id,
        "other_organization_id": other_org.id,
        "idp_id": idp.id,
        "other_idp_id": other_idp.id,
        "site_id": site.id,
        "asset_id": asset.id,
        "component_id": component.id,
        "sensor_id": sensor.id,
        "users": {role: user.id for role, user in users.items()},
        "subjects": subject_by_role,
        "service_principal_id": service_principal.id,
    }


def _claims(subject: str, *, issuer: str = ISSUER_A, audience: str = AUDIENCE) -> TokenClaims:
    return TokenClaims(
        issuer=issuer,
        subject=subject,
        audience=audience,
        expires_at=int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        algorithm="RS256",
        key_id="primary",
    )


def _enterprise_app(monkeypatch, session_factory, jwks=None):
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_CORS_ALLOWED_ORIGINS", "http://testserver,http://ui.example")
    import app.main as app_main

    app_main = importlib.reload(app_main)
    monkeypatch.setattr(app_main, "SessionLocal", session_factory)
    if jwks is not None:
        monkeypatch.setattr(OidcTokenVerifier, "_load_jwks", lambda self, uri, refresh: jwks)
    return app_main, TestClient(app_main.app)


def _seed_ml_training_features(session, fixture) -> None:
    repo = PlatformRepository(session)
    run = repo.create_analytics_run(
        fixture["organization_id"],
        run_kind="sensor",
        sensor_id=fixture["sensor_id"],
        algorithm_version="analytics-v1",
        provenance={"test": "enterprise-security-promotion"},
    )
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for group, offset in [("bearing-a", 0), ("bearing-b", 10)]:
        for index in range(4):
            repo.create_analytics_feature(
                fixture["organization_id"],
                run_id=run.id,
                sensor_id=fixture["sensor_id"],
                batch_id=None,
                source_kind="scalar",
                source_record_id=f"{group}-{index}",
                observed_at=base_time + timedelta(minutes=offset + index),
                feature_name="scalar.rms",
                value=float(index + offset),
                unit="g",
                quality="good",
                algorithm_version="analytics-v1",
                provenance={
                    "target_rul_hours": float(8 - index - offset / 10),
                    "validation_group": group,
                },
            )
    session.commit()


def _register_candidate_model(
    service: MLPlatformService,
    organization_id: str,
    experiment_id: str,
    registry_id: str,
    version: str,
):
    return service.register_model_version(
        organization_id,
        ModelVersionCreate(registry_id=registry_id, experiment_run_id=experiment_id, version=version),
    )


def test_migration_creates_enterprise_security_tables(migrated_db):
    engine, _session_factory = migrated_db
    tables = set(inspect(engine).get_table_names())
    columns = {column["name"]: column for column in inspect(engine).get_columns("service_principals")}
    identity_constraints = {
        constraint["name"] for constraint in inspect(engine).get_unique_constraints("user_identities")
    }
    principal_constraints = {
        constraint["name"] for constraint in inspect(engine).get_unique_constraints("external_principal_identities")
    }
    audit_indexes = {index["name"] for index in inspect(engine).get_indexes("security_audit_events")}

    assert {
        "organization_identity_providers",
        "user_identities",
        "service_principals",
        "external_principal_identities",
        "secret_references",
        "security_audit_events",
    }.issubset(tables)
    assert columns["identity_provider_id"]["nullable"] is False
    assert columns["issuer"]["nullable"] is False
    assert "uq_user_identity_global_issuer_subject" in identity_constraints
    assert "uq_external_principal_global_issuer_subject" in principal_constraints
    assert "ix_security_audit_events_org_occurred_id" in audit_indexes


def test_oidc_verifier_accepts_valid_signed_token_and_rejects_bad_tokens(migrated_db, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        idp = SecurityService(session).get_identity_provider(fixture["organization_id"], fixture["idp_id"])
        verifier = OidcTokenVerifier()
        verifier._jwks_cache[idp.jwks_uri] = (jwks, datetime.now(UTC))

        valid = verifier.verify(_token(rsa_keys["primary"], subject="owner-sub"), idp)
        assert valid.issuer == ISSUER_A
        assert valid.subject == "owner-sub"

        bad_signature = _token(rsa_keys["alternate"], subject="owner-sub")
        expired = _token(rsa_keys["primary"], subject="owner-sub", expires_delta=timedelta(seconds=-5))
        wrong_issuer = _token(rsa_keys["primary"], issuer="https://wrong.example/", subject="owner-sub")
        wrong_audience = _token(rsa_keys["primary"], audience="wrong-audience", subject="owner-sub")
        none_alg = jwt.encode(
            {"iss": ISSUER_A, "aud": AUDIENCE, "sub": "owner-sub", "exp": datetime.now(UTC) + timedelta(minutes=5)},
            key="",
            algorithm="none",
            headers={"kid": "primary"},
        )

        for token in [bad_signature, expired, wrong_issuer, wrong_audience, none_alg]:
            with pytest.raises(AuthenticationError):
                verifier.verify(token, idp)


def test_unknown_kid_refresh_is_rate_limited_but_allows_later_rotation(migrated_db, rsa_keys):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        idp = SecurityService(session).get_identity_provider(fixture["organization_id"], fixture["idp_id"])
        verifier = OidcTokenVerifier(unknown_kid_refresh_cooldown_seconds=60.0)
        calls = []
        allow_rotation = False

        def fake_load_jwks(_jwks_uri, *, refresh):
            calls.append(refresh)
            if refresh and allow_rotation:
                return {"keys": [_jwk_from_public_key(rsa_keys["alternate"].public_key(), kid="rotated")]}
            return {"keys": [_jwk_from_public_key(rsa_keys["primary"].public_key(), kid="primary")]}

        verifier._load_jwks = fake_load_jwks
        for index in range(20):
            with pytest.raises(AuthenticationError):
                verifier.verify(_token(rsa_keys["primary"], subject="owner-sub", kid=f"rotated-{index}"), idp)

        assert calls == [False, True, *([False] * 19)]
        assert calls.count(True) == 1
        assert len(verifier._unknown_kid_cache) == 20
        assert verifier.verify(_token(rsa_keys["primary"], subject="owner-sub"), idp).subject == "owner-sub"
        assert calls == [False, True, *([False] * 20)]

        for index in range(600):
            verifier._record_unknown_kid(idp.jwks_uri, f"extra-{index}", "RS256")
        assert len(verifier._unknown_kid_cache) <= 512

        verifier._jwks_unknown_kid_refresh_cache[idp.jwks_uri] = datetime.now(UTC) - timedelta(seconds=61)
        allow_rotation = True
        assert (
            verifier.verify(
                _token(rsa_keys["alternate"], subject="owner-sub", kid="rotated"),
                idp,
            ).key_id
            == "rotated"
        )
        assert calls == [False, True, *([False] * 20), False, True]


def test_oidc_verifier_expires_cached_jwks(monkeypatch):
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    responses = [
        {"keys": [{"kid": "first"}]},
        {"keys": [{"kid": "second"}]},
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, _uri):
            payload = responses[FakeClient.calls]
            FakeClient.calls += 1
            return FakeResponse(payload)

    monkeypatch.setattr("enterprise_security.service.httpx.Client", FakeClient)
    verifier = OidcTokenVerifier(jwks_cache_ttl_seconds=0)

    assert verifier._load_jwks(JWKS_URI, refresh=False) == {"keys": [{"kid": "first"}]}
    time.sleep(0.01)
    assert verifier._load_jwks(JWKS_URI, refresh=False) == {"keys": [{"kid": "second"}]}
    assert FakeClient.calls == 2


def test_malformed_jwks_entries_fail_closed_as_authentication_errors(migrated_db, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        idp = SecurityService(session).get_identity_provider(fixture["organization_id"], fixture["idp_id"])
        verifier = OidcTokenVerifier()
        valid_token = _token(rsa_keys["primary"], subject="owner-sub")

        verifier._jwks_cache[idp.jwks_uri] = (jwks, datetime.now(UTC))
        assert verifier.verify(valid_token, idp).subject == "owner-sub"

        for malformed_keys in [["not-a-jwk"], [None], [[]]]:
            verifier._jwks_cache[idp.jwks_uri] = ({"keys": malformed_keys}, datetime.now(UTC))
            with pytest.raises(AuthenticationError):
                verifier.verify(valid_token, idp)

        verifier._jwks_cache[idp.jwks_uri] = (
            {"keys": [{"kid": "primary", "kty": "RSA"}]},
            datetime.now(UTC),
        )
        with pytest.raises(AuthenticationError):
            verifier.verify(valid_token, idp)


def test_oidc_verifier_enforces_jwk_signing_restrictions_and_normalizes_numeric_dates(
    migrated_db,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        idp = SecurityService(session).get_identity_provider(fixture["organization_id"], fixture["idp_id"])
        verifier = OidcTokenVerifier()
        valid_token = _token(rsa_keys["primary"], subject="owner-sub")

        base_key = jwks["keys"][0]
        for restricted_key in [
            {**base_key, "use": "enc"},
            {**base_key, "key_ops": ["sign"]},
            {**base_key, "key_ops": "verify"},
            {**base_key, "alg": "RS384"},
        ]:
            verifier._jwks_cache[idp.jwks_uri] = ({"keys": [restricted_key]}, datetime.now(UTC))
            with pytest.raises(AuthenticationError):
                verifier.verify(valid_token, idp)

        verifier._jwks_cache[idp.jwks_uri] = (
            {"keys": [{**base_key, "key_ops": ["verify"]}]},
            datetime.now(UTC),
        )
        assert verifier.verify(valid_token, idp).subject == "owner-sub"

        now = datetime.now(UTC).timestamp()
        fractional_numeric_date_token = jwt.encode(
            {
                "iss": ISSUER_A,
                "aud": AUDIENCE,
                "sub": "owner-sub",
                "iat": now,
                "nbf": now - 1.25,
                "exp": now + 600.75,
            },
            rsa_keys["primary"],
            algorithm="RS256",
            headers={"kid": "primary"},
        )
        verifier._jwks_cache[idp.jwks_uri] = (jwks, datetime.now(UTC))
        claims = verifier.verify(fractional_numeric_date_token, idp)

        assert claims.subject == "owner-sub"
        assert isinstance(claims.expires_at, int)
        assert isinstance(claims.not_before, int)


def test_identity_resolution_uses_issuer_and_subject_not_subject_alone(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org_a = repo.create_organization(OrganizationCreate(slug="a", name="A"))
        org_b = repo.create_organization(OrganizationCreate(slug="b", name="B"))
        user_a = repo.create_user(UserCreate(email="same-a@example.com", external_subject="same-sub"))
        user_b = repo.create_user(UserCreate(email="same-b@example.com"))
        repo.add_membership(org_a.id, user_a.id, "viewer")
        repo.add_membership(org_b.id, user_b.id, "viewer")
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        idp_a = security.create_identity_provider(
            org_a.id,
            IdentityProviderCreate(name="idp-a", issuer=ISSUER_A, audience=AUDIENCE, jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        idp_b = security.create_identity_provider(
            org_b.id,
            IdentityProviderCreate(
                name="idp-b",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/jwks",
            ),
            allow_development_targets=True,
        )
        security.create_user_identity(
            org_a.id,
            UserIdentityCreate(user_id=user_a.id, identity_provider_id=idp_a.id, issuer=ISSUER_A, subject="same-sub"),
        )
        security.create_user_identity(
            org_b.id,
            UserIdentityCreate(user_id=user_b.id, identity_provider_id=idp_b.id, issuer=ISSUER_B, subject="same-sub"),
        )

        context = security.context_from_claims(org_b.id, _claims("same-sub", issuer=ISSUER_B), request_id="req")

        assert context.user_id == user_b.id


def test_authentication_binds_identity_to_verifying_provider(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="provider-binding", name="Provider Binding"))
        user = repo.create_user(UserCreate(email="provider-user@example.com"))
        repo.add_membership(org.id, user.id, "viewer")
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        idp_a = security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience="audience-a", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        idp_b = security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="secondary", issuer=ISSUER_A, audience="audience-b", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        security.create_user_identity(
            org.id,
            UserIdentityCreate(
                user_id=user.id,
                identity_provider_id=idp_a.id,
                issuer=ISSUER_A,
                subject="shared-sub",
            ),
        )
        token = "verified-by-provider-b"
        security.verifier = DeterministicTokenVerifier(
            {
                token: TokenClaims(
                    issuer=ISSUER_A,
                    subject="shared-sub",
                    audience="audience-b",
                    expires_at=int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
                    algorithm="RS256",
                    key_id="primary",
                )
            }
        )

        with pytest.raises(PermissionError):
            security.authenticate_bearer(org.id, token, request_id="req-provider-b")

        assert security.context_from_claims(
            org.id,
            _claims("shared-sub", audience="audience-a"),
            request_id="req-provider-a",
            identity_provider_id=idp_a.id,
        ).user_id == user.id
        with pytest.raises(PermissionError):
            security.context_from_claims(
                org.id,
                _claims("shared-sub", audience="audience-b"),
                request_id="req-provider-b",
                identity_provider_id=idp_b.id,
            )


def test_service_principal_authentication_binds_to_verifying_provider(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="service-provider-binding", name="Service Binding"))
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        idp_a = security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience="audience-a", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="secondary", issuer=ISSUER_A, audience="audience-b", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        service = security.create_service_principal(
            org.id,
            ServicePrincipalCreate(
                name="robot-a",
                identity_provider_id=idp_a.id,
                external_subject="robot-shared-sub",
                issuer=ISSUER_A,
                permissions=[INGESTION_WRITE],
            ),
        )
        token_a = "service-a"
        token_b = "service-b"
        security.verifier = DeterministicTokenVerifier(
            {
                token_a: TokenClaims(
                    issuer=ISSUER_A,
                    subject="robot-shared-sub",
                    audience="audience-a",
                    expires_at=int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
                    algorithm="RS256",
                    key_id="primary",
                ),
                token_b: TokenClaims(
                    issuer=ISSUER_A,
                    subject="robot-shared-sub",
                    audience="audience-b",
                    expires_at=int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
                    algorithm="RS256",
                    key_id="primary",
                ),
            }
        )

        context = security.authenticate_bearer(org.id, token_a, request_id="req-service-a")
        assert context.principal_type == "service"
        assert context.service_principal_id == service.id

        with pytest.raises(PermissionError):
            security.authenticate_bearer(org.id, token_b, request_id="req-service-b")


def test_authentication_continues_across_verified_providers_until_bound_principal(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="multi-provider", name="Multi Provider"))
        user = repo.create_user(UserCreate(email="multi-provider@example.com"))
        repo.add_membership(org.id, user.id, "viewer")
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="first", issuer=ISSUER_A, audience="audience-a", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        idp_b = security.create_identity_provider(
            org.id,
            IdentityProviderCreate(name="second", issuer=ISSUER_A, audience="audience-b", jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        security.create_user_identity(
            org.id,
            UserIdentityCreate(
                user_id=user.id,
                identity_provider_id=idp_b.id,
                issuer=ISSUER_A,
                subject="multi-audience-sub",
            ),
        )
        token = "multi-audience-token"
        security.verifier = DeterministicTokenVerifier(
            {
                token: TokenClaims(
                    issuer=ISSUER_A,
                    subject="multi-audience-sub",
                    audience=["audience-a", "audience-b"],
                    expires_at=int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
                    algorithm="RS256",
                    key_id="primary",
                )
            }
        )

        context = security.authenticate_bearer(org.id, token, request_id="req-multi-provider")

        assert context.principal_type == "user"
        assert context.user_id == user.id


def test_human_and_service_external_principal_collisions_are_rejected(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))

        with pytest.raises(PermissionError):
            security.create_service_principal(
                fixture["organization_id"],
                ServicePrincipalCreate(
                    name="human-collision",
                    identity_provider_id=fixture["idp_id"],
                    external_subject=fixture["subjects"]["owner"],
                    issuer=ISSUER_A,
                    permissions=[INGESTION_WRITE],
                ),
            )

        service = security.create_service_principal(
            fixture["organization_id"],
            ServicePrincipalCreate(
                name="service-first",
                identity_provider_id=fixture["idp_id"],
                external_subject="service-first-sub",
                issuer=ISSUER_A,
                permissions=[INGESTION_WRITE],
            ),
        )
        assert service.external_subject == "service-first-sub"

        with pytest.raises(PermissionError):
            security.create_user_identity(
                fixture["organization_id"],
                UserIdentityCreate(
                    user_id=fixture["users"]["viewer"],
                    identity_provider_id=fixture["idp_id"],
                    issuer=ISSUER_A,
                    subject="service-first-sub",
                ),
            )


def test_ambiguous_legacy_human_service_mapping_fails_authentication_closed(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.add(
            UserIdentity(
                organization_id=fixture["organization_id"],
                user_id=fixture["users"]["viewer"],
                identity_provider_id=fixture["idp_id"],
                issuer=ISSUER_A,
                subject="legacy-ambiguous-sub",
                profile={},
            )
        )
        session.add(
            ServicePrincipal(
                organization_id=fixture["organization_id"],
                name="legacy-ambiguous-service",
                identity_provider_id=fixture["idp_id"],
                external_subject="legacy-ambiguous-sub",
                issuer=ISSUER_A,
                permissions=[INGESTION_WRITE],
                status="active",
                metadata_json={},
            )
        )
        session.flush()
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))

        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["organization_id"],
                _claims("legacy-ambiguous-sub"),
                request_id="req-ambiguous",
                identity_provider_id=fixture["idp_id"],
            )


def test_postgres_serializes_concurrent_human_service_principal_collisions(migrated_db):
    engine, session_factory = migrated_db
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL unique-constraint race regression requires PostgreSQL")

    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="principal-race", name="Principal Race"))
        user = repo.create_user(UserCreate(email="principal-race@example.com"))
        repo.add_membership(org.id, user.id, "viewer")
        idp = SecurityService(session, verifier=DeterministicTokenVerifier({})).create_identity_provider(
            org.id,
            IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience=AUDIENCE, jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        session.commit()

    barrier = threading.Barrier(2)

    def create_human() -> str:
        with session_factory() as session:
            security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
            barrier.wait(timeout=10)
            try:
                security.create_user_identity(
                    org.id,
                    UserIdentityCreate(
                        user_id=user.id,
                        identity_provider_id=idp.id,
                        issuer=ISSUER_A,
                        subject="race-sub",
                    ),
                )
                time.sleep(0.2)
                session.commit()
                return "created-human"
            except (IntegrityError, PermissionError):
                assert session.scalar(select(func.count()).select_from(User)) >= 1
                session.rollback()
                return "blocked"

    def create_service() -> str:
        with session_factory() as session:
            security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
            barrier.wait(timeout=10)
            try:
                security.create_service_principal(
                    org.id,
                    ServicePrincipalCreate(
                        name="race-service",
                        identity_provider_id=idp.id,
                        external_subject="race-sub",
                        issuer=ISSUER_A,
                        permissions=[INGESTION_WRITE],
                    ),
                )
                time.sleep(0.2)
                session.commit()
                return "created-service"
            except (IntegrityError, PermissionError):
                assert session.scalar(select(func.count()).select_from(User)) >= 1
                session.rollback()
                return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create_human), executor.submit(create_service)]
        outcomes = sorted([future.result() for future in futures])

    with session_factory() as session:
        human_count = session.scalar(
            select(func.count()).select_from(UserIdentity).where(UserIdentity.subject == "race-sub")
        )
        service_count = session.scalar(
            select(func.count()).select_from(ServicePrincipal).where(ServicePrincipal.external_subject == "race-sub")
        )
        binding_count = session.scalar(
            select(func.count()).select_from(ExternalPrincipalIdentity).where(
                ExternalPrincipalIdentity.subject == "race-sub"
            )
        )

    assert outcomes.count("blocked") == 1
    assert sum(outcome.startswith("created-") for outcome in outcomes) == 1
    assert int(human_count or 0) + int(service_count or 0) == 1
    assert binding_count == 1


def test_global_identity_binding_is_database_enforced(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org_a = repo.create_organization(OrganizationCreate(slug="global-a", name="Global A"))
        org_b = repo.create_organization(OrganizationCreate(slug="global-b", name="Global B"))
        user_a = repo.create_user(UserCreate(email="global-a@example.com"))
        user_b = repo.create_user(UserCreate(email="global-b@example.com"))
        repo.add_membership(org_a.id, user_a.id, "viewer")
        repo.add_membership(org_b.id, user_b.id, "viewer")
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        idp_a = security.create_identity_provider(
            org_a.id,
            IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience=AUDIENCE, jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        idp_b = security.create_identity_provider(
            org_b.id,
            IdentityProviderCreate(name="primary", issuer=ISSUER_A, audience=AUDIENCE, jwks_uri=JWKS_URI),
            allow_development_targets=True,
        )
        security.create_user_identity(
            org_a.id,
            UserIdentityCreate(user_id=user_a.id, identity_provider_id=idp_a.id, issuer=ISSUER_A, subject="global-sub"),
        )
        session.flush()

        with pytest.raises(IntegrityError):
            session.add(
                UserIdentity(
                    organization_id=org_b.id,
                    user_id=user_b.id,
                    identity_provider_id=idp_b.id,
                    issuer=ISSUER_A,
                    subject="global-sub",
                    profile={},
                )
            )
            session.flush()


def test_membership_and_tenant_authorization_fail_closed(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        viewer = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["viewer"]),
            request_id="req-viewer",
        )
        assert viewer.role == "viewer"
        with pytest.raises(PermissionError):
            security.require_permission(viewer, MAINTENANCE_MANAGE, action="maintenance.case.create")

        membership = session.get(OrganizationMembership, viewer.user_id)
        assert membership is None
        actual_membership = session.query(OrganizationMembership).filter_by(user_id=viewer.user_id).one()
        actual_membership.lifecycle_state = "inactive"
        session.flush()
        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["organization_id"],
                _claims(fixture["subjects"]["viewer"]),
                request_id="req",
            )
        actual_membership.lifecycle_state = "active"
        inactive_user = session.get(User, viewer.user_id)
        inactive_user.lifecycle_state = "inactive"
        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["organization_id"],
                _claims(fixture["subjects"]["viewer"]),
                request_id="req-inactive-user",
            )
        inactive_user.lifecycle_state = "archived"
        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["organization_id"],
                _claims(fixture["subjects"]["viewer"]),
                request_id="req-archived-user",
            )
        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["other_organization_id"],
                _claims(fixture["subjects"]["engineer"]),
                request_id="req-cross",
            )


def test_authentication_rejects_inactive_organizations(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        org = session.get(Organization, fixture["organization_id"])
        org.lifecycle_state = "inactive"
        claims = _claims(fixture["subjects"]["owner"])
        security = SecurityService(session, verifier=DeterministicTokenVerifier({"owner-token": claims}))

        with pytest.raises(PermissionError):
            security.context_from_claims(fixture["organization_id"], claims, request_id="req-inactive-org")
        with pytest.raises(PermissionError):
            security.authenticate_bearer(fixture["organization_id"], "owner-token", request_id="req-inactive-org")


def test_role_policy_and_owner_protection_rules(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        repo = PlatformRepository(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        admin = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["admin"]),
            request_id="req-admin",
        )
        owner = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-owner",
        )

        with pytest.raises(PermissionError):
            security.change_membership_role(
                fixture["organization_id"],
                MembershipChange(user_id=fixture["users"]["admin"], role="owner"),
                actor=admin,
            )
        unreachable_user = repo.create_user(UserCreate(email="unreachable-owner@example.com"))
        repo.add_membership(fixture["organization_id"], unreachable_user.id, "viewer")
        with pytest.raises(PermissionError):
            security.change_membership_role(
                fixture["organization_id"],
                MembershipChange(user_id=unreachable_user.id, role="owner"),
                actor=owner,
            )
        security.create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=unreachable_user.id,
                identity_provider_id=fixture["idp_id"],
                issuer=ISSUER_A,
                subject="reachable-new-owner-sub",
            ),
        )
        reachable_promoted = security.change_membership_role(
            fixture["organization_id"],
            MembershipChange(user_id=unreachable_user.id, role="owner"),
            actor=owner,
        )
        assert reachable_promoted.role == "owner"
        reachable_promoted.role = "viewer"
        session.flush()
        promoted = security.change_membership_role(
            fixture["organization_id"],
            MembershipChange(user_id=fixture["users"]["admin"], role="owner"),
            actor=owner,
        )
        assert promoted.role == "owner"

        original_owner = session.query(OrganizationMembership).filter_by(user_id=fixture["users"]["owner"]).one()
        promoted.lifecycle_state = "inactive"
        legacy_unreachable = repo.create_user(UserCreate(email="legacy-unreachable-owner@example.com"))
        repo.add_membership(fixture["organization_id"], legacy_unreachable.id, "owner")
        session.flush()
        with pytest.raises(PermissionError):
            security.change_membership_role(
                fixture["organization_id"],
                MembershipChange(user_id=fixture["users"]["owner"], role="admin"),
                actor=owner,
            )
        assert original_owner.role == "owner"
        with pytest.raises(PermissionError):
            security.change_membership_status(
                fixture["organization_id"],
                fixture["users"]["owner"],
                MembershipStatusChange(lifecycle_state="inactive"),
                actor=owner,
        )
        assert original_owner.lifecycle_state == "active"


def test_onboarding_uses_centralized_owner_role_policy(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        admin = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["admin"]),
            request_id="req-admin",
        )
        owner = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-owner",
        )

        for role in ["viewer", "technician", "engineer", "admin"]:
            result = security.onboard_user_identity(
                fixture["organization_id"],
                UserIdentityOnboard(
                    email=f"new-{role}@example.com",
                    full_name=f"New {role.title()}",
                    identity_provider_id=fixture["idp_id"],
                    issuer=ISSUER_A,
                    subject=f"new-{role}-sub",
                    role=role,
                ),
                actor=admin,
            )
            assert result["membership"].role == role

        with pytest.raises(PermissionError):
            security.onboard_user_identity(
                fixture["organization_id"],
                UserIdentityOnboard(
                    email="new-owner-denied@example.com",
                    identity_provider_id=fixture["idp_id"],
                    issuer=ISSUER_A,
                    subject="new-owner-denied-sub",
                    role="owner",
                ),
                actor=admin,
            )

        result = security.onboard_user_identity(
            fixture["organization_id"],
            UserIdentityOnboard(
                email="new-owner@example.com",
                identity_provider_id=fixture["idp_id"],
                issuer=ISSUER_A,
                subject="new-owner-sub",
                role="owner",
            ),
            actor=owner,
        )
        assert result["membership"].role == "owner"


def test_onboarding_existing_identity_cannot_bypass_final_owner_protection(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        owner = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-owner",
        )
        for role in ["admin", "engineer", "technician", "viewer"]:
            membership = session.query(OrganizationMembership).filter_by(user_id=fixture["users"][role]).one()
            membership.lifecycle_state = "inactive"
        session.flush()

        with pytest.raises(PermissionError):
            security.onboard_user_identity(
                fixture["organization_id"],
                UserIdentityOnboard(
                    email="owner@example.com",
                    identity_provider_id=fixture["idp_id"],
                    issuer=ISSUER_A,
                    subject=fixture["subjects"]["owner"],
                    role="admin",
                ),
                actor=owner,
            )

        owner_membership = session.query(OrganizationMembership).filter_by(user_id=fixture["users"]["owner"]).one()
        assert owner_membership.role == "owner"
        assert owner_membership.lifecycle_state == "active"


def test_onboarding_validates_identity_binding_before_membership_mutation(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session)
        owner = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-owner",
        )
        conflicting_idp = security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="same-issuer-secondary",
                issuer=ISSUER_A,
                audience="secondary-audience",
                jwks_uri=JWKS_URI,
            ),
            allow_development_targets=True,
        )
        membership = session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == fixture["organization_id"],
                OrganizationMembership.user_id == fixture["users"]["bob"],
            )
        )
        assert membership.role == "technician"

        with pytest.raises(PermissionError):
            security.onboard_user_identity(
                fixture["organization_id"],
                UserIdentityOnboard(
                    email="bob@example.com",
                    identity_provider_id=conflicting_idp.id,
                    issuer=ISSUER_A,
                    subject=fixture["subjects"]["bob"],
                    role="admin",
                ),
                actor=owner,
            )

        assert membership.role == "technician"
        assert membership.lifecycle_state == "active"


def test_postgres_serializes_concurrent_final_owner_demotions(migrated_db):
    engine, session_factory = migrated_db
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock regression requires PostgreSQL")

    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        owner = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-owner",
        )
        security.change_membership_role(
            fixture["organization_id"],
            MembershipChange(user_id=fixture["users"]["admin"], role="owner"),
            actor=owner,
        )
        session.commit()

    barrier = threading.Barrier(2)

    def demote_self(subject: str) -> str:
        with session_factory() as session:
            security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
            actor = security.context_from_claims(fixture["organization_id"], _claims(subject), request_id=subject)
            barrier.wait(timeout=10)
            try:
                security.change_membership_role(
                    fixture["organization_id"],
                    MembershipChange(user_id=actor.user_id, role="admin"),
                    actor=actor,
                )
                time.sleep(0.2)
                session.commit()
                return "demoted"
            except PermissionError:
                session.rollback()
                return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(
            [
                executor.submit(demote_self, fixture["subjects"]["owner"]),
                executor.submit(demote_self, fixture["subjects"]["admin"]),
            ],
            key=lambda future: future.result(),
        )
        outcomes = [future.result() for future in results]

    with session_factory() as session:
        active_owner_count = session.scalar(
            text(
                "select count(*) from organization_memberships "
                "where organization_id = :organization_id and role = 'owner' and lifecycle_state = 'active'"
            ),
            {"organization_id": fixture["organization_id"]},
        )

    assert outcomes == ["blocked", "demoted"]
    assert active_owner_count == 1


def test_postgres_serializes_concurrent_final_identity_provider_deactivation(migrated_db):
    engine, session_factory = migrated_db
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock regression requires PostgreSQL")

    with session_factory() as session:
        fixture = _seed_security(session)
        second_idp = SecurityService(session, verifier=DeterministicTokenVerifier({})).create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/.well-known/jwks.json",
            ),
            allow_development_targets=True,
        )
        SecurityService(session, verifier=DeterministicTokenVerifier({})).create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=fixture["users"]["owner"],
                identity_provider_id=second_idp.id,
                issuer=ISSUER_B,
                subject="owner-concurrent-secondary-sub",
            ),
        )
        idp_ids = [fixture["idp_id"], second_idp.id]
        session.commit()

    barrier = threading.Barrier(2)

    def deactivate(identity_provider_id: str) -> str:
        with session_factory() as session:
            security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
            barrier.wait(timeout=10)
            try:
                security.update_identity_provider(
                    fixture["organization_id"],
                    identity_provider_id,
                    IdentityProviderUpdate(status="inactive"),
                )
                time.sleep(0.2)
                session.commit()
                return "deactivated"
            except PermissionError:
                session.rollback()
                return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = sorted(
            [executor.submit(deactivate, identity_provider_id) for identity_provider_id in idp_ids],
            key=lambda future: future.result(),
        )
        outcomes = [future.result() for future in results]

    with session_factory() as session:
        active_idp_count = session.scalar(
            text(
                "select count(*) from organization_identity_providers "
                "where organization_id = :organization_id and status = 'active'"
            ),
            {"organization_id": fixture["organization_id"]},
        )

    assert outcomes == ["blocked", "deactivated"]
    assert active_idp_count == 1


def test_service_principal_scopes_are_machine_only(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/.well-known/jwks.json",
            ),
            allow_development_targets=True,
        )
        service = security.context_from_claims(
            fixture["organization_id"],
            _claims("robot-sub"),
            request_id="req-service",
        )

        assert service.principal_type == "service"
        security.require_permission(service, INGESTION_WRITE, action="ingestion.write")
        with pytest.raises(PermissionError):
            security.require_permission(service, MAINTENANCE_MANAGE, action="maintenance.note.add")
        with pytest.raises(PermissionError):
            security.context_from_claims(
                fixture["organization_id"],
                _claims("robot-sub", issuer=ISSUER_B),
                request_id="req-wrong-issuer",
            )
        with pytest.raises(ValidationError):
            ServicePrincipalCreate(
                name="unbound-robot",
                external_subject="unbound-robot",
                permissions=[INGESTION_WRITE],
            )
        with pytest.raises(ValueError):
            security.create_service_principal(
                fixture["organization_id"],
                ServicePrincipalCreate(
                    name="bad-robot",
                    identity_provider_id=fixture["idp_id"],
                    external_subject="bad-robot",
                    issuer=ISSUER_A,
                    permissions=[ML_MODEL_PROMOTE_PRODUCTION],
                ),
            )
        assert INGESTION_WRITE in SERVICE_ALLOWED_PERMISSIONS
        assert SERVING_PREDICT in SERVICE_ALLOWED_PERMISSIONS


def test_secret_references_do_not_expose_values_and_plaintext_source_config_is_rejected(migrated_db, monkeypatch):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        secret = security.create_secret_reference(
            fixture["organization_id"],
            SecretReferenceCreate(
                name="abb-token",
                purpose="abb_api_token",
                provider="env",
                locator="ABB_TOKEN",
                rotation_metadata={"last_token": "should-redact"},
            ),
            created_by_user_id=fixture["users"]["owner"],
        )
        monkeypatch.setenv("ABB_TOKEN", "super-secret-value")
        memory_resolver = InMemorySecretResolver({"ABB_TOKEN": "super-secret-value"})
        environment_resolver = EnvironmentSecretResolver()
        resolved = memory_resolver.resolve(secret)

        assert resolved == "super-secret-value"
        assert environment_resolver.resolve(secret) == "super-secret-value"
        secret.status = "rotating"
        assert memory_resolver.resolve(secret) == "super-secret-value"
        assert environment_resolver.resolve(secret) == "super-secret-value"
        for status in ["inactive", "archived"]:
            secret.status = status
            with pytest.raises(SecretResolutionError):
                memory_resolver.resolve(secret)
            with pytest.raises(SecretResolutionError):
                environment_resolver.resolve(secret)
        payload = secret_reference_payload(secret)
        assert "locator" not in payload
        assert "super-secret-value" not in str(payload)
        assert payload["rotation_metadata"]["last_token"] == "[REDACTED]"
        with pytest.raises(ValueError):
            assert_no_plaintext_secrets({"nested": {"api_key": "plaintext"}})
        for invalid_reference in [{}, {"name": "hunter2"}, {"secret_reference_id": ""}]:
            with pytest.raises(ValueError):
                assert_no_plaintext_secrets({"auth": {"password": invalid_reference}})
        secret_reference = {"secret_reference_id": secret.id}
        for key in [
            "private_key",
            "private-key",
            "passphrase",
            "dsn",
            "database_dsn",
            "credential_url",
            "connection_string",
        ]:
            with pytest.raises(ValueError):
                assert_no_plaintext_secrets({"connector": {key: "plaintext"}})
            assert_no_plaintext_secrets({"connector": {key: secret_reference}})
        for value in [
            "https://user:password@vendor.example/api",
            "postgresql://user:password@db.example:5432/pms",
            "https://vendor.example/api?token=plaintext",
            "https://vendor.example/api?key=plaintext",
            "https://vendor.example/api?client_secret=plaintext",
        ]:
            with pytest.raises(ValueError):
                assert_no_plaintext_secrets({"endpoint": value})
        assert_no_plaintext_secrets(
            {
                "endpoint_url": "https://vendor.example/api",
                "callback": "https://vendor.example/callback?mode=health",
                "retry": {"timeout_seconds": 10},
            }
        )
        IngestionService(session).register_source(
            SourceRegistration(
                organization_id=fixture["organization_id"],
                source_type="abb",
                name="ABB",
                config={"auth": {"token": {"secret_reference_id": secret.id}}},
            )
        )
        with pytest.raises(ValueError):
            IngestionService(session).register_source(
                SourceRegistration(
                    organization_id=fixture["organization_id"],
                    source_type="abb",
                    name="Bad ABB",
                    config={"auth": {"token": "plaintext"}},
                )
            )


def test_enterprise_ingestion_source_registration_omits_legacy_plaintext_config(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.add(
            IngestionSource(
                organization_id=fixture["organization_id"],
                name="Legacy ABB",
                source_type="abb",
                status="active",
                config={"auth": {"token": "legacy-plaintext-token"}},
            )
        )
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    response = client.post(
        f"/api/ingestion/{fixture['organization_id']}/sources",
        headers={"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='engineer-sub')}"},
        json={
            "organization_id": fixture["organization_id"],
            "source_type": "abb",
            "name": "Legacy ABB",
            "config": {"auth": {"token": {"secret_reference_id": "safe-reference"}}},
        },
    )

    assert response.status_code == 200
    assert "config" not in response.json()
    assert "legacy-plaintext-token" not in response.text


@pytest.mark.parametrize(
    ("path", "body", "headers"),
    [
        ("/api/ingestion/{org_id}/rest", {"records": []}, {}),
        ("/api/ingestion/{org_id}/mqtt", b"{}", {}),
        ("/api/ingestion/{org_id}/opcua", {"records": []}, {}),
        ("/api/ingestion/{org_id}/abb", {"records": []}, {}),
        ("/api/ingestion/{org_id}/files/csv", b"timestamp,sensor_path,value,unit\n", {}),
        ("/api/ingestion/{org_id}/files/parquet", b"PAR1", {}),
    ],
)
def test_ingestion_authorization_database_failures_are_translated(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
    path,
    body,
    headers,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)

    def failing_authorize(*args, **kwargs):
        raise SQLAlchemyError("audit flush failed")

    monkeypatch.setattr(app_main, "_authorize", failing_authorize)
    request_headers = {
        "Authorization": f"Bearer {_token(rsa_keys['primary'], subject='engineer-sub')}",
        **headers,
    }
    url = path.format(org_id=fixture["organization_id"])
    if isinstance(body, bytes):
        response = client.post(url, headers=request_headers, content=body)
    else:
        response = client.post(url, headers=request_headers, json=body)

    assert response.status_code == 503
    assert response.json()["detail"].startswith("platform database unavailable")


def test_security_configuration_rejects_dangerous_production_settings(monkeypatch):
    monkeypatch.setenv("PMS_ENVIRONMENT", "production")
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(SecurityConfigurationError):
        security_settings()
    monkeypatch.setenv("PMS_CORS_ALLOWED_ORIGINS", "https://studio.example")
    monkeypatch.setenv("PMS_ENABLE_DOCS", "1")
    with pytest.raises(SecurityConfigurationError):
        security_settings()
    monkeypatch.setenv("PMS_ENABLE_DOCS", "0")
    monkeypatch.setenv("PMS_TEST_AUTH", "1")
    with pytest.raises(SecurityConfigurationError):
        security_settings()
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_TEST_AUTH", "0")
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprize")
    with pytest.raises(SecurityConfigurationError):
        security_settings()
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_ENVIRONMENT", "prod")
    with pytest.raises(SecurityConfigurationError):
        security_settings()
    with pytest.raises(SecurityConfigurationError):
        validate_oidc_endpoint("http://127.0.0.1/jwks")
    for unsafe_url in [
        "https://user:pass@issuer.example/jwks",
        "https://issuer.example/jwks?token=plaintext",
        "https://issuer.example/jwks?key=plaintext",
        "https://issuer.example/jwks?secret=plaintext",
        "https://issuer.example/jwks?password=plaintext",
        "https://issuer.example/jwks?tenant=acme",
    ]:
        with pytest.raises(SecurityConfigurationError):
            validate_oidc_endpoint(unsafe_url, allow_development_targets=True)
    validate_oidc_endpoint("https://issuer.example/jwks", allow_development_targets=True)
    validate_oidc_endpoint("https://issuer.example/jwks?version=2026-08", allow_development_targets=True)
    with pytest.raises(ValidationError):
        IdentityProviderCreate(
            name="symmetric",
            issuer=ISSUER_A,
            audience=AUDIENCE,
            jwks_uri=JWKS_URI,
            allowed_algorithms=["HS256"],
        )
    with pytest.raises(ValidationError):
        BootstrapSecurityRequest(
            organization_slug="acme",
            organization_name="Acme",
            owner_email="owner@example.com",
            issuer=ISSUER_A,
            subject="owner-sub",
            audience=AUDIENCE,
            jwks_uri=JWKS_URI,
            allowed_algorithms=["HS256"],
        )


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "-inf", "not-a-number"])
def test_oidc_http_timeout_settings_reject_invalid_values(monkeypatch, timeout):
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_OIDC_HTTP_TIMEOUT_SECONDS", timeout)

    with pytest.raises(SecurityConfigurationError):
        security_settings()


@pytest.mark.parametrize("timeout", ["0.001", "2", "2.5"])
def test_oidc_http_timeout_settings_accept_positive_finite_values(monkeypatch, timeout):
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_OIDC_HTTP_TIMEOUT_SECONDS", timeout)

    assert security_settings().oidc_http_timeout_seconds == float(timeout)


def test_oidc_resolved_address_validation_rejects_unsafe_dns_answers(monkeypatch):
    def resolver_for(addresses: list[str]):
        def fake_getaddrinfo(host, port, *args, **kwargs):
            assert host == "issuer.example"
            assert port == 443
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in addresses
            ]

        return fake_getaddrinfo

    monkeypatch.setattr("enterprise_security.service.socket.getaddrinfo", resolver_for(["93.184.216.34"]))
    assert _validated_oidc_addresses("issuer.example", 443) == ("93.184.216.34",)

    for address in [
        "127.0.0.1",
        "10.0.0.5",
        "100.64.0.1",
        "169.254.169.254",
        "192.0.2.1",
        "203.0.113.1",
    ]:
        monkeypatch.setattr("enterprise_security.service.socket.getaddrinfo", resolver_for([address]))
        with pytest.raises(SecurityConfigurationError):
            _validated_oidc_addresses("issuer.example", 443)
        with pytest.raises(SecurityConfigurationError):
            validate_oidc_endpoint(f"https://{address}/jwks")

    monkeypatch.setattr(
        "enterprise_security.service.socket.getaddrinfo",
        resolver_for(["93.184.216.34", "127.0.0.1"]),
    )
    with pytest.raises(SecurityConfigurationError):
        _validated_oidc_addresses("issuer.example", 443)


def test_oidc_jwks_fetch_pins_validated_connection_destination(monkeypatch):
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 11\r\n"
        b"\r\n"
        b'{"keys":[]}'
    )
    resolver_calls = []
    connected_addresses = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        resolver_calls.append((host, port))
        if len(resolver_calls) == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    class FakeRawSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTlsSocket:
        def __init__(self):
            self._chunks = [response, b""]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def settimeout(self, timeout):
            assert 0 < timeout <= 2.0

        def sendall(self, request):
            assert b"Host: issuer.example\r\n" in request

        def recv(self, _size):
            return self._chunks.pop(0)

    class FakeSslContext:
        def wrap_socket(self, raw_socket, *, server_hostname):
            assert isinstance(raw_socket, FakeRawSocket)
            assert server_hostname == "issuer.example"
            return FakeTlsSocket()

    def fake_create_connection(address, *, timeout):
        connected_addresses.append(address)
        assert timeout == 2.0
        return FakeRawSocket()

    monkeypatch.setattr("enterprise_security.service.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("enterprise_security.service.socket.create_connection", fake_create_connection)
    monkeypatch.setattr("enterprise_security.service.ssl.create_default_context", lambda: FakeSslContext())

    assert _fetch_jwks_with_pinned_address("https://issuer.example/.well-known/jwks.json", timeout_seconds=2.0) == {
        "keys": []
    }
    assert resolver_calls == [("issuer.example", 443)]
    assert connected_addresses == [("93.184.216.34", 443)]


@pytest.mark.parametrize(
    ("uri", "host", "address", "port", "expected_host_header"),
    [
        (
            "https://[2606:4700:4700::1111]/jwks",
            "2606:4700:4700::1111",
            "2606:4700:4700::1111",
            443,
            b"Host: [2606:4700:4700::1111]\r\n",
        ),
        (
            "https://[2606:4700:4700::1111]:8443/jwks",
            "2606:4700:4700::1111",
            "2606:4700:4700::1111",
            8443,
            b"Host: [2606:4700:4700::1111]:8443\r\n",
        ),
        ("https://93.184.216.34/jwks", "93.184.216.34", "93.184.216.34", 443, b"Host: 93.184.216.34\r\n"),
        ("https://issuer.example/jwks", "issuer.example", "93.184.216.34", 443, b"Host: issuer.example\r\n"),
    ],
)
def test_oidc_jwks_fetch_formats_host_header_for_literals_and_hostnames(
    monkeypatch,
    uri,
    host,
    address,
    port,
    expected_host_header,
):
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 11\r\n"
        b"\r\n"
        b'{"keys":[]}'
    )

    def fake_getaddrinfo(resolved_host, resolved_port, *args, **kwargs):
        assert resolved_host == host
        assert resolved_port == port
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, port))]

    class FakeRawSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTlsSocket:
        def __init__(self):
            self._chunks = [response, b""]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def settimeout(self, _timeout):
            return None

        def sendall(self, request):
            assert expected_host_header in request

        def recv(self, _size):
            return self._chunks.pop(0)

    class FakeSslContext:
        def wrap_socket(self, raw_socket, *, server_hostname):
            assert isinstance(raw_socket, FakeRawSocket)
            assert server_hostname == host
            return FakeTlsSocket()

    def fake_create_connection(destination, *, timeout):
        assert destination == (address, port)
        assert timeout == 2.0
        return FakeRawSocket()

    monkeypatch.setattr("enterprise_security.service.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("enterprise_security.service.socket.create_connection", fake_create_connection)
    monkeypatch.setattr("enterprise_security.service.ssl.create_default_context", lambda: FakeSslContext())

    assert _fetch_jwks_with_pinned_address(uri, timeout_seconds=2.0) == {"keys": []}


def test_oidc_jwks_fetch_bounds_response_size_and_overall_deadline(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host == "issuer.example"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    class FakeRawSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeSslContext:
        def __init__(self, tls_socket):
            self.tls_socket = tls_socket

        def wrap_socket(self, raw_socket, *, server_hostname):
            assert isinstance(raw_socket, FakeRawSocket)
            assert server_hostname == "issuer.example"
            return self.tls_socket

    class OversizedTlsSocket:
        def __init__(self):
            self._chunks = [
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
                b"x" * (MAX_JWKS_RESPONSE_BYTES + 1),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def settimeout(self, _timeout):
            return None

        def sendall(self, _request):
            return None

        def recv(self, _size):
            return self._chunks.pop(0)

    monkeypatch.setattr("enterprise_security.service.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("enterprise_security.service.socket.create_connection", lambda *args, **kwargs: FakeRawSocket())
    monkeypatch.setattr(
        "enterprise_security.service.ssl.create_default_context",
        lambda: FakeSslContext(OversizedTlsSocket()),
    )
    with pytest.raises(AuthenticationError):
        _fetch_jwks_with_pinned_address("https://issuer.example/.well-known/jwks.json", timeout_seconds=2.0)

    class TrickleTlsSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def settimeout(self, _timeout):
            return None

        def sendall(self, _request):
            return None

        def recv(self, _size):
            return b"x"

    monotonic_now = {"value": 0.0}

    def fake_monotonic():
        monotonic_now["value"] += 0.6
        return monotonic_now["value"]

    monkeypatch.setattr(
        "enterprise_security.service.ssl.create_default_context",
        lambda: FakeSslContext(TrickleTlsSocket()),
    )
    monkeypatch.setattr("enterprise_security.service.time.monotonic", fake_monotonic)
    with pytest.raises(AuthenticationError):
        _fetch_jwks_with_pinned_address("https://issuer.example/.well-known/jwks.json", timeout_seconds=1.0)


def test_bootstrap_cli_algorithm_override_replaces_default(monkeypatch):
    import scripts.bootstrap_enterprise_security as bootstrap_cli

    base_args = [
        "bootstrap_enterprise_security.py",
        "--organization-slug",
        "acme",
        "--organization-name",
        "Acme",
        "--owner-email",
        "owner@example.com",
        "--issuer",
        ISSUER_A,
        "--subject",
        "owner-sub",
        "--audience",
        AUDIENCE,
        "--jwks-uri",
        JWKS_URI,
    ]
    monkeypatch.setattr("sys.argv", base_args)
    assert bootstrap_cli.parse_args().allowed_algorithm is None

    monkeypatch.setattr("sys.argv", [*base_args, "--allowed-algorithm", "ES256"])
    assert bootstrap_cli.parse_args().allowed_algorithm == ["ES256"]


def test_bootstrap_security_provisioning_is_idempotent_and_org_scoped(migrated_db):
    _engine, session_factory = migrated_db
    request = BootstrapSecurityRequest(
        organization_slug="acme",
        organization_name="Acme",
        owner_email="owner@example.com",
        issuer=ISSUER_A,
        subject="owner-sub",
        audience=AUDIENCE,
        jwks_uri=JWKS_URI,
    )
    with session_factory() as session:
        first = SecurityService(session).bootstrap_initial_owner(request)
        second = SecurityService(session).bootstrap_initial_owner(request)
        session.commit()

        assert first == second
        assert session.query(User).count() == 1
        assert session.query(OrganizationMembership).count() == 1

        with pytest.raises(PermissionError):
            SecurityService(session).bootstrap_initial_owner(
                request.model_copy(update={"organization_slug": "globex", "organization_name": "Globex"})
            )
        session.rollback()

        second_org = SecurityService(session).bootstrap_initial_owner(
            request.model_copy(
                update={
                    "organization_slug": "globex",
                    "organization_name": "Globex",
                    "owner_email": "globex-owner@example.com",
                    "subject": "globex-owner-sub",
                }
            )
        )
        context = SecurityService(session).context_from_claims(
            second_org["organization_id"],
            _claims("globex-owner-sub"),
            request_id="req-second-org",
        )

        assert context.user_id == second_org["user_id"]
        assert session.query(User).count() == 2
        assert session.query(OrganizationMembership).count() == 2
        assert session.query(UserIdentity).count() == 2


def test_bootstrap_rejects_incompatible_existing_provider_binding(migrated_db):
    _engine, session_factory = migrated_db
    request = BootstrapSecurityRequest(
        organization_slug="acme",
        organization_name="Acme",
        owner_email="owner@example.com",
        issuer=ISSUER_A,
        subject="owner-sub",
        audience="audience-a",
        jwks_uri=JWKS_URI,
    )
    with session_factory() as session:
        first = SecurityService(session).bootstrap_initial_owner(request)
        same = SecurityService(session).bootstrap_initial_owner(request)
        session.commit()

        assert same == first

        with pytest.raises(SecurityConfigurationError):
            SecurityService(session).bootstrap_initial_owner(
                request.model_copy(
                    update={
                        "audience": "audience-b",
                        "idp_name": "secondary",
                    }
                )
            )
        session.rollback()

        context = SecurityService(session).context_from_claims(
            first["organization_id"],
            _claims("owner-sub", audience="audience-a"),
            request_id="req-bootstrap-provider",
            identity_provider_id=first["identity_provider_id"],
        )

        assert context.user_id == first["user_id"]


def test_bootstrap_rejects_inactive_existing_owner_user(migrated_db):
    _engine, session_factory = migrated_db
    request = BootstrapSecurityRequest(
        organization_slug="acme",
        organization_name="Acme",
        owner_email="owner@example.com",
        issuer=ISSUER_A,
        subject="owner-sub",
        audience=AUDIENCE,
        jwks_uri=JWKS_URI,
    )
    with session_factory() as session:
        repo = PlatformRepository(session)
        user = repo.create_user(UserCreate(email="owner@example.com"))
        user.lifecycle_state = "inactive"

        with pytest.raises(PermissionError):
            SecurityService(session).bootstrap_initial_owner(request)

        assert session.query(OrganizationMembership).count() == 0
        assert session.query(UserIdentity).count() == 0


@pytest.mark.parametrize("org_state", ["inactive", "archived"])
def test_bootstrap_rejects_inactive_or_archived_existing_organization(migrated_db, org_state):
    _engine, session_factory = migrated_db
    request = BootstrapSecurityRequest(
        organization_slug="acme",
        organization_name="Acme",
        owner_email="owner@example.com",
        issuer=ISSUER_A,
        subject="owner-sub",
        audience=AUDIENCE,
        jwks_uri=JWKS_URI,
    )
    with session_factory() as session:
        org = PlatformRepository(session).create_organization(OrganizationCreate(slug="acme", name="Acme"))
        org.lifecycle_state = org_state
        session.flush()

        with pytest.raises(SecurityConfigurationError):
            SecurityService(session).bootstrap_initial_owner(request)

        assert session.query(OrganizationIdentityProvider).count() == 0
        assert session.query(OrganizationMembership).count() == 0
        assert session.query(UserIdentity).count() == 0


def test_identity_provider_update_preserves_recovery_path(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session)

        with pytest.raises(PermissionError):
            security.update_identity_provider(
                fixture["organization_id"],
                fixture["idp_id"],
                IdentityProviderUpdate(status="inactive"),
            )

        second_idp = security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/jwks",
            ),
            allow_development_targets=True,
        )
        security.create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=fixture["users"]["owner"],
                identity_provider_id=second_idp.id,
                issuer=ISSUER_B,
                subject="owner-b-sub",
            ),
        )
        deactivated = security.update_identity_provider(
            fixture["organization_id"],
            fixture["idp_id"],
            IdentityProviderUpdate(status="inactive"),
        )
        assert deactivated.status == "inactive"

        bootstrap_result = security.bootstrap_initial_owner(
            BootstrapSecurityRequest(
                organization_slug="acme",
                organization_name="Acme Manufacturing",
                owner_email="owner@example.com",
                issuer=ISSUER_A,
                subject=fixture["subjects"]["owner"],
                audience=AUDIENCE,
                jwks_uri=JWKS_URI,
            )
        )
        session.flush()

        assert second_idp.status == "active"
        assert bootstrap_result["identity_provider_id"] == fixture["idp_id"]
        assert session.get(OrganizationIdentityProvider, fixture["idp_id"]).status == "active"


def test_identity_provider_deactivation_requires_owner_reachable_through_another_provider(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session)
        second_idp = security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/jwks",
            ),
            allow_development_targets=True,
        )

        with pytest.raises(PermissionError):
            security.update_identity_provider(
                fixture["organization_id"],
                fixture["idp_id"],
                IdentityProviderUpdate(status="inactive"),
            )

        security.create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=fixture["users"]["owner"],
                identity_provider_id=second_idp.id,
                issuer=ISSUER_B,
                subject="owner-secondary-sub",
            ),
        )

        deactivated = security.update_identity_provider(
            fixture["organization_id"],
            fixture["idp_id"],
            IdentityProviderUpdate(status="inactive"),
        )

        assert deactivated.status == "inactive"


@pytest.mark.parametrize("owner_state", ["inactive", "archived"])
def test_identity_provider_deactivation_ignores_inactive_or_archived_owners(migrated_db, owner_state):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session)
        second_idp = security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/jwks",
            ),
            allow_development_targets=True,
        )
        security.change_membership_role(
            fixture["organization_id"],
            MembershipChange(user_id=fixture["users"]["admin"], role="owner"),
            actor=security.context_from_claims(
                fixture["organization_id"],
                _claims(fixture["subjects"]["owner"]),
                request_id="req-owner",
            ),
        )
        security.create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=fixture["users"]["admin"],
                identity_provider_id=second_idp.id,
                issuer=ISSUER_B,
                subject="admin-secondary-owner-sub",
            ),
        )
        admin_membership = session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == fixture["organization_id"],
                OrganizationMembership.user_id == fixture["users"]["admin"],
            )
        )
        admin_membership.lifecycle_state = owner_state
        session.flush()

        with pytest.raises(PermissionError):
            security.update_identity_provider(
                fixture["organization_id"],
                fixture["idp_id"],
                IdentityProviderUpdate(status="inactive"),
            )

        assert session.get(OrganizationIdentityProvider, fixture["idp_id"]).status == "active"


def test_identity_provider_deactivation_ignores_inactive_recovery_provider(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session)
        second_idp = security.create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/jwks",
            ),
            allow_development_targets=True,
        )
        security.create_user_identity(
            fixture["organization_id"],
            UserIdentityCreate(
                user_id=fixture["users"]["owner"],
                identity_provider_id=second_idp.id,
                issuer=ISSUER_B,
                subject="owner-inactive-provider-sub",
            ),
        )
        second_idp.status = "inactive"
        session.flush()

        with pytest.raises(PermissionError):
            security.update_identity_provider(
                fixture["organization_id"],
                fixture["idp_id"],
                IdentityProviderUpdate(status="inactive"),
            )

        assert session.get(OrganizationIdentityProvider, fixture["idp_id"]).status == "active"


def test_bootstrap_accepts_long_oidc_metadata_without_overflowing_external_subject(migrated_db):
    _engine, session_factory = migrated_db
    long_issuer = f"https://{'i' * 492}.example/"
    long_subject = "s" * 255
    with session_factory() as session:
        result = SecurityService(session).bootstrap_initial_owner(
            BootstrapSecurityRequest(
                organization_slug="long-oidc",
                organization_name="Long OIDC",
                owner_email="owner-long@example.com",
                issuer=long_issuer,
                subject=long_subject,
                audience=AUDIENCE,
                jwks_uri=JWKS_URI,
            )
        )
        user = session.get(User, result["user_id"])

    assert user.external_subject is None


def test_bootstrap_rejects_unsafe_oidc_endpoint_in_production(migrated_db, monkeypatch):
    _engine, session_factory = migrated_db
    monkeypatch.setenv("PMS_ENVIRONMENT", "production")
    request = BootstrapSecurityRequest(
        organization_slug="acme",
        organization_name="Acme",
        owner_email="owner@example.com",
        issuer=ISSUER_A,
        subject="owner-sub",
        audience=AUDIENCE,
        jwks_uri="http://127.0.0.1/jwks",
    )
    with session_factory() as session:
        with pytest.raises(SecurityConfigurationError):
            SecurityService(session).bootstrap_initial_owner(request)


def test_bootstrap_rejects_credential_bearing_oidc_endpoint_without_persisting_it(migrated_db, monkeypatch):
    _engine, session_factory = migrated_db
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    with session_factory() as session:
        with pytest.raises(SecurityConfigurationError):
            SecurityService(session).bootstrap_initial_owner(
                BootstrapSecurityRequest(
                    organization_slug="credential-jwks",
                    organization_name="Credential JWKS",
                    owner_email="owner-credential-jwks@example.com",
                    issuer=ISSUER_A,
                    subject="credential-owner-sub",
                    audience=AUDIENCE,
                    jwks_uri="https://user:pass@issuer.example/jwks",
                )
            )
        assert session.query(OrganizationIdentityProvider).count() == 0
        assert session.query(OrganizationIdentityProvider).filter(
            OrganizationIdentityProvider.jwks_uri.like("%user:pass%")
        ).count() == 0


def test_authenticated_api_enforces_tenant_permissions_and_blocks_actor_spoofing(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_CORS_ALLOWED_ORIGINS", "http://testserver")
    import app.main as app_main

    app_main = importlib.reload(app_main)
    monkeypatch.setattr(app_main, "SessionLocal", session_factory)
    monkeypatch.setattr(OidcTokenVerifier, "_load_jwks", lambda self, uri, refresh: jwks)
    client = TestClient(app_main.app)

    def auth(subject: str, *, issuer: str = ISSUER_A, key=None) -> dict[str, str]:
        return {"Authorization": f"Bearer {_token(key or rsa_keys['primary'], issuer=issuer, subject=subject)}"}

    org_id = fixture["organization_id"]
    viewer_headers = auth(fixture["subjects"]["viewer"])
    owner_headers = auth(fixture["subjects"]["owner"])
    engineer_headers = auth(fixture["subjects"]["engineer"])
    tech_headers = auth(fixture["subjects"]["technician"])
    outsider_headers = auth("outsider-sub")

    denied = client.post(
        f"/api/ingestion/{org_id}/sources",
        headers=viewer_headers,
        json={"organization_id": org_id, "source_type": "rest", "name": "Viewer Source"},
    )
    assert denied.status_code == 403

    other_org_body = fixture["other_organization_id"]
    created = client.post(
        f"/api/ingestion/{org_id}/sources",
        headers=engineer_headers,
        json={
            "organization_id": other_org_body,
            "source_type": "rest",
            "name": "Engineer Source",
            "config": {"auth": {"token": {"secret_reference_id": "ref"}}},
        },
    )
    assert created.status_code == 200
    assert created.json()["organization_id"] == org_id

    inventory = client.get(f"/api/platform/{org_id}/inventory", headers=engineer_headers)
    assert inventory.status_code == 200
    assert [organization["id"] for organization in inventory.json()["organizations"]] == [org_id]
    assert inventory.json()["assets"][0]["site_id"] == fixture["site_id"]

    legacy_profile = client.get("/api/profile", headers=engineer_headers)
    assert legacy_profile.status_code == 403

    cross_tenant = client.get(f"/api/maintenance/{org_id}/cases", headers=outsider_headers)
    assert cross_tenant.status_code == 403

    viewer_audit = client.get(f"/api/security/{org_id}/audit-events", headers=viewer_headers)
    assert viewer_audit.status_code == 403
    owner_audit = client.get(f"/api/security/{org_id}/audit-events", headers=owner_headers)
    assert owner_audit.status_code == 200

    spoofed = client.post(
        f"/api/maintenance/{org_id}/cases",
        headers=tech_headers,
        json={
            "title": "Manual inspection case",
            "priority": "medium",
            "opened_by_user_id": fixture["users"]["bob"],
            "summary": "Operator heard noise.",
        },
    )
    assert spoofed.status_code == 200
    assert spoofed.json()["opened_by_user_id"] == fixture["users"]["technician"]

    with session_factory() as session:
        source = session.query(IngestionSource).filter_by(name="Engineer Source").one()
        case = session.query(MaintenanceCase).one()
        denied_events = session.query(SecurityAuditEvent).filter_by(outcome="denied").all()

    assert source.organization_id == org_id
    assert case.opened_by_user_id == fixture["users"]["technician"]
    assert any(event.required_permission == "ingestion.manage" for event in denied_events)


def test_enterprise_model_promotions_derive_approver_from_auth_context(
    migrated_db,
    monkeypatch,
    tmp_path,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    monkeypatch.setenv("PMS_MODEL_REGISTRY_ROOT", str(tmp_path / "models"))
    with session_factory() as session:
        fixture = _seed_security(session)
        _seed_ml_training_features(session, fixture)
        service = MLPlatformService(session, ModelArtifactStore(tmp_path / "models"))
        dataset = service.create_dataset_version(
            fixture["organization_id"],
            DatasetVersionCreate(
                name="bearing-rul-features",
                version="security-v1",
                feature_names=["scalar.rms"],
                target_provenance_key="target_rul_hours",
                validation_group_provenance_key="validation_group",
            ),
        )
        experiment = service.run_experiment(
            fixture["organization_id"],
            ExperimentCreate(
                dataset_version_id=dataset.id,
                name="security promotion provenance",
                training_config={"n_estimators": 5, "random_state": 17},
            ),
        )
        registry = service.create_registry(
            fixture["organization_id"],
            RegistryCreate(name="security-bearing-rul", task="rul_regression"),
        )
        validated_model = _register_candidate_model(
            service,
            fixture["organization_id"],
            experiment.id,
            registry.id,
            "validated-spoof",
        )
        rejected_model = _register_candidate_model(
            service,
            fixture["organization_id"],
            experiment.id,
            registry.id,
            "rejected-spoof",
        )
        production_model = _register_candidate_model(
            service,
            fixture["organization_id"],
            experiment.id,
            registry.id,
            "production-spoof",
        )
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)

    def auth(subject: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject=subject)}"}

    org_id = fixture["organization_id"]
    outsider_user_id = fixture["users"]["outsider"]
    engineer_headers = auth(fixture["subjects"]["engineer"])
    owner_headers = auth(fixture["subjects"]["owner"])

    validated = client.post(
        f"/api/ml/{org_id}/model-versions/{validated_model.id}/promote",
        headers=engineer_headers,
        json={"target_stage": "validated", "approved_by_user_id": outsider_user_id},
    )
    rejected = client.post(
        f"/api/ml/{org_id}/model-versions/{rejected_model.id}/promote",
        headers=engineer_headers,
        json={"target_stage": "rejected", "approved_by_user_id": outsider_user_id},
    )
    production_ready = client.post(
        f"/api/ml/{org_id}/model-versions/{production_model.id}/promote",
        headers=engineer_headers,
        json={"target_stage": "validated", "approved_by_user_id": outsider_user_id},
    )
    forbidden_production = client.post(
        f"/api/ml/{org_id}/model-versions/{production_model.id}/promote",
        headers=engineer_headers,
        json={"target_stage": "production", "approved_by_user_id": outsider_user_id},
    )
    production = client.post(
        f"/api/ml/{org_id}/model-versions/{production_model.id}/promote",
        headers=owner_headers,
        json={"target_stage": "production", "approved_by_user_id": outsider_user_id},
    )

    assert validated.status_code == 200
    assert rejected.status_code == 200
    assert production_ready.status_code == 200
    assert forbidden_production.status_code == 403
    assert production.status_code == 200

    with session_factory() as session:
        events = session.query(MLModelPromotionEvent).filter_by(organization_id=org_id).all()
        approver_by_target = {
            (event.model_version_id, event.to_stage): event.approved_by_user_id
            for event in events
        }

    assert approver_by_target[(validated_model.id, "validated")] == fixture["users"]["engineer"]
    assert approver_by_target[(rejected_model.id, "rejected")] == fixture["users"]["engineer"]
    assert approver_by_target[(production_model.id, "validated")] == fixture["users"]["engineer"]
    assert approver_by_target[(production_model.id, "production")] == fixture["users"]["owner"]
    assert outsider_user_id not in approver_by_target.values()


def test_enterprise_app_reuses_oidc_verifier_jwks_cache(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return jwks

    class FakeClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def get(self, uri):
            assert uri == JWKS_URI
            FakeClient.calls += 1
            return FakeResponse()

    monkeypatch.setattr("enterprise_security.service.httpx.Client", FakeClient)
    _app_main, client = _enterprise_app(monkeypatch, session_factory)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}

    first = client.get(f"/api/security/{fixture['organization_id']}/me", headers=headers)
    second = client.get(f"/api/security/{fixture['organization_id']}/me", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert FakeClient.calls == 1


def test_cors_allows_patch_and_request_ids_are_bounded(migrated_db, monkeypatch):
    _engine, session_factory = migrated_db
    _app_main, client = _enterprise_app(monkeypatch, session_factory)

    preflight = client.options(
        "/api/security/org-id/memberships/user-id",
        headers={
            "Origin": "http://ui.example",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "Authorization,X-Request-ID",
        },
    )
    too_long_request_id = client.get("/api/health", headers={"X-Request-ID": "x" * 65})
    malformed_request_id = client.get("/api/health", headers={"X-Request-ID": "bad request id"})

    assert preflight.status_code == 200
    assert "PATCH" in preflight.headers["access-control-allow-methods"]
    assert too_long_request_id.status_code == 400
    assert malformed_request_id.status_code == 400


def test_audited_http_paths_are_bounded(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    valid_path = f"/api/security/{fixture['organization_id']}/me"
    oversized_path = f"{valid_path}/{'x' * (MAX_AUDIT_HTTP_PATH_LENGTH + 1)}"

    valid = client.get(valid_path, headers=headers)
    oversized = client.get(oversized_path, headers=headers)

    assert len(valid_path) <= MAX_AUDIT_HTTP_PATH_LENGTH
    assert valid.status_code == 200
    assert oversized.status_code == 414
    assert oversized.status_code not in {500, 503}

    with session_factory() as session:
        paths = [event.http_path for event in session.query(SecurityAuditEvent).all() if event.http_path]

    assert valid_path in paths
    assert all(len(path) <= MAX_AUDIT_HTTP_PATH_LENGTH for path in paths)


def test_health_endpoints_commit_successful_authorization_audits(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]
    paths = [
        f"/api/ingestion/{org_id}/health",
        f"/api/analytics/{org_id}/health",
        f"/api/serving/{org_id}/health",
        f"/api/maintenance/{org_id}/health",
    ]

    for path in paths:
        response = client.get(path, headers=headers)
        assert response.status_code == 200

    with session_factory() as session:
        actions = {
            event.action
            for event in session.query(SecurityAuditEvent).filter_by(
                organization_id=org_id,
                outcome="allowed",
            )
        }

    assert {
        "ingestion.health.read",
        "analytics.health.read",
        "serving.health.read",
        "maintenance.health.read",
    }.issubset(actions)


def test_security_admin_api_onboards_users_updates_idps_and_rotates_secrets(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)

    def auth(subject: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject=subject)}"}

    org_id = fixture["organization_id"]
    owner_headers = auth("owner-sub")
    onboarded = client.post(
        f"/api/security/{org_id}/user-identities",
        headers=owner_headers,
        json={
            "email": "new.viewer@example.com",
            "full_name": "New Viewer",
            "identity_provider_id": fixture["idp_id"],
            "issuer": ISSUER_A,
            "subject": "new-viewer-sub",
            "role": "viewer",
            "profile": {"api_key": "should-redact"},
        },
    )
    assert onboarded.status_code == 200
    assert onboarded.json()["membership"]["role"] == "viewer"

    new_viewer = client.get(f"/api/security/{org_id}/me", headers=auth("new-viewer-sub"))
    assert new_viewer.status_code == 200
    assert new_viewer.json()["role"] == "viewer"

    created_secret = client.post(
        f"/api/security/{org_id}/secret-references",
        headers=owner_headers,
        json={
            "name": "rotating-abb-token",
            "purpose": "abb_api_token",
            "provider": "env",
            "locator": "ABB_TOKEN",
            "rotation_metadata": {"ticket": "SEC-1"},
        },
    )
    assert created_secret.status_code == 200
    rotated_secret = client.patch(
        f"/api/security/{org_id}/secret-references/{created_secret.json()['id']}",
        headers=owner_headers,
        json={
            "status": "rotating",
            "rotation_metadata": {"new_token": "plaintext-token"},
            "last_rotated_at": "2026-08-28T17:00:00Z",
        },
    )
    assert rotated_secret.status_code == 200
    assert rotated_secret.json()["status"] == "rotating"
    assert rotated_secret.json()["rotation_metadata"]["new_token"] == "[REDACTED]"
    assert "plaintext-token" not in str(rotated_secret.json())

    created_idp = client.post(
        f"/api/security/{org_id}/identity-providers",
        headers=owner_headers,
        json={
            "name": "secondary",
            "issuer": ISSUER_B,
            "audience": AUDIENCE,
            "jwks_uri": "https://issuer-b.example/.well-known/jwks.json",
        },
    )
    assert created_idp.status_code == 200
    with session_factory() as session:
        SecurityService(session).create_user_identity(
            org_id,
            UserIdentityCreate(
                user_id=fixture["users"]["owner"],
                identity_provider_id=created_idp.json()["id"],
                issuer=ISSUER_B,
                subject="owner-api-secondary-sub",
            ),
        )
        session.commit()
    updated_idp = client.patch(
        f"/api/security/{org_id}/identity-providers/{fixture['idp_id']}",
        headers=owner_headers,
        json={"status": "inactive"},
    )
    assert updated_idp.status_code == 200
    assert updated_idp.json()["status"] == "inactive"

    with session_factory() as session:
        idp = session.get(OrganizationIdentityProvider, fixture["idp_id"])
        secret = session.get(SecretReference, created_secret.json()["id"])

    assert idp.status == "inactive"
    assert secret.status == "rotating"
    assert secret.rotation_metadata["new_token"] == "[REDACTED]"


def test_denied_owner_onboarding_does_not_persist_orphan_user(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        before_count = session.query(User).count()
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    response = client.post(
        f"/api/security/{fixture['organization_id']}/user-identities",
        headers={"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='admin-sub')}"},
        json={
            "email": "orphan-owner@example.com",
            "identity_provider_id": fixture["idp_id"],
            "issuer": ISSUER_A,
            "subject": "orphan-owner-sub",
            "role": "owner",
        },
    )

    assert response.status_code == 403
    with session_factory() as session:
        assert session.query(User).count() == before_count
        assert session.query(User).filter_by(email="orphan-owner@example.com").one_or_none() is None


def test_security_administration_routes_reject_outside_enterprise_mode(migrated_db, monkeypatch):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        repo = PlatformRepository(session)
        org = repo.create_organization(OrganizationCreate(slug="disabled-security", name="Disabled Security"))
        session.commit()

    monkeypatch.setenv("PMS_SECURITY_MODE", "disabled")
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    import app.main as app_main

    app_main = importlib.reload(app_main)
    monkeypatch.setattr(app_main, "SessionLocal", session_factory)
    client = TestClient(app_main.app)

    me = client.get(f"/api/security/{org.id}/me")
    response = client.post(
        f"/api/security/{org.id}/identity-providers",
        json={
            "name": "attacker",
            "issuer": ISSUER_A,
            "audience": AUDIENCE,
            "jwks_uri": JWKS_URI,
        },
    )

    assert me.status_code == 200
    assert me.json() == {"security_mode": "disabled"}
    assert response.status_code == 403
    assert response.json() == {"detail": "enterprise security administration requires enterprise security mode"}
    with session_factory() as session:
        assert session.query(OrganizationIdentityProvider).count() == 0


def test_service_principal_patch_unknown_and_cross_tenant_are_generic_403(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        other_principal = SecurityService(session).create_service_principal(
            fixture["other_organization_id"],
            ServicePrincipalCreate(
                name="other-robot",
                identity_provider_id=fixture["other_idp_id"],
                external_subject="other-robot-sub",
                issuer=ISSUER_A,
                permissions=[INGESTION_WRITE],
            ),
        )
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]

    for principal_id in ["00000000-0000-4000-8000-000000000000", other_principal.id]:
        response = client.patch(
            f"/api/security/{org_id}/service-principals/{principal_id}",
            headers=headers,
            json={"status": "inactive"},
        )
        assert response.status_code == 403
        assert response.json() == {"detail": "not authorized for this organization"}

    oversized_principal_id = "p" * 300
    oversized_response = client.patch(
        f"/api/security/{org_id}/service-principals/{oversized_principal_id}",
        headers=headers,
        json={"status": "inactive"},
    )
    assert oversized_response.status_code == 403
    assert oversized_response.json() == {"detail": "not authorized for this organization"}

    with session_factory() as session:
        resource_ids = [
            event.resource_id
            for event in session.query(SecurityAuditEvent).filter_by(
                organization_id=org_id,
                action="security.service_principal.update",
                resource_type="service_principal",
            )
        ]
        assert any(resource_id.startswith("sha256:") for resource_id in resource_ids if resource_id)
        assert all(resource_id is None or len(resource_id) <= 255 for resource_id in resource_ids)

    missing_user = client.post(
        f"/api/security/{org_id}/memberships",
        headers=headers,
        json={"user_id": "00000000-0000-4000-8000-000000000000", "role": "viewer"},
    )
    assert missing_user.status_code == 403
    assert missing_user.json() == {"detail": "not authorized for this organization"}


def test_duplicate_identity_provider_creation_returns_stable_conflict(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]

    duplicate_name = client.post(
        f"/api/security/{org_id}/identity-providers",
        headers=headers,
        json={
            "name": "primary",
            "issuer": ISSUER_B,
            "audience": "new-audience",
            "jwks_uri": "https://issuer-b.example/.well-known/jwks.json",
        },
    )
    duplicate_issuer_audience = client.post(
        f"/api/security/{org_id}/identity-providers",
        headers=headers,
        json={
            "name": "unique-name",
            "issuer": ISSUER_A,
            "audience": AUDIENCE,
            "jwks_uri": JWKS_URI,
        },
    )

    assert duplicate_name.status_code == 409
    assert duplicate_name.json() == {"detail": "resource conflict"}
    assert duplicate_issuer_audience.status_code == 409
    assert duplicate_issuer_audience.json() == {"detail": "resource conflict"}
    with session_factory() as session:
        failures = session.query(SecurityAuditEvent).filter_by(
            organization_id=org_id,
            action="security.idp.create",
            outcome="failed",
            reason_code="resource_conflict",
        )
        assert failures.count() == 2


def test_duplicate_secret_reference_creation_returns_stable_conflict(
    migrated_db,
    monkeypatch,
    rsa_keys,
    jwks,
):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]
    payload = {
        "name": "duplicate-secret",
        "purpose": "abb_api_token",
        "provider": "env",
        "locator": "ABB_TOKEN",
    }

    created = client.post(f"/api/security/{org_id}/secret-references", headers=headers, json=payload)
    duplicate = client.post(
        f"/api/security/{org_id}/secret-references",
        headers=headers,
        json={**payload, "locator": "OTHER_TOKEN"},
    )

    assert created.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "resource conflict"}
    with session_factory() as session:
        assert session.query(SecretReference).filter_by(organization_id=org_id, name="duplicate-secret").count() == 1
        assert session.scalar(select(func.count()).select_from(SecretReference)) is not None
        failures = session.query(SecurityAuditEvent).filter_by(
            organization_id=org_id,
            action="security.secret.create",
            outcome="failed",
            reason_code="resource_conflict",
        )
        assert failures.count() == 1


def test_security_mutation_audits_record_target_resource_ids(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        second_idp = SecurityService(session).create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/.well-known/jwks.json",
            ),
            allow_development_targets=True,
        )
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]

    idp_update = client.patch(
        f"/api/security/{org_id}/identity-providers/{second_idp.id}",
        headers=headers,
        json={"status": "inactive"},
    )
    membership_role = client.post(
        f"/api/security/{org_id}/memberships",
        headers=headers,
        json={"user_id": fixture["users"]["bob"], "role": "viewer"},
    )
    membership_status = client.patch(
        f"/api/security/{org_id}/memberships/{fixture['users']['bob']}",
        headers=headers,
        json={"lifecycle_state": "inactive"},
    )
    owner_role = client.post(
        f"/api/security/{org_id}/memberships",
        headers=headers,
        json={"user_id": fixture["users"]["admin"], "role": "owner"},
    )
    service_principal = client.patch(
        f"/api/security/{org_id}/service-principals/{fixture['service_principal_id']}",
        headers=headers,
        json={"status": "inactive"},
    )
    secret = client.post(
        f"/api/security/{org_id}/secret-references",
        headers=headers,
        json={"name": "audited-secret", "purpose": "abb_api_token", "provider": "env", "locator": "ABB_TOKEN"},
    )
    secret_update = client.patch(
        f"/api/security/{org_id}/secret-references/{secret.json()['id']}",
        headers=headers,
        json={"status": "rotating"},
    )

    assert idp_update.status_code == 200
    assert membership_role.status_code == 200
    assert membership_status.status_code == 200
    assert owner_role.status_code == 200
    assert service_principal.status_code == 200
    assert secret.status_code == 200
    assert secret_update.status_code == 200

    with session_factory() as session:
        audit_targets = {
            (event.action, event.required_permission, event.resource_type, event.resource_id)
            for event in session.query(SecurityAuditEvent).filter_by(organization_id=org_id, outcome="allowed")
        }

    assert ("security.idp.update", SECURITY_MANAGE, "identity_provider", second_idp.id) in audit_targets
    assert ("members.role", MEMBERS_MANAGE, "membership", fixture["users"]["bob"]) in audit_targets
    assert ("members.status", MEMBERS_MANAGE, "membership", fixture["users"]["bob"]) in audit_targets
    assert ("members.role", OWNERS_MANAGE, "membership", fixture["users"]["admin"]) in audit_targets
    assert (
        "security.service_principal.update",
        SECURITY_MANAGE,
        "service_principal",
        fixture["service_principal_id"],
    ) in audit_targets
    assert ("security.secret.update", SECRETS_MANAGE, "secret_reference", secret.json()["id"]) in audit_targets


def test_rejected_security_mutations_are_not_audited_as_allowed(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        second_idp = SecurityService(session).create_identity_provider(
            fixture["organization_id"],
            IdentityProviderCreate(
                name="secondary-unreachable",
                issuer=ISSUER_B,
                audience=AUDIENCE,
                jwks_uri="https://issuer-b.example/.well-known/jwks.json",
            ),
            allow_development_targets=True,
        )
        session.commit()

    _app_main, client = _enterprise_app(monkeypatch, session_factory, jwks=jwks)
    headers = {"Authorization": f"Bearer {_token(rsa_keys['primary'], subject='owner-sub')}"}
    org_id = fixture["organization_id"]
    owner_user_id = fixture["users"]["owner"]

    idp_rejection = client.patch(
        f"/api/security/{org_id}/identity-providers/{fixture['idp_id']}",
        headers=headers,
        json={"status": "inactive"},
    )
    owner_rejection = client.patch(
        f"/api/security/{org_id}/memberships/{owner_user_id}",
        headers=headers,
        json={"lifecycle_state": "inactive"},
    )

    assert idp_rejection.status_code == 403
    assert owner_rejection.status_code == 403
    with session_factory() as session:
        allowed_rejection_count = session.query(SecurityAuditEvent).filter(
            SecurityAuditEvent.organization_id == org_id,
            SecurityAuditEvent.outcome == "allowed",
            (
                (
                    (SecurityAuditEvent.action == "security.idp.update")
                    & (SecurityAuditEvent.resource_id == fixture["idp_id"])
                )
                | (
                    (SecurityAuditEvent.action == "members.status")
                    & (SecurityAuditEvent.resource_id == owner_user_id)
                )
            ),
        ).count()
        denied_targets = {
            (event.action, event.resource_type, event.resource_id, event.outcome, event.reason_code)
            for event in session.query(SecurityAuditEvent).filter_by(organization_id=org_id, outcome="denied")
        }
        assert session.get(OrganizationIdentityProvider, second_idp.id).status == "active"

    assert allowed_rejection_count == 0
    assert (
        "security.idp.update",
        "identity_provider",
        fixture["idp_id"],
        "denied",
        "operation_rejected",
    ) in denied_targets
    assert (
        "members.status",
        "membership",
        owner_user_id,
        "denied",
        "operation_rejected",
    ) in denied_targets


def test_service_principal_cannot_create_human_note_via_api(migrated_db, monkeypatch, rsa_keys, jwks):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        case = MaintenanceOperationsService(session).open_case(
            fixture["organization_id"],
            CaseCreate(
                title="Manual case",
                priority="medium",
                opened_by_user_id=fixture["users"]["technician"],
                summary="Needs inspection.",
            ),
        )
        session.commit()

    monkeypatch.setenv("PMS_SECURITY_MODE", "enterprise")
    monkeypatch.setenv("PMS_ENVIRONMENT", "development")
    monkeypatch.setenv("PMS_CORS_ALLOWED_ORIGINS", "http://testserver")
    import app.main as app_main

    app_main = importlib.reload(app_main)
    monkeypatch.setattr(app_main, "SessionLocal", session_factory)
    monkeypatch.setattr(OidcTokenVerifier, "_load_jwks", lambda self, uri, refresh: jwks)
    client = TestClient(app_main.app)
    token = _token(rsa_keys["primary"], subject="robot-sub")

    response = client.post(
        f"/api/maintenance/{fixture['organization_id']}/cases/{case.id}/notes",
        headers={"Authorization": f"Bearer {token}"},
        json={"author_user_id": fixture["users"]["bob"], "body": "Machine wrote this."},
    )

    assert response.status_code == 403
    with session_factory() as session:
        assert session.query(MaintenanceNote).count() == 0


def test_audit_payload_is_sanitized_and_append_only_through_api(migrated_db):
    _engine, session_factory = migrated_db
    with session_factory() as session:
        fixture = _seed_security(session)
        security = SecurityService(session, verifier=DeterministicTokenVerifier({}))
        context = security.context_from_claims(
            fixture["organization_id"],
            _claims(fixture["subjects"]["owner"]),
            request_id="req-audit",
        )
        event = security.record_audit_event(
            context=context,
            action="security.secret.create",
            required_permission=SECURITY_MANAGE,
            resource_type="secret_reference",
            resource_id="secret-1",
            outcome="allowed",
            reason_code="permission_granted",
            request_metadata={"Authorization": "Bearer abc", "safe": "ok"},
        )
        session.commit()

        payload = audit_event_payload(event)
        assert payload["request_metadata"]["Authorization"] == "[REDACTED]"
        assert "abc" not in str(payload)
        assert session.query(SecretReference).count() == 0
