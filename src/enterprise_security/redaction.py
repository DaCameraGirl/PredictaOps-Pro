"""Shared redaction helpers for client-safe security/audit metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

SECRET_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "connection_string",
    "credential",
    "credentials",
    "dsn",
    "passphrase",
    "password",
    "private_key",
    "privatekey",
    "secret",
    "token",
    "access_token",
}
SECRET_QUERY_KEYS = {"key"}


def is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)


def _looks_like_credential_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if not parsed.scheme or not parsed.netloc:
        return False
    if parsed.username or parsed.password:
        return True
    query_keys = {key.lower().replace("-", "_") for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}
    return any(is_secret_key(key) for key in query_keys) or bool(query_keys & SECRET_QUERY_KEYS)


def redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if is_secret_key(str(key)) else redact_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def assert_no_plaintext_secrets(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if is_secret_key(key_text):
                if (
                    isinstance(item, Mapping)
                    and set(item) <= {"secret_reference_id", "provider", "name"}
                    and isinstance(item.get("secret_reference_id"), str)
                    and item["secret_reference_id"].strip()
                ):
                    continue
                raise ValueError(f"{path}.{key_text} must use a secret reference instead of a plaintext value")
            assert_no_plaintext_secrets(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_plaintext_secrets(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _looks_like_credential_url(value):
        raise ValueError(f"{path} must not contain credential-bearing URLs")
