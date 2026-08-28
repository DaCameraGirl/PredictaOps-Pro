"""Shared redaction helpers for client-safe security/audit metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SECRET_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
    "access_token",
}


def is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)


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
                if isinstance(item, Mapping) and set(item) <= {"secret_reference_id", "provider", "name"}:
                    continue
                raise ValueError(f"{path}.{key_text} must use a secret reference instead of a plaintext value")
            assert_no_plaintext_secrets(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_plaintext_secrets(item, path=f"{path}[{index}]")
