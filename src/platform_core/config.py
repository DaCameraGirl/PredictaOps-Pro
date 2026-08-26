"""Database configuration for local, test, and production deployments."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = ROOT / "data" / "platform_core.db"


@dataclass(frozen=True)
class DatabaseSettings:
    url: str
    echo: bool = False


def database_url(*, test: bool = False) -> str:
    env_name = "PMS_TEST_DATABASE_URL" if test else "PMS_DATABASE_URL"
    configured = os.environ.get(env_name)
    if configured:
        return configured
    if test and os.environ.get("PMS_DATABASE_URL"):
        return os.environ["PMS_DATABASE_URL"]
    return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def database_settings(*, test: bool = False) -> DatabaseSettings:
    return DatabaseSettings(
        url=database_url(test=test),
        echo=os.environ.get("PMS_SQL_ECHO") == "1",
    )


def safe_database_label(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
