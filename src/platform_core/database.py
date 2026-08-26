"""SQLAlchemy engine/session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from platform_core.config import database_settings


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    settings = database_settings()
    engine_url = url or settings.url
    engine_echo = settings.echo if echo is None else echo
    connect_args = {"check_same_thread": False} if engine_url.startswith("sqlite") else {}
    engine = create_engine(engine_url, echo=engine_echo, future=True, connect_args=connect_args)
    if engine_url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker[Session] = SessionLocal) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database(session: Session) -> bool:
    session.execute(text("SELECT 1"))
    return True

