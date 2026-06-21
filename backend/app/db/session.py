"""
SQLAlchemy engine + session management.

We deliberately use the *synchronous* SQLAlchemy API in Phase 1 because:
    * It plays naturally with Alembic (which is sync).
    * psycopg2 is rock-solid and well-known.
    * FastAPI tolerates sync dependencies just fine for a project of this size.

A switch to `async` SQLAlchemy + `asyncpg` is a worthwhile follow-up once
throughput becomes a real concern.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# `pool_pre_ping` issues a tiny `SELECT 1` before handing out a connection.
# Costs ~1ms but protects against stale connections after the DB is bounced.
engine: Engine = create_engine(
    settings.sqlalchemy_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a request-scoped Session.

    The `try/finally` guarantees the session is closed even if the handler
    raises. We deliberately don't auto-commit; handlers commit explicitly
    so the boundary between read and write is obvious.

    Usage:
        @router.get("/something")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
