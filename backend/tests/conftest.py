"""
Shared pytest fixtures.

Two flavors of fixture
----------------------
    * `client`              — FastAPI TestClient with the DB dependency
                              swapped for a fake session. Use for tests
                              that only need the HTTP surface (e.g. the
                              OpenAPI smoke test, health endpoints).
    * `db_session`          — real SQLAlchemy Session against the running
                              Postgres, wrapped in a transaction that's
                              rolled back at the end of the test. Skips
                              the test if Postgres is unreachable.
    * `client_with_real_db` — FastAPI TestClient that shares the rolled-
                              back transaction with `db_session`. Lets us
                              integration-test endpoints that actually
                              query the database.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import get_db
from app.main import create_app


# ---------------------------------------------------------------------------
# Force mock providers in the whole test suite
# ---------------------------------------------------------------------------
# Why this exists:
#   The test suite verifies wiring (retrieval logic, citation propagation,
#   refusal path, schema) — NOT generation quality. We do not want the
#   green-ness of the suite to depend on whether Ollama / OpenAI / a
#   network is reachable today. So regardless of what the developer's
#   `.env` says, every pytest invocation runs through the deterministic
#   in-process mock providers.
#
# Mechanism:
#   Monkeypatch the env vars *before* the lru_caches on get_settings /
#   get_embedder / get_chat_provider hold any value, then bust the caches
#   so the next call rebuilds with the override in effect. On teardown we
#   bust them again so a `python -m app.<something>` run from the same
#   process picks up the real .env values.
#
# Side effect: EMBEDDING_DIMENSIONS from .env (e.g. 768 if you're on
# Ollama) still wins — the mock provider produces vectors at whatever
# dimension is configured, which matches the actual DB column.
@pytest.fixture(autouse=True)
def _force_mock_providers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    from app.services.embeddings import get_embedder
    from app.services.generation import get_chat_provider

    get_settings.cache_clear()
    get_embedder.cache_clear()
    get_chat_provider.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
        get_embedder.cache_clear()
        get_chat_provider.cache_clear()


# ---------------------------------------------------------------------------
# Unit-style fixtures (no DB)
# ---------------------------------------------------------------------------
class _FakeSession:
    """Minimal stand-in for an SQLAlchemy Session for non-DB tests."""

    def execute(self, *_args, **_kwargs):
        class _Result:
            def scalar(self) -> int:
                return 1

        return _Result()

    def close(self) -> None:
        pass


def _override_get_db_fake() -> Iterator[_FakeSession]:
    yield _FakeSession()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """FastAPI TestClient with the DB dependency stubbed out."""
    app = create_app()
    app.dependency_overrides[get_db] = _override_get_db_fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Integration fixtures (require a live Postgres)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def db_engine() -> Engine:
    """
    Engine that points at the configured Postgres.

    Skips the depending test if the database isn't reachable, so the suite
    still passes in environments without Docker / Postgres running.
    """
    settings = get_settings()
    engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Postgres not reachable for integration tests: {exc}")
    return engine


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    """
    A transactional session that rolls back at end-of-test.

    Pattern: open a connection, begin a transaction, then build a session
    that *joins* that transaction via a SAVEPOINT
    (`join_transaction_mode="create_savepoint"`). When the application
    code under test calls `session.commit()` it commits the SAVEPOINT,
    not the outer transaction — so writes stay visible to the rest of
    the test but disappear when the fixture rolls the outer transaction
    back at teardown.

    Without this, an endpoint that commits (e.g. /rag/ask writing a
    QueryLog) would escape the rollback and pollute the live database.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client_with_real_db(db_session: Session) -> Iterator[TestClient]:
    """
    TestClient where `get_db` yields the same rolled-back session.

    Use this when the test needs to hit a real endpoint (e.g. /rag/retrieve)
    *and* assert on data the test seeded via `db_session`.
    """

    def _override() -> Iterator[Session]:
        yield db_session

    app = create_app()
    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Shared seeded corpus for retrieval / RAG integration tests
# ---------------------------------------------------------------------------
# Unique document_type so tests can scope retrieval to *only* the fixtures
# and never collide with committed seed_policies data in the developer's DB.
# Postgres READ COMMITTED isolation means our transactional inserts CAN see
# already-committed rows from outside the transaction — filtering by this
# tag is the simplest way to be hermetic.
TEST_CORPUS_TAG = "_pytest_corpus_"


@pytest.fixture()
def seeded_corpus(db_session: Session) -> dict[str, str]:
    """
    Insert four small chunks tagged with `document_type=TEST_CORPUS_TAG`:
        appeals_1 — appeal timeline (60 days).
        appeals_2 — appeal levels.
        denial_1  — denial code HF-022.
        telehealth_1 — modifier 95.

    Returns a dict mapping label -> chunk_id (str). Pass
    `document_type=TEST_CORPUS_TAG` to retrieval calls in your test so
    pre-existing seed data in the DB doesn't leak in.
    """
    from app.db.models import Document, DocumentChunk
    from app.services.embeddings import get_embedder

    embedder = get_embedder()

    docs = {
        "appeals": Document(
            title="Appeal Process Policy",
            source_type="seed",
            document_type=TEST_CORPUS_TAG,
        ),
        "denial": Document(
            title="Claim Denial Policy",
            source_type="seed",
            document_type=TEST_CORPUS_TAG,
        ),
        "telehealth": Document(
            title="Telehealth Billing Policy",
            source_type="seed",
            document_type=TEST_CORPUS_TAG,
        ),
    }
    for d in docs.values():
        db_session.add(d)
    db_session.flush()

    chunks_spec = {
        "appeals_1": (
            docs["appeals"],
            "First-level appeals must be filed within sixty calendar days " "of the denial notice.",
        ),
        "appeals_2": (
            docs["appeals"],
            "There are three levels of appeal: first-level internal, "
            "second-level internal, and external review.",
        ),
        "denial_1": (
            docs["denial"],
            "Denial code HF-022 indicates that prior authorization was not "
            "on file at the time of service.",
        ),
        "telehealth_1": (
            docs["telehealth"],
            "Synchronous telehealth visits require modifier 95 on the " "professional claim.",
        ),
    }

    embeddings = embedder.embed([chunk_text for _, chunk_text in chunks_spec.values()])
    ids: dict[str, str] = {}
    # `text` is shadowed by sqlalchemy.text import at module top — use
    # `chunk_text` here to keep ruff F402 happy.
    for (label, (doc, chunk_text)), vec in zip(chunks_spec.items(), embeddings, strict=True):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_text=chunk_text,
            chunk_index=0,
            token_count=len(chunk_text) // 4,
            chunk_metadata={"document_title": doc.title},
            embedding=vec,
        )
        db_session.add(chunk)
        db_session.flush()
        ids[label] = str(chunk.id)

    db_session.flush()
    return ids
