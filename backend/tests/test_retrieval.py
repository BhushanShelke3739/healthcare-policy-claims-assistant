"""
Integration tests for retrieval — vector, keyword, hybrid.

Requires a live Postgres + pgvector (see `db_session` fixture). The whole
file is skipped when the database isn't reachable.

The `seeded_corpus` fixture (in conftest.py) inserts four small chunks
tagged with `document_type=TEST_CORPUS_TAG`. Every retrieval call here
passes `document_type=TEST_CORPUS_TAG` so the assertions are immune to
whatever the developer happens to have seeded in the live DB.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.services.retrieval import (
    hybrid_search,
    keyword_search,
    retrieve,
    vector_search,
)
from tests.conftest import TEST_CORPUS_TAG


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------
def test_vector_search_finds_semantically_related_chunk(
    db_session: Session, seeded_corpus: dict[str, str]
) -> None:
    hits = vector_search(
        db_session,
        query_text="How many days do I have to appeal a denial?",
        top_k=2,
        document_type=TEST_CORPUS_TAG,
    )

    assert len(hits) >= 1
    top_id = str(hits[0].chunk_id)
    assert top_id in {seeded_corpus["appeals_1"], seeded_corpus["appeals_2"]}
    assert all(h.document_title for h in hits)


def test_vector_search_skips_chunks_without_embeddings(
    db_session: Session, seeded_corpus: dict[str, str]
) -> None:
    # Drop the embedding on the telehealth chunk to simulate an un-backfilled row.
    db_session.execute(
        DocumentChunk.__table__.update()
        .where(DocumentChunk.id == seeded_corpus["telehealth_1"])
        .values(embedding=None)
    )
    db_session.flush()

    hits = vector_search(
        db_session,
        query_text="anything",
        top_k=10,
        document_type=TEST_CORPUS_TAG,
    )
    assert all(str(h.chunk_id) != seeded_corpus["telehealth_1"] for h in hits)


# ---------------------------------------------------------------------------
# Keyword
# ---------------------------------------------------------------------------
def test_keyword_search_finds_exact_identifier(
    db_session: Session, seeded_corpus: dict[str, str]
) -> None:
    """The denial code 'HF-022' is the case keyword search exists for."""
    hits = keyword_search(
        db_session,
        query_text="HF-022",
        top_k=3,
        document_type=TEST_CORPUS_TAG,
    )

    assert hits, "expected at least one keyword hit for HF-022"
    assert str(hits[0].chunk_id) == seeded_corpus["denial_1"]


def test_keyword_search_returns_empty_for_no_match(db_session: Session) -> None:
    hits = keyword_search(
        db_session,
        query_text="zzzzzzz_no_such_term_zzzzzzz",
        top_k=5,
        document_type=TEST_CORPUS_TAG,
    )
    assert hits == []


# ---------------------------------------------------------------------------
# Hybrid + dispatcher
# ---------------------------------------------------------------------------
def test_hybrid_emits_component_scores(db_session: Session, seeded_corpus: dict[str, str]) -> None:
    hits = hybrid_search(
        db_session,
        query_text="appeal denial HF-022",
        top_k=3,
        alpha=0.5,
        document_type=TEST_CORPUS_TAG,
    )
    assert hits
    top = hits[0]
    assert {"vector", "keyword", "alpha"} <= set(top.component_scores)
    assert top.component_scores["alpha"] == 0.5


def test_hybrid_blends_vector_and_keyword_signals(
    db_session: Session, seeded_corpus: dict[str, str]
) -> None:
    """
    Hybrid (with non-trivial keyword weight) should put HF-022 in the
    top-2 even though vector alone might not.
    """
    hits = hybrid_search(
        db_session,
        query_text="What does denial code HF-022 mean?",
        top_k=2,
        alpha=0.5,
        document_type=TEST_CORPUS_TAG,
    )
    top_ids = {str(h.chunk_id) for h in hits}
    assert seeded_corpus["denial_1"] in top_ids


def test_retrieve_dispatches_by_mode(db_session: Session, seeded_corpus: dict[str, str]) -> None:
    for mode in ("vector", "keyword", "hybrid"):
        hits = retrieve(
            db_session,
            query_text="appeal denial",
            top_k=2,
            mode=mode,
            document_type=TEST_CORPUS_TAG,
        )
        assert isinstance(hits, list)


def test_retrieve_rejects_unknown_mode(db_session: Session) -> None:
    with pytest.raises(ValueError):
        retrieve(db_session, query_text="x", top_k=1, mode="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_rag_retrieve_endpoint_returns_expected_schema(client_with_real_db, seeded_corpus) -> None:
    response = client_with_real_db.post(
        "/rag/retrieve",
        json={
            "query": "appeal denial HF-022",
            "top_k": 3,
            "mode": "hybrid",
            "document_type": TEST_CORPUS_TAG,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query"] == "appeal denial HF-022"
    assert payload["mode"] == "hybrid"
    assert isinstance(payload["results"], list)
    if payload["results"]:
        first = payload["results"][0]
        for key in [
            "chunk_id",
            "document_id",
            "document_title",
            "chunk_text",
            "chunk_index",
            "similarity_score",
            "component_scores",
            "metadata",
        ]:
            assert key in first, f"missing key in response: {key}"


def test_rag_retrieve_validates_input(client_with_real_db) -> None:
    # top_k must be >= 1
    r = client_with_real_db.post("/rag/retrieve", json={"query": "anything", "top_k": 0})
    assert r.status_code == 422

    # query must be non-empty
    r = client_with_real_db.post("/rag/retrieve", json={"query": "", "top_k": 3})
    assert r.status_code == 422
