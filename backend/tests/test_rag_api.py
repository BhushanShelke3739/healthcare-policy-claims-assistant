"""
Integration tests for POST /rag/ask.

Uses the mock LLM provider (default) so no API key is needed. The full
flow is exercised end-to-end against a real Postgres via the
`client_with_real_db` + `seeded_corpus` fixtures.
"""

from __future__ import annotations

import uuid

from tests.conftest import TEST_CORPUS_TAG


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_ask_returns_full_schema(client_with_real_db, seeded_corpus) -> None:
    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": "How long do I have to file a first-level appeal?",
            "top_k": 3,
            "mode": "hybrid",
            "document_type": TEST_CORPUS_TAG,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in (
        "question",
        "answer",
        "citations",
        "confidence",
        "grounding_notes",
        "retrieved_chunk_ids",
        "model_name",
        "latency_ms",
    ):
        assert key in payload, f"missing key in response: {key}"

    assert payload["question"] == "How long do I have to file a first-level appeal?"
    assert payload["confidence"] in {"low", "medium", "high"}
    assert isinstance(payload["latency_ms"], int)


def test_ask_returns_citation_when_context_exists(client_with_real_db, seeded_corpus) -> None:
    """Spec: 'answer contains citation when context exists'."""
    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": "What is the appeal filing window?",
            "top_k": 3,
            "mode": "hybrid",
            "document_type": TEST_CORPUS_TAG,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["citations"], "expected at least one citation"
    # Every citation must reference a chunk we actually retrieved — the
    # API layer drops citations whose chunk_id isn't in the retrieved set.
    retrieved_ids = set(payload["retrieved_chunk_ids"])
    for c in payload["citations"]:
        assert c["chunk_id"] in retrieved_ids
        assert c["document_title"]
        assert c["excerpt"]


def test_ask_respects_include_citations_false(client_with_real_db, seeded_corpus) -> None:
    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": "What is the appeal filing window?",
            "top_k": 3,
            "mode": "hybrid",
            "include_citations": False,
            "document_type": TEST_CORPUS_TAG,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["citations"] == []


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------
def test_ask_refuses_when_no_context_matches(client_with_real_db, seeded_corpus) -> None:
    """
    Spec: 'answer refuses or says insufficient context when context is missing'.

    We force this with a `document_type` filter that no document carries.
    """
    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": "What is the appeal filing window?",
            "top_k": 3,
            "mode": "hybrid",
            "document_type": "__no_such_type__",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert "could not find" in payload["answer"].lower()
    assert payload["citations"] == []
    assert payload["confidence"] == "low"
    assert payload["retrieved_chunk_ids"] == []


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_ask_validates_empty_question(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/rag/ask",
        json={"question": "", "top_k": 3},
    )
    assert response.status_code == 422


def test_ask_validates_top_k_range(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/rag/ask",
        json={"question": "any", "top_k": 0},
    )
    assert response.status_code == 422

    response = client_with_real_db.post(
        "/rag/ask",
        json={"question": "any", "top_k": 999},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Side effects: a QueryLog row is written per request
# ---------------------------------------------------------------------------
def test_ask_persists_query_log(client_with_real_db, db_session, seeded_corpus) -> None:
    from app.db.models import QueryLog

    initial_count = db_session.query(QueryLog).count()
    question = f"unique-test-{uuid.uuid4()}: appeal window"

    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": question,
            "top_k": 3,
            "mode": "hybrid",
            "document_type": TEST_CORPUS_TAG,
        },
    )
    assert response.status_code == 200, response.text

    final_count = db_session.query(QueryLog).count()
    assert final_count == initial_count + 1

    log = db_session.query(QueryLog).filter(QueryLog.user_query == question).one_or_none()
    assert log is not None
    assert log.answer  # non-empty
    assert isinstance(log.retrieved_chunk_ids, list)
    assert log.latency_ms is not None and log.latency_ms >= 0
    assert log.model_name == "mock-chat"
