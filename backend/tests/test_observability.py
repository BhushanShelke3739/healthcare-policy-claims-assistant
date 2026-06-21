"""
Phase 9 observability tests.

Covers:
    * Request-id middleware — generation, inbound propagation, response header.
    * /metrics — Prometheus exposition with our named series present.
    * Catch-all exception handler — 500 JSON envelope carrying the request id.
    * Retrieval-score persistence — /rag/ask writes scores into QueryLog.details.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.observability import PROCESS_TIME_HEADER, REQUEST_ID_HEADER
from app.main import create_app
from tests.conftest import TEST_CORPUS_TAG


# ---------------------------------------------------------------------------
# Request id + timing
# ---------------------------------------------------------------------------
def test_request_id_generated_and_returned(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    request_id = response.headers.get(REQUEST_ID_HEADER)
    assert request_id and request_id != "-"
    # Latency header is stamped on every response.
    assert PROCESS_TIME_HEADER in response.headers
    float(response.headers[PROCESS_TIME_HEADER])  # parses as a number


def test_request_id_propagated_from_inbound_header(client: TestClient) -> None:
    """A caller-supplied X-Request-ID is echoed back (trace continuity)."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})
    assert response.headers.get(REQUEST_ID_HEADER) == "trace-abc-123"


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
def test_metrics_endpoint_exposes_prometheus(client: TestClient) -> None:
    # Generate at least one request so the counter has a sample.
    client.get("/health")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "hpca_http_requests_total" in body
    assert "hpca_http_request_duration_seconds" in body


# ---------------------------------------------------------------------------
# Catch-all exception handler
# ---------------------------------------------------------------------------
def test_unhandled_exception_returns_request_id_envelope() -> None:
    app: FastAPI = create_app()

    @app.get("/_boom")
    def _boom() -> None:
        raise RuntimeError("kaboom")

    # raise_server_exceptions=False so TestClient returns the 500 response
    # instead of re-raising it into the test.
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.get("/_boom")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "InternalServerError"
    assert body["error"]["request_id"]
    # Non-prod surfaces the exception text to aid debugging.
    assert "kaboom" in body["error"]["message"]
    # The failed response still carries the correlation header.
    assert response.headers.get(REQUEST_ID_HEADER)


# ---------------------------------------------------------------------------
# Retrieval-score persistence (QueryLog.details)
# ---------------------------------------------------------------------------
def test_ask_persists_retrieval_scores_in_details(
    client_with_real_db, db_session, seeded_corpus
) -> None:
    from app.db.models import QueryLog

    question = f"unique-test-{uuid.uuid4()}: appeal filing window"
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

    log = db_session.query(QueryLog).filter(QueryLog.user_query == question).one()
    details = log.details
    assert isinstance(details, dict)
    assert details["mode"] == "hybrid"
    assert details["top_k"] == 3
    assert details["refused"] is False
    assert details["retrieved"] >= 1
    # Per-chunk scores recorded for after-the-fact debugging.
    scores = details["retrieval_scores"]
    assert isinstance(scores, list) and scores
    first = scores[0]
    assert "chunk_id" in first
    assert "similarity_score" in first
    assert "component_scores" in first


def test_ask_refusal_recorded_in_details(client_with_real_db, db_session, seeded_corpus) -> None:
    from app.db.models import QueryLog

    question = f"unique-test-{uuid.uuid4()}: appeal window"
    response = client_with_real_db.post(
        "/rag/ask",
        json={
            "question": question,
            "top_k": 3,
            "mode": "hybrid",
            "document_type": "__no_such_type__",  # forces an empty retrieval
        },
    )
    assert response.status_code == 200, response.text

    log = db_session.query(QueryLog).filter(QueryLog.user_query == question).one()
    assert log.details["refused"] is True
    assert log.details["retrieved"] == 0
    assert log.details["retrieval_scores"] == []
