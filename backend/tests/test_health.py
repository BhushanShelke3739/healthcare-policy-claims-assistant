"""Tests for the health and readiness endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "healthcare-policy-claims-assistant"
    assert "version" in payload
    assert "environment" in payload


def test_readiness_returns_ready_when_db_responsive(client: TestClient) -> None:
    response = client.get("/readiness")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["database"] == "ok"


def test_openapi_lists_stub_routes(client: TestClient) -> None:
    """Smoke test: the Phase 2-6 stub routes appear in the OpenAPI spec."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for expected in [
        "/health",
        "/readiness",
        "/documents/upload",
        "/rag/retrieve",
        "/rag/ask",
        "/agents/run",
        "/eval/run",
    ]:
        assert expected in paths, f"missing route in OpenAPI: {expected}"
