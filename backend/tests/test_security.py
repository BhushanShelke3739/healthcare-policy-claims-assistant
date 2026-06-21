"""
Phase 9 auth/RBAC placeholder tests.

Covers token issue/verify, the require_role gate, and the /auth/* endpoints in
both the default (auth disabled → anonymous) and enabled modes.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.security import (
    ANONYMOUS_VIEWER,
    CurrentUser,
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    require_role,
)


# ---------------------------------------------------------------------------
# Token mechanics
# ---------------------------------------------------------------------------
def test_token_roundtrip() -> None:
    token = create_access_token(subject="u1", role="analyst", email="u1@example.com")
    claims = decode_access_token(token)
    assert claims["sub"] == "u1"
    assert claims["role"] == "analyst"
    assert claims["email"] == "u1@example.com"
    assert claims["exp"] > claims["iat"]


def test_decode_rejects_tampered_token() -> None:
    token = create_access_token(subject="u1", role="viewer")
    with pytest.raises(InvalidTokenError):
        decode_access_token(token + "tampered")


# ---------------------------------------------------------------------------
# require_role gate (unit — call the dependency directly)
# ---------------------------------------------------------------------------
def test_require_role_allows_matching_role() -> None:
    gate = require_role("admin", "analyst")
    admin = CurrentUser(id="1", email="a@x", role="admin")
    assert gate(user=admin) is admin


def test_require_role_denies_other_role() -> None:
    gate = require_role("admin")
    with pytest.raises(HTTPException) as exc_info:
        gate(user=ANONYMOUS_VIEWER)  # role="viewer"
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# /auth endpoints — default (auth disabled)
# ---------------------------------------------------------------------------
def test_me_is_anonymous_when_auth_disabled(client: TestClient) -> None:
    response = client.get("/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "viewer"
    assert body["email"] == "anonymous@local"
    assert body["auth_enabled"] is False


def test_issue_token_endpoint(client: TestClient) -> None:
    response = client.post("/auth/token", json={"subject": "alice", "role": "admin"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in_minutes"] >= 1
    claims = decode_access_token(body["access_token"])
    assert claims["sub"] == "alice"
    assert claims["role"] == "admin"


# ---------------------------------------------------------------------------
# /auth endpoints — auth enabled
# ---------------------------------------------------------------------------
def test_auth_enforced_when_enabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    get_settings.cache_clear()

    # No token → 401.
    assert client.get("/auth/me").status_code == 401

    # Valid token → identity resolved from claims.
    token = create_access_token(subject="bob", role="analyst")
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "bob"
    assert body["role"] == "analyst"
    assert body["auth_enabled"] is True


def test_invalid_token_rejected_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    get_settings.cache_clear()

    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
