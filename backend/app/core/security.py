"""
Authentication & authorization (Phase 9).

This is a *working* JWT + role-based-access layer that is **disabled by
default** (`AUTH_ENABLED=false`). With auth off, every request resolves to the
anonymous viewer and the demo runs without tokens — exactly as it did before
Phase 9. With auth on, endpoints that depend on `get_current_user` /
`require_role` require a valid bearer token.

Why ship it dormant rather than fully on?
    The project is a portfolio/demo with synthetic data and no user store. A
    real deployment would back this with an identity provider (Google IAP,
    Auth0, Cognito, …) and a users table. What matters here is that the
    *seams* are real and centralized: there is exactly one place that decides
    "who is this caller" (`get_current_user`) and one place that decides "are
    they allowed" (`require_role`). Wiring in a real IdP later means changing
    those two functions, not chasing auth checks across every handler.

Design choices:
    * HS256 symmetric tokens via PyJWT — smallest dependency that does the job.
    * `HTTPBearer(auto_error=False)` so a missing token yields our own 401 with
      a clear message instead of FastAPI's generic one, and so the anonymous
      path works when auth is disabled.
    * Roles are a closed set (`admin` > `analyst` > `viewer`) checked by exact
      membership — no hierarchy magic; a route lists the roles it accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

Role = Literal["admin", "analyst", "viewer"]

# Re-exported so callers catch the right error without importing PyJWT.
InvalidTokenError = jwt.InvalidTokenError


@dataclass(frozen=True)
class CurrentUser:
    """Minimal identity object returned by `get_current_user`."""

    id: str
    email: str
    role: Role


# The placeholder identity used whenever auth is disabled.
ANONYMOUS_VIEWER = CurrentUser(
    id="00000000-0000-0000-0000-000000000000",
    email="anonymous@local",
    role="viewer",
)


# =============================================================================
# Token issue / verify
# =============================================================================
def create_access_token(
    *,
    subject: str,
    role: Role,
    email: str | None = None,
    expires_minutes: int | None = None,
) -> str:
    """
    Mint a signed JWT for `subject` with the given `role`.

    Standard claims: `sub`, `iat`, `exp`. Custom claims: `role`, `email`.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "email": email or f"{subject}@local",
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Verify signature + expiry and return the claims.

    Raises `InvalidTokenError` (PyJWT) on a bad signature, expiry, or malformed
    token — callers translate that into a 401.
    """
    settings = get_settings()
    claims: dict[str, Any] = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    return claims


# =============================================================================
# FastAPI dependencies
# =============================================================================
# auto_error=False: we want to handle "no credentials" ourselves so the
# anonymous path works when auth is disabled.
_bearer_scheme = HTTPBearer(auto_error=False, description="JWT bearer token")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> CurrentUser:
    """
    Resolve the calling user.

    * Auth disabled → always the anonymous viewer (the historical behavior).
    * Auth enabled  → require + verify a bearer token; build a `CurrentUser`
      from its claims. Missing/invalid token → 401.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return ANONYMOUS_VIEWER

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return CurrentUser(
        id=str(claims.get("sub", "unknown")),
        email=str(claims.get("email", "unknown@local")),
        role=claims.get("role", "viewer"),
    )


def require_role(*allowed_roles: Role):
    """
    Dependency factory: gate an endpoint to a set of roles.

    Usage::

        @router.post("/admin/reindex")
        def reindex(user: CurrentUser = Depends(require_role("admin"))):
            ...

    With auth disabled the caller is always the anonymous *viewer*, so routes
    guarded by `require_role("admin")` would 403 — which is why no existing
    route is gated yet; this is wiring for when a real IdP lands.
    """

    def _dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}.",
            )
        return user

    return _dependency
