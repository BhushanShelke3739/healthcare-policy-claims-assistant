"""
Auth placeholder endpoints (Phase 9).

`POST /auth/token` — DEMO login. Mints a signed JWT for the requested subject
                     and role. **No password / no user lookup** — this stands
                     in for whatever real identity provider a production
                     deployment would use.
`GET  /auth/me`    — returns the identity the server resolved for this request
                     (the anonymous viewer when `AUTH_ENABLED=false`, otherwise
                     the bearer token's claims).

See `app/core/security.py` for the token + RBAC mechanics and
`docs/09_security_and_privacy.md` for the threat model and what "real" auth
would add here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.security import CurrentUser, create_access_token, get_current_user
from app.schemas.auth import CurrentUserResponse, TokenRequest, TokenResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="DEMO login — mint a JWT for a subject/role (no password check).",
)
def issue_token(payload: TokenRequest) -> TokenResponse:
    settings = get_settings()
    token = create_access_token(subject=payload.subject, role=payload.role)
    # Audit the issuance. Note we log subject + role but never the token itself.
    logger.info(
        "auth_token_issued",
        extra={"subject": payload.subject, "role": payload.role},
    )
    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Return the identity resolved for this request.",
)
def whoami(user: CurrentUser = Depends(get_current_user)) -> CurrentUserResponse:
    settings = get_settings()
    return CurrentUserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        auth_enabled=settings.auth_enabled,
    )
