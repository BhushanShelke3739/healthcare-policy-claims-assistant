"""
Pydantic schemas for the auth placeholder API (Phase 9).

These back the demo `/auth/token` and `/auth/me` endpoints. There is no user
store and no password check — `/auth/token` mints a token for whatever role
you ask for. It exists to make the JWT + RBAC seam tangible and testable, not
to model real authentication.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["admin", "analyst", "viewer"]


class TokenRequest(BaseModel):
    """Demo login: pick a subject + role and get a signed token. No password."""

    subject: str = Field(
        default="demo-user",
        min_length=1,
        max_length=128,
        description="Identifier embedded as the token's `sub` claim.",
    )
    role: Role = Field(default="viewer", description="Role embedded in the token.")


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_minutes: int


class CurrentUserResponse(BaseModel):
    """Shape returned by `/auth/me`."""

    id: str
    email: str
    role: Role
    auth_enabled: bool = Field(
        description="Whether the server is enforcing auth; when false the caller is anonymous."
    )
