"""
Health endpoints.

`/health`    — liveness: is the process up?
`/readiness` — readiness: are downstream dependencies (Postgres) reachable?

Why two endpoints?
    Kubernetes / Cloud Run distinguish "the container is alive, don't kill
    it" from "the container is ready to serve traffic". A DB outage should
    *not* trigger a liveness restart (that loops forever) — it should mark
    the pod un-ready until the DB returns. Two endpoints encode this.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    """
    Simple liveness probe. Returns 200 as long as the process is running.

    Does *not* touch the database — that's `/readiness`.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
    }


@router.get("/readiness", summary="Readiness probe")
def readiness(db: Session = Depends(get_db)) -> JSONResponse:
    """
    Check that the database is reachable.

    Returns 503 (rather than 500) when the DB is down so an orchestrator
    treats the pod as temporarily un-ready instead of crashed.
    """
    try:
        db.execute(text("SELECT 1"))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "database": "ok"},
        )
    except Exception as exc:
        logger.warning("readiness_db_check_failed", extra={"error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "database": "error", "detail": str(exc)},
        )
