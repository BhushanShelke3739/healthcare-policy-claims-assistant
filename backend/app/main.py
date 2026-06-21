"""
FastAPI application entry point.

Responsibilities:
    * Configure logging before anything else runs.
    * Construct the FastAPI app with OpenAPI metadata.
    * Mount route modules.
    * Wire up application lifecycle hooks (startup / shutdown).

Routes mounted in Phase 1:
    /health       — liveness + dependency check
    /documents    — stubs (Phase 2)
    /rag          — stubs (Phases 3-4)
    /agents       — stubs (Phase 5)
    /eval         — stubs (Phase 6)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_agents,
    routes_auth,
    routes_documents,
    routes_eval,
    routes_health,
    routes_metrics,
    routes_rag,
)
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.observability import RequestContextMiddleware

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Startup / shutdown hook.

    Phase 1 just logs that the app booted. Later phases will use this for:
        * Warm-loading embedding models (Phase 3).
        * Pre-connecting to the vector store.
        * Starting background workers.
    """
    logger.info(
        "application_startup",
        extra={
            "app_env": settings.app_env,
            "app_version": settings.app_version,
        },
    )
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """
    Application factory.

    Using a factory (vs. a module-level `app = FastAPI(...)`) makes it easy
    to construct fresh instances in tests with overridden dependencies.
    """
    app = FastAPI(
        title="Healthcare Policy & Claims Assistant",
        description=(
            "AI-powered assistant for searching healthcare policy documents, "
            "answering claims-related questions with citations, and running "
            "agentic workflows. Synthetic data only — no real PHI/PII."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Middleware order note: Starlette runs middleware in reverse of the order
    # they're added (last added = outermost). We want RequestContextMiddleware
    # outermost so its request-id is set and its timer is running before any
    # other layer (including CORS) executes — so it's added *after* CORS.

    # CORS — the Phase 7 Next.js frontend lives on a different origin
    # (http://localhost:3000) than the API (http://localhost:8000). Without
    # this middleware the browser blocks every cross-origin request. We keep
    # the allowed origin list narrow and configurable rather than `*`, since
    # `allow_credentials=True` cannot be combined with a wildcard.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Phase 9: per-request correlation id + latency timing + Prometheus
    # metric recording. Added last so it wraps everything above.
    app.add_middleware(RequestContextMiddleware)

    # Phase 9: structured, request-id-tagged error responses.
    register_exception_handlers(app)

    # Health + metrics are mounted at the root so probes / scrapers don't need
    # a prefix.
    app.include_router(routes_health.router, tags=["health"])
    app.include_router(routes_metrics.router, tags=["observability"])

    app.include_router(routes_auth.router, prefix="/auth", tags=["auth"])
    app.include_router(routes_documents.router, prefix="/documents", tags=["documents"])
    app.include_router(routes_rag.router, prefix="/rag", tags=["rag"])
    app.include_router(routes_agents.router, prefix="/agents", tags=["agents"])
    app.include_router(routes_eval.router, prefix="/eval", tags=["eval"])

    return app


# The ASGI server (uvicorn) imports this name.
app = create_app()
