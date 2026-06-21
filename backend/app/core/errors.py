"""
Centralized error handling (Phase 9).

Goal: every error leaving the API has a *consistent JSON shape* and carries
the request id, so a caller (or the frontend's ErrorBanner) can quote one id
back to an operator who then greps the logs for the full story.

We register one handler:

    * ``Exception``            — the catch-all. Turns any unhandled exception
      into a 500 with a generic message (no stack trace / internals leaked to
      the client) while the full traceback is logged server-side by the
      observability middleware.

We deliberately leave FastAPI's built-in handlers alone: ``HTTPException`` keeps
its ``{"detail": ...}`` body (clients — including our own tests — depend on
that), and ``RequestValidationError`` keeps its field-level 422 list. The
catch-all only changes the behavior for *unhandled* errors, which previously
returned Starlette's bare ``"Internal Server Error"`` text.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.observability import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger("app.errors")


def _request_id(request: Request) -> str:
    # request.state is set by the middleware; fall back to the contextvar
    # (which is still set when the catch-all runs on the error path).
    return getattr(request.state, "request_id", None) or get_request_id()


def _envelope(*, error_type: str, message: str, request_id: str, status_code: int) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "request_id": request_id,
            }
        },
    )
    # The 500 path builds this response outside the request-id middleware, so
    # echo the header here too — otherwise a failed request wouldn't carry it.
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id(request)
    settings = get_settings()
    # In non-prod, surface the exception text to speed up local debugging; in
    # production keep it generic so internals / data never leak to a client.
    message = (
        "An unexpected error occurred."
        if settings.is_production
        else f"{type(exc).__name__}: {exc}"
    )
    # The middleware already logged the traceback (logger.exception) with the
    # request id attached, so here we only need the envelope.
    return _envelope(
        error_type="InternalServerError",
        message=message,
        request_id=request_id,
        status_code=500,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the Phase 9 catch-all error handler to the app."""
    app.add_exception_handler(Exception, _handle_unexpected)
