"""
Phase 9 observability primitives.

Three concerns, one module, because they all describe the same thing — how a
single HTTP request becomes a log line plus a metric sample:

    1. Request correlation. A contextvar carries a per-request id so every log
       record emitted while handling a request can be tied back together. The
       id is taken from an inbound ``X-Request-ID`` header when present (so a
       trace survives across service hops / a reverse proxy) or generated
       otherwise, and echoed back on the response.

    2. Prometheus metrics. A request counter and a latency histogram, exposed
       at ``/metrics`` for scraping. Labels use the *route template*
       (``/rag/ask``) rather than the raw URL so high-cardinality path
       segments can never blow up the time-series count.

    3. Middleware. One ASGI middleware generates/propagates the id, times the
       request, records the metrics, stamps the response headers, and emits a
       structured ``request_completed`` log line.

Keeping the contextvar, the log filter, the metric objects, and the middleware
in one file means there is exactly one place to read to understand the
request-observability story.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar, Token

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.request")

# Header used both to accept an upstream id and to echo ours back.
REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


# =============================================================================
# Request correlation
# =============================================================================
# Default "-" (rather than None) so records emitted outside any request — app
# startup, a `python -m app.<script>` CLI run — still carry the field with a
# sentinel value instead of raising / showing up as missing.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the id for the request currently being handled (or "-")."""
    return _request_id_ctx.get()


class RequestIdLogFilter(logging.Filter):
    """
    Stamp ``request_id`` onto every LogRecord.

    Installed on the root handler (see ``core.logging.configure_logging``) so
    the JSON formatter picks the field up for *all* log lines — individual
    call sites never have to thread the id through ``extra={...}``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


# =============================================================================
# Prometheus metrics
# =============================================================================
# A dedicated registry (instead of the global default) keeps test runs
# isolated and means /metrics exposes exactly what we register here — nothing
# leaks in from a library that happened to touch the default registry.
REGISTRY = CollectorRegistry()

HTTP_REQUESTS_TOTAL = Counter(
    "hpca_http_requests_total",
    "Total HTTP requests processed, by method, route template, and status.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "hpca_http_request_duration_seconds",
    "HTTP request latency in seconds, by method and route template.",
    labelnames=("method", "path"),
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Return the exposition payload + content type for the /metrics handler."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def _route_template(request: Request) -> str:
    """
    The matched route's path pattern (e.g. ``/documents/{document_id}``).

    Falls back to ``"unmatched"`` for requests that didn't resolve to a route
    (404s, probes for random paths). This is the cardinality guard: without it
    every distinct URL would become its own metric label set.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


# =============================================================================
# Middleware
# =============================================================================
class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Per-request correlation id + latency timing + metric recording.

    Outermost middleware in the stack so its timing brackets everything else
    (including CORS) and so a generated id is in place before any downstream
    code logs.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token: Token[str] = _request_id_ctx.set(request_id)
        # Stash on request.state too: it's the channel the exception handler
        # reads the id from when building a 500 envelope.
        request.state.request_id = request_id
        start = time.perf_counter()
        method = request.method

        try:
            response = await call_next(request)
        except Exception:
            # call_next raised before producing a response. Record the failure
            # as a 500 sample + an error log (with traceback), then re-raise so
            # the registered catch-all builds the response. We intentionally do
            # NOT reset the contextvar here — the catch-all runs *after* this
            # frame and still needs `get_request_id()` to resolve.
            duration = time.perf_counter() - start
            path = _route_template(request)
            HTTP_REQUESTS_TOTAL.labels(method, path, "500").inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method, path).observe(duration)
            logger.exception(
                "request_failed",
                extra={"method": method, "path": path, "duration_ms": round(duration * 1000, 1)},
            )
            raise

        duration = time.perf_counter() - start
        path = _route_template(request)
        HTTP_REQUESTS_TOTAL.labels(method, path, str(response.status_code)).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method, path).observe(duration)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration * 1000:.1f}"

        # Emitted while the contextvar is still set, so this line carries the
        # request_id too (the reset happens immediately after).
        logger.info(
            "request_completed",
            extra={
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 1),
            },
        )
        _request_id_ctx.reset(token)
        return response
