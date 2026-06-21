"""
Prometheus metrics endpoint (Phase 9).

`GET /metrics` returns the current metric values in Prometheus text exposition
format. A scraper (Prometheus, the OpenTelemetry Collector, Grafana Agent, …)
polls this on an interval; the values come from the counters / histograms that
`RequestContextMiddleware` updates on every request.

Mounted at the root (no prefix) because `/metrics` is the conventional path
scrapers default to.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.core.observability import render_metrics

router = APIRouter()


@router.get(
    "/metrics",
    summary="Prometheus metrics exposition",
    # Hidden from the OpenAPI docs: it's a machine endpoint, and its response
    # is plain text, not JSON, so it would only clutter the schema.
    include_in_schema=False,
)
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
