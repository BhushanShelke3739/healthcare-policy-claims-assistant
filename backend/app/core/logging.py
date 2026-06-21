"""
Structured logging setup.

Why structured logs?
    Plain text is fine for a human tailing a terminal, but production
    aggregators (Cloud Logging, Datadog, Loki, ...) parse JSON natively —
    fields like `request_id`, `latency_ms`, `model_name` become queryable
    columns instead of regex targets.

Phase 1 ships a minimal JSON formatter using only the stdlib so the project
has no extra logging dependency. Phase 9 will extend this with request-id
middleware, latency timing, and a `/metrics` endpoint.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, ClassVar

from app.core.observability import RequestIdLogFilter


class JsonFormatter(logging.Formatter):
    """Format log records as a single JSON line per event."""

    # Standard LogRecord attributes we don't want duplicated in the payload.
    # `ClassVar` because this is shared state on the class, not a default
    # for an instance attribute — ruff RUF012.
    _RESERVED: ClassVar[set[str]] = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Anything passed via `logger.info("msg", extra={...})` flows through
        # as record attributes; copy unknown keys into the payload so callers
        # can attach structured context.
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Install a single stdout handler emitting JSON.

    Idempotent: calling it multiple times (e.g. from tests + app startup)
    won't stack handlers.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Replace any handlers added by basicConfig / prior imports.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    # Stamp the current request id (Phase 9) onto every record routed through
    # this handler, so all log lines emitted while serving a request share a
    # `request_id` field without each call site passing it via extra=.
    handler.addFilter(RequestIdLogFilter())
    root.addHandler(handler)

    # Quiet down noisy libraries by default; turn them back up via env if
    # debugging.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
