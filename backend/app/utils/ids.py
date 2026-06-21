"""
Tiny helpers for ID generation.

Centralizing means we can swap UUID4 for ULID or a snowflake without
touching call sites. Helpful when (e.g.) you want timestamp-sortable IDs
for query logs in Phase 9.
"""

from __future__ import annotations

import uuid


def new_uuid() -> uuid.UUID:
    """Return a new UUID4."""
    return uuid.uuid4()


def new_uuid_str() -> str:
    """Return a new UUID4 as a string (handy for JSONB columns / logs)."""
    return str(uuid.uuid4())
