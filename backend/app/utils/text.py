"""
Text-cleaning helpers.

Kept deliberately tiny in Phase 1 — only utilities that are obviously
useful before document ingestion lands. Chunking + token counting move in
in Phase 2.
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace (incl. newlines) into single spaces and trim.

    Useful for ingestion so retrieval matches against a canonical form rather
    than the messy output of PDF extraction.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def approximate_token_count(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English.

    Good enough for chunk-size budgeting and quick-and-dirty cost estimates.
    Phase 4 can swap in `tiktoken` if precise counts are needed.
    """
    return max(1, len(text) // 4)
