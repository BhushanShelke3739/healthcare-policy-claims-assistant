"""
Recursive character chunking.

The core idea
-------------
A long document is split using a *hierarchy* of separators — try the
biggest one first (paragraph breaks), and only fall back to finer ones
(sentences → words → characters) when a piece is still too large.

This preserves semantic boundaries much better than a hard
"every-N-characters" split, which routinely cuts sentences (or even
words) in half — terrible for downstream retrieval because the cited
quote ends up incomplete.

Why we don't just import LangChain's `RecursiveCharacterTextSplitter`
---------------------------------------------------------------------
* No extra dependency until Phase 4 actually needs LangChain for chains
  / prompts.
* Writing it ourselves is ~80 lines and makes the behavior unambiguous
  — useful for explaining retrieval quality in interviews.
* Easy to unit-test with synthetic input.

Public surface
--------------
    chunk_text(text, chunk_size, chunk_overlap, separators=None) -> list[str]
        Pure splitting. Returns the chunks as strings.

    chunk_document(text, chunk_size, chunk_overlap, metadata=None) -> list[Chunk]
        Orchestrator that wraps `chunk_text` and attaches per-chunk
        metadata (index, token count, the caller's metadata dict).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils.text import approximate_token_count

# Default separator hierarchy, biggest semantic unit first. The empty string
# at the end means "fall back to character-level if all else fails".
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced by `chunk_document`."""

    text: str
    index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pure text splitting
# =============================================================================
def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """
    Split `text` into chunks no larger than `chunk_size` characters.

    Args:
        text: input string. May be empty.
        chunk_size: maximum characters per chunk *before* overlap is applied.
            Note: overlap can push a chunk slightly above this size — that's
            the standard tradeoff for preserving cross-boundary context.
        chunk_overlap: how many characters of the previous chunk to prepend
            to the next. Must be < chunk_size.
        separators: hierarchy of separators to try, biggest semantic unit
            first. Defaults to DEFAULT_SEPARATORS.

    Returns:
        List of chunk strings, in document order. Empty input → empty list.

    Raises:
        ValueError: if chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    seps = tuple(separators) if separators is not None else DEFAULT_SEPARATORS

    # 1) Recursively split until every piece fits.
    raw_chunks = _split_recursive(text, seps, chunk_size)

    # 2) Apply character-level overlap as a post-processing step. Doing it
    #    after recursion keeps the recursive logic simple and predictable.
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        return _apply_overlap(raw_chunks, chunk_overlap)
    return raw_chunks


def _split_recursive(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
) -> list[str]:
    """Split `text` using the first separator that helps, recurse on big pieces."""
    if len(text) <= chunk_size:
        # Fits as-is — but drop pure-whitespace chunks (they add no signal).
        return [text] if text.strip() else []

    # Pick the first separator that actually appears in the text. If none
    # do, fall back to character-level (the "" sentinel at the end of the
    # default list).
    chosen_sep = ""
    remaining_seps: tuple[str, ...] = ()
    for i, sep in enumerate(separators):
        if sep == "":
            chosen_sep = ""
            remaining_seps = ()
            break
        if sep in text:
            chosen_sep = sep
            remaining_seps = separators[i + 1 :]
            break

    splits = list(text) if chosen_sep == "" else text.split(chosen_sep)

    # Merge consecutive splits back together until adding the next one
    # would exceed chunk_size. If a single split is itself too big, recurse
    # into it with the finer separators.
    chunks: list[str] = []
    buffer = ""
    for piece in splits:
        # Reconstitute the separator that .split() consumed.
        candidate = piece if not buffer else f"{buffer}{chosen_sep}{piece}"

        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        # candidate doesn't fit. Flush buffer (if any) and decide what to
        # do with `piece`.
        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(piece) <= chunk_size:
            buffer = piece
        else:
            # `piece` alone is still too big — recurse with finer separators.
            chunks.extend(_split_recursive(piece, remaining_seps, chunk_size))

    if buffer:
        chunks.append(buffer)

    # Strip empties / whitespace-only that may have slipped through.
    return [c for c in chunks if c.strip()]


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """
    Prepend the last `chunk_overlap` characters of each chunk to the next.

    This adds context across chunk boundaries — retrieval often misses a
    sentence that straddles a split point unless the next chunk repeats a
    bit of the prior one.
    """
    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        overlapped.append(tail + chunks[i])
    return overlapped


# =============================================================================
# Document-level orchestration
# =============================================================================
def chunk_document(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None = None,
    separators: tuple[str, ...] | list[str] | None = None,
) -> list[Chunk]:
    """
    Chunk `text` and wrap each piece in a `Chunk` with index, token count,
    and the caller-provided metadata.

    Metadata is *copied* into each chunk so subsequent code can mutate the
    chunk's metadata without affecting siblings or the caller's dict.
    """
    base_metadata = dict(metadata or {})
    texts = chunk_text(text, chunk_size, chunk_overlap, separators=separators)

    return [
        Chunk(
            text=t,
            index=i,
            token_count=approximate_token_count(t),
            metadata=dict(base_metadata),
        )
        for i, t in enumerate(texts)
    ]
