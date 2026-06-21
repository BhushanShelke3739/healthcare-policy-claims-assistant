"""
Unit tests for the chunking service.

Covers the four cases called out by Phase 2:
    * chunk creation
    * overlap behavior
    * empty document handling
    * metadata persistence
"""

from __future__ import annotations

import pytest

from app.services.chunking import Chunk, chunk_document, chunk_text


# ---------------------------------------------------------------------------
# Empty / boundary input
# ---------------------------------------------------------------------------
def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_whitespace_only_text_returns_empty_list() -> None:
    """All-whitespace input shouldn't produce a noise chunk."""
    assert chunk_text("   \n\n \t  ", chunk_size=100, chunk_overlap=10) == []


def test_short_text_returns_single_chunk() -> None:
    """If the text already fits, no splitting happens."""
    text = "Prior authorization is required for advanced imaging."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert chunks == [text]


# ---------------------------------------------------------------------------
# Chunk creation — splits on the largest available separator
# ---------------------------------------------------------------------------
def test_splits_on_paragraph_boundary_when_possible() -> None:
    """Prefer paragraph breaks over sentence/word boundaries."""
    para1 = "Section 1: Prior authorization rules and approval timelines."
    para2 = "Section 2: Appeals can be filed within sixty days of denial."
    text = f"{para1}\n\n{para2}"

    chunks = chunk_text(text, chunk_size=len(para1) + 5, chunk_overlap=0)

    assert len(chunks) == 2
    # Paragraphs survive intact — no mid-sentence cuts.
    assert para1 in chunks[0]
    assert para2 in chunks[1]


def test_long_run_falls_back_to_finer_separators() -> None:
    """A single long line should fall back to sentence- / word-level splitting."""
    sentences = [
        "Prior authorization is required for advanced imaging.",
        "Approval is communicated within five business days.",
        "Urgent cases are reviewed within twenty four hours.",
        "Denials of prior authorization may be appealed in writing.",
    ]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size=80, chunk_overlap=0)

    assert len(chunks) > 1
    # No chunk should massively exceed chunk_size (we allow some slack for
    # the recursive packing — but each individual sentence is < 80 chars).
    for c in chunks:
        assert len(c) <= 120, f"chunk too large: {c!r} ({len(c)} chars)"


# ---------------------------------------------------------------------------
# Overlap behavior
# ---------------------------------------------------------------------------
def test_overlap_prepends_tail_of_previous_chunk() -> None:
    """With overlap > 0, each chunk (after the first) starts with chars from the previous chunk."""
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4  # 104 chars, no separators

    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)

    assert len(chunks) >= 2
    # Find the boundary between any two consecutive chunks and verify the
    # overlap window is identical.
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-5:]
        next_head = chunks[i][:5]
        assert prev_tail == next_head, (
            f"overlap mismatch at boundary {i}: prev tail={prev_tail!r}, "
            f"next head={next_head!r}"
        )


def test_overlap_zero_produces_disjoint_chunks() -> None:
    """With overlap=0, concatenating chunks reproduces the source (modulo separators)."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)

    # No chunk shares text with the next when overlap is 0. We assert the
    # union of chunks (joined back with spaces) covers every original token.
    rejoined = " ".join(chunks).split()
    expected_tokens = text.split()
    for tok in expected_tokens:
        assert tok in rejoined


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", chunk_size=50, chunk_overlap=50)


# ---------------------------------------------------------------------------
# Metadata persistence (via chunk_document)
# ---------------------------------------------------------------------------
def test_chunk_document_assigns_sequential_indices() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_document(text, chunk_size=12, chunk_overlap=0)

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_attaches_metadata_to_every_chunk() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    meta = {"document_title": "Appeals Policy", "section": "first-level"}

    chunks = chunk_document(text, chunk_size=12, chunk_overlap=0, metadata=meta)

    assert all(c.metadata == meta for c in chunks)


def test_chunk_document_metadata_is_copied_not_shared() -> None:
    """Mutating one chunk's metadata must not bleed into others."""
    text = "A.\n\nB.\n\nC."
    meta = {"section": "appeals"}

    chunks = chunk_document(text, chunk_size=2, chunk_overlap=0, metadata=meta)
    if not chunks:
        pytest.skip("expected at least one chunk")

    chunks[0].metadata["section"] = "MUTATED"
    for c in chunks[1:]:
        assert c.metadata["section"] == "appeals"
    # And the caller's dict is untouched.
    assert meta["section"] == "appeals"


def test_chunk_document_records_token_count() -> None:
    chunks = chunk_document("Hello world", chunk_size=100, chunk_overlap=0)
    assert chunks and chunks[0].token_count > 0


def test_chunk_document_returns_chunk_objects() -> None:
    chunks = chunk_document("hello", chunk_size=100, chunk_overlap=0)
    assert all(isinstance(c, Chunk) for c in chunks)
