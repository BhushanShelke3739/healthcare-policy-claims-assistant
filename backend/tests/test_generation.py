"""
Unit tests for the generation service.

Only the mock provider is exercised — the OpenAI provider needs network
or a stub server. We keep this file network-free.
"""

from __future__ import annotations

import uuid

from app.core.config import get_settings
from app.schemas.rag import GeneratedAnswer
from app.services.generation import MockChatProvider, build_user_message
from app.services.retrieval import RetrievedChunk


def _make_chunk(
    text: str,
    score: float = 0.8,
    title: str = "Test Policy",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        chunk_text=text,
        chunk_index=0,
        similarity_score=score,
        component_scores={"vector": score},
        metadata={},
    )


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------
def test_mock_returns_refusal_when_no_chunks() -> None:
    provider = MockChatProvider(get_settings())
    result = provider.generate("What is the appeal window?", chunks=[])

    assert isinstance(result, GeneratedAnswer)
    assert "could not find" in result.answer.lower()
    assert result.citations == []
    assert result.confidence == "low"


def test_mock_uses_refusal_phrase_from_settings() -> None:
    """The exact phrase is part of the spec — must match settings."""
    settings = get_settings()
    provider = MockChatProvider(settings)
    result = provider.generate("anything", chunks=[])
    assert result.answer == settings.refusal_phrase


def test_mock_produces_citations_when_chunks_exist() -> None:
    chunks = [
        _make_chunk("First-level appeals filed within sixty days.", score=0.9),
        _make_chunk("Three levels of appeal.", score=0.5),
        _make_chunk("Modifier 95 required.", score=0.4),
    ]
    provider = MockChatProvider(get_settings())
    result = provider.generate("How long to appeal?", chunks=chunks)

    assert result.citations, "expected at least one citation"
    # Mock cites the top-2 chunks.
    assert len(result.citations) == 2
    # Each citation references one of the retrieved chunk_ids verbatim.
    cited_ids = {c.chunk_id for c in result.citations}
    chunk_ids = {str(c.chunk_id) for c in chunks[:2]}
    assert cited_ids == chunk_ids


def test_mock_confidence_tracks_top_score() -> None:
    provider = MockChatProvider(get_settings())

    high = provider.generate("q", chunks=[_make_chunk("text", score=0.85)])
    med = provider.generate("q", chunks=[_make_chunk("text", score=0.5)])
    low = provider.generate("q", chunks=[_make_chunk("text", score=0.1)])

    assert high.confidence == "high"
    assert med.confidence == "medium"
    assert low.confidence == "low"


def test_mock_answer_is_top_chunk_text() -> None:
    """
    The mock is honest about being a mock: its answer is the top chunk
    verbatim, not invented text.
    """
    top_text = "Synchronous telehealth visits require modifier 95."
    chunks = [_make_chunk(top_text)]
    provider = MockChatProvider(get_settings())
    result = provider.generate("anything", chunks=chunks)
    assert result.answer == top_text


def test_mock_grounding_notes_identifies_itself() -> None:
    chunks = [_make_chunk("any text", score=0.5)]
    result = MockChatProvider(get_settings()).generate("q", chunks=chunks)
    assert "mock" in result.grounding_notes.lower()


# ---------------------------------------------------------------------------
# build_user_message — prompt construction
# ---------------------------------------------------------------------------
def test_user_message_contains_chunk_ids_for_citation() -> None:
    """
    The LLM produces citations by copying chunk_id literals from the
    prompt. The prompt must therefore include each chunk_id as a label.
    """
    chunks = [
        _make_chunk("chunk text A"),
        _make_chunk("chunk text B"),
    ]
    message = build_user_message("How long to appeal?", chunks=chunks)

    for c in chunks:
        assert str(c.chunk_id) in message, "chunk_id should appear in prompt"
    assert "How long to appeal?" in message
    assert "chunk text A" in message
    assert "chunk text B" in message


def test_user_message_handles_no_chunks() -> None:
    message = build_user_message("any question", chunks=[])
    # Doesn't crash; explicitly notes "no policy context".
    assert "question" in message.lower()
    assert "no policy context" in message.lower()
