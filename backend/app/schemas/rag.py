"""
Pydantic schemas for the RAG API.

Phase 3 — /rag/retrieve schemas (RetrieveRequest, RetrieveResponse).
Phase 4 — /rag/ask schemas (AskRequest, AskResponse, Citation) +
          GeneratedAnswer (what the LLM produces; the API wraps it).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalMode = Literal["vector", "keyword", "hybrid"]
Confidence = Literal["low", "medium", "high"]


# =============================================================================
# /rag/retrieve
# =============================================================================
class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="Free-text query.")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode = Field(
        default="hybrid",
        description=(
            "vector  — pgvector cosine search.\n"
            "keyword — PostgreSQL full-text search.\n"
            "hybrid  — weighted combination (default; see `alpha`)."
        ),
    )
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid weight on the vector score (1 - alpha is the keyword "
            "weight). When omitted, falls back to settings.hybrid_alpha."
        ),
    )
    document_type: str | None = Field(
        default=None,
        max_length=64,
        description='Optional filter (e.g. "policy" | "claim").',
    )


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    chunk_index: int
    similarity_score: float = Field(
        description="Combined score (mode-dependent). Higher is better."
    )
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description='Per-mode score components, e.g. {"vector": 0.81, "keyword": 0.42, "alpha": 0.6}.',
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    results: list[RetrievedChunk]
    total: int


# =============================================================================
# /rag/ask
# =============================================================================
class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question about the policy corpus.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many chunks to retrieve as grounding context.",
    )
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Retrieval mode handed to /rag/retrieve under the hood.",
    )
    include_citations: bool = Field(
        default=True,
        description="If False, citations are stripped from the response.",
    )
    document_type: str | None = Field(
        default=None,
        max_length=64,
        description="Optional retrieval filter (e.g. only search 'policy' docs).",
    )


class Citation(BaseModel):
    """
    A single citation backing a claim in the answer.

    `chunk_id` always references a chunk that was actually retrieved —
    server-side validation drops any citation pointing at a chunk_id the
    LLM made up.
    """

    document_title: str
    chunk_id: uuid.UUID
    excerpt: str = Field(
        description=(
            "Verbatim excerpt from the cited chunk, ideally the sentence(s) "
            "directly supporting the claim."
        )
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: Confidence
    grounding_notes: str | None = Field(
        default=None,
        description=(
            "Short explanation of how the answer maps to the retrieved "
            "evidence, plus any server-side notes (e.g. dropped citations)."
        ),
    )
    retrieved_chunk_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="Every chunk that was retrieved, in retrieval order.",
    )
    model_name: str = Field(description="Which LLM produced the answer (or 'mock').")
    latency_ms: int


# =============================================================================
# Structured-output schema produced by the LLM
# =============================================================================
class _LLMCitation(BaseModel):
    """
    What the LLM emits. Uses str for chunk_id because (a) OpenAI strict
    structured output supports str better than uuid, and (b) we'll validate
    + convert server-side anyway.
    """

    chunk_id: str
    excerpt: str


class GeneratedAnswer(BaseModel):
    """
    The schema we ask the LLM to produce. Kept narrow / unambiguous to
    play nicely with both OpenAI strict mode and Ollama JSON mode.
    """

    answer: str
    citations: list[_LLMCitation]
    confidence: Confidence
    grounding_notes: str
