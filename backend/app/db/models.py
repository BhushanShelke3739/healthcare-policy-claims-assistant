"""
SQLAlchemy ORM models for Healthcare Policy & Claims Assistant.

Modeling notes
--------------
* **UUID primary keys** make rows safe to share across logs / messages /
  external systems without leaking row counts.
* **`created_at` defaults to now()** at the DB layer so it's correct even
  when rows are inserted by SQL directly (e.g. migrations).
* **`DocumentChunk.embedding`** uses pgvector's `Vector` type. The
  dimension comes from settings so we can swap embedding providers
  without re-typing the column. An IVFFLAT index is added in the
  Phase 3 migration once we have real vectors to index.
* **JSONB** is used for chunk metadata so we can attach arbitrary
  per-chunk context (section, page, source URL, ...) without schema
  changes every time a new field is needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base

_settings = get_settings()
EMBEDDING_DIM = _settings.embedding_dimensions


def _uuid() -> uuid.UUID:
    """UUID4 default for PK columns (Python-side, portable)."""
    return uuid.uuid4()


# =============================================================================
# User
# =============================================================================
class User(Base):
    """
    Application user.

    Phase 1 stores only the minimum fields needed to associate audit data
    with an identity. Authentication / password hashing arrives in Phase 9.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="viewer", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# =============================================================================
# Document
# =============================================================================
class Document(Base):
    """
    A single source document (policy PDF, markdown file, etc.).

    The actual binary/text content lives elsewhere — only metadata is
    persisted here. Chunks (`DocumentChunk`) hold the text that retrieval
    will operate on.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    # Where the doc came from: "upload", "url", "seed", ...
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, default="upload")
    file_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Healthcare-domain tag: "policy", "claim", "appeal", "compliance", ...
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, default="policy")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        # Deleting a document removes its chunks (and their embeddings) too.
    )


# =============================================================================
# DocumentChunk
# =============================================================================
class DocumentChunk(Base):
    """
    A chunk of a document — the atomic unit retrieval operates on.

    The `embedding` column is nullable because Phase 2 ingests text before
    Phase 3 introduces embeddings. After Phase 3 the ingestion flow should
    populate it inline.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        # Attribute is renamed from `metadata` because SQLAlchemy's
        # Declarative base reserves the name `metadata` on the class itself.
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # `tsv` is a Postgres GENERATED ALWAYS column populated from
    # chunk_text (see Alembic migration 0002). `Computed(...)` tells
    # SQLAlchemy this is a generated column so it's excluded from
    # INSERT/UPDATE statements — Postgres would otherwise reject any
    # explicit value, even NULL.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


# =============================================================================
# QueryLog
# =============================================================================
class QueryLog(Base):
    """
    Audit log of every question routed through the RAG pipeline.

    Useful for:
        * Replaying questions when iterating on prompts.
        * Cost / latency analysis.
        * Debugging hallucinations after the fact.
    """

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # IDs of chunks retrieved for this query. Stored as JSONB list of UUID
    # strings so we don't need a join table just to record provenance.
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Phase 9 observability grab-bag: retrieval mode/top_k, per-chunk scores,
    # citation keep/drop counts, confidence, refusal flag. JSONB so new signals
    # don't cost a migration (see Alembic 0004).
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# =============================================================================
# EvaluationRun + EvaluationResult
# =============================================================================
class EvaluationRun(Base):
    """
    A single execution of the evaluation harness against a question set.

    A run holds many results (one per question). Aggregate metrics for the
    run are computed by querying the results table.
    """

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    results: Mapped[list[EvaluationResult]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvaluationResult(Base):
    """Per-question evaluation outcome attached to a run."""

    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Approximations / signals — all floats in [0.0, 1.0] when present.
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)

    hallucination_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # JSONB grab-bag for anything that doesn't deserve its own column:
    # retrieved_chunk_ids, expected_document, keyword recall breakdown,
    # retrieval_hit boolean. Adding a new metric here costs zero
    # migrations.
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")
