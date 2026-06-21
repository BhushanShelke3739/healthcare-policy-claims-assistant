"""
Retrieval service — vector, keyword, and hybrid search over `document_chunks`.

Three modes
-----------
    vector   : pgvector cosine similarity. Good for semantic / paraphrase
               match.
    keyword  : PostgreSQL full-text search (`tsvector @@ plainto_tsquery`).
               Good for identifier / acronym match (e.g. "HF-022").
    hybrid   : weighted linear combination of the two, with per-mode
               min-max normalization so the scores are comparable.

Why we do the score combination ourselves
-----------------------------------------
We could push everything into one SQL query with a CTE — that's faster.
But computing it in Python keeps the math obvious and lets us emit the
per-mode component scores in the response (useful for debugging *why* a
chunk ranked highly). Two queries against an HNSW + GIN index at our
scale is still sub-millisecond.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_text: str
    chunk_index: int
    similarity_score: float
    # Per-mode component scores. Populated for "hybrid" so callers can see
    # which signal carried each result; for the single-mode searches, only
    # the matching key is populated.
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Vector search
# =============================================================================
def vector_search(
    db: Session,
    *,
    query_text: str,
    top_k: int,
    document_type: str | None = None,
) -> list[RetrievedChunk]:
    """Cosine-distance nearest neighbors via pgvector + HNSW."""
    embedder = get_embedder()
    [query_vec] = embedder.embed([query_text])

    # `cosine_distance` is provided by the pgvector SQLAlchemy adapter.
    # Distance in [0, 2]; we convert to similarity = 1 - distance, which
    # is the convention most users expect.
    distance = DocumentChunk.embedding.cosine_distance(query_vec)

    stmt = (
        select(
            DocumentChunk,
            Document.title,
            distance.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        # NULL embeddings (rows ingested before Phase 3 / before backfill)
        # don't have a meaningful distance — exclude them explicitly.
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)

    rows = db.execute(stmt).all()
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            chunk_text=chunk.chunk_text,
            chunk_index=chunk.chunk_index,
            similarity_score=float(1.0 - dist),
            component_scores={"vector": float(1.0 - dist)},
            metadata=chunk.chunk_metadata or {},
        )
        for chunk, title, dist in rows
    ]


# =============================================================================
# Keyword search (Postgres FTS)
# =============================================================================
def _build_or_tsquery(query_text: str):
    """
    Turn a natural-language query into an OR-joined `websearch_to_tsquery`.

    Why not `plainto_tsquery`?
        plainto_tsquery ANDs every content word: a 7-word question like
        "How many days for an external IRO decision?" demands chunks that
        contain ALL of {many, days, external, IRO, decision}, and basically
        no policy chunk does. The eval harness surfaced this empirically —
        15/16 in-scope questions returned zero chunks under pure-keyword.

    Solution
        Build a websearch_to_tsquery from " or "-joined content words
        (≥3 chars). Any chunk containing ANY of the terms participates;
        ts_rank_cd (cover-density rank) still surfaces the most relevant
        ones first. Net effect: keyword recall goes from ~6% to ~80%+ on
        natural-language questions, at no precision cost on the top
        results.

    Edge cases
        * If the query has no content words, fall back to the original
          query string — Postgres will likely return nothing, but at
          least the syntax is valid.
        * Hyphenated identifiers like "HF-022" are split by Postgres'
          English text-search config into the parts: the whole token
          "hf-022", the letter part "hf", and the number part — but
          `hword_numpart` ("022") is NOT mapped to any dictionary in
          the default English config, so it never lands in the tsv.
          That's why we keep the threshold at 2 chars (not 3): so the
          "hf" part survives our content-word filter and matches the
          tsv. Also covers other common short acronyms (PA, CT, OON).
    """
    settings = get_settings()
    content_words = [w for w in re.findall(r"[A-Za-z0-9]+", query_text.lower()) if len(w) >= 2]
    # Include the original query verbatim too — that lets
    # `websearch_to_tsquery` apply its hyphenated-word handling to
    # identifiers like "HF-022" before we OR-decompose into parts.
    or_text = f"{query_text} or " + " or ".join(content_words) if content_words else query_text
    return func.websearch_to_tsquery(settings.fts_language, or_text)


def keyword_search(
    db: Session,
    *,
    query_text: str,
    top_k: int,
    document_type: str | None = None,
) -> list[RetrievedChunk]:
    """
    Full-text search via the generated `tsv` column.

    Queries are OR-joined (see `_build_or_tsquery`) so a long natural-
    language question still finds chunks that match any meaningful term.
    `ts_rank_cd` (cover-density rank) handles the relevance ordering.
    """
    ts_query = _build_or_tsquery(query_text)
    score = func.ts_rank_cd(DocumentChunk.tsv, ts_query)

    stmt = (
        select(
            DocumentChunk,
            Document.title,
            score.label("score"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.tsv.op("@@")(ts_query))
        .order_by(score.desc())
        .limit(top_k)
    )
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)

    rows = db.execute(stmt).all()
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            chunk_text=chunk.chunk_text,
            chunk_index=chunk.chunk_index,
            similarity_score=float(s),
            component_scores={"keyword": float(s)},
            metadata=chunk.chunk_metadata or {},
        )
        for chunk, title, s in rows
    ]


# =============================================================================
# Hybrid search — weighted combination
# =============================================================================
def hybrid_search(
    db: Session,
    *,
    query_text: str,
    top_k: int,
    alpha: float | None = None,
    document_type: str | None = None,
) -> list[RetrievedChunk]:
    """
    Combine vector + keyword scores with `alpha * vector + (1-alpha) * keyword`.

    Both sides are min-max normalized to [0, 1] over their own result set
    before combining; otherwise the keyword `ts_rank_cd` (typically < 1)
    and the vector similarity (~0.0-1.0) aren't on the same scale.

    Over-fetch from both sides (4x top_k) before fusion so chunks that
    rank just outside the top of one mode but very high in the other
    still survive.
    """
    settings = get_settings()
    effective_alpha = settings.hybrid_alpha if alpha is None else alpha
    over_fetch = max(top_k * 4, 20)

    vec_hits = vector_search(
        db, query_text=query_text, top_k=over_fetch, document_type=document_type
    )
    kw_hits = keyword_search(
        db, query_text=query_text, top_k=over_fetch, document_type=document_type
    )

    vec_norm = _minmax_normalize({h.chunk_id: h.similarity_score for h in vec_hits})
    kw_norm = _minmax_normalize({h.chunk_id: h.similarity_score for h in kw_hits})

    # Union of chunk_ids seen by either side.
    all_ids = set(vec_norm) | set(kw_norm)

    # Build a quick id -> hit lookup so we can borrow text/metadata from
    # whichever side surfaced each chunk first.
    by_id: dict[uuid.UUID, RetrievedChunk] = {}
    for h in vec_hits + kw_hits:
        by_id.setdefault(h.chunk_id, h)

    fused: list[RetrievedChunk] = []
    for cid in all_ids:
        v = vec_norm.get(cid, 0.0)
        k = kw_norm.get(cid, 0.0)
        combined = effective_alpha * v + (1.0 - effective_alpha) * k
        src = by_id[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=src.chunk_id,
                document_id=src.document_id,
                document_title=src.document_title,
                chunk_text=src.chunk_text,
                chunk_index=src.chunk_index,
                similarity_score=combined,
                component_scores={"vector": v, "keyword": k, "alpha": effective_alpha},
                metadata=src.metadata,
            )
        )

    fused.sort(key=lambda r: r.similarity_score, reverse=True)
    return fused[:top_k]


def _minmax_normalize(scores: dict[uuid.UUID, float]) -> dict[uuid.UUID, float]:
    """
    Rescale `scores` into [0, 1].

    If all scores are equal (degenerate but possible — e.g. one result),
    every item gets 1.0 so the upstream weighted sum still works.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# =============================================================================
# Dispatcher
# =============================================================================
def retrieve(
    db: Session,
    *,
    query_text: str,
    top_k: int,
    mode: RetrievalMode = "hybrid",
    alpha: float | None = None,
    document_type: str | None = None,
) -> list[RetrievedChunk]:
    """Mode-aware entry point used by the API layer."""
    if mode == "vector":
        return vector_search(db, query_text=query_text, top_k=top_k, document_type=document_type)
    if mode == "keyword":
        return keyword_search(db, query_text=query_text, top_k=top_k, document_type=document_type)
    if mode == "hybrid":
        return hybrid_search(
            db,
            query_text=query_text,
            top_k=top_k,
            alpha=alpha,
            document_type=document_type,
        )
    raise ValueError(f"unknown retrieval mode: {mode!r}")
