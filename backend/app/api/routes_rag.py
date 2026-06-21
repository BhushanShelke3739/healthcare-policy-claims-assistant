"""
RAG endpoints.

Phase 3: POST /rag/retrieve  — vector / keyword / hybrid retrieval.
Phase 4: POST /rag/ask       — grounded answer generation with citations.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import QueryLog
from app.db.session import get_db
from app.schemas.rag import (
    AskRequest,
    AskResponse,
    Citation,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.generation import get_chat_provider
from app.services.retrieval import retrieve

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# /rag/retrieve
# =============================================================================
@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve the top-k chunks for a query (vector / keyword / hybrid).",
)
def retrieve_chunks(
    payload: RetrieveRequest,
    db: Session = Depends(get_db),
) -> RetrieveResponse:
    start = time.perf_counter()

    hits = retrieve(
        db,
        query_text=payload.query,
        top_k=payload.top_k,
        mode=payload.mode,
        alpha=payload.alpha,
        document_type=payload.document_type,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "rag_retrieve",
        extra={
            "query_length": len(payload.query),
            "top_k": payload.top_k,
            "mode": payload.mode,
            "result_count": len(hits),
            "latency_ms": latency_ms,
        },
    )

    return RetrieveResponse(
        query=payload.query,
        mode=payload.mode,
        total=len(hits),
        results=[
            RetrievedChunk(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                document_title=h.document_title,
                chunk_text=h.chunk_text,
                chunk_index=h.chunk_index,
                similarity_score=h.similarity_score,
                component_scores=h.component_scores,
                metadata=h.metadata,
            )
            for h in hits
        ],
    )


# =============================================================================
# /rag/ask
# =============================================================================
@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question and get a grounded answer with citations.",
)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    The full RAG flow:
        1. Retrieve top-K chunks (using hybrid by default).
        2. Short-circuit to refusal if retrieval is empty or below the
           configured score floor — no need to spend an LLM call.
        3. Call the chat provider with the system prompt + retrieved
           context.
        4. Validate citations: drop any chunk_id the LLM made up; record
           the drop in `grounding_notes`.
        5. Persist a QueryLog row for replay / audit.
    """
    settings = get_settings()
    start = time.perf_counter()

    hits = retrieve(
        db,
        query_text=payload.question,
        top_k=payload.top_k,
        mode=payload.mode,
        document_type=payload.document_type,
    )

    # Score-floor short-circuit. Spending LLM tokens on a low-confidence
    # retrieval doesn't get us a better answer.
    above_floor = [h for h in hits if h.similarity_score >= settings.refusal_score_floor]
    chat = get_chat_provider()

    if not above_floor:
        generated = _build_refusal(
            phrase=settings.refusal_phrase,
            reason=(
                "Retrieval returned no chunks."
                if not hits
                else f"All retrieved chunks scored below the refusal floor "
                f"({settings.refusal_score_floor:.2f})."
            ),
        )
        model_name = chat.model_name
    else:
        generated = chat.generate(question=payload.question, chunks=above_floor)
        model_name = chat.model_name

    # Citation validation. The LLM may emit a chunk_id that doesn't match
    # any retrieved chunk — that's a hallucination. We drop it and add a
    # note so the operator can see something was filtered.
    retrieved_ids_by_str = {str(h.chunk_id): h for h in hits}
    valid_citations: list[Citation] = []
    dropped = 0
    for c in generated.citations:
        hit = retrieved_ids_by_str.get(c.chunk_id)
        if hit is None:
            dropped += 1
            continue
        valid_citations.append(
            Citation(
                document_title=hit.document_title,
                chunk_id=hit.chunk_id,
                excerpt=c.excerpt,
            )
        )

    notes = generated.grounding_notes
    if dropped:
        suffix = (
            f" [server: dropped {dropped} citation(s) referencing chunk_ids "
            "not present in the retrieved set]"
        )
        notes = (notes or "") + suffix
        logger.warning(
            "rag_dropped_citations",
            extra={"dropped": dropped, "question_length": len(payload.question)},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Phase 9: record the retrieval scores + outcome alongside the audit row so
    # a low-quality answer can be diagnosed after the fact without re-running.
    details = {
        "mode": payload.mode,
        "top_k": payload.top_k,
        "retrieved": len(hits),
        "above_floor": len(above_floor),
        "refused": not above_floor,
        "confidence": generated.confidence,
        "kept_citations": len(valid_citations),
        "dropped_citations": dropped,
        "retrieval_scores": [
            {
                "chunk_id": str(h.chunk_id),
                "document_title": h.document_title,
                "similarity_score": round(h.similarity_score, 4),
                "component_scores": h.component_scores,
            }
            for h in hits
        ],
    }

    # Persist for audit / replay. The same Session retrieved + generated,
    # so one commit at the end is the whole interaction.
    db.add(
        QueryLog(
            user_query=payload.question,
            answer=generated.answer,
            retrieved_chunk_ids=[str(h.chunk_id) for h in hits],
            latency_ms=latency_ms,
            model_name=model_name,
            details=details,
        )
    )
    db.commit()

    logger.info(
        "rag_ask",
        extra={
            "question_length": len(payload.question),
            "top_k": payload.top_k,
            "mode": payload.mode,
            "retrieved": len(hits),
            "kept_citations": len(valid_citations),
            "dropped_citations": dropped,
            "confidence": generated.confidence,
            "model_name": model_name,
            "latency_ms": latency_ms,
        },
    )

    return AskResponse(
        question=payload.question,
        answer=generated.answer,
        citations=valid_citations if payload.include_citations else [],
        confidence=generated.confidence,
        grounding_notes=notes,
        retrieved_chunk_ids=[h.chunk_id for h in hits],
        model_name=model_name,
        latency_ms=latency_ms,
    )


# =============================================================================
# Helpers
# =============================================================================
def _build_refusal(*, phrase: str, reason: str):
    """Return a GeneratedAnswer-shaped refusal so the API path is uniform."""
    from app.schemas.rag import GeneratedAnswer

    return GeneratedAnswer(
        answer=phrase,
        citations=[],
        confidence="low",
        grounding_notes=f"Refused: {reason}",
    )
