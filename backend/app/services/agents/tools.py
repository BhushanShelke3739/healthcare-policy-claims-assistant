"""
Tools used by the agent workflows.

Two flavors of tool:
    1. Pure-Python deterministic helpers (no LLM call):
         classify_claim_issue, run_grounding_check, rewrite_query (fallback)
       These exist so the system has predictable behavior even when the
       LLM is the deterministic mock provider, and so we don't burn tokens
       on tasks that don't need them.
    2. LLM-backed helpers that go through the existing ChatProvider:
         agent_generate (wraps generation.get_chat_provider().generate())
       The ChatProvider abstraction means these work the same against
       OpenAI, Ollama, vLLM, etc. — and the autouse pytest fixture forces
       the mock provider so the test suite is hermetic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.schemas.agents import ClaimDenialCategory
from app.services.generation import get_chat_provider
from app.services.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


# =============================================================================
# Query rewriting (deterministic fallback)
# =============================================================================
def rewrite_query(query: str) -> str:
    """
    Normalize / lightly expand the query before retrieval.

    The original spec lists "rewrite query" as the first step of the
    policy_lookup agent. A full implementation would use an LLM to expand
    domain abbreviations (e.g. PA → prior authorization) and add synonyms.
    Here we do the safer subset: whitespace cleanup + a tiny known-good
    abbreviation map. Predictable, fast, and never makes the query worse.
    """
    cleaned = re.sub(r"\s+", " ", query).strip()

    abbreviations = {
        r"\bPA\b": "prior authorization",
        r"\bOON\b": "out-of-network",
        r"\bEOB\b": "explanation of benefits",
        r"\bIRO\b": "independent review organization",
        r"\bCOB\b": "coordination of benefits",
    }
    for pattern, replacement in abbreviations.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


# =============================================================================
# Retrieval (thin wrapper)
# =============================================================================
def search_policy_documents(
    db: Session,
    *,
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
) -> list[RetrievedChunk]:
    """Wraps the Phase 3 retrieval service; hybrid by default."""
    return retrieve(
        db,
        query_text=query,
        top_k=top_k,
        mode="hybrid",
        document_type=document_type,
    )


# =============================================================================
# Claim denial classification (rule-based)
# =============================================================================
# Each category has a list of (case-insensitive) patterns. First match wins.
_DENIAL_PATTERNS: list[tuple[ClaimDenialCategory, list[str]]] = [
    (
        "authorization",
        [
            r"\bprior\s+authorization\b",
            r"\bpre[-\s]?auth(orization)?\b",
            r"\bauth\s+(?:was\s+)?(?:not|missing)\b",
            r"\bno\s+(?:pa|auth)\b",
            r"\bhf-022\b",
        ],
    ),
    (
        "medical_necessity",
        [
            r"\bmedical(?:ly)?\s+necessar(?:y|ity)\b",
            r"\bnot\s+medically\s+necessary\b",
            r"\bdoes\s+not\s+meet\s+criteria\b",
            r"\bhf-031\b",
        ],
    ),
    (
        "coordination_of_benefits",
        [
            r"\bcoordination\s+of\s+benefits\b",
            r"\bprimary\s+payer\b",
            r"\bcob\b",
            r"\bhf-045\b",
        ],
    ),
    (
        "coverage",
        [
            r"\bnot\s+(?:a\s+)?covered\b",
            r"\bservice\s+not\s+covered\b",
            r"\bexcluded\b",
            r"\bcosmetic\b",
            r"\bexperimental\b",
            r"\bbenefit\s+maximum\b",
            r"\bhf-014\b",
        ],
    ),
    (
        "administrative",
        [
            r"\binvalid\s+member\s+id\b",
            r"\bduplicate\s+claim\b",
            r"\beligibility\b",
            r"\bnot\s+credentialed\b",
            r"\bhf-001\b",
            r"\bhf-058\b",
            r"\bhf-072\b",
        ],
    ),
]


def classify_claim_issue(claim_summary: str) -> tuple[ClaimDenialCategory, str]:
    """
    Map a free-text claim summary to one of the denial categories.

    Returns (category, rationale). When nothing matches, returns
    ("unclassified", reason). Rule-based on purpose:
        * Deterministic, so tests don't flake with model changes.
        * Cheap — no token cost.
        * The categories are narrow enough that regex covers >90% of
          real-world phrasings.

    A future enhancement could fall back to an LLM call when no regex
    matches, but the rules cover all six synthetic-policy denial codes.
    """
    if not claim_summary or not claim_summary.strip():
        return "unclassified", "Empty claim summary."

    text = claim_summary.lower()
    for category, patterns in _DENIAL_PATTERNS:
        for pat in patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if match:
                rationale = f"Matched '{match.group(0)}' in the claim summary → {category}."
                return category, rationale

    return (
        "unclassified",
        "No known denial pattern matched. Manual review recommended.",
    )


# =============================================================================
# Grounding check (rule-based)
# =============================================================================
def run_grounding_check(answer_text: str, chunks: list[RetrievedChunk]) -> float:
    """
    Estimate how grounded the answer is in the retrieved chunks.

    Score in [0.0, 1.0] = fraction of "content tokens" in the answer that
    also appear in any retrieved chunk. Content tokens = lowercased words
    of length >= 4, excluding a small stop list.

    This is intentionally simple — a sentence-by-sentence LLM-judge
    would be more accurate but costs tokens and isn't deterministic. The
    rule-based check is good enough to flag the easy hallucinations
    (answers that share no vocabulary with their context).
    """
    if not answer_text.strip() or not chunks:
        return 0.0

    stop = {
        "this",
        "that",
        "with",
        "from",
        "your",
        "have",
        "will",
        "must",
        "they",
        "them",
        "their",
        "there",
        "which",
        "would",
        "could",
        "should",
        "been",
        "into",
        "than",
        "then",
        "what",
        "when",
    }
    answer_tokens = {t for t in re.findall(r"[a-z]{4,}", answer_text.lower()) if t not in stop}
    if not answer_tokens:
        return 1.0  # nothing meaningful claimed → trivially grounded

    chunk_text = " ".join(c.chunk_text for c in chunks).lower()
    chunk_tokens = set(re.findall(r"[a-z]{4,}", chunk_text))

    overlap = answer_tokens & chunk_tokens
    return round(len(overlap) / len(answer_tokens), 3)


# =============================================================================
# LLM-backed answer generation (wrapper)
# =============================================================================
def agent_generate(
    *,
    question: str,
    chunks: list[RetrievedChunk],
):
    """
    Call the configured chat provider with the agent's question + context.

    Reuses the Phase 4 generation service (system prompt, structured
    output, refusal rules) so agent answers share the same grounding
    guardrails as the main /rag/ask endpoint. Returns a GeneratedAnswer.
    """
    provider = get_chat_provider()
    return provider.generate(question=question, chunks=chunks)


# Helpers used by graph nodes for serializing RetrievedChunk → state dict.
def chunk_to_dict(c: RetrievedChunk) -> dict:
    """Convert a RetrievedChunk dataclass to a JSON-safe dict.

    LangGraph state needs to be cheap to copy / serialize, so we flatten
    the dataclass and stringify UUIDs.
    """
    d = asdict(c)
    d["chunk_id"] = str(c.chunk_id)
    d["document_id"] = str(c.document_id)
    return d


def chunks_from_state(state_chunks: list[dict]) -> list[RetrievedChunk]:
    """Inverse of `chunk_to_dict` — used by nodes that need real
    RetrievedChunk objects (e.g. to hand to agent_generate)."""
    import uuid as _uuid

    return [
        RetrievedChunk(
            chunk_id=_uuid.UUID(d["chunk_id"]),
            document_id=_uuid.UUID(d["document_id"]),
            document_title=d["document_title"],
            chunk_text=d["chunk_text"],
            chunk_index=d["chunk_index"],
            similarity_score=d["similarity_score"],
            component_scores=d.get("component_scores", {}),
            metadata=d.get("metadata", {}),
        )
        for d in state_chunks
    ]
