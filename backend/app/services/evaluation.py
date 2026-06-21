"""
Evaluation framework.

Runs every item in an eval dataset through the same Phase 3 retrieval +
Phase 4 generation pipeline used by /rag/ask, then computes a battery
of metrics per question and aggregates them into a run-level summary.

Persistence:
    EvaluationRun        — one row per `/eval/run` call.
    EvaluationResult     — one row per question, with typed metric
                           columns + a JSONB `details` bag for the
                           rest (retrieved_chunk_ids, expected_document
                           echo, keyword-recall breakdown).

Why not call the /rag/ask HTTP endpoint internally?
    Doing the work in-process keeps the eval in one DB transaction per
    run and avoids serialization overhead for what could be hundreds of
    questions. Same code paths, just one layer further down the stack.

Metrics implemented
-------------------
    retrieval_hit            (bool)   expected_document appeared in retrieved chunks?
    context_precision        (float)  retrieved chunks judged relevant / retrieved
    context_recall           (float)  expected keywords found in retrieved text / expected
    answer_relevancy         (float)  question/answer content-word overlap
    faithfulness             (float)  answer/retrieved-context content-word overlap
                                      (reuses agents.tools.run_grounding_check)
    hallucination_flag       (bool)   faithfulness below floor AND answer wasn't refusal
    refusal_correct          (bool)   for expected_refusal=true items, did we refuse?
    latency_ms               (int)    wall clock for retrieval+generation

Refusal questions get a different scoring path — only `refusal_correct`
and `latency_ms` are populated; the float metrics stay null so they
don't pollute the averages.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import EvaluationResult, EvaluationRun
from app.schemas.eval import EvalDataset, EvalDatasetItem
from app.schemas.rag import RetrievalMode
from app.services.agents.tools import run_grounding_check
from app.services.generation import get_chat_provider
from app.services.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


# =============================================================================
# Constants / tuning
# =============================================================================
# Below this faithfulness score AND given a non-empty non-refusal
# answer, flag the result as hallucinated.
_HALLUCINATION_FAITHFULNESS_FLOOR = 0.30

# Stop list shared with run_grounding_check — content-only tokens.
_STOP = {
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
    "does",
    "about",
    "where",
    "while",
    "after",
    "before",
}

# Default location of the bundled eval set. Resolved relative to this
# file so the path is correct regardless of the caller's CWD.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent
DEFAULT_DATASET_PATH = _REPO_ROOT / "sample_data" / "eval_questions" / "healthcare_policy_eval.json"


# =============================================================================
# Dataset loading
# =============================================================================
def load_dataset(path: str | Path | None = None) -> EvalDataset:
    """Load and validate an eval dataset JSON file."""
    if path is None:
        resolved = DEFAULT_DATASET_PATH
    else:
        p = Path(path)
        resolved = p if p.is_absolute() else (_REPO_ROOT / p)
    if not resolved.exists():
        raise FileNotFoundError(f"eval dataset not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(payload)


# =============================================================================
# Per-question scoring
# =============================================================================
@dataclass
class _ScoredItem:
    """Internal: everything needed to write one EvaluationResult row."""

    question: str
    expected_answer: str | None
    generated_answer: str
    is_refusal_expected: bool

    retrieval_hit: bool | None
    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    hallucination_flag: bool
    refusal_correct: bool | None

    retrieved_chunks: list[RetrievedChunk]
    expected_document: str | None
    expected_keywords: list[str]
    keyword_hits: list[str]
    keyword_misses: list[str]
    latency_ms: int
    notes: str


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in _STOP}


def _is_refusal_text(answer: str, refusal_phrase: str) -> bool:
    """Did the system return the configured refusal phrase?"""
    return refusal_phrase.lower().strip() in answer.lower().strip()


def _chunk_is_relevant(
    chunk: RetrievedChunk,
    *,
    expected_document: str | None,
    expected_keywords: list[str],
) -> bool:
    """
    A chunk is judged relevant if (any of):
        - it's from the expected document, or
        - it contains at least one of the expected keywords (case-insensitive).
    Used for context_precision.
    """
    if expected_document and chunk.document_title.lower() == expected_document.lower():
        return True
    text_lc = chunk.chunk_text.lower()
    return any(kw.lower() in text_lc for kw in expected_keywords)


def _keyword_recall(
    chunks: list[RetrievedChunk],
    expected_keywords: list[str],
) -> tuple[float | None, list[str], list[str]]:
    """
    What fraction of expected_keywords appear (case-insensitive) anywhere
    in the retrieved chunk text?

    Returns (recall_or_None, hits, misses). Recall is None when there
    are no expected keywords — averaging None-valued recall would be
    misleading.
    """
    if not expected_keywords:
        return None, [], []
    all_text = " ".join(c.chunk_text for c in chunks).lower()
    hits: list[str] = []
    misses: list[str] = []
    for kw in expected_keywords:
        if kw.lower() in all_text:
            hits.append(kw)
        else:
            misses.append(kw)
    return round(len(hits) / len(expected_keywords), 3), hits, misses


def _answer_relevancy(question: str, answer: str) -> float | None:
    """
    Fraction of question content-words also present in the answer.

    None when the question has no content tokens (defensive — never
    happens with real questions but guards against empty input).
    """
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return None
    a_tokens = _content_tokens(answer)
    if not a_tokens:
        return 0.0
    overlap = q_tokens & a_tokens
    return round(len(overlap) / len(q_tokens), 3)


def _score_one(
    *,
    db: Session,
    item: EvalDatasetItem,
    top_k: int,
    mode: RetrievalMode,
    alpha: float | None,
    refusal_phrase: str,
) -> _ScoredItem:
    """Run one eval item through retrieve+generate and compute metrics."""
    start = time.perf_counter()

    chunks = retrieve(
        db,
        query_text=item.question,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
    )

    chat = get_chat_provider()
    generated = chat.generate(question=item.question, chunks=chunks)
    answer_text = generated.answer

    latency_ms = int((time.perf_counter() - start) * 1000)

    # --------- branch on refusal-expected vs in-scope ----------
    is_refusal_expected = bool(item.expected_refusal)

    if is_refusal_expected:
        refused = _is_refusal_text(answer_text, refusal_phrase)
        return _ScoredItem(
            question=item.question,
            expected_answer=item.expected_answer,
            generated_answer=answer_text,
            is_refusal_expected=True,
            retrieval_hit=None,
            context_precision=None,
            context_recall=None,
            faithfulness=None,
            answer_relevancy=None,
            hallucination_flag=not refused,  # answering an out-of-scope is a hallucination
            refusal_correct=refused,
            retrieved_chunks=list(chunks),
            expected_document=None,
            expected_keywords=[],
            keyword_hits=[],
            keyword_misses=[],
            latency_ms=latency_ms,
            notes=(
                "refusal-expected; refused correctly."
                if refused
                else "refusal-expected; answered instead of refusing."
            ),
        )

    # --------- in-scope metrics ----------
    retrieval_hit = (
        bool(
            item.expected_document
            and any(c.document_title.lower() == item.expected_document.lower() for c in chunks)
        )
        if item.expected_document
        else None
    )

    # context_precision: fraction of retrieved chunks judged relevant.
    if chunks:
        relevant_count = sum(
            1
            for c in chunks
            if _chunk_is_relevant(
                c,
                expected_document=item.expected_document,
                expected_keywords=item.expected_keywords,
            )
        )
        context_precision = round(relevant_count / len(chunks), 3)
    else:
        context_precision = 0.0

    context_recall, hits, misses = _keyword_recall(chunks, item.expected_keywords)
    answer_relevancy = _answer_relevancy(item.question, answer_text)
    faithfulness = run_grounding_check(answer_text, chunks)

    answered_with_refusal = _is_refusal_text(answer_text, refusal_phrase)
    hallucination_flag = (
        not answered_with_refusal
        and answer_text.strip() != ""
        and faithfulness < _HALLUCINATION_FAITHFULNESS_FLOOR
    )

    notes_parts: list[str] = []
    if retrieval_hit is False:
        notes_parts.append(f"expected document {item.expected_document!r} not in retrieved set")
    if hallucination_flag:
        notes_parts.append(
            f"low faithfulness ({faithfulness:.2f} < {_HALLUCINATION_FAITHFULNESS_FLOOR})"
        )
    if answered_with_refusal:
        notes_parts.append("system returned refusal phrase")

    return _ScoredItem(
        question=item.question,
        expected_answer=item.expected_answer,
        generated_answer=answer_text,
        is_refusal_expected=False,
        retrieval_hit=retrieval_hit,
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        hallucination_flag=hallucination_flag,
        refusal_correct=None,
        retrieved_chunks=list(chunks),
        expected_document=item.expected_document,
        expected_keywords=list(item.expected_keywords),
        keyword_hits=hits,
        keyword_misses=misses,
        latency_ms=latency_ms,
        notes="; ".join(notes_parts) or "ok",
    )


# =============================================================================
# Run orchestration
# =============================================================================
@dataclass
class _AggregateMetrics:
    """Aggregate counters across all scored items."""

    num_questions: int
    in_scope_count: int
    refusal_count: int
    retrieval_hits: int
    context_precision_sum: float
    context_precision_n: int
    context_recall_sum: float
    context_recall_n: int
    faithfulness_sum: float
    faithfulness_n: int
    answer_relevancy_sum: float
    answer_relevancy_n: int
    hallucinations: int
    refusals_correct: int
    latency_sum: int


def _aggregate(scored: list[_ScoredItem]) -> _AggregateMetrics:
    """One pass to produce the run-level summary numbers."""
    agg = _AggregateMetrics(
        num_questions=len(scored),
        in_scope_count=0,
        refusal_count=0,
        retrieval_hits=0,
        context_precision_sum=0.0,
        context_precision_n=0,
        context_recall_sum=0.0,
        context_recall_n=0,
        faithfulness_sum=0.0,
        faithfulness_n=0,
        answer_relevancy_sum=0.0,
        answer_relevancy_n=0,
        hallucinations=0,
        refusals_correct=0,
        latency_sum=0,
    )
    for s in scored:
        agg.latency_sum += s.latency_ms
        if s.hallucination_flag:
            agg.hallucinations += 1
        if s.is_refusal_expected:
            agg.refusal_count += 1
            if s.refusal_correct:
                agg.refusals_correct += 1
            continue
        agg.in_scope_count += 1
        if s.retrieval_hit:
            agg.retrieval_hits += 1
        if s.context_precision is not None:
            agg.context_precision_sum += s.context_precision
            agg.context_precision_n += 1
        if s.context_recall is not None:
            agg.context_recall_sum += s.context_recall
            agg.context_recall_n += 1
        if s.faithfulness is not None:
            agg.faithfulness_sum += s.faithfulness
            agg.faithfulness_n += 1
        if s.answer_relevancy is not None:
            agg.answer_relevancy_sum += s.answer_relevancy
            agg.answer_relevancy_n += 1
    return agg


def _safe_div(num: float, den: int) -> float | None:
    """Avoid dividing by zero — return None when there were no samples."""
    return round(num / den, 3) if den else None


def summarize(scored: list[_ScoredItem]) -> dict:
    """Return the EvalSummary as a plain dict."""
    if not scored:
        return {"num_questions": 0}
    agg = _aggregate(scored)
    return {
        "num_questions": agg.num_questions,
        "retrieval_hit_rate": _safe_div(float(agg.retrieval_hits), agg.in_scope_count),
        "avg_context_precision": _safe_div(agg.context_precision_sum, agg.context_precision_n),
        "avg_context_recall": _safe_div(agg.context_recall_sum, agg.context_recall_n),
        "avg_faithfulness": _safe_div(agg.faithfulness_sum, agg.faithfulness_n),
        "avg_answer_relevancy": _safe_div(agg.answer_relevancy_sum, agg.answer_relevancy_n),
        "hallucination_rate": _safe_div(float(agg.hallucinations), agg.num_questions),
        "refusal_accuracy": _safe_div(float(agg.refusals_correct), agg.refusal_count),
        "avg_latency_ms": _safe_div(float(agg.latency_sum), agg.num_questions),
        "token_cost": None,  # Phase 6 placeholder — see docs/06.
    }


def run_evaluation(
    db: Session,
    *,
    name: str,
    description: str | None = None,
    dataset: EvalDataset | None = None,
    dataset_path: str | None = None,
    top_k: int = 5,
    mode: RetrievalMode = "hybrid",
    alpha: float | None = None,
) -> EvaluationRun:
    """
    Execute one evaluation run end-to-end and persist results.

    The function loads (or accepts) a dataset, scores every item in
    order, persists a parent `EvaluationRun` row plus one
    `EvaluationResult` per question, and returns the parent (with
    `results` available via the ORM relationship).

    Caller is responsible for committing the session. We flush so IDs
    are populated and the in-memory ORM tree is consistent.
    """
    settings = get_settings()
    refusal_phrase = settings.refusal_phrase

    if dataset is None:
        dataset = load_dataset(dataset_path)

    if not dataset.items:
        raise ValueError("eval dataset is empty")

    run = EvaluationRun(name=name, description=description)
    db.add(run)
    db.flush()  # populate run.id

    scored: list[_ScoredItem] = []
    for item in dataset.items:
        s = _score_one(
            db=db,
            item=item,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
            refusal_phrase=refusal_phrase,
        )
        scored.append(s)

        row = EvaluationResult(
            evaluation_run_id=run.id,
            question=s.question,
            expected_answer=s.expected_answer,
            generated_answer=s.generated_answer,
            context_precision=s.context_precision,
            context_recall=s.context_recall,
            faithfulness=s.faithfulness,
            answer_relevancy=s.answer_relevancy,
            hallucination_flag=s.hallucination_flag,
            notes=s.notes,
            latency_ms=s.latency_ms,
            details={
                "retrieval_hit": s.retrieval_hit,
                "refusal_expected": s.is_refusal_expected,
                "refusal_correct": s.refusal_correct,
                "expected_document": s.expected_document,
                "expected_keywords": s.expected_keywords,
                "keyword_hits": s.keyword_hits,
                "keyword_misses": s.keyword_misses,
                "retrieved_chunk_ids": [str(c.chunk_id) for c in s.retrieved_chunks],
                "top_chunk_titles": [c.document_title for c in s.retrieved_chunks[:5]],
                # Echo the run-level knobs so a future reader of the row
                # knows what configuration produced it.
                "run_params": {
                    "mode": mode,
                    "top_k": top_k,
                    "alpha": alpha,
                },
            },
        )
        db.add(row)

    db.flush()
    logger.info(
        "evaluation_run_completed",
        extra={
            "run_id": str(run.id),
            "num_questions": len(scored),
            "summary": summarize(scored),
        },
    )
    return run


def summarize_run(run: EvaluationRun) -> dict:
    """
    Reconstruct an `EvalSummary`-shaped dict from a loaded `EvaluationRun`.

    Used by GET /eval/runs/{id}: we don't store the summary on the run
    row, so we recompute it from the persisted per-result data when the
    client asks for it. Cheap (small N) and immune to schema drift.
    """
    items: Iterable[EvaluationResult] = run.results
    scored = [
        _ScoredItem(
            question=r.question,
            expected_answer=r.expected_answer,
            generated_answer=r.generated_answer or "",
            is_refusal_expected=bool(r.details.get("refusal_expected", False)),
            retrieval_hit=r.details.get("retrieval_hit"),
            context_precision=r.context_precision,
            context_recall=r.context_recall,
            faithfulness=r.faithfulness,
            answer_relevancy=r.answer_relevancy,
            hallucination_flag=bool(r.hallucination_flag),
            refusal_correct=r.details.get("refusal_correct"),
            retrieved_chunks=[],  # not needed for aggregation
            expected_document=r.details.get("expected_document"),
            expected_keywords=list(r.details.get("expected_keywords", []) or []),
            keyword_hits=list(r.details.get("keyword_hits", []) or []),
            keyword_misses=list(r.details.get("keyword_misses", []) or []),
            latency_ms=int(r.latency_ms or 0),
            notes=r.notes or "",
        )
        for r in items
    ]
    return summarize(scored)
