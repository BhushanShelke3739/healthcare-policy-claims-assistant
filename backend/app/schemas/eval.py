"""
Pydantic schemas for /eval/*.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.rag import RetrievalMode


# =============================================================================
# Request
# =============================================================================
class EvalRunRequest(BaseModel):
    name: str = Field(
        default="ad-hoc",
        max_length=256,
        description="Human-readable label for this run (shows up in /eval/runs).",
    )
    description: str | None = Field(default=None, max_length=2000)
    # Optional override of the eval dataset. Defaults to
    # sample_data/eval_questions/healthcare_policy_eval.json.
    dataset_path: str | None = Field(
        default=None,
        description="Path (absolute or relative to repo root). When omitted, the bundled healthcare_policy_eval.json is used.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode = Field(default="hybrid")
    # Per-run override of `HYBRID_ALPHA` for A/B comparisons without
    # restarting uvicorn. None = use the value from settings.
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Hybrid weight on vector score (1-alpha keyword). Only meaningful when mode='hybrid'. None = use settings.hybrid_alpha.",
    )


# =============================================================================
# Dataset item (what each row of the JSON file looks like)
# =============================================================================
class EvalDatasetItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    question: str
    expected_answer: str | None = None
    expected_document: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    # When true, scoring treats "the refusal phrase" as the correct
    # answer. Used for out-of-scope questions (e.g. "what is the
    # capital of France?").
    expected_refusal: bool = False


class EvalDataset(BaseModel):
    name: str | None = None
    description: str | None = None
    items: list[EvalDatasetItem]


# =============================================================================
# Per-result + run summary
# =============================================================================
class EvalResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    question: str
    expected_answer: str | None
    generated_answer: str | None

    context_precision: float | None
    context_recall: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    hallucination_flag: bool

    latency_ms: int | None
    notes: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvalSummary(BaseModel):
    num_questions: int
    retrieval_hit_rate: float | None = Field(
        default=None,
        description="Fraction of in-scope questions where expected_document appeared in the retrieved chunks.",
    )
    avg_context_precision: float | None = None
    avg_context_recall: float | None = None
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    hallucination_rate: float | None = None
    refusal_accuracy: float | None = Field(
        default=None,
        description="For questions marked expected_refusal=true: fraction the system actually refused.",
    )
    avg_latency_ms: float | None = None
    token_cost: float | None = Field(
        default=None,
        description="Placeholder. Always None today — Ollama doesn't expose usage stats over the OpenAI compat API.",
    )


class EvalRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    summary: EvalSummary | None = None
    results: list[EvalResultRead] = Field(default_factory=list)


class EvalRunSummaryRow(BaseModel):
    """Light-weight row returned by GET /eval/runs (no per-question results)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    num_questions: int


class EvalRunList(BaseModel):
    items: list[EvalRunSummaryRow]
    total: int
