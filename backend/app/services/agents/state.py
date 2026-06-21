"""
Shared LangGraph state for every workflow.

LangGraph builds graphs around a typed state dict that flows through
nodes — each node returns the keys it produced, and LangGraph merges
them into the running state. We use `total=False` so every field is
optional; each workflow uses a subset.

Why one shared state vs. per-workflow state classes?
    The state surface is small and the workflows share most fields
    (chunks, citations, model_name). Forcing per-workflow types would
    multiply boilerplate without catching real bugs.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- inputs ---
    workflow_name: str
    input: dict[str, Any]
    top_k: int

    # --- per-node outputs (workflow-dependent) ---
    rewritten_query: str
    classification: str
    classification_rationale: str

    # Retrieval outputs. chunks = the main result;
    # chunks_a / chunks_b are used by policy_comparison.
    chunks: list[dict[str, Any]]
    chunks_a: list[dict[str, Any]]
    chunks_b: list[dict[str, Any]]

    # LLM-produced text/structured outputs.
    answer_text: str
    checklist_items: list[dict[str, Any]]
    validated_item_count: int
    comparison_summary: str
    differences: list[dict[str, Any]]
    next_steps: list[dict[str, Any]]

    grounding_score: float
    uncertainty_flag: bool

    # --- always-accumulating ---
    steps: list[dict[str, Any]]  # execution trace
    citations: list[dict[str, Any]]  # consolidated citations
    confidence: str
    model_name: str
