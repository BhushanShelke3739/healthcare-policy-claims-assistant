"""
Public entry point for the agent layer.

`run_workflow(name, input, db, top_k)` builds the appropriate LangGraph,
runs it to completion, and packages the final state into a workflow-
specific `final_output` plus shared metadata (steps, citations,
confidence, model_name, latency).

Why this lives in its own module:
    Keeps the workflow definitions in `workflows.py` focused on graph
    construction. Keeps the API route in `routes_agents.py` thin —
    it just delegates here and serializes the result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.agents.state import AgentState
from app.services.agents.workflows import WORKFLOW_BUILDERS
from app.services.generation import get_chat_provider

logger = logging.getLogger(__name__)

AGENT_WORKFLOW_NAMES = tuple(WORKFLOW_BUILDERS.keys())


class UnknownWorkflowError(ValueError):
    """Raised when a workflow name is not in the registry."""


@dataclass(frozen=True)
class AgentRunResult:
    """Internal result returned by `run_workflow` before HTTP serialization."""

    workflow: str
    final_output: dict[str, Any]
    steps: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    confidence: str
    model_name: str
    latency_ms: int


def run_workflow(
    *,
    name: str,
    input_payload: dict[str, Any],
    db: Session,
    top_k: int | None = None,
) -> AgentRunResult:
    """Execute a registered workflow and return its packaged result."""
    if name not in WORKFLOW_BUILDERS:
        raise UnknownWorkflowError(f"Unknown workflow {name!r}. Valid: {AGENT_WORKFLOW_NAMES}")

    settings = get_settings()
    effective_top_k = top_k if top_k is not None else settings.agent_default_top_k

    builder = WORKFLOW_BUILDERS[name]
    graph = builder(db, default_top_k=effective_top_k)

    initial: AgentState = {
        "workflow_name": name,
        "input": dict(input_payload),
        "top_k": effective_top_k,
        "steps": [],
        "chunks": [],
        "citations": [],
        "confidence": "low",
    }

    start = time.perf_counter()
    # Cap the number of node executions LangGraph will run — guards
    # against a buggy workflow looping forever. Comfortably above the
    # longest legitimate workflow (claim_triage with one reflection
    # loop ≈ 5 steps).
    config = {"recursion_limit": settings.agent_max_steps * 2}
    final_state: AgentState = graph.invoke(initial, config=config)
    latency_ms = int((time.perf_counter() - start) * 1000)

    model_name = get_chat_provider().model_name
    final_output = _build_final_output(name, final_state)
    result = AgentRunResult(
        workflow=name,
        final_output=final_output,
        steps=list(final_state.get("steps", [])),
        citations=list(final_state.get("citations", [])),
        confidence=str(final_state.get("confidence", "low")),
        model_name=model_name,
        latency_ms=latency_ms,
    )

    logger.info(
        "agent_run",
        extra={
            "workflow": name,
            "latency_ms": latency_ms,
            "steps": len(result.steps),
            "citations": len(result.citations),
            "confidence": result.confidence,
            "model_name": model_name,
        },
    )
    return result


def _build_final_output(workflow: str, state: AgentState) -> dict[str, Any]:
    """Project the right slice of state into the workflow-specific output."""
    if workflow == "policy_lookup":
        return {
            "answer": state.get("answer_text", ""),
            "rewritten_query": state.get("rewritten_query", ""),
        }
    if workflow == "claim_triage":
        return {
            "classification": state.get("classification", "unclassified"),
            "classification_rationale": state.get("classification_rationale", ""),
            "next_steps": state.get("next_steps", []),
            "uncertainty_flag": bool(state.get("uncertainty_flag", False)),
            "grounding_score": float(state.get("grounding_score", 0.0)),
            "answer": state.get("answer_text", ""),
        }
    if workflow == "policy_comparison":
        inp = state.get("input", {})
        return {
            "document_a_title": inp.get("document_a_title", ""),
            "document_b_title": inp.get("document_b_title", ""),
            "summary": state.get("comparison_summary", state.get("answer_text", "")),
            "differences": state.get("differences", []),
        }
    if workflow == "compliance_checklist":
        return {
            "topic": state.get("input", {}).get("topic", ""),
            "items": state.get("checklist_items", []),
            "validated_item_count": int(state.get("validated_item_count", 0)),
            "answer": state.get("answer_text", ""),
        }
    # Shouldn't reach here — registry is exhaustive.
    return {}
