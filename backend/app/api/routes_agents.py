"""
Agent endpoints — Phase 5.

`POST /agents/run` dispatches to one of four LangGraph workflows:
    policy_lookup, claim_triage, policy_comparison, compliance_checklist

The body shape is uniform — `{workflow, input, top_k?}` — but the
`input` dict and the response's `final_output` are workflow-specific
(see `app/schemas/agents.py` for the per-workflow shapes).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.agents import AgentRunRequest, AgentRunResponse, AgentStep
from app.schemas.rag import Citation
from app.services.agents import AGENT_WORKFLOW_NAMES, run_workflow
from app.services.agents.runner import UnknownWorkflowError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Run a named agent workflow over the policy corpus.",
)
def run_agent(
    payload: AgentRunRequest,
    db: Session = Depends(get_db),
) -> AgentRunResponse:
    try:
        result = run_workflow(
            name=payload.workflow,
            input_payload=payload.input,
            db=db,
            top_k=payload.top_k,
        )
    except UnknownWorkflowError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{exc}. Valid: {', '.join(AGENT_WORKFLOW_NAMES)}",
        ) from exc

    # Convert internal dict-shaped steps/citations into Pydantic models
    # for the response.
    steps = [
        AgentStep(
            name=str(s.get("name", "")),
            summary=str(s.get("summary", "")),
            **{k: v for k, v in s.items() if k not in {"name", "summary"}},
        )
        for s in result.steps
    ]
    citations = [
        Citation(
            document_title=c.get("document_title", ""),
            chunk_id=c["chunk_id"],
            excerpt=c.get("excerpt", ""),
        )
        for c in result.citations
        if c.get("chunk_id")
    ]

    confidence = result.confidence if result.confidence in {"low", "medium", "high"} else "low"

    return AgentRunResponse(
        workflow=payload.workflow,
        final_output=result.final_output,
        steps=steps,
        citations=citations,
        confidence=confidence,  # type: ignore[arg-type]
        model_name=result.model_name,
        latency_ms=result.latency_ms,
    )
