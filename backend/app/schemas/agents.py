"""
Pydantic schemas for /agents/run.

Each workflow has its own input and output shape, but everything goes
through one endpoint. The request lists the workflow name and an
`input` payload whose shape depends on which workflow; the response
includes a workflow-specific `final_output` plus shared meta-fields
(steps, citations, confidence, latency, model_name).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.rag import Citation, Confidence

WorkflowName = Literal[
    "policy_lookup",
    "claim_triage",
    "policy_comparison",
    "compliance_checklist",
]

# Denial categories used by the claim_triage agent. Kept narrow on
# purpose — small list, easy to reason about. The classification tool
# is rule-based, so adding categories means adding a regex too.
ClaimDenialCategory = Literal[
    "administrative",
    "coverage",
    "medical_necessity",
    "authorization",
    "coordination_of_benefits",
    "unclassified",
]


# =============================================================================
# Request / response envelope
# =============================================================================
class AgentRunRequest(BaseModel):
    workflow: WorkflowName
    input: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Workflow-specific input. See examples per workflow:\n"
            ' - policy_lookup: {"query": "..."}\n'
            ' - claim_triage: {"claim_summary": "...", "question": "..."}\n'
            ' - policy_comparison: {"document_a_title": "...", "document_b_title": "...", "focus": "..."}\n'
            ' - compliance_checklist: {"topic": "..."}'
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20)


class AgentStep(BaseModel):
    """One node in the workflow's execution trace."""

    model_config = ConfigDict(extra="allow")
    name: str
    summary: str = Field(description="Short human-readable summary of what this step produced.")


class AgentRunResponse(BaseModel):
    workflow: WorkflowName
    final_output: dict[str, Any] = Field(
        description="Workflow-specific result. Shape depends on `workflow`."
    )
    steps: list[AgentStep] = Field(
        default_factory=list,
        description="Execution trace, in order. Each entry is one node's output summary.",
    )
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = "low"
    model_name: str
    latency_ms: int


# =============================================================================
# Per-workflow output shapes
# (Internal — surfaced via AgentRunResponse.final_output)
# =============================================================================
class PolicyLookupOutput(BaseModel):
    answer: str
    rewritten_query: str


class TriageNextStep(BaseModel):
    order: int
    action: str
    rationale: str = ""


class ClaimTriageOutput(BaseModel):
    classification: ClaimDenialCategory
    classification_rationale: str
    next_steps: list[TriageNextStep]
    uncertainty_flag: bool = False


class PolicyDifference(BaseModel):
    dimension: str = Field(
        description='What the difference is about — e.g. "timeline", "documentation", "scope".'
    )
    document_a: str
    document_b: str


class PolicyComparisonOutput(BaseModel):
    summary: str
    differences: list[PolicyDifference]
    document_a_title: str
    document_b_title: str


class ChecklistItem(BaseModel):
    order: int
    text: str
    supporting_chunk_ids: list[uuid.UUID] = Field(default_factory=list)


class ComplianceChecklistOutput(BaseModel):
    topic: str
    items: list[ChecklistItem]
    validated_item_count: int = Field(
        ge=0,
        description="How many items survived the grounding-validation step.",
    )
