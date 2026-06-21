"""
Tests for Phase 5 — agent workflows.

Two tiers:
    * Pure-unit tests for deterministic tools (classify, rewrite,
      grounding check, ordered-step extraction).
    * Integration tests that run a workflow end-to-end through LangGraph
      against the real Postgres (via `db_session`) using the autouse
      mock LLM — no Ollama / OpenAI calls.
"""

from __future__ import annotations

import pytest

from app.services.agents import AGENT_WORKFLOW_NAMES, run_workflow
from app.services.agents.runner import UnknownWorkflowError
from app.services.agents.tools import (
    classify_claim_issue,
    rewrite_query,
    run_grounding_check,
)
from app.services.agents.workflows import _extract_ordered_steps
from app.services.retrieval import RetrievedChunk
from tests.conftest import TEST_CORPUS_TAG


# ---------------------------------------------------------------------------
# rewrite_query — deterministic
# ---------------------------------------------------------------------------
def test_rewrite_expands_known_abbreviations() -> None:
    assert "prior authorization" in rewrite_query("Find policy for PA on MRI").lower()
    assert "out-of-network" in rewrite_query("OON claim").lower()


def test_rewrite_normalizes_whitespace() -> None:
    assert rewrite_query("  appeal  process\n\n window ") == "appeal process window"


def test_rewrite_leaves_normal_text_alone() -> None:
    assert rewrite_query("how long do I have to appeal") == "how long do I have to appeal"


# ---------------------------------------------------------------------------
# classify_claim_issue — deterministic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "summary,expected",
    [
        ("Claim denied because prior authorization was missing.", "authorization"),
        ("Denial code HF-022 returned by adjudication.", "authorization"),
        ("The reviewer said it was not medically necessary.", "medical_necessity"),
        ("Member has primary payer; need their EOB.", "coordination_of_benefits"),
        ("Service excluded as cosmetic per benefit summary.", "coverage"),
        ("Duplicate claim with HF-058 code.", "administrative"),
        ("Nothing meaningful here.", "unclassified"),
        ("", "unclassified"),
    ],
)
def test_classify_claim_issue_routes_correctly(summary: str, expected: str) -> None:
    category, rationale = classify_claim_issue(summary)
    assert category == expected
    assert isinstance(rationale, str)


# ---------------------------------------------------------------------------
# run_grounding_check — deterministic
# ---------------------------------------------------------------------------
def _make_chunk(text: str) -> RetrievedChunk:
    import uuid

    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="X",
        chunk_text=text,
        chunk_index=0,
        similarity_score=1.0,
        component_scores={},
        metadata={},
    )


def test_grounding_check_high_when_answer_overlaps_chunks() -> None:
    chunks = [
        _make_chunk(
            "First-level appeals must be filed within sixty calendar days of the denial notice."
        )
    ]
    score = run_grounding_check(
        "Appeals must be filed within sixty calendar days of the denial notice.",
        chunks,
    )
    assert score >= 0.7


def test_grounding_check_low_when_answer_invents_words() -> None:
    chunks = [_make_chunk("Appeals must be filed within sixty days.")]
    score = run_grounding_check(
        "Submit ferromagnetic samples to the unicorn registry.",
        chunks,
    )
    assert score <= 0.1


def test_grounding_check_zero_when_no_chunks() -> None:
    assert run_grounding_check("anything", []) == 0.0


# ---------------------------------------------------------------------------
# _extract_ordered_steps — robust to a few formats
# ---------------------------------------------------------------------------
def test_extract_ordered_steps_numbered() -> None:
    text = "1. Call provider\n2. Request medical records\n3. Resubmit"
    steps = _extract_ordered_steps(text)
    assert [s["order"] for s in steps] == [1, 2, 3]
    assert steps[0]["action"] == "Call provider"


def test_extract_ordered_steps_bulleted() -> None:
    text = "- File appeal within 60 days\n- Include denial notice\n* Attach chart notes"
    steps = _extract_ordered_steps(text)
    assert len(steps) == 3


def test_extract_ordered_steps_sentence_fallback() -> None:
    text = "File the appeal first. Include the denial notice next. Attach records."
    steps = _extract_ordered_steps(text)
    assert len(steps) >= 2


# ---------------------------------------------------------------------------
# Workflow registry
# ---------------------------------------------------------------------------
def test_workflow_registry_lists_all_four() -> None:
    assert set(AGENT_WORKFLOW_NAMES) == {
        "policy_lookup",
        "claim_triage",
        "policy_comparison",
        "compliance_checklist",
    }


def test_run_workflow_rejects_unknown_name(db_session) -> None:
    with pytest.raises(UnknownWorkflowError):
        run_workflow(name="nonexistent", input_payload={}, db=db_session)


# ---------------------------------------------------------------------------
# End-to-end: policy_lookup
# ---------------------------------------------------------------------------
def test_policy_lookup_runs_end_to_end(db_session, seeded_corpus) -> None:
    result = run_workflow(
        name="policy_lookup",
        input_payload={"query": "How long do I have to appeal a PA denial?"},
        db=db_session,
        top_k=3,
    )

    # Schema shape
    assert result.workflow == "policy_lookup"
    assert "answer" in result.final_output
    assert "rewritten_query" in result.final_output

    # PA → "prior authorization" expansion happened
    assert "prior authorization" in result.final_output["rewritten_query"].lower()

    # Trace records each node
    step_names = [s["name"] for s in result.steps]
    assert step_names == ["rewrite", "retrieve", "generate"]

    # Mock chat provider returns chunks → citations should be populated
    # whenever retrieval returned chunks.
    if any(s.get("chunk_count", 0) for s in result.steps):
        assert result.citations
        for c in result.citations:
            assert c.get("chunk_id")
            assert c.get("document_title")


# ---------------------------------------------------------------------------
# End-to-end: claim_triage
# ---------------------------------------------------------------------------
def test_claim_triage_classifies_and_produces_next_steps(db_session, seeded_corpus) -> None:
    result = run_workflow(
        name="claim_triage",
        input_payload={
            "claim_summary": "Claim denied because prior authorization was missing for MRI.",
            "question": "What should the billing team do next?",
        },
        db=db_session,
        top_k=3,
    )

    assert result.workflow == "claim_triage"
    out = result.final_output
    assert out["classification"] == "authorization"
    assert out["classification_rationale"]
    # Mock provider returns top-chunk verbatim; at least one parsed step.
    assert isinstance(out["next_steps"], list)
    assert isinstance(out["grounding_score"], float)

    step_names = [s["name"] for s in result.steps]
    # Must include the four canonical nodes (possibly with retrieve twice
    # if the grounding floor triggered a reflection loop).
    assert "classify" in step_names
    assert "retrieve" in step_names
    assert "generate_checklist" in step_names
    assert "ground_check" in step_names


# ---------------------------------------------------------------------------
# End-to-end: compliance_checklist
# ---------------------------------------------------------------------------
def test_compliance_checklist_runs_end_to_end(db_session, seeded_corpus) -> None:
    result = run_workflow(
        name="compliance_checklist",
        input_payload={"topic": "filing a first-level appeal for a denied claim"},
        db=db_session,
        top_k=3,
    )

    assert result.workflow == "compliance_checklist"
    out = result.final_output
    assert "items" in out
    assert "validated_item_count" in out
    assert isinstance(out["items"], list)

    step_names = [s["name"] for s in result.steps]
    assert step_names == ["retrieve", "generate_checklist", "validate"]


# ---------------------------------------------------------------------------
# End-to-end: policy_comparison
# ---------------------------------------------------------------------------
def test_policy_comparison_runs_end_to_end(db_session, seeded_corpus) -> None:
    result = run_workflow(
        name="policy_comparison",
        input_payload={
            "document_a_title": "Appeal Process Policy",
            "document_b_title": "Claim Denial Policy",
            "focus": "timelines and documentation requirements",
        },
        db=db_session,
        top_k=3,
    )

    assert result.workflow == "policy_comparison"
    out = result.final_output
    assert out["document_a_title"] == "Appeal Process Policy"
    assert out["document_b_title"] == "Claim Denial Policy"
    assert "summary" in out
    assert isinstance(out["differences"], list)

    step_names = [s["name"] for s in result.steps]
    assert step_names == ["retrieve_a", "retrieve_b", "compare"]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------
def test_agents_run_endpoint_returns_expected_schema(client_with_real_db, seeded_corpus) -> None:
    response = client_with_real_db.post(
        "/agents/run",
        json={
            "workflow": "policy_lookup",
            "input": {"query": "How long do I have to appeal?"},
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in (
        "workflow",
        "final_output",
        "steps",
        "citations",
        "confidence",
        "model_name",
        "latency_ms",
    ):
        assert key in payload, f"missing key: {key}"
    assert payload["workflow"] == "policy_lookup"


def test_agents_run_rejects_unknown_workflow(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/agents/run",
        json={"workflow": "made_up_workflow", "input": {}},
    )
    # Pydantic-level Literal validation kicks in first → 422.
    assert response.status_code == 422
