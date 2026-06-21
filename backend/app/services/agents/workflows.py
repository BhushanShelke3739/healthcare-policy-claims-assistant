"""
LangGraph workflow definitions.

Each workflow is a `StateGraph` whose nodes are small functions that
(a) read from `AgentState`, (b) call one tool, and (c) return the new
keys to merge into state. Edges are mostly linear; the claim_triage
workflow has one conditional edge that loops back into retrieval when
the first generation is poorly grounded (the "reflection" step).

Why build the graphs per-request (vs. compile once at import)?
    Nodes need to call retrieval, which needs a DB Session. The
    Session is request-scoped (yielded by FastAPI's `get_db`
    dependency). We close over `db` when building the graph for one
    request — slightly more work per request than caching the graph
    structure, but it keeps the graph nodes pure functions and avoids
    juggling thread-locals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.services.agents.state import AgentState
from app.services.agents.tools import (
    agent_generate,
    chunk_to_dict,
    chunks_from_state,
    classify_claim_issue,
    rewrite_query,
    run_grounding_check,
    search_policy_documents,
)

logger = logging.getLogger(__name__)


def _record_step(state: AgentState, *, name: str, summary: str, **extra) -> dict:
    """Helper to append one step trace to the state."""
    steps = list(state.get("steps", []))
    step = {"name": name, "summary": summary, **extra}
    steps.append(step)
    return {"steps": steps}


# =============================================================================
# Workflow 1 — Policy lookup
#   rewrite → retrieve → generate → END
# =============================================================================
def build_policy_lookup_graph(db: Session, *, default_top_k: int):
    def node_rewrite(state: AgentState) -> dict:
        original = state["input"].get("query", "")
        rewritten = rewrite_query(original)
        return {
            "rewritten_query": rewritten,
            **_record_step(
                state,
                name="rewrite",
                summary=f"normalized query → {rewritten!r}",
                rewritten_query=rewritten,
            ),
        }

    def node_retrieve(state: AgentState) -> dict:
        top_k = state.get("top_k", default_top_k)
        hits = search_policy_documents(db, query=state["rewritten_query"], top_k=top_k)
        return {
            "chunks": [chunk_to_dict(h) for h in hits],
            **_record_step(
                state,
                name="retrieve",
                summary=f"retrieved {len(hits)} chunk(s) via hybrid search",
                chunk_count=len(hits),
            ),
        }

    def node_generate(state: AgentState) -> dict:
        chunks = chunks_from_state(state.get("chunks", []))
        result = agent_generate(question=state["rewritten_query"], chunks=chunks)
        return {
            "answer_text": result.answer,
            "confidence": result.confidence,
            "citations": [
                {
                    "document_title": next(
                        (c.document_title for c in chunks if str(c.chunk_id) == cit.chunk_id),
                        "",
                    ),
                    "chunk_id": cit.chunk_id,
                    "excerpt": cit.excerpt,
                }
                for cit in result.citations
                if any(str(c.chunk_id) == cit.chunk_id for c in chunks)
            ],
            **_record_step(
                state,
                name="generate",
                summary=f"answer generated (confidence={result.confidence})",
            ),
        }

    g = StateGraph(AgentState)
    g.add_node("rewrite", node_rewrite)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


# =============================================================================
# Workflow 2 — Claim denial triage
#   classify → retrieve → generate_checklist → ground_check → END
#                                                  │
#                                                  └─ (if poorly grounded → retrieve once more)
# =============================================================================
# Routing the classification to a retrieval query. Keeps the LLM out of
# the loop for a step that benefits from determinism.
_CATEGORY_TO_QUERY = {
    "authorization": "prior authorization required missing claim denial appeal",
    "medical_necessity": "medical necessity criteria appeal denial",
    "coordination_of_benefits": "coordination of benefits primary payer claim",
    "coverage": "service not covered benefit exclusion appeal",
    "administrative": "claim adjudication eligibility duplicate denial appeal",
    "unclassified": "claim denial appeal process timeline documentation",
}

# Below this rule-based grounding score, we loop back into retrieval
# with a broadened query before producing the final checklist.
_GROUNDING_FLOOR = 0.25


def build_claim_triage_graph(db: Session, *, default_top_k: int):
    def node_classify(state: AgentState) -> dict:
        claim_summary = state["input"].get("claim_summary", "")
        category, rationale = classify_claim_issue(claim_summary)
        return {
            "classification": category,
            "classification_rationale": rationale,
            **_record_step(
                state,
                name="classify",
                summary=f"classified as {category}",
                rationale=rationale,
            ),
        }

    def node_retrieve(state: AgentState) -> dict:
        category = state.get("classification", "unclassified")
        base_query = _CATEGORY_TO_QUERY.get(category, _CATEGORY_TO_QUERY["unclassified"])
        # Mix in the user's actual question so the search benefits from
        # their phrasing too.
        user_question = state["input"].get("question", "")
        query = f"{base_query} {user_question}".strip()

        # If we've already retrieved once and are looping for grounding,
        # broaden the query.
        already_retrieved = state.get("chunks") is not None and len(state.get("chunks", [])) > 0
        if already_retrieved:
            query = f"{query} policy appeal next steps documentation"

        hits = search_policy_documents(db, query=query, top_k=state.get("top_k", default_top_k))
        return {
            "chunks": [chunk_to_dict(h) for h in hits],
            **_record_step(
                state,
                name="retrieve",
                summary=f"retrieved {len(hits)} chunk(s) for category={category}",
                query=query,
                chunk_count=len(hits),
                broadened=already_retrieved,
            ),
        }

    def node_generate_checklist(state: AgentState) -> dict:
        chunks = chunks_from_state(state.get("chunks", []))
        question = state["input"].get(
            "question",
            "What should the billing team do next on this claim?",
        )
        ask = (
            f"Claim category: {state.get('classification', 'unclassified')}. "
            f"Claim summary: {state['input'].get('claim_summary', '')}. "
            f"Operational question: {question}. "
            "Produce a short ordered list of next steps the billing team "
            "should take, each grounded in the policy context."
        )
        result = agent_generate(question=ask, chunks=chunks)

        steps_list = _extract_ordered_steps(result.answer)
        return {
            "answer_text": result.answer,
            "next_steps": steps_list,
            "confidence": result.confidence,
            "citations": [
                {
                    "document_title": next(
                        (c.document_title for c in chunks if str(c.chunk_id) == cit.chunk_id),
                        "",
                    ),
                    "chunk_id": cit.chunk_id,
                    "excerpt": cit.excerpt,
                }
                for cit in result.citations
                if any(str(c.chunk_id) == cit.chunk_id for c in chunks)
            ],
            **_record_step(
                state,
                name="generate_checklist",
                summary=f"produced {len(steps_list)} next-step(s)",
                next_step_count=len(steps_list),
            ),
        }

    def node_ground_check(state: AgentState) -> dict:
        chunks = chunks_from_state(state.get("chunks", []))
        score = run_grounding_check(state.get("answer_text", ""), chunks)
        return {
            "grounding_score": score,
            "uncertainty_flag": score < _GROUNDING_FLOOR,
            **_record_step(
                state,
                name="ground_check",
                summary=f"grounding score = {score:.2f}",
                score=score,
                floor=_GROUNDING_FLOOR,
                flagged=score < _GROUNDING_FLOOR,
            ),
        }

    def route_after_ground_check(state: AgentState) -> str:
        # Loop back into retrieval at most once. If we already broadened
        # and grounding is still poor, accept it (with uncertainty_flag)
        # rather than spinning forever.
        any_broadened = any(s.get("broadened") for s in state.get("steps", []))
        if state.get("grounding_score", 1.0) < _GROUNDING_FLOOR and not any_broadened:
            return "retrieve"
        return END

    g = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate_checklist", node_generate_checklist)
    g.add_node("ground_check", node_ground_check)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate_checklist")
    g.add_edge("generate_checklist", "ground_check")
    g.add_conditional_edges(
        "ground_check",
        route_after_ground_check,
        {"retrieve": "retrieve", END: END},
    )
    return g.compile()


def _extract_ordered_steps(text: str) -> list[dict]:
    """
    Pull ordered next-steps out of an LLM answer (numbered or bulleted lines).

    Robust to a few common formats. When nothing structured is detected,
    treats sentence breaks as a single bullet each so the agent still
    has something useful to return.
    """
    import re

    steps: list[dict] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(?:\d+\.|\d+\)|[-*•])\s*(.+)$", line)
        if m:
            steps.append({"order": len(steps) + 1, "action": m.group(1).strip()})

    if not steps:
        # Fallback: one bullet per sentence (rough but better than nothing).
        for sent in re.split(r"(?<=[.!?])\s+", text or ""):
            sent = sent.strip()
            if sent and len(sent) > 8:
                steps.append({"order": len(steps) + 1, "action": sent})

    return steps[:10]  # cap to keep responses reasonable


# =============================================================================
# Workflow 3 — Policy comparison
#   retrieve_a → retrieve_b → compare → END
# =============================================================================
def build_policy_comparison_graph(db: Session, *, default_top_k: int):
    def _retrieve_for_doc(
        state: AgentState,
        *,
        title_key: str,
        dest_key: str,
        step_name: str,
    ) -> dict:
        title = state["input"].get(title_key, "")
        focus = state["input"].get("focus") or ""
        query = f"{title} {focus}".strip() or title
        hits = search_policy_documents(
            db,
            query=query,
            top_k=state.get("top_k", default_top_k),
        )
        # Filter to the named document, if found. If we don't find any
        # chunks belonging to it, keep the top hybrid hits — better than
        # nothing.
        named = [h for h in hits if h.document_title.lower() == title.lower()]
        chunks = named if named else hits
        return {
            dest_key: [chunk_to_dict(c) for c in chunks],
            **_record_step(
                state,
                name=step_name,
                summary=f"retrieved {len(chunks)} chunk(s) for {title!r}",
                title=title,
                chunk_count=len(chunks),
                exact_title_match=bool(named),
            ),
        }

    def node_retrieve_a(state: AgentState) -> dict:
        return _retrieve_for_doc(
            state,
            title_key="document_a_title",
            dest_key="chunks_a",
            step_name="retrieve_a",
        )

    def node_retrieve_b(state: AgentState) -> dict:
        return _retrieve_for_doc(
            state,
            title_key="document_b_title",
            dest_key="chunks_b",
            step_name="retrieve_b",
        )

    def node_compare(state: AgentState) -> dict:
        chunks_a = chunks_from_state(state.get("chunks_a", []))
        chunks_b = chunks_from_state(state.get("chunks_b", []))
        focus = state["input"].get("focus") or "requirements, timelines, and exceptions"

        title_a = state["input"].get("document_a_title", "Document A")
        title_b = state["input"].get("document_b_title", "Document B")

        # Build a synthetic "comparison context" by labeling each chunk
        # with which document it came from before handing to the LLM.
        labeled: list = []
        for c in chunks_a:
            labeled.append(_relabel_chunk(c, prefix=f"[{title_a}] "))
        for c in chunks_b:
            labeled.append(_relabel_chunk(c, prefix=f"[{title_b}] "))

        ask = (
            f"Compare these two policy documents on the following dimensions: {focus}. "
            f"For each dimension, state what {title_a} says and what {title_b} says. "
            "Quote verbatim from the context where possible. Only include dimensions "
            "actually addressed in the context."
        )
        result = agent_generate(question=ask, chunks=labeled)

        return {
            "answer_text": result.answer,
            "comparison_summary": result.answer,
            "differences": _extract_differences(result.answer, title_a, title_b),
            "confidence": result.confidence,
            "citations": [
                {
                    "document_title": next(
                        (c.document_title for c in labeled if str(c.chunk_id) == cit.chunk_id),
                        "",
                    ),
                    "chunk_id": cit.chunk_id,
                    "excerpt": cit.excerpt,
                }
                for cit in result.citations
                if any(str(c.chunk_id) == cit.chunk_id for c in labeled)
            ],
            **_record_step(
                state,
                name="compare",
                summary=f"compared {title_a!r} vs {title_b!r}",
                a_chunks=len(chunks_a),
                b_chunks=len(chunks_b),
            ),
        }

    g = StateGraph(AgentState)
    g.add_node("retrieve_a", node_retrieve_a)
    g.add_node("retrieve_b", node_retrieve_b)
    g.add_node("compare", node_compare)
    g.add_edge(START, "retrieve_a")
    g.add_edge("retrieve_a", "retrieve_b")
    g.add_edge("retrieve_b", "compare")
    g.add_edge("compare", END)
    return g.compile()


def _relabel_chunk(c, *, prefix: str):
    """Return a RetrievedChunk with the prefix prepended to chunk_text.

    The LLM sees `[Title] actual text...` which lets it attribute the
    quote in its answer. We don't change chunk_id/document_id so
    server-side citation validation still works.
    """
    from dataclasses import replace

    return replace(c, chunk_text=prefix + c.chunk_text)


def _extract_differences(text: str, title_a: str, title_b: str) -> list[dict]:
    """
    Heuristically split the LLM's comparison prose into structured rows.

    Looks for lines/paragraphs containing both titles and parses them.
    Falls back to a single-row summary if no structured pattern is found.
    """
    rows: list[dict] = []
    for block in (text or "").split("\n"):
        block = block.strip()
        if not block:
            continue
        if title_a in block and title_b in block:
            # crude: split on the second mention
            rows.append(
                {
                    "dimension": "comparison",
                    "document_a": _extract_clause(block, title_a),
                    "document_b": _extract_clause(block, title_b),
                }
            )
    if not rows:
        rows.append(
            {
                "dimension": "summary",
                "document_a": title_a,
                "document_b": title_b,
            }
        )
    return rows[:10]


def _extract_clause(block: str, title: str) -> str:
    """Take a substring of `block` starting at `title` and trimmed."""
    idx = block.find(title)
    if idx < 0:
        return block.strip()
    return block[idx : idx + 240].strip()


# =============================================================================
# Workflow 4 — Compliance checklist
#   retrieve → generate_checklist → validate → END
# =============================================================================
def build_compliance_checklist_graph(db: Session, *, default_top_k: int):
    def node_retrieve(state: AgentState) -> dict:
        topic = state["input"].get("topic", "")
        if not topic:
            return {
                "chunks": [],
                **_record_step(
                    state,
                    name="retrieve",
                    summary="no topic provided — skipping retrieval",
                ),
            }
        hits = search_policy_documents(db, query=topic, top_k=state.get("top_k", default_top_k))
        return {
            "chunks": [chunk_to_dict(h) for h in hits],
            **_record_step(
                state,
                name="retrieve",
                summary=f"retrieved {len(hits)} chunk(s) for topic {topic!r}",
                chunk_count=len(hits),
            ),
        }

    def node_generate_checklist(state: AgentState) -> dict:
        chunks = chunks_from_state(state.get("chunks", []))
        topic = state["input"].get("topic", "the relevant workflow")
        ask = (
            f"Create a numbered checklist for {topic}. Each item should be a "
            "single concrete action. Ground each item in the policy context "
            "and prefer items that mention specific timelines, codes, or "
            "documentation requirements when available."
        )
        result = agent_generate(question=ask, chunks=chunks)
        raw_items = _extract_ordered_steps(result.answer)
        # Attach supporting chunk ids based on simple keyword overlap.
        items = _attach_supporting_chunks(raw_items, chunks)
        return {
            "answer_text": result.answer,
            "checklist_items": items,
            "confidence": result.confidence,
            "citations": [
                {
                    "document_title": next(
                        (c.document_title for c in chunks if str(c.chunk_id) == cit.chunk_id),
                        "",
                    ),
                    "chunk_id": cit.chunk_id,
                    "excerpt": cit.excerpt,
                }
                for cit in result.citations
                if any(str(c.chunk_id) == cit.chunk_id for c in chunks)
            ],
            **_record_step(
                state,
                name="generate_checklist",
                summary=f"produced {len(items)} item(s)",
                item_count=len(items),
            ),
        }

    def node_validate(state: AgentState) -> dict:
        """
        Drop any checklist item that doesn't share content words with the
        retrieved chunks. This is the "self-validate" reflection step.
        """
        chunks = chunks_from_state(state.get("chunks", []))
        items = state.get("checklist_items", [])
        validated: list[dict] = []
        for item in items:
            score = run_grounding_check(item.get("text", ""), chunks)
            if score >= 0.15:  # lower bar than the triage workflow
                validated.append(item)
        return {
            "checklist_items": validated,
            "validated_item_count": len(validated),
            **_record_step(
                state,
                name="validate",
                summary=f"{len(validated)}/{len(items)} item(s) survived grounding check",
                kept=len(validated),
                dropped=len(items) - len(validated),
            ),
        }

    g = StateGraph(AgentState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate_checklist", node_generate_checklist)
    g.add_node("validate", node_validate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate_checklist")
    g.add_edge("generate_checklist", "validate")
    g.add_edge("validate", END)
    return g.compile()


def _attach_supporting_chunks(items: list[dict], chunks) -> list[dict]:
    """For each checklist item, attach IDs of chunks that share content words."""
    import re

    chunk_tokens_by_id: dict[str, set[str]] = {}
    for c in chunks:
        chunk_tokens_by_id[str(c.chunk_id)] = set(re.findall(r"[a-z]{4,}", c.chunk_text.lower()))

    out: list[dict] = []
    for item in items:
        item_tokens = set(re.findall(r"[a-z]{4,}", item.get("action", "").lower()))
        supporting = [cid for cid, ctoks in chunk_tokens_by_id.items() if item_tokens & ctoks]
        out.append(
            {
                "order": item.get("order"),
                "text": item.get("action", ""),
                "supporting_chunk_ids": supporting[:3],
            }
        )
    return out


# =============================================================================
# Registry
# =============================================================================
# Returns a compiled LangGraph. We type the return as `Any` (rather than
# `object`) because the concrete `CompiledStateGraph` type comes from langgraph,
# which we treat with `ignore_missing_imports`; `object` would hide `.invoke`
# from mypy at the call site in runner.py.
BuilderFn = Callable[..., Any]
WORKFLOW_BUILDERS: dict[str, BuilderFn] = {
    "policy_lookup": build_policy_lookup_graph,
    "claim_triage": build_claim_triage_graph,
    "policy_comparison": build_policy_comparison_graph,
    "compliance_checklist": build_compliance_checklist_graph,
}
