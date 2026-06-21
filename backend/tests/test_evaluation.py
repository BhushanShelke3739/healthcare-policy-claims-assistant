"""
Tests for Phase 6 — evaluation framework.

Two tiers:
    * Pure-unit tests for metric helpers (no DB).
    * Integration tests that run a small eval dataset end-to-end
      through the real Postgres (`db_session`) using the autouse
      mock LLM. No Ollama / OpenAI calls.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.db.models import EvaluationResult, EvaluationRun
from app.schemas.eval import EvalDataset, EvalDatasetItem
from app.services.evaluation import (
    _answer_relevancy,
    _chunk_is_relevant,
    _content_tokens,
    _is_refusal_text,
    _keyword_recall,
    load_dataset,
    run_evaluation,
    summarize_run,
)
from app.services.retrieval import RetrievedChunk
from tests.conftest import TEST_CORPUS_TAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chunk(text: str, title: str = "Appeal Process Policy") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        chunk_text=text,
        chunk_index=0,
        similarity_score=1.0,
        component_scores={},
        metadata={},
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def test_content_tokens_strips_stopwords_and_short_tokens() -> None:
    tokens = _content_tokens("this is the appeal process policy with denial")
    assert "appeal" in tokens
    assert "denial" in tokens
    assert "this" not in tokens  # stop word
    assert "is" not in tokens  # < 4 chars


def test_is_refusal_text_matches_substring() -> None:
    phrase = "I could not find this in the available policy documents."
    assert _is_refusal_text(phrase, phrase) is True
    assert _is_refusal_text("Wrapper.  " + phrase + "  end.", phrase) is True
    assert _is_refusal_text("A normal answer", phrase) is False


def test_chunk_is_relevant_by_document_match() -> None:
    chunk = _make_chunk("anything", title="Appeal Process Policy")
    assert (
        _chunk_is_relevant(chunk, expected_document="Appeal Process Policy", expected_keywords=[])
        is True
    )


def test_chunk_is_relevant_by_keyword_match() -> None:
    chunk = _make_chunk("Sixty calendar days from the denial notice.", title="X")
    assert (
        _chunk_is_relevant(chunk, expected_document="Different Policy", expected_keywords=["sixty"])
        is True
    )


def test_chunk_is_relevant_false_when_no_match() -> None:
    chunk = _make_chunk("unrelated text", title="Other")
    assert (
        _chunk_is_relevant(
            chunk, expected_document="Appeal Process Policy", expected_keywords=["sixty"]
        )
        is False
    )


def test_keyword_recall_full_hit() -> None:
    chunks = [_make_chunk("Sixty calendar days from the denial notice.")]
    recall, hits, misses = _keyword_recall(chunks, ["sixty", "denial notice"])
    assert recall == 1.0
    assert set(hits) == {"sixty", "denial notice"}
    assert misses == []


def test_keyword_recall_partial() -> None:
    chunks = [_make_chunk("Sixty calendar days from the notice.")]
    recall, hits, misses = _keyword_recall(chunks, ["sixty", "denial notice"])
    assert recall == 0.5
    assert "sixty" in hits
    assert "denial notice" in misses


def test_keyword_recall_none_when_no_expected_keywords() -> None:
    recall, hits, misses = _keyword_recall([_make_chunk("x")], [])
    assert recall is None
    assert hits == []
    assert misses == []


def test_answer_relevancy_overlap() -> None:
    # Question content tokens: appeal, denied, claim
    # Answer covers "appeal" and "denied" → 2/3
    score = _answer_relevancy(
        "How do I appeal a denied claim?",
        "Filing an appeal after a denied determination is allowed.",
    )
    assert score is not None
    assert 0.3 < score <= 1.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def test_load_default_dataset_parses() -> None:
    ds = load_dataset()
    assert isinstance(ds, EvalDataset)
    assert ds.items
    # Spot-check known items exist.
    ids = {item.id for item in ds.items}
    assert "denial-code-hf022" in ids
    assert "out-of-scope-france" in ids
    # Refusal item should have expected_refusal=True.
    refusal_item = next(i for i in ds.items if i.id == "out-of-scope-france")
    assert refusal_item.expected_refusal is True


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing)


def test_load_dataset_from_explicit_relative_path(tmp_path, monkeypatch) -> None:
    # Build a tiny custom dataset and verify it round-trips.
    custom = tmp_path / "tiny_eval.json"
    custom.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "question": "anything",
                        "expected_document": None,
                        "expected_keywords": [],
                        "expected_refusal": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ds = load_dataset(custom)
    assert len(ds.items) == 1
    assert ds.items[0].expected_refusal is True


# ---------------------------------------------------------------------------
# End-to-end: run_evaluation with a tiny in-memory dataset
# ---------------------------------------------------------------------------
def _make_tiny_dataset() -> EvalDataset:
    """A 3-item dataset hitting the seeded TEST_CORPUS_TAG fixture."""
    return EvalDataset(
        name="tiny",
        description="3 items for integration testing",
        items=[
            EvalDatasetItem(
                id="hit",
                question="When must a first-level appeal be filed?",
                expected_answer="Within sixty calendar days.",
                expected_document="Appeal Process Policy",
                expected_keywords=["sixty", "calendar days"],
            ),
            EvalDatasetItem(
                id="identifier",
                question="What does HF-022 mean?",
                expected_answer="prior authorization was not on file",
                expected_document="Claim Denial Policy",
                expected_keywords=["HF-022", "prior authorization"],
            ),
            EvalDatasetItem(
                id="refusal",
                question="What is the capital of France?",
                expected_refusal=True,
            ),
        ],
    )


def test_run_evaluation_persists_run_and_per_question_results(db_session, seeded_corpus) -> None:
    run = run_evaluation(
        db_session,
        name="integration-tiny",
        description="end-to-end smoke",
        dataset=_make_tiny_dataset(),
        top_k=3,
        mode="hybrid",
    )

    # ORM tree is consistent immediately.
    assert isinstance(run, EvaluationRun)
    assert run.name == "integration-tiny"
    assert len(run.results) == 3

    # Every persisted result has its expected basic columns populated.
    by_question = {r.question: r for r in run.results}
    assert any("first-level appeal" in q for q in by_question)
    for r in run.results:
        assert isinstance(r, EvaluationResult)
        assert r.evaluation_run_id == run.id
        assert r.latency_ms is not None and r.latency_ms >= 0
        assert isinstance(r.details, dict)


def test_run_evaluation_summary_has_expected_fields(db_session, seeded_corpus) -> None:
    run = run_evaluation(
        db_session,
        name="integration-summary",
        dataset=_make_tiny_dataset(),
        top_k=3,
    )
    summary = summarize_run(run)
    for key in (
        "num_questions",
        "retrieval_hit_rate",
        "avg_context_precision",
        "avg_faithfulness",
        "avg_answer_relevancy",
        "hallucination_rate",
        "refusal_accuracy",
        "avg_latency_ms",
    ):
        assert key in summary, f"missing summary key: {key}"
    assert summary["num_questions"] == 3


def test_run_evaluation_refusal_path(db_session, seeded_corpus) -> None:
    """
    The dedicated 'refusal' item should be marked with refusal_correct
    in its details — mock provider with empty / unrelated retrieval
    should produce the refusal phrase (or close to it).
    """
    ds = EvalDataset(
        items=[
            EvalDatasetItem(
                id="solo-refusal",
                question="What is the capital of France?",
                expected_refusal=True,
            )
        ]
    )
    run = run_evaluation(db_session, name="refusal-only", dataset=ds, top_k=3)
    r = run.results[0]
    assert r.details.get("refusal_expected") is True
    # `refusal_correct` is True OR False — but it must be set (not None).
    assert "refusal_correct" in r.details


def test_run_evaluation_empty_dataset_rejected(db_session) -> None:
    ds = EvalDataset(items=[])
    with pytest.raises(ValueError):
        run_evaluation(db_session, name="empty", dataset=ds)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_eval_run_endpoint_returns_summary_and_results(
    client_with_real_db, seeded_corpus, tmp_path: Path, monkeypatch
) -> None:
    # Use a tiny custom dataset via dataset_path so the API doesn't try
    # to run the full 18-item bundled file (still cheap with mock, but
    # this is more isolated).
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "x",
                        "question": "When must a first-level appeal be filed?",
                        "expected_document": "Appeal Process Policy",
                        "expected_keywords": ["sixty"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    response = client_with_real_db.post(
        "/eval/run",
        json={
            "name": "api-test",
            "dataset_path": str(custom),
            "top_k": 3,
            "mode": "hybrid",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    for key in ("id", "name", "created_at", "summary", "results"):
        assert key in payload, f"missing key: {key}"
    assert payload["name"] == "api-test"
    assert payload["summary"]["num_questions"] == 1
    assert len(payload["results"]) == 1


def test_eval_runs_list_and_get_roundtrip(client_with_real_db, db_session, seeded_corpus) -> None:
    # Create one run directly via the service so we don't depend on the
    # full bundled dataset.
    run = run_evaluation(
        db_session,
        name="list-test",
        dataset=_make_tiny_dataset(),
        top_k=3,
    )

    # GET /eval/runs lists it
    r = client_with_real_db.get("/eval/runs")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert any(item["id"] == str(run.id) for item in items)

    # GET /eval/runs/{id} returns the full result
    r2 = client_with_real_db.get(f"/eval/runs/{run.id}")
    assert r2.status_code == 200, r2.text
    payload = r2.json()
    assert payload["id"] == str(run.id)
    assert len(payload["results"]) == 3
    assert payload["summary"]["num_questions"] == 3


def test_eval_runs_get_404_for_missing_id(client_with_real_db) -> None:
    fake_id = uuid.uuid4()
    r = client_with_real_db.get(f"/eval/runs/{fake_id}")
    assert r.status_code == 404
