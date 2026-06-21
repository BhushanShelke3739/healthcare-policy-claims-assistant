"""
Unit tests for the embeddings service.

The mock provider is exercised directly. The OpenAI provider is not
exercised here — it requires network or a stub server, and we keep this
file network-free.
"""

from __future__ import annotations

import math

import pytest

from app.services.embeddings import MockEmbeddingProvider


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


# ---------------------------------------------------------------------------
# Determinism + shape
# ---------------------------------------------------------------------------
def test_mock_embeddings_are_deterministic() -> None:
    p = MockEmbeddingProvider(dimensions=128)
    a = p.embed(["Prior authorization is required for advanced imaging."])[0]
    b = p.embed(["Prior authorization is required for advanced imaging."])[0]
    assert a == b


def test_mock_embeddings_have_correct_dimension() -> None:
    p = MockEmbeddingProvider(dimensions=384)
    [vec] = p.embed(["anything"])
    assert len(vec) == 384


def test_mock_embeddings_are_l2_normalized() -> None:
    p = MockEmbeddingProvider(dimensions=128)
    [vec] = p.embed(["claim denial appeal process"])
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_empty_text_returns_zero_vector() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    [vec] = p.embed([""])
    assert vec == [0.0] * 64


def test_batch_preserves_order() -> None:
    p = MockEmbeddingProvider(dimensions=64)
    texts = ["alpha", "beta", "gamma"]
    vectors = p.embed(texts)
    # Reconstructed batch matches per-item embeds in the same order.
    for i, t in enumerate(texts):
        assert vectors[i] == p.embed([t])[0]


# ---------------------------------------------------------------------------
# Semantic-ish properties
# ---------------------------------------------------------------------------
def test_similar_texts_have_higher_cosine_than_unrelated() -> None:
    """
    The mock isn't a real semantic model, but two texts sharing tokens
    should still cluster closer than two texts with disjoint vocabularies.
    """
    p = MockEmbeddingProvider(dimensions=1024)

    appeal_a = p.embed(["How do I file an appeal for a denied claim?"])[0]
    appeal_b = p.embed(["The appeal process for denied claims has 60 days."])[0]
    unrelated = p.embed(["Telehealth modifier 95 is required for video visits."])[0]

    sim_related = _cosine(appeal_a, appeal_b)
    sim_unrelated = _cosine(appeal_a, unrelated)

    assert (
        sim_related > sim_unrelated
    ), f"expected related > unrelated, got {sim_related:.3f} vs {sim_unrelated:.3f}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimensions=0)


def test_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimensions=-1)
