"""
Embeddings service.

Why pluggable providers?
    Real OpenAI embeddings cost money and require an API key. CI and unit
    tests need a deterministic, network-free alternative. A small Protocol
    + factory pattern lets us swap the backing model without code changes
    elsewhere — and lets future providers (e.g. sentence-transformers,
    Vertex AI) slot in without touching the retrieval layer.

Providers
---------
    mock     : deterministic hashed-bag-of-tokens embeddings. Same input
               → same vector, no network. Two texts that share many tokens
               score high cosine similarity, so this is good enough for
               smoke-testing the retrieval plumbing.
    openai   : any OpenAI-compatible /v1/embeddings endpoint (works against
               OpenAI, vLLM, Ollama-OpenAI, LM Studio, ...).

Public surface
--------------
    get_embedder()              -> EmbeddingProvider singleton
    EmbeddingProvider.embed(texts) -> list[list[float]]
"""

from __future__ import annotations

import hashlib
import logging
import math
from functools import lru_cache
from typing import Protocol

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Provider protocol
# =============================================================================
class EmbeddingProvider(Protocol):
    """Anything that turns a batch of strings into a batch of vectors."""

    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order. Vectors are L2-normalized."""
        ...


# =============================================================================
# Mock provider — deterministic hashed bag-of-tokens
# =============================================================================
class MockEmbeddingProvider:
    """
    Deterministic, network-free embeddings for tests / local dev.

    Algorithm
    ---------
    For each lowercased whitespace token of the input, derive eight
    `(index, sign)` pairs from a SHA-256 hash of the token. Each pair adds
    ±1 to one position in a `dim`-length vector. Repeat for character
    bigrams to capture sub-word similarity (so "appeal" and "appeals"
    still overlap). Finally L2-normalize.

    Properties this gets us
    -----------------------
    * Same text → same vector.
    * Texts sharing tokens land closer in cosine space than unrelated ones.
    * No external dependency, no API key, no GPU.
    * Behavior is independent of `dim`, so the same code works at any
      target dimension.

    Not appropriate for production retrieval — only for smoke tests and
    local development.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        if not text:
            return vec

        tokens = text.lower().split()
        self._accumulate(tokens, vec, salt="tok")
        # Character bigrams give a little robustness to morphological variation
        # ("appeal"/"appeals") and to tokens that share a substring.
        bigrams = [text[i : i + 2].lower() for i in range(len(text) - 1)]
        self._accumulate(bigrams, vec, salt="bi")

        # L2 normalize so cosine similarity reduces to a dot product.
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _accumulate(self, features: list[str], vec: list[float], salt: str) -> None:
        for feature in features:
            digest = hashlib.sha256(f"{salt}:{feature}".encode()).digest()
            # Use 8 4-byte windows per feature → 8 (index, sign) updates.
            for i in range(8):
                window = digest[i * 4 : (i + 1) * 4]
                idx = int.from_bytes(window, "big") % self.dimensions
                sign = 1.0 if (digest[i] & 1) else -1.0
                vec[idx] += sign


# =============================================================================
# OpenAI-compatible provider
# =============================================================================
class OpenAIEmbeddingProvider:
    """
    OpenAI-compatible embeddings.

    Reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `EMBEDDING_MODEL` from
    settings. Anything that speaks the OpenAI embeddings API works here
    (vLLM, Ollama-OpenAI, LM Studio, etc.) — just point `OPENAI_BASE_URL`
    at the right URL.
    """

    # Hard-coded batch cap. OpenAI accepts up to ~2048 inputs in one call;
    # 96 is a safe value for self-hosted endpoints with smaller windows.
    _BATCH_SIZE = 96

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            # We don't fall back silently — a missing key when the operator
            # explicitly chose OpenAI is almost always a misconfiguration.
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Either set the key or switch EMBEDDING_PROVIDER=mock for "
                "local development."
            )
        # Import lazily so the package is only required when the provider
        # is actually used.
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Batch to respect provider input limits.
        out: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[start : start + self._BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            out.extend(d.embedding for d in response.data)

        # Sanity-check dimensions — if the operator misconfigured
        # EMBEDDING_DIMENSIONS, we want to fail loudly here rather than at
        # the pgvector INSERT (which gives a confusing error).
        if out and len(out[0]) != self.dimensions:
            raise RuntimeError(
                f"Embedding dimension mismatch: model {self._model!r} returned "
                f"{len(out[0])}-dim vectors but EMBEDDING_DIMENSIONS is "
                f"{self.dimensions}. Update settings and re-run migrations."
            )
        return out


# =============================================================================
# Factory
# =============================================================================
def _build_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider
    if provider == "mock":
        return MockEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if provider == "sentence-transformers":  # pragma: no cover — future work
        raise NotImplementedError(
            "sentence-transformers provider will be added in a later phase. "
            "Use EMBEDDING_PROVIDER=mock or =openai for now."
        )
    raise ValueError(f"unknown EMBEDDING_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    """Process-wide embedding provider singleton."""
    settings = get_settings()
    embedder = _build_provider(settings)
    logger.info(
        "embedding_provider_initialized",
        extra={
            "provider": settings.embedding_provider,
            "dimensions": embedder.dimensions,
            "model": settings.embedding_model,
        },
    )
    return embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper around `get_embedder().embed(...)`."""
    return get_embedder().embed(texts)
