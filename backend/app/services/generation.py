"""
Grounded answer generation.

Pluggable provider, same shape as embeddings: `ChatProvider` Protocol +
factory. Two implementations:
    * MockChatProvider          — deterministic, no LLM call. Default for
                                  dev + tests.
    * OpenAIChatProvider        — any OpenAI-compatible chat endpoint
                                  (OpenAI, Ollama, vLLM, LM Studio, ...).

System prompt + grounding rules live here. Every behavior the user can
observe from /rag/ask (citation discipline, refusal phrasing,
no-medical-advice) is enforced in one of three places:
    1. The system prompt (this file).
    2. Mock provider behavior (this file).
    3. Server-side citation validation (the API layer in routes_rag.py).

Tip: the LLM cannot be fully trusted to follow rules — server-side
validation is the cheap belt to the prompt's suspenders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.rag import Confidence, GeneratedAnswer, _LLMCitation
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# =============================================================================
# System prompt
# =============================================================================
SYSTEM_PROMPT = """\
You are a healthcare policy assistant for operations and billing teams at a
health plan. You help staff find what an internal policy *says*. You do not
make clinical, legal, or financial decisions on the user's behalf.

# GROUNDING RULES (do not violate)
1. Use ONLY the information in the "Policy Context" section to answer.
2. If the Policy Context does not contain enough information to answer the
   user's question, set "answer" to the exact phrase:
   "I could not find this in the available policy documents."
   In that case, return an empty citations list and confidence = "low".
3. Do not draw on outside knowledge, regulations not mentioned in the
   context, or general healthcare facts.

# CITATION RULES
4. Every factual claim in your answer must be backed by a citation.
5. Each citation references the `chunk_id` shown in the Policy Context.
   Copy the chunk_id exactly — do not invent IDs.
6. The `excerpt` field on each citation must be a verbatim substring of
   the cited chunk's text, ideally the sentence directly supporting your
   claim.

# SCOPE RULES
7. Do not give medical advice or recommend medical treatment.
8. Do not give legal advice. You may explain what a policy specifies; you
   may not advise the user on how to act outside what the policy says.
9. Distinguish between EXPLAINING what a policy says ("the policy
   specifies a 60-day filing window") and MAKING A DECISION for the user
   ("you should file by next Tuesday") — only the former is allowed.

# STYLE
10. Answer in clear, direct operational language using 2-4 complete
    sentences. One-word answers are not acceptable.
11. Always include the specific timelines, dollar amounts, and codes
    from the context AND the situation/clause they apply to. For
    example, "60" alone is not a complete answer — write "First-level
    appeals must be filed within 60 calendar days of the denial notice."
12. Lead with the direct answer in the first sentence, then add any
    relevant qualifiers, exceptions, or related deadlines in the
    following sentences (still drawn only from the context).

# CONFIDENCE
13. "high"   = answer is directly stated in the context with no ambiguity.
    "medium" = answer is supported but you had to combine context across
               chunks, or the context phrasing is indirect.
    "low"    = context is thin / mostly off-topic; use this when in doubt,
               and prefer the refusal phrase when context is genuinely
               insufficient.

# OUTPUT
Return JSON conforming to this schema (and nothing else — no markdown,
no preamble). Each field has a specific semantic, listed below.

  - "answer"           = the COMPLETE natural-language reply the user
                         reads. 2-4 full sentences. Never just a number
                         or a single word.
  - "citations"        = list of supporting evidence pulled FROM the
                         Policy Context. Each entry has the exact
                         chunk_id (UUID) from the prompt and a verbatim
                         excerpt copied from that chunk. Empty list ONLY
                         when there is no relevant context at all.
  - "confidence"       = "low" | "medium" | "high".
  - "grounding_notes"  = short meta-commentary about how the answer maps
                         to evidence ("direct quote from the appeal
                         policy", "combined two chunks"). NOT a
                         restatement of the answer.

# WORKED EXAMPLE

If the Policy Context contained:

[chunk_id: 11111111-1111-1111-1111-111111111111] First-level appeals must be filed within sixty calendar days of the denial notice.

[chunk_id: 22222222-2222-2222-2222-222222222222] Expedited appeals are decided within seventy-two hours when delay would jeopardize the member's life or health.

And the user question was:
How long do I have to appeal a denied claim?

The correct response is exactly:

{
  "answer": "First-level appeals must be filed within 60 calendar days of the denial notice. For urgent situations where delay would jeopardize the member's health, an expedited appeal can be requested and is decided within 72 hours.",
  "citations": [
    {"chunk_id": "11111111-1111-1111-1111-111111111111", "excerpt": "First-level appeals must be filed within sixty calendar days of the denial notice."},
    {"chunk_id": "22222222-2222-2222-2222-222222222222", "excerpt": "Expedited appeals are decided within seventy-two hours when delay would jeopardize the member's life or health."}
  ],
  "confidence": "high",
  "grounding_notes": "Primary answer from the appeal policy chunk; expedited timeline added as a related qualifier from a second chunk."
}

Now produce the JSON response for the ACTUAL question below using the
ACTUAL Policy Context chunks shown.
"""


# =============================================================================
# Provider protocol
# =============================================================================
class ChatProvider(Protocol):
    """Anything that turns a (question, retrieved chunks) into a GeneratedAnswer."""

    model_name: str

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer: ...


# =============================================================================
# Prompt building (shared)
# =============================================================================
def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Render the user-facing prompt: context chunks (with chunk_id labels)
    followed by the question.

    chunk_id labels are how the LLM produces citations — the model copies
    the literal chunk_id it sees in the prompt.
    """
    if not chunks:
        context_block = "(no policy context retrieved)"
    else:
        context_block = "\n\n".join(
            f"[chunk_id: {chunk.chunk_id}] {chunk.chunk_text}" for chunk in chunks
        )

    return "Policy Context:\n" f"{context_block}\n\n" "User Question:\n" f"{question}"


# =============================================================================
# Mock provider
# =============================================================================
@dataclass
class _MockHeuristics:
    """Score thresholds the mock uses to map retrieval scores → confidence."""

    high_floor: float = 0.7
    medium_floor: float = 0.3


class MockChatProvider:
    """
    Deterministic, no-LLM provider.

    Used for:
        * Local dev without an API key.
        * Unit tests (no network).
        * Smoke testing the full /rag/ask flow.

    Behavior:
        * Empty retrieval → refusal phrase, confidence=low, no citations.
        * Otherwise → "answer" is the top chunk's text verbatim;
                      citations are the top-2 chunks with chunk_text as
                      excerpt; confidence is derived from the top score.

    The mock is honest about being a mock — `grounding_notes` says so.
    """

    model_name = "mock-chat"

    def __init__(self, settings: Settings) -> None:
        self._refusal = settings.refusal_phrase
        self._heuristics = _MockHeuristics()

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=self._refusal,
                citations=[],
                confidence="low",
                grounding_notes=("Mock provider — retrieval returned no chunks for this question."),
            )

        top = chunks[0]
        top_score = top.similarity_score
        # Annotated explicitly so mypy keeps the Literal type instead of
        # widening the ternary chain to plain `str`.
        confidence: Confidence = (
            "high"
            if top_score >= self._heuristics.high_floor
            else "medium"
            if top_score >= self._heuristics.medium_floor
            else "low"
        )

        # Cite up to two chunks. The excerpt is a (truncated) verbatim
        # slice of the chunk text — never something the mock made up.
        citations = [
            _LLMCitation(
                chunk_id=str(c.chunk_id),
                excerpt=c.chunk_text[:240],
            )
            for c in chunks[:2]
        ]

        # The "answer" is the top chunk verbatim. Honest about being a
        # mock — no fake generation.
        answer = top.chunk_text.strip()

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounding_notes=(
                "Mock provider — answer is the top retrieved chunk verbatim, "
                f"top similarity={top_score:.3f}. Swap LLM_PROVIDER=openai for "
                "real generation."
            ),
        )


# =============================================================================
# OpenAI-compatible provider (works against OpenAI, Ollama, vLLM, ...)
# =============================================================================
class OpenAIChatProvider:
    """
    OpenAI-compatible chat provider.

    Supports two structured-output modes via `LLM_STRUCTURED_OUTPUT`:
        "strict"      — OpenAI proper. Uses `response_format` with a JSON
                        schema. The server enforces schema adherence;
                        parsing won't fail.
        "json_object" — Ollama / vLLM / LM Studio. Uses
                        `response_format={"type": "json_object"}` and we
                        parse + validate the result ourselves. Includes
                        one retry on parse failure with a corrective
                        follow-up message.

    Read `OPENAI_BASE_URL` to point at the right endpoint.
    """

    _MAX_PARSE_RETRIES = 1

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "(For Ollama / local servers, set OPENAI_API_KEY to any "
                "non-empty value, e.g. 'ollama'.)"
            )

        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model_name = settings.llm_model
        self._temperature = settings.llm_temperature
        self._strict = settings.llm_structured_output == "strict"

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        # `list[Any]` because the openai client's accepted message types
        # are TypedDicts in a private namespace; constructing them inline
        # as plain dicts is the idiomatic pattern and OpenAI accepts them
        # fine at runtime.
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks)},
        ]

        if self._strict:
            return self._generate_strict(messages)
        return self._generate_json_object(messages)

    # --- strict (OpenAI proper) -------------------------------------------
    def _generate_strict(self, messages: list[Any]) -> GeneratedAnswer:
        """Use OpenAI's `parse` API — server enforces our Pydantic schema."""
        completion = self._client.beta.chat.completions.parse(
            model=self.model_name,
            temperature=self._temperature,
            messages=messages,
            response_format=GeneratedAnswer,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("LLM returned no parsed content under strict mode.")
        return parsed

    # --- json_object (Ollama / vLLM / others) -----------------------------
    def _generate_json_object(self, messages: list[Any]) -> GeneratedAnswer:
        """Ask for JSON, parse + validate; retry once on failure."""
        last_error: str | None = None
        for attempt in range(self._MAX_PARSE_RETRIES + 1):
            completion = self._client.chat.completions.create(
                model=self.model_name,
                temperature=self._temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or ""
            try:
                payload = json.loads(raw)
                return GeneratedAnswer.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "llm_json_parse_failed",
                    extra={
                        "attempt": attempt,
                        "error": last_error,
                        "raw_head": raw[:200],
                    },
                )
                if attempt >= self._MAX_PARSE_RETRIES:
                    break
                # Append a corrective message and try once more.
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON matching the "
                            "required schema. Return ONLY the JSON object, with no "
                            "preamble or markdown. Error was: " + last_error[:200]
                        ),
                    },
                ]

        raise RuntimeError(
            "LLM failed to return valid JSON after retries. " f"Last error: {last_error}"
        )


# =============================================================================
# Factory
# =============================================================================
def _build_provider(settings: Settings) -> ChatProvider:
    if settings.llm_provider == "mock":
        return MockChatProvider(settings)
    if settings.llm_provider == "openai":
        return OpenAIChatProvider(settings)
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider!r}")


@lru_cache(maxsize=1)
def get_chat_provider() -> ChatProvider:
    settings = get_settings()
    provider = _build_provider(settings)
    logger.info(
        "chat_provider_initialized",
        extra={
            "provider": settings.llm_provider,
            "model": provider.model_name,
            "base_url": settings.openai_base_url if settings.llm_provider == "openai" else None,
        },
    )
    return provider
