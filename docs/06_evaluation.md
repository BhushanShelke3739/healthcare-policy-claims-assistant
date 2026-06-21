# 06 — Evaluation

> *"You can't improve what you don't measure."* The whole point of
> Phase 6 is to turn the question *"did that change help?"* into a
> number.

Phases 1–5 built the RAG and agent layers. Each phase ended with a
manual smoke test ("ask a question, eyeball the answer"). That's good
for the first iteration. It does not scale: by the time you're
choosing between four chunk sizes × three embedding models × two LLMs,
hand-testing is hopeless. Phase 6 introduces a **deterministic eval
harness**: a fixed set of questions with known expected answers, run
through the pipeline on demand, scored on multiple axes, and persisted
so you can compare runs over time.

## Why RAG eval is hard

Classic ML eval has one signal — *"did the model predict the right
label?"* — and aggregates over thousands of examples. RAG eval has
**at least four** signals, all of which can fail independently:

1. **Retrieval** — did we find the right chunk(s)?
2. **Precision / recall of context** — were the retrieved chunks
   actually useful, or full of noise?
3. **Faithfulness** — does the answer stay within the retrieved
   chunks, or wander off into model-training-data territory?
4. **Answer relevancy** — does the answer actually address the
   question, regardless of whether it's grounded?

A pipeline can score high on retrieval and still produce hallucinated
answers (the LLM ignored the context). Or it can have perfect
faithfulness on garbage retrieval (it cited the wrong policy
correctly). You need to inspect all four to know what to fix.

## The metric battery

We compute the following per question, then aggregate at the run level:

| Metric | What it measures | How it's computed |
|---|---|---|
| `retrieval_hit_rate` | Did `expected_document` appear in retrieved chunks? | Binary per Q, average across in-scope Qs. |
| `context_precision` | Fraction of retrieved chunks judged "relevant." | A chunk is relevant if it's from the expected doc OR contains an expected keyword. |
| `context_recall` | Fraction of expected keywords present in retrieved text. | Substring match, case-insensitive. Null when no keywords expected. |
| `answer_relevancy` | Does the answer address the question? | Token overlap between question content-words and answer content-words. |
| `faithfulness` | Is the answer grounded in the retrieved chunks? | Reuses `run_grounding_check` from the agent layer — token overlap between answer and combined chunk text. |
| `hallucination_flag` | Boolean — did the model invent something? | True when `faithfulness < 0.3` AND the answer wasn't the refusal phrase AND the answer is non-empty. |
| `refusal_accuracy` | For `expected_refusal=true` questions: did we refuse? | Binary per refusal Q, average across the refusal set. |
| `latency_ms` | Wall-clock time. | Measured around retrieval + generation. |
| `token_cost` | Placeholder. | Always None today — Ollama doesn't surface usage stats via the OpenAI-compatible API. Reserved for when this project hits real OpenAI. |

### Why rule-based metrics (and not an LLM judge)?

LLM-as-judge (asking a strong model to score each answer against a
rubric) is the gold standard for production RAG eval — see
[RAGAS](https://github.com/explodinggradients/ragas), TruLens,
DeepEval. It's also:

- **Expensive** (one LLM call per metric per question = O(N × 5)).
- **Non-deterministic** (different runs produce different scores).
- **Circular** (the judge model has its own biases).

For a portfolio / dev-loop tool, a deterministic rule-based judge is
the right tradeoff. The metrics are coarse but **monotonic** — if your
chunk-size change drops `avg_context_precision` from 0.78 to 0.52,
something got worse, even if the absolute numbers are an
approximation. That's enough to drive iteration.

A future enhancement could add an `LLM_JUDGE` mode that re-scores
faithfulness with a strong model — the `details` JSONB column on
`EvaluationResult` is already shaped to accept it.

## The dataset format

[`sample_data/eval_questions/healthcare_policy_eval.json`](../sample_data/eval_questions/healthcare_policy_eval.json)
contains 18 hand-built items covering the six synthetic policies. Each
looks like:

```json
{
  "id": "appeal-window-first-level",
  "question": "How long do I have to file a first-level appeal?",
  "expected_answer": "Within 60 calendar days of the denial notice.",
  "expected_document": "Appeal Process Policy",
  "expected_keywords": ["60", "sixty", "calendar days", "denial notice"]
}
```

Two items have `expected_refusal: true` for out-of-scope questions
("what is the capital of France?", "should I take ibuprofen?"). The
refusal-correct metric is computed only against these — answering them
*at all* is a hallucination.

### What makes a good eval item

- **One clear factual claim per question.** Multi-part questions are
  fine but pick *one* primary answer.
- **`expected_keywords` should be distinctive.** "Days" is a bad
  keyword because every policy mentions days. "Sixty calendar days
  from the denial notice" is good.
- **Mix difficulty.** Direct quotes, paraphrases, identifier lookups
  (HF-022), cross-doc questions, refusal cases.
- **Synthetic only.** No real PHI / PII / claim IDs.

## The iteration loop

Phase 6 is the foundation for **comparing changes over time**. The
intended workflow is:

```
   ┌───────────────────────────────────────────────────────┐
   │   1. Baseline: POST /eval/run → record summary        │
   │                                                       │
   │   2. Make ONE change (chunk size, LLM, alpha, ...)    │
   │                                                       │
   │   3. POST /eval/run again                             │
   │                                                       │
   │   4. Diff the two summaries                           │
   │       - Did retrieval_hit_rate move?                  │
   │       - Did avg_faithfulness move?                    │
   │       - Did hallucination_rate move?                  │
   │       - Did avg_latency_ms blow up?                   │
   │                                                       │
   │   5. Keep / revert / iterate                          │
   └───────────────────────────────────────────────────────┘
```

The `EvaluationRun.name` and `EvaluationRun.description` fields are
where you record what changed:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/eval/run -Method POST -ContentType "application/json" -Body (@{
    name        = "chunk-size-1200-vs-baseline"
    description = "Increased CHUNK_SIZE from 800 to 1200; otherwise unchanged"
} | ConvertTo-Json)
```

Later, `GET /eval/runs` shows the full history; `GET /eval/runs/{id}`
shows one run's per-question metrics so you can drill into which
question types got worse.

## Comparing chunking strategies (concrete example)

The most common use case is sweeping `CHUNK_SIZE` / `CHUNK_OVERLAP` /
`HYBRID_ALPHA`. The recipe:

1. Set the baseline values in `.env`.
2. Re-seed the corpus: `python -m app.seed_policies --replace`.
3. `POST /eval/run` with `name="baseline-c800-o120-a0.6"`.
4. Change ONE setting in `.env`.
5. Re-seed (if you changed CHUNK_SIZE) and re-embed (if you changed
   the embedding provider/model).
6. `POST /eval/run` with a name that describes the change.
7. Diff the two summaries.

Common patterns to expect:

| Change | What usually moves |
|---|---|
| Bigger chunks (1200+) | Recall ↑ (more context per chunk), precision ↓ (more noise), latency ↑ |
| Smaller chunks (400) | Precision ↑, recall ↓ (answers split across chunks), retrieval_hit ↓ |
| More overlap | Both precision/recall ↑ slightly, storage cost ↑ |
| Higher hybrid alpha (more vector) | Better on paraphrase questions, worse on identifier questions (HF-022) |
| Lower alpha (more keyword) | Inverse |

You won't get clear "X is universally best" answers — that's the
whole point of evaluating against *your specific corpus and question
distribution*.

## Latency

With `LLM_PROVIDER=mock` (the default), a 15-question eval finishes
in under a second. With Ollama on CPU, expect roughly
`num_questions × per-question-LLM-latency` — e.g. 15 × 30s = ~7
minutes. The harness writes the run incrementally, so even an
interrupted Ollama run produces partial results.

> **Dev tip:** keep `LLM_PROVIDER=mock` while iterating on the metric
> formulas themselves or adding new eval items. Switch to Ollama /
> OpenAI only when you want a real-quality answer in the summary.

## What this enables next

- **Phase 7 (Frontend):** an eval dashboard page that shows the
  history of runs as a table with sparkline-style trend deltas.
- **Phase 9 (Observability):** log metric snapshots to
  `query_logs.metadata` per `/rag/ask` call so you can detect
  production drift, not just batch evals.
- **CI gating (later):** every PR runs `/eval/run` and fails if
  metric averages drop more than a threshold.
