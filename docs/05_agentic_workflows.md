# 05 — Agentic Workflows

Phase 4 gave us a **RAG endpoint** — one question, one retrieval, one
generation. Phase 5 layers **agents** on top: multi-step workflows that
chain tools (retrieval, classification, generation, validation) to
handle tasks that don't fit a single Q&A round trip.

## RAG vs. agentic RAG

| Single-shot RAG (`/rag/ask`) | Agentic RAG (`/agents/run`) |
|---|---|
| One question → one retrieval → one answer | A plan with multiple steps |
| Stateless | Carries a `state` through nodes |
| Same prompt every call | Different prompts at different steps |
| No reflection | Can re-retrieve / refine when grounding is poor |
| Best for clear factual lookups | Best for multi-step operational tasks |

A claim triage question is the canonical example: it isn't "what does
the policy say?" — it's *classify the denial reason → look up the right
policy → produce next-step checklist → make sure each step is grounded
in the policy*. That's a procedure, not a question.

## What "agentic" actually means here

A pop-science "agent" is an LLM that picks its own tools step by step.
That works on big frontier models but breaks on the tiny local models
we ship by default — and is hard to debug because the LLM's plan
changes on every call.

So Phase 5 takes a more **disciplined** version of the pattern:

- The plan (which nodes run, in which order) is **defined in code**
  per workflow. Predictable, debuggable, testable.
- The LLM is used at the nodes where natural-language reasoning is
  irreplaceable (drafting a checklist, comparing two policies).
- Deterministic tools cover the steps where the LLM would only add
  noise (classification by regex, grounding check by token overlap).
- A **reflection node** can loop back when output quality is too low
  (claim_triage workflow uses this once if grounding score < 0.25).

This is sometimes called "structured agents" or "scripted agents."
LangGraph is the right framework for this style — its `StateGraph`
makes the workflow a directed graph you can literally draw.

## LangGraph in 60 seconds

A LangGraph workflow is:

1. **State** — a `TypedDict` (or Pydantic model) describing what each
   step might write.
2. **Nodes** — small functions that take state and return the keys they
   produced. LangGraph merges the returned keys into state.
3. **Edges** — either fixed (`A → B`) or conditional (`A → B if X else
   C`). `START` is the entry node, `END` is the exit.
4. **Compile** — `graph.compile()` returns a runnable object. Call
   `.invoke(initial_state, config={...})` to execute.

```python
from langgraph.graph import StateGraph, START, END

g = StateGraph(AgentState)
g.add_node("rewrite", rewrite_node)
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.add_edge(START, "rewrite")
g.add_edge("rewrite", "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
runnable = g.compile()
```

That's the entire mental model. Our four workflows differ only in
which nodes exist and how they're connected.

## Tools

Each node in a workflow calls one tool. Tools live in
[backend/app/services/agents/tools.py](../backend/app/services/agents/tools.py):

| Tool | Backed by | Used in |
|---|---|---|
| `rewrite_query(query)` | regex + abbreviation map | policy_lookup |
| `search_policy_documents(query, top_k)` | Phase 3 retrieval (hybrid) | all |
| `classify_claim_issue(summary)` | regex pattern match → category | claim_triage |
| `agent_generate(question, chunks)` | Phase 4 chat provider | all that produce text |
| `run_grounding_check(answer, chunks)` | content-word overlap | claim_triage, compliance_checklist |

A future enhancement: replace `classify_claim_issue` with an LLM call
that emits a `ClaimDenialCategory` via strict structured output. The
rule-based version is good enough for our six synthetic policies and
makes the test suite deterministic.

## The four workflows

### 1. `policy_lookup` — natural-language policy finder

```
START → rewrite → retrieve → generate → END
```

- **rewrite**: expand abbreviations ("PA" → "prior authorization"),
  normalize whitespace.
- **retrieve**: hybrid search for top-k chunks.
- **generate**: hand the question + chunks to the chat provider,
  collect a grounded answer + citations.

**Input:**  `{"query": "Find the policy for prior authorization on imaging."}`
**Output:** `{answer, rewritten_query}` plus citations + confidence.

### 2. `claim_triage` — denial → next steps

```
START → classify → retrieve → generate_checklist → ground_check → END
                       ↑                                 │
                       └── (loop once if grounding < 0.25)
```

- **classify**: rule-based regex → one of six denial categories.
- **retrieve**: query templated from the classification + user's
  question. On the loop-back, the query is broadened.
- **generate_checklist**: chat provider drafts an ordered list of
  next-step actions.
- **ground_check**: token-overlap score; flips an `uncertainty_flag`
  when low. Causes one retrieval loop if needed.

**Input:**  `{"claim_summary": "...", "question": "..."}`
**Output:** `{classification, classification_rationale, next_steps,
            uncertainty_flag, grounding_score, answer}`.

### 3. `policy_comparison` — diff two policies

```
START → retrieve_a → retrieve_b → compare → END
```

- **retrieve_a / retrieve_b**: separate retrievals per document title.
  If we can't find chunks tagged with the exact title, we fall back to
  hybrid hits across the corpus.
- **compare**: the chunks from both documents are labeled with their
  source title (e.g. `[Appeal Process Policy] First-level appeals must…`)
  before being handed to the chat provider. The LLM produces a
  dimension-by-dimension comparison.

**Input:**  `{"document_a_title": "...", "document_b_title": "...", "focus": "..."}`
**Output:** `{document_a_title, document_b_title, summary, differences}`.

### 4. `compliance_checklist` — generate-then-validate

```
START → retrieve → generate_checklist → validate → END
```

- **retrieve**: chunks for the topic.
- **generate_checklist**: chat provider drafts items.
- **validate**: rule-based grounding check per item — drop items that
  share no content words with the retrieved chunks. Surface the count
  in `validated_item_count`.

**Input:**  `{"topic": "filing a denied-claim appeal"}`
**Output:** `{topic, items, validated_item_count, answer}`.

## Guardrails (defense-in-depth)

The same three layers from Phase 4 apply here:

1. **System prompt** — every LLM call goes through the Phase 4
   `generate()` which carries the healthcare-specific system prompt
   (no medical/legal advice, cite every claim, refuse on insufficient
   context).
2. **Server-side citation validation** — citations whose `chunk_id`
   doesn't match a retrieved chunk are dropped, same as `/rag/ask`.
3. **Workflow-level grounding checks** — claim_triage flags low-
   grounding outputs; compliance_checklist drops un-grounded items
   outright; policy_comparison labels chunks by source so the LLM has
   what it needs to attribute claims correctly.

The `agent_max_steps` config setting and LangGraph's `recursion_limit`
prevent a buggy workflow from looping forever.

## Latency reality check

Each LLM-touching node is one round-trip to the chat provider. On
local Ollama (`llama3.2:3b` CPU), that's 30–360s per call.

- `policy_lookup`: 1 LLM call.
- `claim_triage`: 1 LLM call (2 if the reflection loop fires).
- `policy_comparison`: 1 LLM call.
- `compliance_checklist`: 1 LLM call.

So even worst-case, a workflow is bounded by ~2 LLM round-trips. With
`LLM_PROVIDER=mock` (the test default), all workflows complete in
<100ms.

**Recommendation for dev:** set `LLM_PROVIDER=mock` in `.env` while
working on Phase 5 / Phase 6. Switch to Ollama (or OpenAI) only when
you want a realistic demo.

## Mapping back to job-spec language

| Spec requirement | Where it lives |
|---|---|
| Multi-step tool use | LangGraph nodes calling tool functions |
| Structured outputs | Pydantic models in [schemas/agents.py](../backend/app/schemas/agents.py) |
| Reflection step | `ground_check` node + conditional edge in `claim_triage` |
| Final answer validation | `validate` node in `compliance_checklist`; citation filter in routes |
| Planning | The workflow graphs themselves — the plan is the graph |
| Guardrails | Healthcare system prompt + retrieval scope + grounding checks |
| Tool calling | Each node maps to one tool from `tools.py` |
