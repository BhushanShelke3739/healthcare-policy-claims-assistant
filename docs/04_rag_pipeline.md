# 04 — RAG Pipeline

Retrieval-Augmented Generation (RAG) is the architectural pattern this
project is built around. This doc explains the pipeline at the level of
detail that matters when you're staring at a confused answer in
production and trying to figure out why.

> Phase 2 builds the first stage of the pipeline — **ingestion**. Phases
> 3, 4, and 5 build the later stages. This doc is written end-to-end so
> the early decisions make sense in light of what's coming.

## The pipeline at a glance

```
┌──────────────┐   ┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐
│  Document    │ → │  Chunk     │ → │  Embed      │ → │  Retrieve    │ → │  Generate  │
│  (PDF / TXT) │   │  (split)   │   │  (vector)   │   │  (top-k)     │   │  (LLM)     │
└──────────────┘   └────────────┘   └─────────────┘   └──────────────┘   └────────────┘
       ↑                  ↑                ↑                  ↑                 ↑
   Phase 2            Phase 2          Phase 3            Phase 3           Phase 4
                                                          + Hybrid
                                                          (FTS + vector)
```

A user question (e.g., *"Can a denied claim be appealed after 30 days?"*)
flows through **Retrieve → Generate** at query time. Everything left of
the dashed line happens **once at ingest time** per document.

## Stage 1 — Document loading (Phase 2)

The loader takes raw bytes and returns one normalized string.

- `.txt` / `.md`: UTF-8 decode with replacement for stray bytes.
- `.pdf`: per-page text extraction via `pypdf`, joined with double
  newlines so downstream chunking treats page boundaries as paragraph
  boundaries.

What the loader does **not** do:
- OCR (a Phase 9/10 enhancement when we plug Document AI / Tesseract in).
- Strip headers/footers (we lean on chunking + retrieval to be robust
  against this rather than pre-processing, because page-furniture rules
  differ document-to-document).
- Detect tables. (Phase 9 candidate — table chunks have very different
  retrieval characteristics from prose.)

> **Code reference:** [backend/app/services/document_loader.py](../backend/app/services/document_loader.py)

## Stage 2 — Chunking (Phase 2)

Chunking is the most impactful step for retrieval quality. The single
parameter that matters most is **chunk size**, with **overlap** a close
second.

### What chunking is

A document gets split into smaller pieces ("chunks") that get embedded
and searched independently. We chunk because:

1. **Embeddings encode meaning at a scale.** A 30-page policy embedded
   as a single vector smears every topic into one point — retrieval
   degenerates to a popularity contest. A handful of paragraph-sized
   chunks lets retrieval pick the most relevant *passage*, not the most
   relevant document.
2. **LLM context windows are finite (and expensive).** Even with
   100k-token windows, you don't want to stuff irrelevant pages in —
   that's straight-up paying for noise.
3. **Citations need granularity.** A user wants to see the specific
   paragraph that answers their question, not a 5-page PDF link.

### Why recursive character chunking

A naive splitter ("every 800 characters") will routinely cut sentences
in half — terrible for both embedding quality and citation usefulness.

**Recursive character chunking** tries a *hierarchy* of separators,
biggest semantic unit first:

1. Paragraph breaks (`\n\n`)
2. Line breaks (`\n`)
3. Sentence ends (`. `)
4. Word boundaries (` `)
5. Individual characters (last resort)

For each piece that's still bigger than `chunk_size`, fall back to the
next finer separator and recurse. The result respects natural
boundaries whenever possible.

> **Code reference:** [backend/app/services/chunking.py](../backend/app/services/chunking.py)

### Why chunk size matters

| Chunk size | Effect on retrieval                                                                                          |
|------------|--------------------------------------------------------------------------------------------------------------|
| Too small  | Each chunk lacks the surrounding context that makes it interpretable — embeddings become noisy, recall drops |
| Just right | Each chunk is a self-contained passage about one idea                                                        |
| Too large  | Embeddings smear multiple topics together → precision drops; LLM prompt gets bloated                         |

The Phase 2 default is **800 characters** with **120-character overlap**.
For our policy documents (most paragraphs 200–600 chars), that lands
each chunk at one or two paragraphs — usually the natural unit a
policy answer lives in. The right value is corpus-specific and is one
of the first things we revisit during Phase 6 evaluation.

### What overlap does

Without overlap, a sentence that straddles a chunk boundary is half in
chunk N and half in chunk N+1. If the answer happens to live in that
straddling sentence, *neither* chunk reads cleanly. Overlap fixes that
by repeating the last `chunk_overlap` characters of each chunk at the
start of the next.

Common pitfalls:

- **Overlap too small (<50 chars):** rarely helps; you still cut
  important phrases.
- **Overlap too large (>30% of chunk_size):** chunks get repetitive,
  storage and retrieval cost climbs, and top-k results become
  duplicates of each other.

A 10–20% overlap is the usual sweet spot. The Phase 2 default is 120/800
= 15%.

### Metadata preservation

Every chunk carries a metadata dict that includes:

- `document_title`, `document_type`
- `source_file_name`, `source_extension`
- `page_count` (for PDFs)
- Anything the caller of `chunk_document` adds

This metadata is what makes citations possible in Phase 4 — when the
retriever returns chunk N, we already know which document and section it
came from without re-querying.

## Healthcare-specific chunking considerations

Policy documents have failure modes generic-corpus advice misses:

1. **Don't split a clause from its qualifier.** *"Members may file an
   appeal within 60 days"* — if the next sentence is *"unless the denial
   was for emergency services, in which case the timeframe is 90 days"*,
   landing those two sentences in different chunks is actively
   dangerous. Larger chunk sizes (800+) help; overlap helps more.

2. **Treat tables atomically.** A denial-code table (HF-001 through
   HF-091 in our synthetic data) should ideally live in one chunk.
   Phase 2 doesn't detect tables; we mitigate by keeping chunks big
   enough to typically swallow short tables whole. Phase 9 adds
   table-aware extraction.

3. **Section boundaries matter.** A user question about "appeals" should
   not retrieve a chunk that ends mid-appeals and starts mid-billing.
   The recursive splitter's preference for `\n\n` lines up with the
   numbered-section layout of our synthetic policies.

4. **Numbers and units must survive.** *"within 60 calendar days of the
   denial notice"* must stay together. Word-level fallback handles this;
   character-level fallback risks splitting *"60"* from *"days"*.

5. **Stamp policy provenance on every chunk.** Operations staff are
   answering specific questions for specific cases — the answer is only
   actionable if they can name the policy it came from. Metadata
   propagation is the cheap insurance against this.

## Stage 3 — Embeddings & vector storage (Phase 3)

An **embedding** is a fixed-size vector of floats that represents the
*meaning* of a piece of text. Two texts about the same topic land close
together in vector space; unrelated texts land far apart. "Close" is
measured by cosine similarity (or its complement, cosine distance).

> *Mental model:* PCA / word2vec turned up to 1500-ish dimensions and
> trained on the whole internet. The geometry isn't human-interpretable,
> but distances in that space track meaning surprisingly well.

### Pluggable providers

The `EmbeddingProvider` Protocol in
[backend/app/services/embeddings.py](../backend/app/services/embeddings.py)
hides the model choice behind one method:

```python
provider.embed(["text one", "text two"]) -> list[list[float]]
```

Built-in implementations:

| Provider | When to use                                                     | Cost           |
|----------|-----------------------------------------------------------------|----------------|
| `mock`   | Tests, CI, local dev without keys. Deterministic hashed BoW.    | Free, offline  |
| `openai` | Real semantic quality. Works with any OpenAI-compatible server. | API calls $/M  |

The mock is good enough to verify the retrieval plumbing — two texts
sharing tokens land closer than unrelated texts — but won't catch
paraphrases ("appeals" vs. "challenge"). Swap to `openai` (or the
sentence-transformers provider in a later phase) when you need real
semantic quality.

### Why store embeddings in pgvector

| Option              | Trade-off                                                          |
|---------------------|---------------------------------------------------------------------|
| pgvector (chosen)   | Same DB as the rest of the data → one transaction, no extra ops.    |
| Dedicated vector DB | Better at billions of vectors; another service to operate.          |
| Search engines (ES) | Fine for ~millions; another service; uneven hybrid story.           |

For corpora that fit on one machine, pgvector is the simplest "good
enough" answer. The `vector_store` module is intentionally thin so a
later phase can swap the backend without touching ingestion or
retrieval.

### Indexing — HNSW vs IVFFlat

pgvector supports two ANN index types:

- **HNSW** (Hierarchical Navigable Small World) — better recall at the
  same speed budget; works on empty tables. We use this.
- **IVFFlat** — older; requires data in the table at build time.

We create the HNSW index in Alembic migration `0002` with cosine ops:

```sql
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

Default `m=16`, `ef_construction=64` are fine at our scale. Bump
`ef_construction` to ~200 for higher recall on very large corpora.

### Backfill workflow

Phase 1 created the `embedding` column nullable; Phase 2 left it as
NULL during ingestion. Phase 3 wires `get_embedder()` into the
ingestion pipeline so all *new* uploads embed automatically, and
provides a CLI for the existing rows:

```bash
python -m app.backfill_embeddings              # only NULL embeddings
python -m app.backfill_embeddings --all        # re-embed everything
```

Use `--all` when you change `EMBEDDING_PROVIDER` (e.g. mock → openai)
or `EMBEDDING_MODEL` — old vectors are incompatible with new queries.

## Stage 4 — Retrieval — vector, keyword, hybrid (Phase 3)

### Vector search

Embed the query with the same provider used at ingest time, then ask
Postgres for the chunks whose stored embeddings are closest in cosine
space. The pgvector SQLAlchemy adapter exposes
`DocumentChunk.embedding.cosine_distance(query_vec)` as an order-by
expression.

```python
distance = DocumentChunk.embedding.cosine_distance(query_vec)
stmt = select(DocumentChunk, distance).order_by(distance).limit(top_k)
```

We translate distance → similarity = 1 − distance so higher always
means better in the API response.

### Keyword search

Stage 3's Alembic migration adds a generated `tsv` column on
`document_chunks`:

```sql
ALTER TABLE document_chunks
ADD COLUMN tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
```

`GENERATED ALWAYS … STORED` means Postgres keeps `tsv` in sync with
`chunk_text` on every insert/update. The application never writes to
it. A GIN index makes lookups fast.

At query time:

```python
ts_query = func.plainto_tsquery('english', query_text)
score = func.ts_rank_cd(DocumentChunk.tsv, ts_query)
stmt = select(DocumentChunk, score).where(
    DocumentChunk.tsv.op("@@")(ts_query)
).order_by(score.desc())
```

### Hybrid search and why it matters in healthcare

Vector search shines on paraphrase, synonym, and topical match. It
stumbles on **identifiers** — codes (HF-022, CPT 99213), modifiers
(95, GT), drug names. Embedding models tend to represent
`HF-022` similarly to other capital-letter-and-digit strings; that's
not what a user asking *"What does HF-022 mean?"* wants.

Keyword search has the opposite profile: perfect on identifiers,
weak on paraphrase. *"Can I challenge a rejected claim?"* with no
"appeal" / "denied" tokens scores poorly on FTS even though the
appeals doc is the right hit.

**Hybrid** runs both and combines scores:

```
hybrid_score = alpha * normalized_vector_score
             + (1 - alpha) * normalized_keyword_score
```

Each side is min-max normalized to [0, 1] over its own results before
combining — otherwise the FTS rank (typically < 1) and the cosine
similarity (~0–1 but distributed differently) aren't on the same scale.
We over-fetch (4× top_k from each side) and re-rank, so a chunk that
ranks 30th in vector but 1st in keyword can still surface.

The default `HYBRID_ALPHA=0.6` leans on semantics while leaving real
weight for exact-identifier matches. Tune it per corpus during
Phase 6 evaluation.

#### Healthcare-specific reasons hybrid wins

1. **Denial / modifier / CPT codes** are dense identifiers; vector
   alone misses them often.
2. **Policy section numbers** ("Section 3.4") often phrase the legal
   constraint — keyword match keeps these reachable.
3. **Drug names** are often non-words to general embedders; keyword
   indexing keeps them findable.
4. **Compliance language** ("minimum necessary") is a regulated phrase
   — users will type it verbatim and expect a verbatim match.

### Component scores in the response

The `/rag/retrieve` response includes a `component_scores` dict per
result:

```json
"component_scores": {"vector": 0.81, "keyword": 0.42, "alpha": 0.6}
```

Great for debugging *why* a chunk surfaced — if a result is mostly
keyword-driven, you'll see it.

## Stage 5 — Grounded answer generation (Phase 4)

### What grounded generation means

The LLM is given the retrieved chunks as context and asked to answer
*only* from them. That sounds obvious. It's also the line between RAG
that works and RAG that hallucinates.

We enforce grounding in three places, defense-in-depth:

1. **System prompt** ([backend/app/services/generation.py](../backend/app/services/generation.py))
   — explicit rules: use only the provided context, cite every claim,
   refuse with a fixed phrase when context is insufficient, no medical
   or legal advice.
2. **Mock provider behavior** — when run without a real LLM, the mock
   provider literally returns the top retrieved chunk so there's no
   path to fabrication.
3. **Server-side citation validation** ([backend/app/api/routes_rag.py](../backend/app/api/routes_rag.py))
   — the model might cite a `chunk_id` it hallucinated. We drop any
   citation whose `chunk_id` isn't in the retrieved set and note the
   drop in `grounding_notes`.

### Structured output

The LLM is asked to return JSON matching `GeneratedAnswer`:

```python
class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[_LLMCitation]   # chunk_id (str) + excerpt (str)
    confidence: Literal["low", "medium", "high"]
    grounding_notes: str
```

Two ways the request to the LLM enforces this:

- **`strict` mode (OpenAI proper).** Uses `client.beta.chat.completions.parse(response_format=GeneratedAnswer)`.
  OpenAI's server constrains generation to a JSON Schema derived from
  the Pydantic model. Parsing cannot fail.
- **`json_object` mode (Ollama / vLLM / LM Studio).** Uses
  `response_format={"type": "json_object"}` which just forces JSON-only
  output. We parse + validate against `GeneratedAnswer` ourselves, with
  one retry on failure (the retry gets a corrective message).

The two modes are selected via `LLM_STRUCTURED_OUTPUT` env var. Default
is `json_object` because it's the lowest common denominator that works
across all OpenAI-compatible servers.

### Refusal path

Two ways the request short-circuits to the refusal phrase ("I could not
find this in the available policy documents."):

1. Retrieval returned no chunks. Skip the LLM call entirely.
2. Retrieval returned chunks but all scored below `REFUSAL_SCORE_FLOOR`
   (default `0.0` — disabled). Bumping this to, say, `0.3` is a useful
   guardrail when running against the mock embedder.

### Healthcare-specific guardrails

The system prompt explicitly prohibits:

- **Medical advice.** The assistant can explain what a policy says
  about a service category but cannot recommend treatment.
- **Legal advice.** It can describe an appeals timeline but cannot
  advise the user how to act outside what the policy specifies.
- **Decision-making for the user.** "The policy specifies a 60-day
  filing window" is fine. "You should file by next Tuesday" is not.

These are not just style notes — in a real healthcare deployment they
map directly to regulatory + liability concerns.

### Audit log

Every `/rag/ask` writes a `QueryLog` row (see [03_database_design.md](03_database_design.md))
containing `user_query`, `answer`, `retrieved_chunk_ids`, `latency_ms`,
and `model_name`. That gives us replay (rerun a question against a new
model or new chunks), cost analysis, and the seed of a Phase 6
evaluation dataset.

### Switching to Ollama (or another OpenAI-compatible server)

The same code path serves OpenAI proper and self-hosted servers. The
`openai` Python client only cares about `OPENAI_BASE_URL`:

| Provider     | `OPENAI_BASE_URL`                      | `OPENAI_API_KEY`     |
|--------------|-----------------------------------------|----------------------|
| OpenAI       | `https://api.openai.com/v1` (default)   | real key (`sk-...`)  |
| Ollama       | `http://localhost:11434/v1`             | any non-empty string |
| vLLM         | `http://<host>:8000/v1`                 | any non-empty string |
| LM Studio    | `http://localhost:1234/v1`              | any non-empty string |
| llamafile    | `http://localhost:8080/v1`              | any non-empty string |

For Ollama:

```bash
ollama pull llama3.2:3b                  # or a larger model if you can
ollama pull nomic-embed-text             # only if you want real embeddings too
```

```env
# .env
LLM_PROVIDER=openai
LLM_MODEL=llama3.2:3b
LLM_STRUCTURED_OUTPUT=json_object        # Ollama doesn't support strict json_schema
OPENAI_API_KEY=ollama                    # any non-empty value
OPENAI_BASE_URL=http://localhost:11434/v1
```

If you also want Ollama embeddings (vs. mock or OpenAI), the embedding
dimension changes (`nomic-embed-text` is 768, not 1536). Rebuild the
embedding column:

```bash
python -m app.recreate_embedding_column 768
# Then update .env: EMBEDDING_PROVIDER=openai, EMBEDDING_MODEL=nomic-embed-text,
# EMBEDDING_DIMENSIONS=768
python -m app.backfill_embeddings --all
```

## Stage 6 — Reranking (later phase — optional)

After retrieving the top-K (say 20) chunks, an optional reranker scores
each candidate against the query at higher fidelity — typically a
cross-encoder or an LLM-as-judge. The top-N (say 5) are then handed to
the generator.

Reranking is cheap insurance for high-stakes answers and is the easiest
single lever for improving precision once vector + hybrid are tuned.

## Evaluation — closing the loop (Phase 6)

The point of writing this pipeline as discrete, observable stages is so
each stage can be evaluated independently:

- **Retrieval hit rate** — for a known good question, did the right
  chunk land in top-K?
- **Context precision / recall** — how much of the retrieved context is
  actually relevant?
- **Faithfulness** — does the answer stay within the retrieved context?
- **Hallucination flag** — does the answer contain claims not supported
  by the context?

Tuning chunk size, overlap, top-K, and retrieval weights without an
evaluation harness is guesswork. With one, it's an engineering exercise.
