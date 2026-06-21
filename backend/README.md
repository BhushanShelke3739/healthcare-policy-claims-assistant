# Backend — Healthcare Policy & Claims Assistant

FastAPI service powering the assistant.

## Layout

```
backend/
├── app/
│   ├── main.py            # FastAPI app factory + lifespan
│   ├── core/              # config, logging, security
│   ├── db/                # session, declarative base, ORM models
│   ├── api/               # route modules grouped by domain
│   ├── services/          # ingestion / RAG / agents / eval (filled in by phase)
│   ├── schemas/           # Pydantic request/response schemas
│   └── utils/             # text + id helpers
├── tests/                 # pytest suite
├── alembic/               # database migrations
├── alembic.ini
├── pytest.ini
├── requirements.txt
└── Dockerfile
```

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Point at a local Postgres+pgvector (e.g. one started by docker compose)
export DATABASE_URL=postgresql+psycopg2://hpca:hpca_password@localhost:5432/hpca

alembic upgrade head
uvicorn app.main:app --reload
```

## Run via Docker Compose (recommended)

From the repository root:

```bash
docker compose up --build
docker compose exec backend alembic upgrade head
```

## Tests

```bash
# from backend/
pytest -v
```

The Phase 1 test suite covers the `/health` + `/readiness` endpoints and
verifies that the stubbed routes for later phases show up in the OpenAPI
schema. Tests use a fake DB session, so no live Postgres is required.

## Seed the synthetic policy documents (Phase 2)

After running migrations, load the six synthetic policy documents into
the database with one command:

```bash
# from backend/ with .venv activated
python -m app.seed_policies

# replace existing rows with the same file_name:
python -m app.seed_policies --replace
```

You'll see a line per document with the number of chunks it produced.
Then verify via Swagger:

- `GET /documents` — list with chunk counts.
- `GET /documents/{id}` — fetch one.

## Embeddings + retrieval (Phase 3)

After Phase 3 lands, ingestion auto-embeds new chunks. To embed existing
chunks (e.g. policies seeded before Phase 3, or after switching the
embedding provider/model):

```bash
# Embed only chunks whose embedding column is NULL:
python -m app.backfill_embeddings

# Re-embed every chunk (use after changing EMBEDDING_PROVIDER or EMBEDDING_MODEL):
python -m app.backfill_embeddings --all
```

Then try a retrieval query:

```bash
curl -s http://localhost:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do I have to appeal a denied claim?", "top_k": 3, "mode": "hybrid"}' | python -m json.tool
```

Switch `mode` between `vector`, `keyword`, and `hybrid` to compare. The
`component_scores` field in each result shows the vector / keyword
contribution.

### Switching embedding providers

Two options today: `mock` (default, deterministic, offline) and `openai`
(any OpenAI-compatible endpoint). Edit `.env`:

```
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # change to point at vLLM / Ollama / LM Studio
```

After changing the provider/model, **re-run the backfill with `--all`**
— old vectors aren't comparable to ones produced by a different model.

If the new model has a different output dimension (e.g. switching to
Ollama's 768-dim `nomic-embed-text`), rebuild the embedding column too:

```powershell
python -m app.recreate_embedding_column 768
# Update .env: EMBEDDING_DIMENSIONS=768
python -m app.backfill_embeddings --all
```

## Grounded answer generation (Phase 4)

`POST /rag/ask` takes a question, retrieves top-k chunks, and asks an
LLM to answer using *only* that context — with citations. Defaults to a
mock LLM so it works without any API key.

```powershell
$body = @{
    question = "How long do I have to file a first-level appeal?"
    top_k = 5
    mode  = "hybrid"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/rag/ask `
    -Method POST -ContentType "application/json" -Body $body
```

You'll get back `answer`, `citations` (each with `chunk_id`,
`document_title`, `excerpt`), `confidence`, `grounding_notes`,
`retrieved_chunk_ids`, `model_name`, and `latency_ms`. The mock LLM
returns the top retrieved chunk as the answer with citations to the
top-2 chunks — enough to verify the wiring works end-to-end.

### Switch to a real LLM (OpenAI, Ollama, or other OpenAI-compatible)

Edit `.env`:

For OpenAI proper:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_STRUCTURED_OUTPUT=strict
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

For Ollama (free, local — `ollama pull llama3.2:3b` first):

```env
LLM_PROVIDER=openai
LLM_MODEL=llama3.2:3b
LLM_STRUCTURED_OUTPUT=json_object
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
```

For vLLM / LM Studio / llamafile, point `OPENAI_BASE_URL` at the right URL and pick the model name they serve. Avoid inline comments after values in `.env` — keep comments on their own lines.

## Agent workflows (Phase 5)

`POST /agents/run` dispatches to one of four LangGraph workflows. The
body is uniform — `{workflow, input, top_k?}` — but `input` shape and
the response's `final_output` shape are workflow-specific. See
[docs/05_agentic_workflows.md](../docs/05_agentic_workflows.md) for
the full walkthrough.

> **Dev tip.** Each workflow chains 1–2 LLM calls. With Ollama on CPU
> (30–360s per call) a single `/agents/run` can take many minutes. For
> day-to-day dev, set `LLM_PROVIDER=mock` in `.env` — workflows run in
> <100ms with the deterministic mock. Switch back to Ollama / OpenAI
> for the real demo.

### policy_lookup

```powershell
$body = @{
    workflow = "policy_lookup"
    input    = @{ query = "Find the policy for prior authorization on imaging." }
    top_k    = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/agents/run `
    -Method POST -ContentType "application/json" -Body $body |
    Format-List workflow, final_output, confidence, model_name, latency_ms
```

### claim_triage

```powershell
$body = @{
    workflow = "claim_triage"
    input    = @{
        claim_summary = "Claim denied because prior authorization was missing for MRI."
        question      = "What should the billing team do next?"
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8000/agents/run `
    -Method POST -ContentType "application/json" -Body $body |
    ConvertTo-Json -Depth 6
```

### policy_comparison

```powershell
$body = @{
    workflow = "policy_comparison"
    input    = @{
        document_a_title = "Appeal Process Policy"
        document_b_title = "Claim Denial Policy"
        focus            = "timelines and required documentation"
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8000/agents/run `
    -Method POST -ContentType "application/json" -Body $body |
    ConvertTo-Json -Depth 6
```

### compliance_checklist

```powershell
$body = @{
    workflow = "compliance_checklist"
    input    = @{ topic = "filing a first-level appeal for a denied claim" }
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/agents/run `
    -Method POST -ContentType "application/json" -Body $body |
    ConvertTo-Json -Depth 6
```

## Evaluation harness (Phase 6)

Runs the bundled eval dataset (or a custom one) through the same
retrieval + generation pipeline used by `/rag/ask`, scores each
question on multiple metrics, persists the run, and returns a
summary. Full design in
[docs/06_evaluation.md](../docs/06_evaluation.md).

> **Dev tip.** With `LLM_PROVIDER=mock` an eval run takes under a
> second. With Ollama on CPU it's roughly `num_questions ×
> per-question-LLM-latency` (≈7 minutes for the bundled 18 items at
> 30s/call). Use mock while iterating on the harness or metrics.

```powershell
# 1. Run the bundled eval set
$body = @{ name = "baseline"; description = "first run, defaults" } | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://localhost:8000/eval/run `
    -Method POST -ContentType "application/json" -Body $body
$r.summary
$r.results | Select-Object question, hallucination_flag, faithfulness, latency_ms |
    Format-Table -AutoSize

# 2. List past runs
Invoke-RestMethod -Uri http://localhost:8000/eval/runs |
    Select-Object -ExpandProperty items |
    Format-Table id, name, created_at, num_questions

# 3. Drill into one run
Invoke-RestMethod -Uri "http://localhost:8000/eval/runs/$($r.id)" |
    Select-Object -ExpandProperty summary
```

### Compare two runs (the actual point of all this)

```powershell
# Baseline
$baseline = Invoke-RestMethod -Uri http://localhost:8000/eval/run `
    -Method POST -ContentType "application/json" `
    -Body (@{ name = "baseline-c800" } | ConvertTo-Json)

# Change CHUNK_SIZE in .env, re-seed: python -m app.seed_policies --replace
# Then:
$bigger = Invoke-RestMethod -Uri http://localhost:8000/eval/run `
    -Method POST -ContentType "application/json" `
    -Body (@{ name = "c1200-overlap-200" } | ConvertTo-Json)

# Diff the summaries
"baseline:", ($baseline.summary | ConvertTo-Json -Depth 3)
"bigger:",   ($bigger.summary   | ConvertTo-Json -Depth 3)
```

## Testing & quality (Phase 8)

Install dev tools once:

```powershell
pip install -r requirements-dev.txt
```

Then:

```powershell
# Lint (also sorts imports)
ruff check .
ruff check --fix .

# Format
ruff format .
# or: black .

# Type check (loose mode — won't fail on existing un-annotated test code)
mypy app

# Tests (already 100+ passing; ~95 hermetic, the rest skip without Postgres)
pytest -v
pytest --cov                 # with coverage report

# All of the above in one shot (after installing pre-commit hooks)
pre-commit run --all-files
```

See [docs/07_testing_strategy.md](../docs/07_testing_strategy.md) for
the test pyramid, the hermetic-mock pattern, the SAVEPOINT isolation
trick, and how to test RAG systems specifically.

### Pre-commit hook (optional, recommended)

```powershell
# Once
pip install pre-commit
pre-commit install

# Now every `git commit` runs ruff + black + mypy on the staged files.
# Run on the whole repo at any time:
pre-commit run --all-files
```

Restart uvicorn after changing `.env`. The generation code is the same
across all backends — only the URL and model name change.

## Useful URLs

| Path                     | What it does                                |
|--------------------------|---------------------------------------------|
| `/health`                | Liveness probe                              |
| `/readiness`             | Readiness probe (DB ping)                   |
| `POST /documents/upload` | Upload a `.txt` / `.md` / `.pdf` (Phase 2)  |
| `GET /documents`         | List with chunk counts (Phase 2)            |
| `GET /documents/{id}`    | Fetch one document (Phase 2)                |
| `DELETE /documents/{id}` | Delete a document and its chunks (Phase 2)  |
| `POST /rag/retrieve`     | Vector / keyword / hybrid search (Phase 3)  |
| `POST /rag/ask`          | Grounded answer + citations (Phase 4)       |
| `POST /agents/run`       | LangGraph agent workflows (Phase 5)         |
| `POST /eval/run`         | Run the evaluation harness (Phase 6)        |
| `GET /eval/runs`         | List past evaluation runs (Phase 6)         |
| `GET /eval/runs/{id}`    | One run with per-question metrics (Phase 6) |
| `/docs`                  | Swagger UI                                  |
| `/redoc`                 | ReDoc                                       |
| `/openapi.json`          | Raw OpenAPI spec                            |

## What's in each module

| Module                          | Role                                                      | Filled in by |
|---------------------------------|-----------------------------------------------------------|--------------|
| `app/main.py`                   | App factory, lifespan, router wiring                      | Phase 1 ✅   |
| `app/core/config.py`            | Pydantic-settings, env var loading                        | Phase 1 ✅   |
| `app/core/logging.py`           | JSON formatter, root logger setup                         | Phase 1 ✅   |
| `app/core/security.py`          | Placeholder `CurrentUser` + dependency                    | Phase 1 ✅   |
| `app/db/session.py`             | SQLAlchemy engine + session factory + `get_db`            | Phase 1 ✅   |
| `app/db/models.py`              | ORM models (User, Document, Chunk, QueryLog, Eval*)       | Phase 1 ✅   |
| `app/api/routes_health.py`      | `/health`, `/readiness`                                   | Phase 1 ✅   |
| `app/api/routes_documents.py`   | Upload / list / get / delete                              | Phase 2 ✅   |
| `app/services/document_loader.py` | Extract text from .txt / .md / .pdf                     | Phase 2 ✅   |
| `app/services/chunking.py`      | Recursive character chunking + metadata propagation       | Phase 2 ✅   |
| `app/services/ingestion.py`     | Load → chunk → embed → persist orchestrator               | Phase 2/3 ✅ |
| `app/services/embeddings.py`    | Pluggable embedding providers (mock / OpenAI)             | Phase 3 ✅   |
| `app/services/retrieval.py`     | Vector / keyword / hybrid search                          | Phase 3 ✅   |
| `app/services/generation.py`    | Mock + OpenAI-compatible chat providers, system prompt    | Phase 4 ✅   |
| `app/services/agents/`          | LangGraph workflows (lookup, triage, comparison, checklist) | Phase 5 ✅ |
| `app/services/evaluation.py`    | Eval harness: metrics, dataset loader, run orchestration  | Phase 6 ✅   |
| `app/seed_policies.py`          | One-command load of `sample_data/policies/`               | Phase 2 ✅   |
| `app/backfill_embeddings.py`    | CLI to populate `embedding` for existing chunks           | Phase 3 ✅   |
| `app/recreate_embedding_column.py` | CLI to rebuild `embedding` at a new vector dimension   | Phase 4 ✅   |
| `app/api/routes_rag.py`         | `/rag/retrieve`, `/rag/ask`                               | Phase 3–4    |
| `app/api/routes_agents.py`      | `/agents/run`                                             | Phase 5      |
| `app/api/routes_eval.py`        | `/eval/run`, `/eval/runs`                                 | Phase 6      |
| `app/services/*`                | Domain logic (chunking, embeddings, RAG, agents)          | Phase 2–6    |
| `app/schemas/*`                 | Pydantic request/response models                          | Phase 1+ ✅  |
| `app/utils/text.py`             | `normalize_whitespace`, `approximate_token_count`         | Phase 1 ✅   |
| `app/utils/ids.py`              | UUID helpers                                              | Phase 1 ✅   |
