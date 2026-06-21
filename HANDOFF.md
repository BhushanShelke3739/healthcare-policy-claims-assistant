# Handoff — Healthcare Policy & Claims Assistant (HPCA)

> **For a new AI assistant session:** paste this entire file into the
> first message. It is the smallest self-contained context that lets
> you pick up where the previous session left off without re-deriving
> anything from scratch.
>
> **For a human teammate:** start at "How to run it" and then read
> [docs/01_project_overview.md](docs/01_project_overview.md).
>
> **Last updated:** 2026-06-21 (end of Phase 9). Always update this
> doc at the end of a working session.

---

## 30-second project pitch

A **production-style RAG portfolio project** that helps a healthcare
operations team (claims billing, compliance) ask plain-English
questions of their internal policy documents and get **grounded
answers with citations**. Layered on top: **agent workflows** (claim
denial triage, policy comparison, compliance checklists), a real
**evaluation harness** with persisted runs and metrics, and a small
**Next.js demo UI**.

Built phase-by-phase for **resume / GitHub portfolio / interview
discussion** value. The user is intentionally using this to learn how
each component fits together; explanations and design rationale are
as important as working code.

**Synthetic data only.** Six invented policy docs in
[sample_data/policies/](sample_data/policies/). No real PHI, PII,
patient records, or claim IDs anywhere.

---

## Current state (end of Phase 9)

9 of 11 phases done. Tests are green (135 passing). Backend + frontend + DB all work end-to-end. Real Ollama integration confirmed working (`llama3.2:3b` answer with citations). `mypy app` and `ruff check` are both clean.

| # | Phase | Status | What it added |
|---|---|---|---|
| 1 | Foundation | ✅ | FastAPI factory, /health + /readiness, Postgres 16 + pgvector via Docker Compose, SQLAlchemy 2.x ORM, Alembic, JSON logging, pydantic-settings. |
| 2 | Document ingestion | ✅ | `POST /documents/upload`, `.txt`/`.md`/`.pdf` loaders, recursive chunking, 6 synthetic policies, `python -m app.seed_policies`. |
| 3 | Embeddings + retrieval | ✅ | Pluggable embedding providers (mock + OpenAI-compatible), HNSW vector index, FTS `tsv` GENERATED column + GIN index, vector/keyword/hybrid retrieval, `POST /rag/retrieve`. |
| 4 | Grounded answer generation | ✅ | `POST /rag/ask` with system prompt + citations + refusal path + server-side citation validation + QueryLog audit. Works with OpenAI / Ollama / vLLM / LM Studio. |
| 5 | Agentic workflows (LangGraph) | ✅ | `POST /agents/run` with 4 workflows: `policy_lookup`, `claim_triage` (with reflection loop), `policy_comparison`, `compliance_checklist`. |
| 6 | Evaluation framework | ✅ | 18-question synthetic eval set, `POST /eval/run` + `/eval/runs` + `/eval/runs/{id}`, 8 metrics. **Caught a real `plainto_tsquery` bug** (kw hit rate 6.2% → 87.5% after switching to `websearch_to_tsquery` + OR-joined content words). |
| 7 | Next.js demo frontend | ✅ | 5 pages: overview / documents / ask / agents / eval. TypeScript + Tailwind. Backend gained CORS middleware. |
| 8 | Testing & quality | ✅ | ruff + black + mypy + pre-commit + pytest-cov configured via `pyproject.toml`; frontend ESLint + Prettier + `npm run check`; `/documents/*` API tests + util tests; [docs/07_testing_strategy.md](docs/07_testing_strategy.md). |
| 9 | Observability & security | ✅ | Request-ID middleware (`X-Request-ID` in/out) + latency timing + `request_completed` logs; `/metrics` (prometheus-client, custom registry); retrieval scores persisted to new `query_logs.details` JSONB (Alembic 0004); catch-all 500 JSON error envelope with request_id; JWT + RBAC placeholders (`auth_enabled=false` default) + demo `/auth/token` & `/auth/me`; [docs/09_security_and_privacy.md](docs/09_security_and_privacy.md). |
| 10 | Deployment | ⏳ | **NEXT.** GitHub Actions CI workflow, prod Dockerfile (multi-stage), Cloud Run deployment notes, optional GCP architecture diagram. |
| 11 | Docs polish | ⏳ | Root README polish, screenshots, `docs/10_learning_notes.md` (REST / pgvector / RAG / LangGraph / eval explained for beginners), resume bullets. |

---

## How to run it (live developer setup)

**OS / shell:** Windows 11 + PowerShell. Default to PowerShell-native command syntax (`Invoke-RestMethod`, `$env:NAME`, backtick line continuation). Bash-style escapes (`-d '{\"foo\": ...}'`) break here.

**Python:** 3.10.11 in a venv at `.venv/`. Code is written to support 3.10+; pyproject.toml targets `py310` so ruff doesn't apply 3.11-only idioms like `datetime.UTC`.

### Ports

| Port | What | When | URL |
|---|---|---|---|
| 5432 | Postgres + pgvector | `docker compose up -d db` | (internal) |
| 8000 | FastAPI backend | `uvicorn app.main:app --reload` from `backend/` | <http://localhost:8000>, [/docs](http://localhost:8000/docs) |
| 3000 | Next.js frontend | `npm run dev` from `frontend/` | <http://localhost:3000> |
| 8081 | Adminer DB browser | `docker compose --profile tools up -d adminer` | <http://localhost:8081> |
| 11434 | Ollama | Background service (Windows installer auto-starts it) | <http://localhost:11434> |

### Three-terminal demo loop

```powershell
# Terminal 1 — DB
docker compose up -d db

# Terminal 2 — backend (in backend/ with .venv active)
uvicorn app.main:app --reload

# Terminal 3 — frontend (in frontend/)
npm run dev
```

Then open <http://localhost:3000>.

### `.env` at repo root

Current state on this machine (no secrets here — Ollama API key is literally the string `"ollama"`):

```env
POSTGRES_HOST=localhost
DATABASE_URL=postgresql+psycopg2://hpca:hpca_password@localhost:5432/hpca

# LLM — Ollama via OpenAI-compatible API
LLM_PROVIDER=openai
LLM_MODEL=llama3.2:3b
LLM_STRUCTURED_OUTPUT=json_object
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1

# Embeddings — Ollama nomic-embed-text (768 dim)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

**Critical convention:** `pytest` has an autouse fixture in
[backend/tests/conftest.py](backend/tests/conftest.py) that **forces
mock providers regardless of `.env`**. So tests stay green even when
Ollama is down. For development, recommend setting
`LLM_PROVIDER=mock` in `.env` during phases that don't specifically
need real generation (Phase 8 / Phase 9 dev loops are much faster
that way).

### Other knobs worth knowing

- `HYBRID_ALPHA=0.6` (default) — weight on vector score in hybrid retrieval. Can be overridden per-request to `/eval/run` and `/rag/retrieve` without restarting.
- `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120` — set at config-load time; changing requires re-seed (`python -m app.seed_policies --replace`).
- `CORS_ORIGINS` — list of allowed origins for the frontend. Defaults allow `http://localhost:3000`.

---

## Repo layout

```
healthcare-policy-claims-assistant/
├── README.md                   # human-facing entry point
├── HANDOFF.md                  # this file
├── docker-compose.yml          # db, backend, adminer (profile=tools)
├── .env.example
├── .pre-commit-config.yaml
│
├── backend/
│   ├── pyproject.toml          # ruff + black + mypy + coverage config
│   ├── pytest.ini
│   ├── requirements.txt        # runtime
│   ├── requirements-dev.txt    # ruff, black, mypy, pre-commit, pytest-cov
│   ├── Dockerfile
│   ├── alembic/                # migrations 0001 (schema), 0002 (FTS+HNSW), 0003 (eval cols), 0004 (query_logs.details)
│   ├── app/
│   │   ├── main.py             # FastAPI factory + CORS + RequestContext middleware + error handlers + routers
│   │   ├── core/               # config, logging (+request_id filter), observability (P9), errors (P9), security (JWT+RBAC, P9)
│   │   ├── db/                 # session, base, models (User, Document, DocumentChunk, QueryLog(+details), EvaluationRun, EvaluationResult)
│   │   ├── api/                # routes_health, routes_metrics (P9), routes_auth (P9), routes_documents, routes_rag, routes_agents, routes_eval
│   │   ├── schemas/            # pydantic request/response per domain
│   │   ├── services/
│   │   │   ├── chunking.py     # recursive char chunking
│   │   │   ├── document_loader.py   # .txt/.md/.pdf
│   │   │   ├── embeddings.py   # MockEmbeddingProvider + OpenAIEmbeddingProvider
│   │   │   ├── generation.py   # MockChatProvider + OpenAIChatProvider (strict + json_object modes)
│   │   │   ├── ingestion.py    # load → chunk → embed → persist orchestrator
│   │   │   ├── retrieval.py    # vector_search, keyword_search (websearch_to_tsquery OR), hybrid_search
│   │   │   ├── evaluation.py   # 8 metrics + run orchestration
│   │   │   └── agents/         # state.py, tools.py, workflows.py (4 LangGraph graphs), runner.py
│   │   ├── utils/              # text, ids
│   │   ├── seed_policies.py    # CLI: python -m app.seed_policies
│   │   ├── backfill_embeddings.py
│   │   └── recreate_embedding_column.py
│   └── tests/                  # ~120 tests; autouse mock fixture; transactional SAVEPOINT pattern
│
├── frontend/                   # Next.js 14 App Router + TS + Tailwind
│   ├── package.json            # npm run dev / build / check (lint+format+typecheck)
│   ├── src/
│   │   ├── lib/                # api.ts (fetch wrapper), types.ts (mirror backend pydantic)
│   │   ├── components/         # Nav, Card, ConfidenceBadge, CitationCard, ErrorBanner
│   │   └── app/                # layout + globals.css + page.tsx + documents/ + ask/ + agents/ + eval/
│   └── README.md
│
├── sample_data/
│   ├── policies/               # 6 synthetic .txt files
│   ├── claims/                 # README only (Phase 5 used these conceptually)
│   └── eval_questions/         # healthcare_policy_eval.json (18 items: 16 in-scope + 2 refusal)
│
└── docs/
    ├── 01_project_overview.md
    ├── 02_architecture.md
    ├── 04_rag_pipeline.md      # ingestion, chunking, embeddings, hybrid, generation
    ├── 05_agentic_workflows.md # RAG vs agentic, LangGraph, per-workflow walk
    ├── 06_evaluation.md        # metrics, iteration loop, why eval matters
    └── 07_testing_strategy.md  # test pyramid, mocking, isolation patterns
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI 0.115 + uvicorn | Async + sync; pydantic everywhere; free OpenAPI |
| DB | Postgres 16 + pgvector | Same store for relational + vectors → single transaction ingest; HNSW index |
| ORM | SQLAlchemy 2.x | Modern `Mapped[...]` typing; mypy-friendly |
| Migrations | Alembic | Three migrations: 0001 schema, 0002 FTS+HNSW, 0003 eval columns |
| Config | pydantic-settings | One Settings class; loads from `.env`; resolved by file path so works from any CWD |
| Embeddings | Mock (default) OR OpenAI-compatible (Ollama / OpenAI / vLLM / LM Studio) | Pluggable Protocol |
| LLM | Mock (default) OR OpenAI-compatible | Same pluggable Protocol; strict json_schema for OpenAI, json_object for Ollama |
| RAG | Built from scratch (no LangChain in core path) | Smaller dep surface; clearer code |
| Agents | LangGraph 0.2+ | StateGraph per workflow; built per-request to close over `db` |
| Frontend | Next.js 14 App Router + TS + Tailwind v3 | App Router; no shadcn / no tanstack-query (keeps deps minimal) |
| Tests | pytest + httpx | Autouse mock fixture for hermetic runs; SAVEPOINT transactional isolation |
| Quality | ruff + black + mypy + pre-commit + pytest-cov | All configured via `pyproject.toml` |

---

## Conventions and gotchas (the institutional knowledge)

**Default to PowerShell in command examples.** Bash-style `curl -d '{\"foo\": ...}'` doesn't survive PowerShell's argument quoting. Use either `Invoke-RestMethod` with a hashtable + `ConvertTo-Json`, or `curl.exe` (not the PowerShell alias) with proper single-quoted JSON. Same goes for env vars (`$env:NAME = "x"`, not `export`), backtick (`` ` ``) for line continuation (not backslash), and here-strings (`@'...'@`, not heredocs).

**Avoid inline comments in `.env`.** Keep comments on their own lines. python-dotenv mostly handles inline `# comment` fine but it's fragile and caused confusion once already.

**The autouse mock fixture in `tests/conftest.py` is load-bearing.** It forces `EMBEDDING_PROVIDER=mock` + `LLM_PROVIDER=mock` for every test and clears `lru_cache` on `get_settings` / `get_embedder` / `get_chat_provider`. Without it, Ollama outages flake the suite. **Don't disable it.**

**Tests use real Postgres via `db_session` fixture with `join_transaction_mode="create_savepoint"`.** This means endpoints that call `db.commit()` (like `/rag/ask`, `/eval/run`) commit a SAVEPOINT — the outer transaction rolls back at fixture teardown. If you ever see test pollution, check whether your new endpoint commits and whether the savepoint pattern still holds.

**Hermetic seeding via `TEST_CORPUS_TAG`.** Retrieval tests seed docs with `document_type="_pytest_corpus_"` and pass that filter to all retrieval calls so the user's live-DB seeded policies don't leak into assertions.

**Python 3.10 local venv, 3.11 Docker image.** Keep `pyproject.toml` targets at `py310` so ruff doesn't auto-fix into 3.11-only idioms (like `datetime.UTC`). The code is otherwise version-agnostic.

**Generated `tsv` column** on `document_chunks` is `GENERATED ALWAYS … STORED`. SQLAlchemy must know this — annotated with `Computed("to_tsvector('english', chunk_text)", persisted=True)` so SQLAlchemy excludes it from INSERTs. Otherwise Postgres rejects every chunk insert.

**Embedding dimension is set at config time (`EMBEDDING_DIMENSIONS=768` for Ollama, 1536 for OpenAI text-embedding-3-small).** If you change models, run `python -m app.recreate_embedding_column <new_dim>` then `python -m app.backfill_embeddings --all`.

**`/eval/run` accepts a per-request `alpha` override** so A/B comparisons don't need a uvicorn restart.

**Server-side citation validation in `routes_rag.py`** drops citations whose chunk_id isn't in the retrieved set. This is the primary defense against LLMs hallucinating chunk_ids. The drop count is surfaced in `grounding_notes`.

**Phase 9: `RequestContextMiddleware` is added LAST in `create_app`** so it's the *outermost* layer (Starlette runs middleware in reverse of add order). It sets a `request_id` contextvar + `request.state.request_id`, times the request, records Prometheus metrics, and stamps `X-Request-ID` + `X-Process-Time-Ms` on the response. The `RequestIdLogFilter` (installed in `configure_logging`) then puts `request_id` on every JSON log line for free.

**Phase 9: the 500 error envelope reads the request id from `request.state`, not the contextvar.** On an unhandled exception the response is built by Starlette's outer `ServerErrorMiddleware` *after* our middleware's frame unwinds — so the catch-all in `core/errors.py` reads `request.state.request_id` (and the middleware deliberately does NOT reset the contextvar on the error path as a fallback). We only register a catch-all `Exception` handler; `HTTPException`/422 keep FastAPI's native `{"detail": ...}` shape because `test_documents_api.py` asserts on `detail`.

**Phase 9: Prometheus uses a custom `CollectorRegistry`** (in `observability.py`), not the global default — cleaner test isolation, and `/metrics` exposes exactly our two series. Metric labels use the **route template** (`request.scope["route"].path`), never the raw URL, so path params can't explode cardinality.

**Phase 9: auth is real but dormant (`AUTH_ENABLED=false`).** `get_current_user` returns the anonymous viewer unless auth is on; then it requires a verified JWT bearer token. No existing route is gated with `require_role` yet — the mechanism is proven by tests, ready to apply. Toggling auth in a test = `monkeypatch.setenv("AUTH_ENABLED","true")` + `get_settings.cache_clear()`.

**Phase 9 mypy fix:** `BuilderFn` in `agents/workflows.py` was `Callable[..., object]`, which hid `.invoke` on the compiled LangGraph and made `mypy app` fail at `runner.py:84`. Changed to `Callable[..., Any]` (the concrete `CompiledStateGraph` type comes from langgraph, which we already treat with `ignore_missing_imports`). `mypy app` is now clean across 47 files.

---

## Memorable anecdotes (for the portfolio writeup)

**The `plainto_tsquery` bug caught by the eval harness.** Phase 6 first A/B compared pure-keyword vs pure-vector retrieval. `pure-keyword.retrieval_hit_rate` came back as 0.062 = exactly 1/16. Investigation: `plainto_tsquery('english', 'How long do I have to file a first-level appeal?')` ANDs every content word, and no chunk contains all of them. Fix: `websearch_to_tsquery` with OR-joined terms (≥2 chars; the `>=3` initial cut killed `"hf"` which IS in the tsv for HF-022 chunks while `"022"` is dropped by the English config's missing `hword_numpart` mapping). Hit rate climbed to 0.875. Classic "manual smoke-testing wouldn't have caught this — hand-crafted queries are always shorter than real user phrasing" story.

**The 1B LLM putting the answer in `grounding_notes`.** `llama3.2:1b` interpreted the field names by their sound: *"answer"* = the punchline (just "60"), *"grounding_notes"* = the explanation ("First-level appeals must be filed within sixty calendar days…"). Fix was a one-shot worked example in the system prompt showing correct field semantics. Worth telling — illustrates how small models drift under structured output and how few-shot prompting is the right hammer.

**Mock vector + small corpus = retrieval modes converge.** When the user A/B tested mode=hybrid vs mode=vector vs mode=keyword (after the `websearch_to_tsquery` fix), all three produced identical aggregate metrics. Reason: over-fetch=20 chunks from each side, total corpus ~50, so the union covers nearly everything. Real semantic embeddings (Ollama nomic-embed-text) would differentiate. This is itself an eval finding.

---

## How to start a new working session

The user prefers a tight rhythm: confirm where we are, smoke-check the previous phase, then move forward.

**At session start, when the user says "Phase N":**

1. Acknowledge what phase is next, briefly.
2. Suggest a 30-second smoke check: `pytest -v` (should be ~120 passing) + a quick sanity hit to `/rag/ask` if Ollama is up. If `LLM_PROVIDER=mock` recommend keeping it there during dev unless the phase specifically benefits from real generation.
3. Lay out the work as a TodoWrite list before writing code.
4. Build incrementally, marking todos done as you go.
5. End with a verification script (PowerShell), a recap of what changed, and an offer of 2-3 reasonable next moves.

**During the session:**

- **PowerShell-first** in all command examples (`Invoke-RestMethod`, `$env:VAR`, backtick continuation, `curl.exe` if curl is needed).
- **Skip TodoWrite for tiny tasks** (bug fixes, README tweaks, single-file edits). It's only worth it for multi-step phases.
- **When tests fail, treat it as a portfolio-grade opportunity.** Phase 6's `plainto_tsquery` story came from exactly this. Surface the diagnosis cleanly, fix it, re-verify, and mention the framing if it applies.
- **Update this file at session end** if anything material changed: new phase status, new gotchas, new memorable anecdotes, new env config, new ports.

**The kickoff prompt for the project** (referenced multiple times by the user) is the original 11-phase plan. Each phase has a short spec; see `docs/01_project_overview.md` for the abbreviated version. Key acceptance criteria across the project:

- Synthetic data only, no PHI/PII.
- Mock providers as defaults so the system works without any API key.
- Hermetic test suite.
- One commit-worthy chunk of work per phase, ending with the user verifying it works.

---

## Phase 9 — DONE (recap)

Shipped: request-ID middleware (`X-Request-ID` accepted/generated + echoed, `X-Process-Time-Ms`), `RequestIdLogFilter` so every JSON log line carries `request_id`, `request_completed` latency logs; `/metrics` via prometheus-client (custom registry, route-template labels) — `app/api/routes_metrics.py`; retrieval-score persistence in new `query_logs.details` JSONB (Alembic 0004, written in `routes_rag.py`); catch-all 500 JSON error envelope with request_id (`app/core/errors.py`); JWT + RBAC placeholders in `app/core/security.py` (`create_access_token`/`decode_access_token`/`require_role`, `auth_enabled=false` default) with demo `/auth/token` + `/auth/me` (`routes_auth.py`); `docs/09_security_and_privacy.md`. New deps: `prometheus-client`, `PyJWT`. Tests +14 (`test_observability.py`, `test_security.py`) → 135 total; `mypy app` + `ruff` clean.

## Phase 10 — what's coming next

From the original kickoff:

```
PHASE 10: Deployment

- GitHub Actions CI workflow (lint + typecheck + test on push/PR)
- Production Dockerfile (multi-stage; the backend Dockerfile exists but
  is dev-oriented — make a slim prod build)
- Cloud Run deployment notes (build image → push to Artifact Registry →
  deploy; autoscale to zero)
- Cloud SQL for Postgres with the `vector` extension (or managed PG)
- Secret Manager for OPENAI_API_KEY / JWT_SECRET / DB password
- Optional: GCP architecture diagram (Mermaid)
- Create docs/08_deployment.md
```

When the user says "Phase 10", start with a smoke-check (`pytest -q` → 135) + TodoWrite. Recommended order: GitHub Actions CI first (codifies the lint+typecheck+test gate that already passes locally), then the multi-stage prod Dockerfile, then the Cloud Run / Cloud SQL / Secret Manager notes + diagram in `docs/08_deployment.md`. Note the CI workflow should run the hermetic test subset (the autouse mock fixture means most tests need no Postgres; the 14 integration tests skip cleanly without a DB, or spin up a `postgres:16` + pgvector service container to run them).

---

## Quick command cheat sheet

```powershell
# Backend dev loop
cd backend
uvicorn app.main:app --reload                       # http://localhost:8000
pytest -v                                            # ~120 tests
pytest --cov                                         # with coverage
ruff check --fix .
ruff format .
mypy app

# Seed / re-embed
python -m app.seed_policies                          # idempotent by file_name
python -m app.seed_policies --replace                # wipe + re-ingest
python -m app.backfill_embeddings                    # only NULL embeddings
python -m app.backfill_embeddings --all              # re-embed everything
python -m app.recreate_embedding_column 768          # change vector dim

# Frontend dev loop
cd frontend
npm install
npm run dev                                          # http://localhost:3000
npm run check                                        # lint + format:check + typecheck

# DB browsing
docker compose --profile tools up -d adminer        # http://localhost:8081

# Hit the API from PowerShell (the right way)
$body = @{ question = "How long do I have to file a first-level appeal?"; top_k = 5; mode = "hybrid" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/rag/ask -Method POST -ContentType "application/json" -Body $body |
    Format-List answer, confidence, model_name, latency_ms
```

---

## Where to read next (from this doc forward)

1. [README.md](README.md) — quick start, tech stack, roadmap.
2. [docs/01_project_overview.md](docs/01_project_overview.md) — full project vision, mapping to job requirements.
3. [docs/02_architecture.md](docs/02_architecture.md) — Mermaid system + ER diagrams.
4. [docs/04_rag_pipeline.md](docs/04_rag_pipeline.md) — RAG explained end-to-end.
5. [docs/05_agentic_workflows.md](docs/05_agentic_workflows.md) — LangGraph + per-workflow walk.
6. [docs/06_evaluation.md](docs/06_evaluation.md) — eval metrics and iteration loop.
7. [docs/07_testing_strategy.md](docs/07_testing_strategy.md) — test pyramid + isolation patterns.
8. [docs/09_security_and_privacy.md](docs/09_security_and_privacy.md) — PHI/PII risks, secure config, audit logging, access control, prompt-injection + retrieval-poisoning threats, healthcare disclaimer.

Phases 10-11 will add `08_deployment.md` and `10_learning_notes.md`.
