# 07 — Testing Strategy

This project has ~100 tests across 11 files. This document explains how
they're organized, what each tier is *for*, the patterns that make a
RAG system testable, and how to extend the suite as the project grows.

## TL;DR

```powershell
# Backend (from backend/ with .venv activated)
pytest -v                 # run everything
pytest --cov              # with coverage report
ruff check .              # lint
ruff format .             # format
mypy app                  # type check
pre-commit run --all      # all of the above on every file

# Frontend (from frontend/)
npm run check             # lint + format check + typecheck (one command)
```

## The test pyramid

```
                 ┌──────────────────────┐
                 │  /eval/run live test │  ← few, slow, comprehensive
                 │  (manual; portfolio  │
                 │   anecdote-grade)    │
                 ├──────────────────────┤
                 │   API tests          │  ← real HTTP + real DB,
                 │   (TestClient)       │     1-2 dozen, ~1s each
                 ├──────────────────────┤
                 │   Integration tests  │  ← real DB, no HTTP,
                 │   (db_session)       │     20-30 tests, ~0.1s each
                 ├──────────────────────┤
                 │   Unit tests         │  ← pure functions, no I/O,
                 │   (chunking, util,   │     40-50 tests, <0.01s each
                 │    embeddings mock)  │
                 └──────────────────────┘
```

We deliberately have **more unit tests than integration**, and **more
integration than API**. The opposite is a common anti-pattern — every
test wrapped in `TestClient + Postgres` looks rigorous but actually
exercises the same code paths over and over while being 100× slower.

## Tier 1 — Unit tests

What they test:
- Chunking algorithm (paragraph-preferred boundaries, overlap, metadata)
- Document loader (.txt/.md/.pdf extraction, error paths)
- Mock embedding provider (determinism, L2-normalization, dimension)
- Mock chat provider (refusal path, citations, confidence calibration)
- Agent tool functions (regex classifier, grounding check, query rewrite)
- Eval metric helpers (keyword recall, content tokens, relevance check)
- Text utilities (whitespace, token count, uuid)

What makes them units:
- No database access
- No network
- No file I/O (PDF tests build PDFs in memory)
- Sub-millisecond per test

Files:
[test_chunking.py](../backend/tests/test_chunking.py),
[test_document_loader.py](../backend/tests/test_document_loader.py),
[test_embeddings.py](../backend/tests/test_embeddings.py),
[test_generation.py](../backend/tests/test_generation.py),
[test_text_utils.py](../backend/tests/test_text_utils.py).

## Tier 2 — Integration tests

What they test:
- Vector / keyword / hybrid retrieval against a real pgvector index
- Agent workflows end-to-end through LangGraph nodes
- Evaluation harness running a small custom dataset

What makes them integration:
- Run against a live Postgres + pgvector via the `db_session` fixture
- Don't go through FastAPI — call services directly
- Each test runs in a **rolled-back transaction** so it can't pollute the DB

Files:
[test_retrieval.py](../backend/tests/test_retrieval.py),
[test_agents.py](../backend/tests/test_agents.py),
[test_evaluation.py](../backend/tests/test_evaluation.py).

## Tier 3 — API tests

What they test:
- HTTP status codes (201, 404, 413, 415, 422)
- Request/response schema shape
- Multipart upload mechanics
- Cross-endpoint workflows (upload → list → get → delete)

What makes them API:
- Use FastAPI's `TestClient` (via the `client_with_real_db` fixture)
- Real Postgres underneath (same rolled-back transaction)
- Assert on JSON bodies

Files:
[test_health.py](../backend/tests/test_health.py),
[test_documents_api.py](../backend/tests/test_documents_api.py),
[test_rag_api.py](../backend/tests/test_rag_api.py),
[test_retrieval.py](../backend/tests/test_retrieval.py) (mixed),
[test_evaluation.py](../backend/tests/test_evaluation.py) (mixed),
[test_agents.py](../backend/tests/test_agents.py) (mixed).

## Tier 4 — Live evaluation

The 18-question eval set in
[sample_data/eval_questions/healthcare_policy_eval.json](../sample_data/eval_questions/healthcare_policy_eval.json)
isn't part of `pytest`. It's run manually via `POST /eval/run` and the
result is stored in Postgres. Think of it as "macro-tests" — they
validate the system as a whole rather than any specific function.

This is the tier that **caught the `plainto_tsquery` AND-joining bug**
that no amount of unit-testing would have surfaced.

## Key patterns

### 1. Hermetic test suite via autouse provider mocking

The single most important fixture in the project. From
[backend/tests/conftest.py](../backend/tests/conftest.py):

```python
@pytest.fixture(autouse=True)
def _force_mock_providers(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    get_embedder.cache_clear()
    get_chat_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_embedder.cache_clear()
    get_chat_provider.cache_clear()
```

**Why it matters:**
- `autouse=True` runs it on every test, no opt-in needed.
- Forces mock providers regardless of `.env`. So a test never depends
  on whether Ollama is up, whether an API key is present, or whether
  the network is reachable.
- Clears the `lru_cache` so the *next* call to `get_embedder()` rebuilds
  with the override in effect, then clears again on teardown so a
  follow-up `python -m app.<x>` from the same process picks up real
  `.env` values.
- The mock providers are **deterministic** — same input always
  produces the same vector / answer / classification. Without
  determinism, "did this change break anything?" becomes statistical.

### 2. Transactional isolation via SQLAlchemy SAVEPOINTs

`db_session` opens a connection, begins a transaction, and binds a
session to that connection with
`join_transaction_mode="create_savepoint"`. When the application code
calls `session.commit()` (which `/rag/ask` does, and `/eval/run`
does), it commits the SAVEPOINT — not the outer transaction. End of
test: outer transaction rolls back. Net effect: writes are visible to
the rest of the test but disappear before the next test.

Without this, an endpoint that commits would pollute the database;
the next test would see leftover rows.

### 3. Test-corpus isolation via `TEST_CORPUS_TAG`

The seeded test corpus uses `document_type="_pytest_corpus_"` so
retrieval tests can pass `document_type=TEST_CORPUS_TAG` and ignore
whatever the developer has loaded into their live DB.

Postgres's default `READ COMMITTED` isolation means our transactional
inserts can *see* already-committed rows from outside the
transaction. Filtering by tag is the cheapest way to be hermetic
without resetting the database.

### 4. Real PDFs built in memory

[test_document_loader.py](../backend/tests/test_document_loader.py)
constructs valid minimal PDFs in-memory via `pypdf.PdfWriter` so
nothing on disk has to drift over time. The pattern generalizes:
**don't checkin binary fixtures unless they're genuinely complex**.

## How to test RAG systems specifically

### Three things you can actually assert

1. **Schema** — the response has the right shape.
2. **Provenance** — citations reference chunks that were actually
   retrieved (server-side validation in
   [routes_rag.py](../backend/app/api/routes_rag.py) drops invented
   chunk_ids; tests assert the drop happened).
3. **Refusal** — when retrieval returns nothing, the system returns
   exactly the configured refusal phrase.

### Three things you can't (cheaply) assert

1. **Answer correctness** — "did the LLM give the right answer?" is
   nondeterministic on real models. Use the eval harness instead.
2. **Citation goodness** — "is this the *best* excerpt to cite?" is
   subjective.
3. **Tone / style** — same.

The pattern that emerges: unit tests for schema + provenance + refusal,
eval harness for quality. Don't try to make pytest measure quality.

### Mocking LLM calls

The `MockChatProvider` returns the top retrieved chunk verbatim as the
answer, with the top-2 chunks cited. This:

- Lets `test_rag_api` assert the citation field is populated when
  retrieval returns chunks.
- Lets `test_rag_api` assert refusal when retrieval returns nothing.
- Doesn't pretend to be a real LLM (the answer is honest about being
  mock content via `grounding_notes`).

A common alternative — recording real LLM responses and replaying
them — would be more "realistic" but break every time we change the
system prompt. Determinism > realism for unit tests.

## Coverage

Run with:

```powershell
pytest --cov
```

Coverage config lives in
[backend/pyproject.toml](../backend/pyproject.toml) and excludes CLI
scripts (`seed_policies.py`, `backfill_embeddings.py`,
`recreate_embedding_column.py`) since those are exercised manually.

We don't enforce a minimum threshold yet because that ratchet is too
easy to game (test what's easy, not what matters). Coverage is a
**signal**, not a goal. When you see a percentage drop, ask *which
lines* lost coverage and whether they should have been covered.

## Linting + formatting + type-checking

| Tool   | What it does                                  | Where configured           |
|--------|-----------------------------------------------|----------------------------|
| ruff   | Lints + sorts imports (replaces isort/flake8) | `pyproject.toml[tool.ruff]`|
| ruff format / black | Formats code                       | `pyproject.toml`           |
| mypy   | Static type check (loose mode)                | `pyproject.toml[tool.mypy]`|
| eslint | Frontend lint                                 | `frontend/.eslintrc.json`  |
| prettier | Frontend formatter                          | `frontend/.prettierrc.json`|
| tsc    | Frontend type check                           | `frontend/tsconfig.json`   |
| pre-commit | Runs ruff + black + mypy on staged files | `.pre-commit-config.yaml`  |

To install everything in one go:

```powershell
# Backend
pip install -r requirements-dev.txt

# Pre-commit hooks (optional but recommended)
pre-commit install

# Frontend
cd frontend
npm install
```

## CI considerations (Phase 10)

Phase 10 will add `.github/workflows/ci.yml`. The minimal job is:

```yaml
- ruff check .
- ruff format --check .
- mypy app
- pytest -v --cov
```

Plus the frontend job:

```yaml
- npm ci
- npm run check
- npm run build
```

Both run in <2 minutes against the test corpus. The eval harness
(Tier 4) stays a manual gate for now — running real LLMs in CI gets
expensive fast.

## Extending the suite

When you add a feature, ask **which tier covers it**:

| You added… | Test it at tier… |
|---|---|
| A pure function | Unit |
| A new SQLAlchemy query | Integration (`db_session`) |
| A new endpoint | API (`client_with_real_db`) |
| A new LLM-touching node | Add a mock-provider test at Integration |
| A new metric formula | Unit (against synthetic inputs) + the harness |
| A bug found in production | Write the test that would have caught it, *then* fix |

The last one is the most important habit. The
`plainto_tsquery` → `websearch_to_tsquery` bug only existed because we
had no test asserting that long natural-language questions returned
*any* keyword hits. Adding such a test alongside the fix prevents
regression. (Phase 6 doesn't have that exact assertion yet — it's a
good first contribution.)
