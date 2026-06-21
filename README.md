# Healthcare Policy & Claims Assistant

An end-to-end, AI-powered web application that helps healthcare operations teams
search internal policy documents, analyze insurance/claims-related questions,
generate grounded answers with citations, and run agentic workflows such as
policy lookup, claim issue triage, document comparison, and compliance
checklist generation.

> **Picking this up in a new chat or onboarding a teammate?**
> Read [HANDOFF.md](HANDOFF.md) first — it's a self-contained context
> dump (current phase, live setup, conventions, what's next) designed
> to be pasteable into a fresh assistant session.

> **Status: Phase 8 — Testing & quality complete.** Backend, frontend,
> retrieval, RAG, agents, evaluation, and quality gates all working
> end-to-end. Phases 9 (observability + security) and 10 (deployment)
> are next.

> **Healthcare disclaimer.** This project uses only **synthetic** policy and
> claims data. Do not upload real PHI, PII, or patient records. The system is
> for operational policy lookup — it does not provide medical or legal advice.

---

## Project goals

This repo is designed as a production-style portfolio project demonstrating:

- Python 3 backend development with FastAPI
- RESTful API design and OpenAPI documentation
- PostgreSQL relational modeling with SQLAlchemy + Alembic
- Vector search via pgvector
- Full RAG pipeline (ingestion → chunking → embeddings → retrieval → generation)
- Hybrid search and reranking
- Agentic workflows with planning, tool use, and reflection (LangGraph)
- Structured outputs and grounding checks
- Evaluation framework with retrieval and answer-quality metrics
- Observability, logging, monitoring
- Containerization, CI/CD, and cloud-ready deployment
- Security best practices for healthcare-adjacent systems

## Tech stack

| Layer            | Choice                                       |
|------------------|----------------------------------------------|
| Backend          | Python 3.11, FastAPI, Uvicorn                |
| Database         | PostgreSQL 16                                |
| Vector store     | pgvector extension on the same Postgres      |
| ORM / migrations | SQLAlchemy 2.x, Alembic                      |
| Config           | pydantic-settings                            |
| Testing          | pytest, httpx                                |
| Containerization | Docker, Docker Compose                       |
| RAG (later)      | LangChain / LlamaIndex                       |
| Agents (later)   | LangGraph                                    |
| Frontend (later) | React or Next.js (simple demo UI)            |

## Repository layout

```
healthcare-policy-claims-assistant/
├── backend/                  # FastAPI service
│   ├── app/
│   │   ├── main.py           # FastAPI app entry
│   │   ├── core/             # config, logging, security
│   │   ├── db/               # session, base, ORM models
│   │   ├── api/              # route modules
│   │   ├── services/         # ingestion, RAG, agents, eval (stubs in P1)
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   └── utils/            # text, id helpers
│   ├── tests/                # pytest suite
│   ├── alembic/              # DB migrations
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/                 # Optional UI (later phase)
├── sample_data/              # Synthetic policy + claims fixtures
│   ├── policies/
│   ├── claims/
│   └── eval_questions/
├── docs/                     # Architecture, learning notes
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quick start

### 1. Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Python 3.11+ if you want to run the backend outside Docker

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change defaults (DB password, ports, etc.)
```

### 3. Start the stack

```bash
docker compose up --build
```

This launches:

- `db` — PostgreSQL 16 with the pgvector extension on port `5432`
- `backend` — FastAPI on port `8000`

### 4. Apply database migrations

In a separate terminal:

```bash
docker compose exec backend alembic upgrade head
```

### 5. Verify it's working

- Open <http://localhost:8000/health> — should return `{"status": "ok", ...}`
- Open <http://localhost:8000/docs> — Swagger UI for the API
- Open <http://localhost:8000/redoc> — ReDoc API documentation

### Optional: browse the database with Adminer

Adminer is an opt-in dev tool (not part of the application) for inspecting
tables, running ad-hoc SQL, and watching `document_chunks` / `query_logs`
fill up while you work.

```powershell
docker compose --profile tools up -d adminer
```

Then open <http://localhost:8081> and log in:

| Field    | Value         |
|----------|---------------|
| System   | PostgreSQL    |
| Server   | `db`          |
| Username | `hpca`        |
| Password | `hpca_password` |
| Database | `hpca`        |

(Server is pre-filled to `db` thanks to `ADMINER_DEFAULT_SERVER` in the
compose file. Other fields are the defaults from `.env.example`.)

Stop it again with `docker compose stop adminer`.

### Optional: run the Next.js demo frontend

Five-page UI for the API. Lives in [frontend/](frontend/), requires
Node 20+ and npm.

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Then open <http://localhost:3000>. The backend must be running on
port 8000. See [frontend/README.md](frontend/README.md) for the full
walkthrough.

### Running tests

```bash
docker compose exec backend pytest -v
```

Or locally (after creating a venv and `pip install -r backend/requirements.txt`):

```bash
cd backend
pytest -v
```

## Environment variables

See [.env.example](.env.example) for the full list. The key variables are:

| Variable        | Purpose                                              |
|-----------------|------------------------------------------------------|
| `APP_ENV`       | `development`, `staging`, or `production`            |
| `DATABASE_URL`  | SQLAlchemy connection string for Postgres + pgvector |
| `LOG_LEVEL`     | `DEBUG`, `INFO`, `WARNING`, `ERROR`                  |
| `EMBEDDING_*`   | (later) Embedding provider + model config            |
| `LLM_*`         | (later) Chat model config                            |

## Documentation

In-depth docs live in [docs/](docs/):

- [01 — Project overview](docs/01_project_overview.md)
- [02 — Architecture](docs/02_architecture.md)
- 03 — Database design *(Phase 1, in this doc set)*
- 04 — RAG pipeline *(Phase 2/3)*
- 05 — Agentic workflows *(Phase 5)*
- 06 — Evaluation *(Phase 6)*
- 07 — Testing strategy *(Phase 8)*
- 08 — Deployment *(Phase 10)*
- 09 — Security and privacy *(Phase 9)*
- 10 — Learning notes *(Phase 11)*

## Roadmap

- [x] **Phase 1 — Foundation:** FastAPI, health endpoint, Postgres + pgvector, models, Alembic, Docker Compose
- [x] **Phase 2 — Document ingestion:** upload + list + delete endpoints, .txt/.md/.pdf loaders, recursive chunking, 6 synthetic policies, seed script
- [x] **Phase 3 — Embeddings + retrieval:** pluggable embedding providers (mock + OpenAI-compatible), pgvector HNSW index, FTS tsvector index, hybrid `/rag/retrieve`, backfill CLI
- [x] **Phase 4 — Grounded answer generation:** `POST /rag/ask` with citations + confidence + refusal path, healthcare guardrails system prompt, mock + OpenAI-compatible LLM providers (works with OpenAI / Ollama / vLLM / LM Studio), server-side citation validation, QueryLog audit trail
- [x] **Phase 5 — Agentic workflows (LangGraph):** `POST /agents/run` with four workflows — `policy_lookup`, `claim_triage` (with reflection loop), `policy_comparison`, `compliance_checklist` (with self-validation); rule-based denial classifier; structured outputs per workflow; same healthcare guardrails as Phase 4
- [x] **Phase 6 — Evaluation framework:** 18-item synthetic eval dataset, `/eval/run` + `/eval/runs` + `/eval/runs/{id}` endpoints, metrics (retrieval_hit_rate, context_precision/recall, faithfulness, answer_relevancy, hallucination_flag, refusal_accuracy, latency), persisted runs with JSONB detail bag, ready for chunk/model A/B comparisons
- [x] **Phase 7 — Demo frontend:** Next.js 14 + TypeScript + Tailwind UI with five pages (overview / documents / ask / agents / eval). Hits the backend via `NEXT_PUBLIC_API_URL`; backend ships CORS middleware allowing `http://localhost:3000`.
- [x] **Phase 8 — Testing & quality:** ruff + black + mypy + pre-commit + pytest-cov configured via `pyproject.toml`; frontend ESLint + Prettier + Tailwind plugin; `npm run check` aggregate command; new `/documents/*` API tests + text-utility unit tests; `docs/07_testing_strategy.md` (test pyramid, hermetic-mock pattern, SAVEPOINT isolation, RAG-specific testing).
- [ ] Phase 9 — Observability & security hardening
- [ ] Phase 10 — Deployment (Cloud Run / GCP notes)
- [ ] Phase 11 — Polished docs & learning material

## License

Educational / portfolio project. Use synthetic data only.
