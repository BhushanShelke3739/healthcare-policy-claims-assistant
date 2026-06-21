# 01 — Project Overview

## What is this?

**Healthcare Policy & Claims Assistant** is a portfolio-grade web
application that helps a healthcare operations team get fast, *grounded*
answers from their own policy and claims documents.

A real billing or compliance team typically deals with:

- Dozens (or hundreds) of internal policy PDFs that change every quarter.
- Frequent questions like *"Can this denied claim still be appealed?"* or
  *"What documentation do we need for prior authorization on imaging?"*
- High cost when the wrong answer is given — denied appeals, compliance
  exposure, rework.

A generic LLM can answer those questions confidently — and often wrongly.
A **RAG pipeline** grounded in the team's own documents, with citations
and an explicit "I don't know" path, is far safer.

This project shows how to build that system end to end, then layers
**agents** on top so the assistant can do multi-step work (look up policy,
classify a claim issue, generate a compliance checklist, compare two
versions of a policy).

> **Synthetic data only.** Every fixture in `sample_data/` is invented
> for this project. No real PHI, PII, claims, or patient records are
> used. The system is for operational policy lookup — it does **not**
> give medical or legal advice.

## What it does (by the end of all phases)

1. Upload `.txt` / `.md` / `.pdf` policy documents.
2. Chunk, embed, and store them in PostgreSQL with `pgvector`.
3. Answer free-form questions:
   - Retrieves the most relevant chunks via vector + keyword (hybrid) search.
   - Reranks.
   - Generates an answer that quotes only retrieved context, with citations.
   - Refuses to answer when the context is insufficient.
4. Runs **agentic workflows**:
   - **Policy Lookup** — find the policy that applies to a situation.
   - **Claim Denial Triage** — given a denial reason, produce a next-step
     checklist grounded in policy.
   - **Policy Comparison** — diff two versions of a policy across
     timelines, requirements, exceptions.
   - **Compliance Checklist** — generate and self-validate a checklist
     for a workflow.
5. Runs **evaluations** on a held-out set of questions and exposes the
   results via API.

## How to read this repo

This project is structured for *learning* as much as for shipping. Every
phase produces working software, and every major file has a header
explaining **what** it does and **why** the design was chosen.

The phases match the implementation order in the README roadmap. Each
phase has a corresponding doc in this folder.

| Phase | Topic                                 | Doc                              |
|-------|---------------------------------------|----------------------------------|
| 1     | Foundation (you are here)             | `01_project_overview.md`, `02_architecture.md` |
| 2     | Document ingestion                    | `04_rag_pipeline.md`             |
| 3     | Embeddings + vector + hybrid search   | `04_rag_pipeline.md`             |
| 4     | Grounded answer generation            | `04_rag_pipeline.md`             |
| 5     | Agentic workflows                     | `05_agentic_workflows.md`        |
| 6     | Evaluation                            | `06_evaluation.md`               |
| 7     | Frontend / demo UI                    | *(later)*                        |
| 8     | Testing & quality                     | `07_testing_strategy.md`         |
| 9     | Observability & security              | `09_security_and_privacy.md`     |
| 10    | Deployment                            | `08_deployment.md`               |
| 11    | Learning notes                        | `10_learning_notes.md`           |

## Mapping to real job requirements

| Job requirement                                        | Where it shows up                                  |
|--------------------------------------------------------|----------------------------------------------------|
| Python 3 / FastAPI                                     | `backend/app/main.py`, every route module          |
| RESTful API design + OpenAPI                            | `app/api/routes_*` + auto-generated `/docs`        |
| PostgreSQL + SQLAlchemy + Alembic                       | `app/db/`, `alembic/`                              |
| RAG (ingestion → chunking → embeddings → retrieval → gen) | `app/services/`                                  |
| Vector search                                           | `pgvector` column on `document_chunks.embedding`   |
| Hybrid search + reranking                               | Phase 3                                            |
| Agent workflows / tool use / structured outputs         | Phase 5                                            |
| Evaluation / hallucination checks                       | Phase 6                                            |
| Testing                                                 | `backend/tests/`                                   |
| Docker + Compose                                        | `Dockerfile`, `docker-compose.yml`                 |
| CI/CD                                                   | Phase 10 (`.github/workflows/`)                    |
| Observability                                           | `app/core/logging.py`, Phase 9 middleware          |
| Security best practices                                 | `app/core/security.py`, Phase 9                    |

## What Phase 1 delivers

A runnable, *empty-but-correct* foundation:

- ✅ FastAPI app with `/health`, `/readiness`, OpenAPI docs at `/docs`.
- ✅ PostgreSQL 16 + pgvector spun up via Docker Compose.
- ✅ SQLAlchemy 2.x ORM models for every entity the project will need.
- ✅ Alembic migrations, with the first revision enabling `pgvector`.
- ✅ Pydantic-settings-driven configuration via `.env`.
- ✅ Structured JSON logging.
- ✅ Pytest suite that runs without a live database.
- ✅ Documentation (this file + `02_architecture.md`).

What's intentionally **not** here yet: ingestion, embeddings, RAG,
agents, evaluation. Those routes exist in the OpenAPI as `501 Not
Implemented` so the API surface is documented from day one, but the
business logic lands in subsequent phases.
