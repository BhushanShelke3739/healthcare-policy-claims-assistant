# Frontend — Healthcare Policy & Claims Assistant

Next.js + TypeScript + Tailwind demo UI for the HPCA backend. Five
pages, no auth, single-user dev tool.

## Pages

| Route          | What it does                                                                 |
|----------------|------------------------------------------------------------------------------|
| `/`            | Backend health, doc count, eval-run count, quick-start steps.                |
| `/documents`   | Upload `.txt`/`.md`/`.pdf`, list current documents, delete.                  |
| `/ask`         | Submit a question, see grounded answer + confidence badge + citations.       |
| `/agents`      | Pick one of 4 LangGraph workflows, fill form, view final output + trace.     |
| `/eval`        | Kick off an eval run; browse past runs; drill into per-question metrics.     |

## Prerequisites

- **Node.js 20+** and npm. Check with `node --version`.
- **Backend running** on `http://localhost:8000` (see [../backend/README.md](../backend/README.md)).
- Backend `.env` should leave `CORS_ORIGINS` at its default — it already
  allows `http://localhost:3000`.

## First run

```powershell
# from frontend/
copy .env.example .env
npm install
npm run dev
```

Open <http://localhost:3000>. The home page will show the backend's
health response if everything is wired up.

## Tech notes

- **App Router** (Next.js 14). Pages are mostly `"use client"` because
  every page makes an API call from the browser — server-rendering
  buys us nothing here and adds complexity.
- **No state-management library.** `useState` + mount-time `fetch` is
  enough for a single-user demo. If pages start chaining 5+ requests
  with cache invalidation, swap to tanstack-query.
- **Tailwind v3** with a narrow brand palette. No design system
  (shadcn / Radix / etc.) — keeps the dependency tree small.
- **TypeScript types** in [src/lib/types.ts](src/lib/types.ts) mirror
  the backend Pydantic schemas. Hand-maintained, deliberately narrow —
  adding fields on the backend won't break the frontend, but removing
  fields will (which is the right tradeoff).
- **API client** in [src/lib/api.ts](src/lib/api.ts) — plain `fetch`,
  no axios. ~150 lines covering every endpoint.

## File layout

```
frontend/
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── next.config.js
├── .env.example
└── src/
    ├── app/
    │   ├── layout.tsx                 # nav + global styles
    │   ├── globals.css                # tailwind imports
    │   ├── page.tsx                   # /
    │   ├── documents/page.tsx         # /documents
    │   ├── ask/page.tsx               # /ask
    │   ├── agents/page.tsx            # /agents
    │   └── eval/page.tsx              # /eval
    ├── components/
    │   ├── Nav.tsx
    │   ├── Card.tsx
    │   ├── ConfidenceBadge.tsx
    │   ├── CitationCard.tsx
    │   └── ErrorBanner.tsx
    └── lib/
        ├── api.ts
        └── types.ts
```

## Common commands

```powershell
npm run dev          # http://localhost:3000 with hot reload
npm run typecheck    # tsc --noEmit
npm run lint         # next lint (eslint)
npm run lint:fix     # next lint --fix
npm run format       # prettier --write
npm run format:check # prettier --check (CI-friendly)
npm run check        # lint + format:check + typecheck (use this before committing)
npm run build        # production build
npm run start        # run the production build
```

## Troubleshooting

**"Failed to fetch"** on every page → the backend isn't running or
`CORS_ORIGINS` doesn't include `http://localhost:3000`. Confirm:

```powershell
curl.exe http://localhost:8000/health
```

**Document upload fails with 413** → the file is larger than
`MAX_UPLOAD_SIZE_MB` (default 10 MB). Bump the backend setting.

**Ask / agent latency feels broken** → if the backend is on
`LLM_PROVIDER=openai` pointing at Ollama, single calls take 30s+. Set
`LLM_PROVIDER=mock` in backend `.env` and restart uvicorn for instant
responses while iterating.
