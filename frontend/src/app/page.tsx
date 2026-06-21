"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Card } from "@/components/Card";
import { ErrorBanner } from "@/components/ErrorBanner";
import { getHealth, listDocuments, listEvalRuns } from "@/lib/api";

interface Stats {
  health: { status: string; environment: string; version: string };
  docCount: number;
  evalCount: number;
}

export default function HomePage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // One mount-time fetch for the three quick-stat blocks. Pages are
    // tiny so we don't bother with tanstack-query / SWR.
    let cancelled = false;
    Promise.all([
      getHealth(),
      listDocuments({ limit: 1 }),
      listEvalRuns({ limit: 1 }),
    ])
      .then(([health, docs, evals]) => {
        if (cancelled) return;
        setStats({
          health,
          docCount: docs.total,
          evalCount: evals.total,
        });
      })
      .catch((e) => !cancelled && setError(String(e?.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          A grounded RAG assistant over synthetic healthcare policy
          documents. Upload policies, ask questions and get answers with
          citations, run multi-step agent workflows, and measure
          retrieval and answer quality with an evaluation harness.
        </p>
      </header>

      {error && <ErrorBanner message={error} />}

      <div className="grid gap-4 md:grid-cols-3">
        <Card title="Backend">
          {stats ? (
            <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-sm">
              <dt className="text-slate-500">Status</dt>
              <dd className="font-medium text-green-700">
                {stats.health.status}
              </dd>
              <dt className="text-slate-500">Env</dt>
              <dd>{stats.health.environment}</dd>
              <dt className="text-slate-500">Version</dt>
              <dd className="font-mono text-xs">{stats.health.version}</dd>
            </dl>
          ) : (
            <Skeleton lines={3} />
          )}
        </Card>

        <Card title="Documents">
          {stats ? (
            <div>
              <div className="text-3xl font-semibold">{stats.docCount}</div>
              <p className="text-sm text-slate-500">in the corpus</p>
              <Link
                href="/documents"
                className="mt-3 inline-block text-sm text-brand-600 hover:underline"
              >
                Manage →
              </Link>
            </div>
          ) : (
            <Skeleton lines={2} />
          )}
        </Card>

        <Card title="Evaluation runs">
          {stats ? (
            <div>
              <div className="text-3xl font-semibold">{stats.evalCount}</div>
              <p className="text-sm text-slate-500">runs persisted</p>
              <Link
                href="/eval"
                className="mt-3 inline-block text-sm text-brand-600 hover:underline"
              >
                View →
              </Link>
            </div>
          ) : (
            <Skeleton lines={2} />
          )}
        </Card>
      </div>

      <Card title="Quick start" subtitle="Where to begin">
        <ol className="ml-5 list-decimal space-y-2 text-sm text-slate-700">
          <li>
            <Link href="/documents" className="text-brand-600 hover:underline">
              Documents
            </Link>{" "}
            — confirm the six synthetic policies are loaded.
          </li>
          <li>
            <Link href="/ask" className="text-brand-600 hover:underline">
              Ask
            </Link>{" "}
            — try a question like &ldquo;How long do I have to file a first-level
            appeal?&rdquo; The answer must come from a retrieved chunk and is
            shown with citations.
          </li>
          <li>
            <Link href="/agents" className="text-brand-600 hover:underline">
              Agents
            </Link>{" "}
            — run the <code>claim_triage</code> workflow on a denial summary to
            see the multi-step trace.
          </li>
          <li>
            <Link href="/eval" className="text-brand-600 hover:underline">
              Evaluation
            </Link>{" "}
            — kick off a run on the bundled 18-question dataset and compare
            summary metrics across modes / alpha settings.
          </li>
        </ol>
      </Card>
    </div>
  );
}

function Skeleton({ lines }: { lines: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="h-3 w-3/4 rounded bg-slate-100" />
      ))}
    </div>
  );
}
