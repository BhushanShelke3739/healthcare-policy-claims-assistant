"use client";

import { useEffect, useState } from "react";

import { Card } from "@/components/Card";
import { ErrorBanner } from "@/components/ErrorBanner";
import { getEvalRun, listEvalRuns, runEval } from "@/lib/api";
import type {
  EvalRunRead,
  EvalRunSummaryRow,
  EvalSummary,
  RetrievalMode,
} from "@/lib/types";

const MODES: RetrievalMode[] = ["hybrid", "vector", "keyword"];

export default function EvalPage() {
  const [runs, setRuns] = useState<EvalRunSummaryRow[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<EvalRunRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setError(null);
    try {
      const list = await listEvalRuns({ limit: 50 });
      setRuns(list.items);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setSelectedRun(null);
      return;
    }
    let cancelled = false;
    getEvalRun(selectedId)
      .then((r) => !cancelled && setSelectedRun(r))
      .catch((e) => !cancelled && setError(String((e as Error).message)));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  async function handleRunEval(payload: {
    name: string;
    description?: string;
    mode: RetrievalMode;
    top_k: number;
    alpha?: number;
  }) {
    setBusy(true);
    setError(null);
    try {
      const r = await runEval(payload);
      await refresh();
      setSelectedId(r.id);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          Run the bundled 18-question dataset through the same pipeline
          used by /rag/ask. Per-question and aggregate metrics
          (retrieval_hit_rate, context_precision/recall, faithfulness,
          answer_relevancy, hallucination_flag, refusal_accuracy,
          latency) are persisted for run-to-run comparison.
        </p>
      </header>

      {error && <ErrorBanner message={error} />}

      <RunForm onSubmit={handleRunEval} disabled={busy} />

      <Card
        title="History"
        subtitle={
          runs === null
            ? undefined
            : runs.length === 0
            ? "No runs yet — kick one off above."
            : `${runs.length} run(s)`
        }
      >
        {runs === null ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-slate-500">Empty.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {runs.map((r) => (
              <li
                key={r.id}
                className={
                  "flex items-center justify-between gap-4 py-2 px-2 -mx-2 rounded cursor-pointer hover:bg-slate-50 " +
                  (r.id === selectedId ? "bg-brand-50/40" : "")
                }
                onClick={() =>
                  setSelectedId(r.id === selectedId ? null : r.id)
                }
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-800">
                    {r.name}
                  </p>
                  <p className="truncate text-xs text-slate-500">
                    {new Date(r.created_at).toLocaleString()} ·{" "}
                    {r.num_questions} question(s)
                  </p>
                </div>
                <span className="text-xs text-brand-600">
                  {r.id === selectedId ? "selected" : "view →"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {selectedRun && <RunDetail run={selectedRun} />}
    </div>
  );
}

function RunForm({
  onSubmit,
  disabled,
}: {
  onSubmit: (p: {
    name: string;
    description?: string;
    mode: RetrievalMode;
    top_k: number;
    alpha?: number;
  }) => void;
  disabled: boolean;
}) {
  const [name, setName] = useState("ad-hoc");
  const [description, setDescription] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [alphaText, setAlphaText] = useState("");

  return (
    <Card title="Run evaluation" subtitle="Uses the bundled dataset (sample_data/eval_questions/healthcare_policy_eval.json)">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const alpha = alphaText.trim() === "" ? undefined : Number(alphaText);
          onSubmit({
            name,
            description: description.trim() || undefined,
            mode,
            top_k: topK,
            alpha,
          });
        }}
        className="grid gap-3 sm:grid-cols-2"
      >
        <label className="text-xs font-medium text-slate-600">
          Name
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm"
          />
        </label>
        <label className="text-xs font-medium text-slate-600">
          Description (optional)
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. baseline c800-o120"
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm"
          />
        </label>
        <label className="text-xs font-medium text-slate-600">
          Mode
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as RetrievalMode)}
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm"
          >
            {MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs font-medium text-slate-600">
          top_k
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm"
          />
        </label>
        <label className="text-xs font-medium text-slate-600 sm:col-span-2">
          alpha (hybrid only — leave blank to use settings.hybrid_alpha)
          <input
            type="text"
            inputMode="decimal"
            value={alphaText}
            onChange={(e) => setAlphaText(e.target.value)}
            placeholder="0.0 — 1.0"
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm"
          />
        </label>
        <div className="sm:col-span-2 flex justify-end">
          <button
            type="submit"
            disabled={disabled}
            className="rounded-md bg-brand-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {disabled ? "Running…" : "Run"}
          </button>
        </div>
      </form>
    </Card>
  );
}

function RunDetail({ run }: { run: EvalRunRead }) {
  return (
    <div className="space-y-4">
      <Card title={run.name} subtitle={run.description ?? undefined}>
        <p className="mb-3 text-xs text-slate-500">
          {new Date(run.created_at).toLocaleString()}
        </p>
        {run.summary ? <SummaryGrid summary={run.summary} /> : null}
      </Card>

      <Card title="Per-question results" subtitle={`${run.results.length} row(s)`}>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="py-2 pr-4">Question</th>
                <th className="py-2 pr-4">Hit</th>
                <th className="py-2 pr-4">Prec.</th>
                <th className="py-2 pr-4">Recall</th>
                <th className="py-2 pr-4">Faith.</th>
                <th className="py-2 pr-4">Rel.</th>
                <th className="py-2 pr-4">Halluc.</th>
                <th className="py-2 pr-4 text-right">ms</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {run.results.map((r) => {
                const hit = r.details?.["retrieval_hit"];
                return (
                  <tr key={r.id}>
                    <td className="max-w-md truncate py-2 pr-4 text-slate-800" title={r.question}>
                      {r.question}
                    </td>
                    <td className="py-2 pr-4">
                      {hit === true ? (
                        <span className="text-green-600">✓</span>
                      ) : hit === false ? (
                        <span className="text-red-600">✗</span>
                      ) : (
                        <span className="text-slate-400">–</span>
                      )}
                    </td>
                    <td className="py-2 pr-4 tabular-nums">{fmt(r.context_precision)}</td>
                    <td className="py-2 pr-4 tabular-nums">{fmt(r.context_recall)}</td>
                    <td className="py-2 pr-4 tabular-nums">{fmt(r.faithfulness)}</td>
                    <td className="py-2 pr-4 tabular-nums">{fmt(r.answer_relevancy)}</td>
                    <td className="py-2 pr-4">
                      {r.hallucination_flag ? (
                        <span className="rounded bg-red-100 px-1.5 py-0.5 text-[10px] text-red-700">
                          flag
                        </span>
                      ) : (
                        <span className="text-slate-400">–</span>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {r.latency_ms ?? "–"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function SummaryGrid({ summary }: { summary: EvalSummary }) {
  const items: Array<[string, number | null]> = [
    ["num_questions", summary.num_questions],
    ["retrieval_hit_rate", summary.retrieval_hit_rate],
    ["avg_context_precision", summary.avg_context_precision],
    ["avg_context_recall", summary.avg_context_recall],
    ["avg_faithfulness", summary.avg_faithfulness],
    ["avg_answer_relevancy", summary.avg_answer_relevancy],
    ["hallucination_rate", summary.hallucination_rate],
    ["refusal_accuracy", summary.refusal_accuracy],
    ["avg_latency_ms", summary.avg_latency_ms],
  ];
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-md bg-slate-50 p-3">
          <dt className="text-[10px] uppercase tracking-wide text-slate-500">{k}</dt>
          <dd className="mt-0.5 font-mono text-sm font-semibold text-slate-800">
            {fmt(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  return v.toFixed(3);
}
