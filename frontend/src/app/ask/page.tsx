"use client";

import { useState } from "react";

import { Card } from "@/components/Card";
import { CitationCard } from "@/components/CitationCard";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ErrorBanner } from "@/components/ErrorBanner";
import { askQuestion } from "@/lib/api";
import type { AskResponse, RetrievalMode } from "@/lib/types";

const MODES: RetrievalMode[] = ["hybrid", "vector", "keyword"];

const SAMPLES = [
  "How long do I have to file a first-level appeal?",
  "What does denial code HF-022 mean?",
  "Which imaging services require prior authorization?",
  "What modifier is required for synchronous telehealth visits?",
];

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await askQuestion({
        question,
        top_k: topK,
        mode,
        include_citations: true,
      });
      setResult(r);
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          Free-text policy questions. The system retrieves the most
          relevant chunks (hybrid by default), passes them to the
          configured LLM, and returns an answer grounded only in those
          chunks &mdash; with citations.
        </p>
      </header>

      {error && <ErrorBanner message={error} />}

      <Card title="Question">
        <form onSubmit={handleSubmit} className="space-y-3">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. How long do I have to file a first-level appeal?"
            rows={3}
            className="w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
          <div className="flex flex-wrap items-end gap-4">
            <label className="text-xs font-medium text-slate-600">
              top_k
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="ml-2 w-16 rounded-md border border-slate-200 bg-white px-2 py-1 text-sm shadow-sm"
              />
            </label>
            <label className="text-xs font-medium text-slate-600">
              mode
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as RetrievalMode)}
                className="ml-2 rounded-md border border-slate-200 bg-white px-2 py-1 text-sm shadow-sm"
              >
                {MODES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="submit"
              disabled={busy || !question.trim()}
              className="ml-auto rounded-md bg-brand-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Thinking…" : "Ask"}
            </button>
          </div>
          <div className="flex flex-wrap gap-2 pt-2 text-xs text-slate-500">
            <span>Try:</span>
            {SAMPLES.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setQuestion(s)}
                className="rounded-full border border-slate-200 px-2 py-0.5 hover:bg-slate-100"
              >
                {s}
              </button>
            ))}
          </div>
        </form>
      </Card>

      {result && <ResultPanel result={result} />}
    </div>
  );
}

function ResultPanel({ result }: { result: AskResponse }) {
  return (
    <div className="space-y-4">
      <Card
        title="Answer"
        actions={<ConfidenceBadge value={result.confidence} />}
      >
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
          {result.answer}
        </p>
        {result.grounding_notes && (
          <p className="mt-3 border-l-2 border-slate-200 pl-3 text-xs italic text-slate-500">
            {result.grounding_notes}
          </p>
        )}
        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-500 sm:grid-cols-4">
          <Meta label="model" value={result.model_name} />
          <Meta label="latency" value={`${result.latency_ms} ms`} />
          <Meta label="chunks" value={String(result.retrieved_chunk_ids.length)} />
          <Meta label="citations" value={String(result.citations.length)} />
        </dl>
      </Card>

      {result.citations.length > 0 && (
        <Card title="Citations" subtitle="Verbatim excerpts the answer is grounded in">
          <div className="grid gap-3 sm:grid-cols-2">
            {result.citations.map((c, i) => (
              <CitationCard key={`${c.chunk_id}-${i}`} citation={c} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="uppercase tracking-wide">{label}</dt>
      <dd className="font-mono text-[11px] text-slate-700">{value}</dd>
    </div>
  );
}
