"use client";

import { useState } from "react";

import { Card } from "@/components/Card";
import { CitationCard } from "@/components/CitationCard";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { ErrorBanner } from "@/components/ErrorBanner";
import { runAgent } from "@/lib/api";
import type { AgentRunResponse, WorkflowName } from "@/lib/types";

/**
 * Each workflow's input shape is declared here. Keeps the form
 * rendering generic — the page doesn't need a switch statement on
 * workflow name, and adding a new workflow is just a config entry.
 */
interface FieldConfig {
  key: string;
  label: string;
  placeholder?: string;
  multiline?: boolean;
}

const WORKFLOWS: Record<
  WorkflowName,
  { title: string; description: string; fields: FieldConfig[]; sample: Record<string, string> }
> = {
  policy_lookup: {
    title: "Policy Lookup",
    description:
      "Natural-language policy finder. Rewrites the query (abbreviation expansion), retrieves, generates a grounded answer.",
    fields: [
      {
        key: "query",
        label: "Query",
        placeholder: "Find the PA policy for advanced imaging.",
      },
    ],
    sample: { query: "Find the PA policy for advanced imaging." },
  },
  claim_triage: {
    title: "Claim Denial Triage",
    description:
      "Classifies the denial reason (rule-based), retrieves the relevant policy, produces a next-steps checklist, runs a grounding self-check.",
    fields: [
      {
        key: "claim_summary",
        label: "Claim summary",
        placeholder: "Claim denied because prior authorization was missing for MRI.",
        multiline: true,
      },
      {
        key: "question",
        label: "Operational question",
        placeholder: "What should the billing team do next?",
      },
    ],
    sample: {
      claim_summary:
        "Claim denied because prior authorization was missing for MRI.",
      question: "What should the billing team do next?",
    },
  },
  policy_comparison: {
    title: "Policy Comparison",
    description:
      "Retrieves chunks from two named documents, asks the LLM to compare them on the focus dimensions.",
    fields: [
      {
        key: "document_a_title",
        label: "Document A title",
        placeholder: "Appeal Process Policy",
      },
      {
        key: "document_b_title",
        label: "Document B title",
        placeholder: "Claim Denial Policy",
      },
      {
        key: "focus",
        label: "Focus (dimensions to compare)",
        placeholder: "timelines and required documentation",
      },
    ],
    sample: {
      document_a_title: "Appeal Process Policy",
      document_b_title: "Claim Denial Policy",
      focus: "timelines and required documentation",
    },
  },
  compliance_checklist: {
    title: "Compliance Checklist",
    description:
      "Retrieves chunks for the topic, generates a numbered checklist, drops items that fail a grounding check.",
    fields: [
      {
        key: "topic",
        label: "Topic",
        placeholder: "filing a first-level appeal for a denied claim",
      },
    ],
    sample: { topic: "filing a first-level appeal for a denied claim" },
  },
};

const WORKFLOW_NAMES = Object.keys(WORKFLOWS) as WorkflowName[];

export default function AgentsPage() {
  const [workflow, setWorkflow] = useState<WorkflowName>("policy_lookup");
  const [values, setValues] = useState<Record<string, string>>(
    WORKFLOWS["policy_lookup"].sample,
  );
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentRunResponse | null>(null);

  const cfg = WORKFLOWS[workflow];

  function selectWorkflow(name: WorkflowName) {
    setWorkflow(name);
    setValues(WORKFLOWS[name].sample);
    setResult(null);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await runAgent({ workflow, input: values, top_k: topK });
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
        <h1 className="text-2xl font-semibold tracking-tight">Agents</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          LangGraph-orchestrated multi-step workflows over the policy
          corpus. Each workflow has a predefined plan (a directed graph
          of nodes); the LLM is called at the nodes where natural-
          language reasoning helps.
        </p>
      </header>

      {error && <ErrorBanner message={error} />}

      <div className="flex flex-wrap gap-2">
        {WORKFLOW_NAMES.map((name) => {
          const active = name === workflow;
          return (
            <button
              key={name}
              onClick={() => selectWorkflow(name)}
              className={
                "rounded-md border px-3 py-1.5 text-sm transition-colors " +
                (active
                  ? "border-brand-500 bg-brand-50 text-brand-700"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50")
              }
            >
              {WORKFLOWS[name].title}
            </button>
          );
        })}
      </div>

      <Card title={cfg.title} subtitle={cfg.description}>
        <form onSubmit={handleSubmit} className="space-y-3">
          {cfg.fields.map((field) => (
            <div key={field.key}>
              <label className="block text-xs font-medium text-slate-600">
                {field.label}
              </label>
              {field.multiline ? (
                <textarea
                  value={values[field.key] ?? ""}
                  onChange={(e) =>
                    setValues({ ...values, [field.key]: e.target.value })
                  }
                  rows={3}
                  placeholder={field.placeholder}
                  className="mt-1 w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
                />
              ) : (
                <input
                  type="text"
                  value={values[field.key] ?? ""}
                  onChange={(e) =>
                    setValues({ ...values, [field.key]: e.target.value })
                  }
                  placeholder={field.placeholder}
                  className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
                />
              )}
            </div>
          ))}
          <div className="flex items-center gap-4 pt-1">
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
            <button
              type="submit"
              disabled={busy}
              className="ml-auto rounded-md bg-brand-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Running…" : "Run workflow"}
            </button>
          </div>
        </form>
      </Card>

      {result && <ResultPanel result={result} />}
    </div>
  );
}

function ResultPanel({ result }: { result: AgentRunResponse }) {
  return (
    <div className="space-y-4">
      <Card
        title="Final output"
        actions={<ConfidenceBadge value={result.confidence} />}
      >
        <pre className="overflow-x-auto rounded-md bg-slate-50 p-3 text-xs text-slate-800">
          {JSON.stringify(result.final_output, null, 2)}
        </pre>
        <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-500 sm:grid-cols-4">
          <div>
            <dt className="uppercase tracking-wide">workflow</dt>
            <dd className="font-mono text-[11px] text-slate-700">
              {result.workflow}
            </dd>
          </div>
          <div>
            <dt className="uppercase tracking-wide">model</dt>
            <dd className="font-mono text-[11px] text-slate-700">
              {result.model_name}
            </dd>
          </div>
          <div>
            <dt className="uppercase tracking-wide">latency</dt>
            <dd className="font-mono text-[11px] text-slate-700">
              {result.latency_ms} ms
            </dd>
          </div>
          <div>
            <dt className="uppercase tracking-wide">steps</dt>
            <dd className="font-mono text-[11px] text-slate-700">
              {result.steps.length}
            </dd>
          </div>
        </dl>
      </Card>

      <Card title="Trace" subtitle="One row per LangGraph node">
        <ol className="space-y-2">
          {result.steps.map((step, i) => (
            <li
              key={`${step.name}-${i}`}
              className="flex gap-3 rounded-md bg-slate-50 p-2 text-sm"
            >
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-100 text-xs font-medium text-brand-700">
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-slate-800">{step.name}</p>
                <p className="text-slate-600">{step.summary}</p>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      {result.citations.length > 0 && (
        <Card title="Citations">
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
