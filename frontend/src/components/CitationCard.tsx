import type { Citation } from "@/lib/types";

export function CitationCard({ citation }: { citation: Citation }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <p className="text-sm font-semibold text-brand-700">
          {citation.document_title || "(untitled)"}
        </p>
        <p className="font-mono text-[10px] text-slate-400" title={citation.chunk_id}>
          {citation.chunk_id.slice(0, 8)}…
        </p>
      </div>
      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-700">
        {citation.excerpt}
      </p>
    </div>
  );
}
