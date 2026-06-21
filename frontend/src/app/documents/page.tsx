"use client";

import { useEffect, useState } from "react";

import { Card } from "@/components/Card";
import { ErrorBanner } from "@/components/ErrorBanner";
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "@/lib/api";
import type { DocumentRead } from "@/lib/types";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentRead[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setError(null);
    try {
      const list = await listDocuments({ limit: 200 });
      setDocs(list.items);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload(file: File, title: string) {
    setBusy(true);
    setError(null);
    try {
      await uploadDocument(file, { title: title || undefined });
      await refresh();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this document and all of its chunks?")) return;
    setBusy(true);
    setError(null);
    try {
      await deleteDocument(id);
      await refresh();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          The policy corpus. Each upload is text-extracted, chunked
          (recursive character splitting with paragraph-preferred
          boundaries), embedded under the configured embedding model,
          and inserted into Postgres.
        </p>
      </header>

      {error && <ErrorBanner message={error} />}

      <UploadForm onSubmit={handleUpload} disabled={busy} />

      <Card
        title="Loaded documents"
        subtitle={
          docs === null
            ? undefined
            : docs.length === 0
            ? "No documents yet — upload one above, or run `python -m app.seed_policies` from the backend."
            : `${docs.length} document(s) in the corpus`
        }
      >
        {docs === null ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : docs.length === 0 ? (
          <p className="text-sm text-slate-500">Empty corpus.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2 pr-4">Title</th>
                  <th className="py-2 pr-4">Type</th>
                  <th className="py-2 pr-4">Source</th>
                  <th className="py-2 pr-4">Chunks</th>
                  <th className="py-2 pr-4">Created</th>
                  <th className="py-2 pr-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {docs.map((d) => (
                  <tr key={d.id}>
                    <td className="py-2 pr-4 font-medium text-slate-800">
                      {d.title}
                    </td>
                    <td className="py-2 pr-4">{d.document_type}</td>
                    <td className="py-2 pr-4 text-slate-500">
                      {d.source_type}
                    </td>
                    <td className="py-2 pr-4 tabular-nums">
                      {d.chunk_count ?? "–"}
                    </td>
                    <td className="py-2 pr-4 text-slate-500">
                      {new Date(d.created_at).toLocaleString()}
                    </td>
                    <td className="py-2 pr-4 text-right">
                      <button
                        onClick={() => handleDelete(d.id)}
                        disabled={busy}
                        className="text-red-600 hover:underline disabled:cursor-not-allowed disabled:text-slate-400"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function UploadForm({
  onSubmit,
  disabled,
}: {
  onSubmit: (file: File, title: string) => void;
  disabled: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");

  return (
    <Card title="Upload a document" subtitle=".txt, .md, or .pdf — max 10 MB">
      <form
        className="grid gap-3 sm:grid-cols-[1fr_2fr_auto] sm:items-end"
        onSubmit={(e) => {
          e.preventDefault();
          if (file) onSubmit(file, title);
        }}
      >
        <div>
          <label className="block text-xs font-medium text-slate-600">
            File
          </label>
          <input
            type="file"
            accept=".txt,.md,.pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 w-full text-sm"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600">
            Title (optional)
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Defaults to the file name"
            className="mt-1 w-full rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </div>
        <button
          type="submit"
          disabled={!file || disabled}
          className="rounded-md bg-brand-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {disabled ? "Working…" : "Upload"}
        </button>
      </form>
    </Card>
  );
}
