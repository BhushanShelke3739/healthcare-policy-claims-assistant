import type { Metadata } from "next";
import { Nav } from "@/components/Nav";

import "./globals.css";

export const metadata: Metadata = {
  title: "HPCA — Healthcare Policy & Claims Assistant",
  description:
    "Demo UI for the Healthcare Policy & Claims Assistant — RAG over synthetic policy documents with grounded answers, agent workflows, and an evaluation harness.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-6 pb-8 pt-2 text-xs text-slate-400">
          Synthetic data only — no real PHI or PII. Demo project; the
          assistant does not provide medical or legal advice.
        </footer>
      </body>
    </html>
  );
}
