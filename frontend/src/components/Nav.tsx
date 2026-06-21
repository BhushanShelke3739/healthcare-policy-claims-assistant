"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/documents", label: "Documents" },
  { href: "/ask", label: "Ask" },
  { href: "/agents", label: "Agents" },
  { href: "/eval", label: "Evaluation" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center gap-1 px-6 py-3">
        <Link href="/" className="mr-6 text-sm font-semibold tracking-tight">
          <span className="text-brand-700">HPCA</span>
          <span className="ml-2 text-slate-500 font-normal">
            Healthcare Policy &amp; Claims Assistant
          </span>
        </Link>
        <div className="flex gap-1">
          {LINKS.slice(1).map((link) => {
            const active =
              pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={
                  "rounded-md px-3 py-1.5 text-sm transition-colors " +
                  (active
                    ? "bg-brand-50 text-brand-700"
                    : "text-slate-600 hover:bg-slate-100")
                }
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
