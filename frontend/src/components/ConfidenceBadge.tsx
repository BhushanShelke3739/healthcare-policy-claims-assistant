import type { Confidence } from "@/lib/types";

const CLASSES: Record<Confidence, string> = {
  high: "bg-green-100 text-green-800 ring-green-200",
  medium: "bg-amber-100 text-amber-800 ring-amber-200",
  low: "bg-red-100 text-red-800 ring-red-200",
};

export function ConfidenceBadge({ value }: { value: Confidence }) {
  return (
    <span
      className={
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 " +
        CLASSES[value]
      }
    >
      {value}
    </span>
  );
}
