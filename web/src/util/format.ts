import type { OutputMetadata } from "../types";

/** Decimal gigabytes, matching server and storage-vendor reporting. */
export function formatGB(bytes: number | null | undefined): string {
  if (!bytes) return "—";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/** Short relative time like "3m", "2h", "4d", "3w", or a date for older items. */
export function formatRelativeTime(unixSeconds: number): string {
  if (!unixSeconds) return "";
  const now = Date.now() / 1000;
  const diff = Math.max(0, now - unixSeconds);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d`;
  if (diff < 86400 * 30) return `${Math.floor(diff / (86400 * 7))}w`;
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatResolution(meta: OutputMetadata): string {
  if (meta.width && meta.height) return `${meta.width}×${meta.height}`;
  return "";
}

/** Trim `flux-dev:q8` → `flux-dev` when we only want the family name. */
export function shortModel(model: string): string {
  return model || "unknown model";
}
