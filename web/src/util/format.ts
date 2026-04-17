import type { OutputMetadata, Scheduler } from "../types";

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

export function formatAbsoluteTime(unixSeconds: number): string {
  if (!unixSeconds) return "";
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatResolution(meta: OutputMetadata): string {
  if (meta.width && meta.height) return `${meta.width}×${meta.height}`;
  return "";
}

/** 1.23 MB / 845 KB style. */
export function formatFileSize(bytes: number): string {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

/** Normalize the `scheduler` enum — it can serialize as a string or object. */
export function formatScheduler(s: Scheduler | null | undefined): string {
  if (!s) return "";
  if (typeof s === "string") return s;
  return Object.keys(s)[0] ?? "";
}

/** Trim `flux-dev:q8` → `flux-dev` when we only want the family name. */
export function shortModel(model: string): string {
  return model || "unknown model";
}
