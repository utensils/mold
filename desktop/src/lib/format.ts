/** Decimal units, one decimal place — matches mold's shared byte formatting. */
export function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/** Adaptive decimal units for totals that may be small (gallery sizes). */
export function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return formatGB(bytes);
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${bytes} B`;
}

/** Compact counts for card metadata: 999 → "999", 12_300 → "12.3k", 4_200_000 → "4.2M". */
export function formatCount(count: number): string {
  const compact = (value: number, suffix: string): string =>
    `${value.toFixed(1).replace(/\.0$/, "")}${suffix}`;
  // The unit is chosen from the ROUNDED mantissa, so 999_950 (whose k
  // mantissa rounds to "1000.0") rolls over to "1M" instead of "1000k".
  if (count >= 1_000_000 || Number((count / 1_000).toFixed(1)) >= 1_000) {
    return compact(count / 1_000_000, "M");
  }
  if (count >= 1_000) return compact(count / 1_000, "k");
  return String(count);
}

export type MeterLevel = "ok" | "critical";

/** VRAM meter turns Stop-red past 92% (design spec §3, the Bench rail). */
export function vramLevel(used: number, total: number): MeterLevel {
  if (total <= 0) return "ok";
  return used / total >= 0.92 ? "critical" : "ok";
}

export function percent(used: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

/** Compact relative timestamp for MRU lists ("just now", "5m ago", "3d ago"). */
export function timeAgo(thenMs: number, nowMs: number = Date.now()): string {
  const seconds = Math.max(0, Math.floor((nowMs - thenMs) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(thenMs).toLocaleDateString();
}
