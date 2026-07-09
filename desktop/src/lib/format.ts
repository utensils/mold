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
