/** Decimal units, one decimal place — matches mold's shared byte formatting. */
export function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/** A used/total pair sharing one unit — `12.1 / 36.0 GB`. Meters already say
 *  what they measure, so the reading beside them repeats neither the unit nor
 *  the word (style guide: units stay tight and mono). */
export function formatGBPair(used: number, total: number): string {
  return `${(used / 1_000_000_000).toFixed(1)} / ${formatGB(total)}`;
}

/** Adaptive decimal units for totals that may be small (gallery sizes). */
export function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return formatGB(bytes);
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${bytes} B`;
}

/** Human name for a metadata scheduler — plain kebab-case string or a
 * serde-tagged object (`{ ddim: … }`). "default" and empty render as "". */
export function formatScheduler(s: string | Record<string, unknown> | null | undefined): string {
  if (!s || s === "default") return "";
  if (typeof s === "string") return s;
  return Object.keys(s)[0] ?? "";
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

/**
 * The READOUT for a `percent()` value: one whole number and the sign. The
 * float belongs to the meter's width, never to a person — a machine card that
 * printed the raw getter read "59.10320281982422%". Round here and only here.
 */
export function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

/** The mono line under a VRAM meter: `31.6 / 51.5 GB graphics memory`. Says
 *  what the percent beside the meter is measuring, without decimals of its own. */
export function formatGraphicsMemory(used: number, total: number): string {
  return `${formatGBPair(used, total)} graphics memory`;
}

/** Transfer rate for the download banner's mono line: 12_400_000 → "12.4 MB/s". */
export function formatRate(bytesPerSecond: number | null): string {
  if (bytesPerSecond == null || !Number.isFinite(bytesPerSecond)) return "—";
  return `${(Math.max(0, bytesPerSecond) / 1_000_000).toFixed(1)} MB/s`;
}

/** Compact ETA for download rows: 42 → "42s", 192 → "3m 12s", 3845 → "1h 4m". */
export function formatEta(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const whole = Math.max(0, Math.round(seconds));
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes}m ${whole % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** Compact uptime for host telemetry: 42 → "42s", 300 → "5m", 5400 → "1h 30m", 200_000 → "2d 7h". */
export function formatUptime(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
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
