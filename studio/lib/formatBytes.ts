const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"] as const;

/**
 * Compact decimal byte count shared by web, desktop, and mobile surfaces.
 * Decimal units match Mold's download and disk-size presentation.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1_000) return `${Math.round(bytes)} B`;

  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1_000)),
    BYTE_UNITS.length - 1,
  );
  return `${(bytes / 1_000 ** exponent).toFixed(1)} ${BYTE_UNITS[exponent]}`;
}

/** Human-readable completed/total bytes for live progress rows. */
export function formatByteProgress(completed: number, total: number): string {
  return `${formatBytes(completed)} / ${formatBytes(total)}`;
}
