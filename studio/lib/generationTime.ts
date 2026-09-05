/**
 * How long a print took, as every surface spells it: `4.0s` under a minute,
 * `1m 12s` from a minute up, and `null` when the print does not know — an
 * older host, a synthesized row, or a zero the server uses for "not
 * measured". A caller renders nothing for `null` rather than `0.0s`.
 */
export function formatGenerationTime(
  ms: number | null | undefined,
): string | null {
  if (ms == null || !Number.isFinite(ms) || ms <= 0) return null;
  if (ms < 60_000) return `${(Math.floor(ms / 100) / 10).toFixed(1)}s`;
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}
