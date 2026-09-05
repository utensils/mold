import { formatGenerationTime } from "./generationTime";

/** The least a print needs to say how fast its style is. */
export interface TimedPrint {
  model: string;
  generation_time_ms?: number | null;
}

/** How many recent timed prints per style the typical time is read from. */
export const STYLE_SPEED_SAMPLE = 10;

/**
 * How long each style typically takes, from the prints already made with it:
 * the MEDIAN of the newest `sample` timed prints per style, so one cold-cache
 * first render or one tiny draft cannot stand for the style. Prints that do
 * not know how long they took (older hosts, synthesized rows, zero) do not
 * count, and a style with no timed print has no entry at all — the column
 * says nothing rather than guessing. `prints` is newest first, as the gallery
 * lists them.
 */
export function typicalGenerationTimes(
  prints: readonly TimedPrint[],
  sample: number = STYLE_SPEED_SAMPLE,
): Map<string, number> {
  const samples = new Map<string, number[]>();
  for (const print of prints) {
    const ms = print.generation_time_ms;
    if (!print.model || ms == null || !Number.isFinite(ms) || ms <= 0) continue;
    const list = samples.get(print.model) ?? [];
    if (list.length >= sample) continue;
    list.push(ms);
    samples.set(print.model, list);
  }
  const typical = new Map<string, number>();
  for (const [model, list] of samples) {
    const sorted = [...list].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    typical.set(
      model,
      sorted.length % 2 === 1
        ? sorted[mid]!
        : (sorted[mid - 1]! + sorted[mid]!) / 2,
    );
  }
  return typical;
}

/** `~4s` / `~1m 12s` — the Styles column's spelling of a typical time. */
export function formatTypicalTime(
  ms: number | null | undefined,
): string | null {
  if (ms == null || !Number.isFinite(ms) || ms <= 0) return null;
  if (ms < 60_000) return `~${Math.max(1, Math.round(ms / 1000))}s`;
  return `~${formatGenerationTime(ms)}`;
}
