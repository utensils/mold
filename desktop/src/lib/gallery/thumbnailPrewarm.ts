/**
 * Pre-warm planning for the persistent thumbnail cache.
 *
 * After a listing lands, the tiles the user is NOT looking at yet are worth
 * fetching quietly so a scroll finds them on disk — the Lightroom "build
 * previews" pass. This module is the pure half: it orders candidates by
 * distance from the viewport, marks the nearby band `near` and the rest
 * `background`, skips what is already cached or on screen, and caps the work
 * per host so a 5 000-print machine does not queue 5 000 requests. The view
 * owns probing, scheduling, and cancellation.
 */
import type { ThumbnailPriority } from "@studio/lib/thumbnailScheduler";

/** Rows within this many viewport heights of the visible span are `near`. */
export const PREWARM_NEAR_VIEWPORTS = 2;
/** Upper bound on tiles prepared per host per planning pass. */
export const PREWARM_MAX_PER_HOST = 600;
/** Probe batch size — one IPC round trip stats this many entries. */
export const PREWARM_PROBE_BATCH = 64;

export interface PrewarmCandidate {
  sourceKey: string;
  filename: string;
  mediaVersion: string;
  /** Row the tile sits in (the virtualizer's unit). */
  rowIndex: number;
}

export interface PrewarmViewport {
  /** First and last row currently on screen (inclusive). */
  startRow: number;
  endRow: number;
  /** Rows per viewport height, used to size the `near` band. */
  rowsPerViewport: number;
}

export interface PrewarmPlanEntry {
  candidate: PrewarmCandidate;
  priority: Exclude<ThumbnailPriority, "visible">;
}

/** Distance in rows from the visible span; 0 inside it. */
function rowDistance(row: number, viewport: PrewarmViewport): number {
  if (row < viewport.startRow) return viewport.startRow - row;
  if (row > viewport.endRow) return row - viewport.endRow;
  return 0;
}

/**
 * Order and classify what to warm. On-screen rows are excluded (their tiles
 * request themselves at `visible`), nearby rows come first at `near`, then
 * everything else at `background`, in scroll order below the viewport
 * before above it — the direction a user usually keeps scrolling.
 */
export function planPrewarm(
  candidates: readonly PrewarmCandidate[],
  viewport: PrewarmViewport,
  options: { maxPerHost?: number; nearViewports?: number } = {},
): PrewarmPlanEntry[] {
  const maxPerHost = options.maxPerHost ?? PREWARM_MAX_PER_HOST;
  const nearRows =
    Math.max(1, viewport.rowsPerViewport) * (options.nearViewports ?? PREWARM_NEAR_VIEWPORTS);
  const ranked = candidates
    .filter((c) => rowDistance(c.rowIndex, viewport) > 0)
    .map((c, order) => ({
      c,
      distance: rowDistance(c.rowIndex, viewport),
      below: c.rowIndex > viewport.endRow ? 0 : 1,
      order,
    }))
    .sort((a, b) => a.distance - b.distance || a.below - b.below || a.order - b.order);
  const perHost = new Map<string, number>();
  const plan: PrewarmPlanEntry[] = [];
  for (const { c, distance } of ranked) {
    const used = perHost.get(c.sourceKey) ?? 0;
    if (used >= maxPerHost) continue;
    perHost.set(c.sourceKey, used + 1);
    plan.push({ candidate: c, priority: distance <= nearRows ? "near" : "background" });
  }
  return plan;
}

/** Split a list into probe-sized batches, preserving order. */
export function chunkForProbe<T>(list: readonly T[], size = PREWARM_PROBE_BATCH): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < list.length; i += size) out.push(list.slice(i, i + size));
  return out;
}
