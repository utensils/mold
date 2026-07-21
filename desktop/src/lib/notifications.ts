import type { Job } from "./generationJob";

/**
 * Cross-surface notification helpers (§08 G11). Pure so the "toast only when
 * off-canvas" and "offline fires once" guarantees are testable without wiring
 * Vue watchers. The App shell owns the seen-sets and the actual toast/native
 * dispatch.
 */

/** Jobs that completed since the last pass (by clientId). The caller adds the
 *  returned ids to `seen` after handling so each completion notifies once. */
export function newlyCompletedJobs(jobs: Job[], seen: Set<number>): Job[] {
  return jobs.filter((j) => j.status === "complete" && !seen.has(j.clientId));
}

/** A finished generation raises a foreground toast only when the user isn't
 *  already watching the Create canvas (there the print appears in place). */
export function shouldToastGenerationComplete(routePath: string): boolean {
  return routePath !== "/create";
}

export interface HostStatusSnapshot {
  id: string;
  label: string;
  status: string;
}

/**
 * Hosts that just went from reachable to offline (ready → error). Restricting
 * to that edge means a boot-time failure (handled by the hosts store) and a
 * host that stays errored across polls don't re-toast — the transition fires
 * exactly once.
 */
export function detectOfflineTransitions(
  previous: Record<string, string>,
  current: HostStatusSnapshot[],
): HostStatusSnapshot[] {
  return current.filter((h) => h.status === "error" && previous[h.id] === "ready");
}

/** Snapshot the current statuses so the next poll can diff against them. */
export function snapshotHostStatuses(current: HostStatusSnapshot[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const h of current) out[h.id] = h.status;
  return out;
}

/** Cap a badge count so the nav pill stays legible. */
export function badgeCount(n: number): string | number | undefined {
  if (n <= 0) return undefined;
  return n > 99 ? "99+" : n;
}
