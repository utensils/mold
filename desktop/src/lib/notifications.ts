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

/*
 * Host reachability lives in the shared studio policy so web and desktop
 * narrate the same two edges — a drop is a WARNING (the poll keeps retrying on
 * its own), a recovery is a SUCCESS. Re-exported here because the App shell
 * reads its notification helpers from one module.
 */
export {
  HOST_OFFLINE_DESCRIPTION,
  HOST_RECONNECTING_LABEL,
  detectOfflineTransitions,
  detectReconnectTransitions,
  hostOfflineTitle,
  hostReconnectedTitle,
  snapshotHostStatuses,
  type HostStatusSnapshot,
} from "@studio/lib/hostConnectivity";

/** Cap a badge count so the nav pill stays legible. */
export function badgeCount(n: number): string | number | undefined {
  if (n <= 0) return undefined;
  return n > 99 ? "99+" : n;
}
