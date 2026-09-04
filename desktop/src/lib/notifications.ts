import { reconcileHostConnectivity, type HostStatusSnapshot } from "@studio/lib/hostConnectivity";
import type { Job } from "./generationJob";

/**
 * Cross-surface notification helpers. Pure so the "toast only when
 * off-canvas" and "offline fires once" guarantees are testable without wiring
 * Vue watchers. The App shell owns the seen-sets and the actual toast/native
 * dispatch.
 */

/** Jobs that completed since the last pass (by clientId). The caller adds the
 *  returned ids to `seen` after handling so each completion notifies once. */
export function newlyCompletedJobs(jobs: Job[], seen: Set<number>): Job[] {
  return jobs.filter(
    (j) => j.status === "complete" && !j.suppressFreshCompletion && !seen.has(j.clientId),
  );
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
  hostOfflineTitle,
  hostReconnectedTitle,
  type HostStatusSnapshot,
} from "@studio/lib/hostConnectivity";

export interface HostConnectivityEffects {
  /** Raise the sticky offline warning; returns the toast id to retain. */
  warn: (host: HostStatusSnapshot) => number;
  /** Confirm a recovery for a host we previously warned about. */
  announceRecovery: (host: HostStatusSnapshot) => void;
  /** Withdraw a retained toast (a no-op if it is already gone). */
  dismiss: (toastId: number) => void;
}

/**
 * Apply one pass of the shared reachability policy to the desktop toast shelf
 * and return the snapshot for the next pass. The retained toast ids live in
 * `warned`, which this mutates: a warning is remembered until the host answers
 * or leaves the list, and both of those withdraw it.
 *
 * Extracted from App.vue so the id bookkeeping is testable without mounting
 * the whole shell.
 */
export function applyHostConnectivity(
  previous: Readonly<Record<string, string>>,
  current: readonly HostStatusSnapshot[],
  warned: Map<string, number>,
  effects: HostConnectivityEffects,
): Record<string, string> {
  const changes = reconcileHostConnectivity({
    previous,
    current,
    warned: warned.keys(),
    // The boot probe is deliberately quiet; the Machines workspace already
    // shows an errored row for a host that was never reachable.
    warnOnFirstContact: false,
  });
  const retire = (id: string) => {
    const toastId = warned.get(id);
    if (toastId !== undefined) effects.dismiss(toastId);
    warned.delete(id);
  };
  for (const host of changes.offline) warned.set(host.id, effects.warn(host));
  for (const host of changes.reconnected) {
    retire(host.id);
    effects.announceRecovery(host);
  }
  // A host disconnected or forgotten while offline stops being polled, so
  // nothing could ever withdraw its warning — retire it with the entry.
  for (const id of changes.retired) retire(id);
  return changes.next;
}

/** Cap a badge count so the nav pill stays legible. */
export function badgeCount(n: number): string | number | undefined {
  if (n <= 0) return undefined;
  return n > 99 ? "99+" : n;
}
