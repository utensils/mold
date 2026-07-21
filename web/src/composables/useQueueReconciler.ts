import type { Ref } from "vue";
import { fetchQueue } from "../api";
import type { Job } from "./useGenerateStream";

/// How long to poll `GET /api/queue` between rounds when the server is
/// reachable. Reconciliation is a safety net behind the L1 SSE silent-close
/// fix — every individual stream drop should already be caught there. This
/// loop only sweeps the edge cases where the SSE never errored *and* the
/// server-side registry has no record (page reload, server restart, browser
/// tab suspended past keepalive).
///
/// 30 s is a deliberately conservative cadence — high enough that the SPA
/// doesn't hammer `/api/queue` while a long generation churns; low enough
/// that a stuck card surfaces well before the user starts wondering whether
/// to refresh.
export const RECONCILE_INTERVAL_MS = 30_000;

/// Grace window before a "running" card without server backing is flipped
/// to error. Covers the gap between submit() opening an SSE connection and
/// the first `queued` event landing — during that gap the server already
/// has the job registered but the client hasn't captured its `serverId`
/// yet. Without a grace period a poll that fires inside that window would
/// dead-letter a perfectly healthy job.
export const RECONCILE_GRACE_MS = 30_000;

/// On poll failure (network blip, server restart) we never dead-letter —
/// the SSE error path is the canonical signal for "this card is dead." The
/// reconciler just sleeps a bit longer and retries.
export const RECONCILE_BACKOFF_MAX_MS = 120_000;

/// One round of reconciliation. Exported so unit tests can drive the
/// logic synchronously without wrangling timers.
///
/// Takes the current `now` as a parameter (rather than reading `Date.now()`
/// internally) so tests can simulate the grace window deterministically.
export function reconcileRound(
  jobs: Job[],
  serverIds: ReadonlySet<string>,
  now: number,
): void {
  for (const j of jobs) {
    if (j.state !== "running") continue;
    if (!j.serverId) continue; // Request has not received its queued frame yet.
    if (serverIds.has(j.serverId)) continue; // server confirms it's alive
    if (now - j.lastProgressAt < RECONCILE_GRACE_MS) continue;
    j.state = "error";
    j.error = "job not found on server — connection lost";
  }
}

export interface QueueReconcilerHandle {
  /** Stop the polling loop. Idempotent. */
  stop: () => void;
}

/// Start the reconciliation loop. Returns a handle whose `stop()` tears it
/// down — call this from `App.vue`'s `onBeforeUnmount` (or never, since the
/// SPA root unmount only happens on page navigation, which destroys the
/// timer with the tab anyway).
///
/// The loop exits early when there are no running jobs to check — no
/// `/api/queue` call goes out unless we have something to reconcile.
export function startQueueReconciler(
  jobs: Ref<Job[]>,
  options: { intervalMs?: number } = {},
): QueueReconcilerHandle {
  const intervalMs = options.intervalMs ?? RECONCILE_INTERVAL_MS;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let consecutiveFailures = 0;

  async function tick() {
    if (stopped) return;
    const candidates = jobs.value.filter(
      (j) => j.state === "running" && j.serverId,
    );
    if (candidates.length === 0) {
      schedule(intervalMs);
      return;
    }
    try {
      const listing = await fetchQueue();
      consecutiveFailures = 0;
      const known = new Set(listing.entries.map((e) => e.id));
      reconcileRound(jobs.value, known, Date.now());
    } catch {
      // Server unreachable or 5xx. Don't dead-letter — could be a blip;
      // the per-job SSE error path is the real source of truth for "this
      // job died." Back off exponentially up to RECONCILE_BACKOFF_MAX_MS.
      consecutiveFailures += 1;
    }
    const backoff = Math.min(
      intervalMs * Math.pow(2, consecutiveFailures),
      RECONCILE_BACKOFF_MAX_MS,
    );
    schedule(backoff);
  }

  function schedule(delay: number) {
    timer = setTimeout(tick, delay);
  }

  schedule(intervalMs);

  return {
    stop() {
      stopped = true;
      if (timer) clearTimeout(timer);
      timer = null;
    },
  };
}
