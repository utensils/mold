import type { Ref } from "vue";
import { fetchQueue, listGalleryFrom } from "../api";
import { getHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import type { StreamTarget } from "../api";
import type { Job, UseGenerateStream } from "./useGenerateStream";

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

/// One read-only round of reconciliation. Returns jobs that are no longer
/// present server-side; the generation-stream owner performs settlement so
/// terminal metadata and canvas error authority cannot diverge.
///
/// Takes the current `now` as a parameter (rather than reading `Date.now()`
/// internally) so tests can simulate the grace window deterministically.
export function reconcileRound(
  jobs: Job[],
  serverIds: ReadonlySet<string>,
  now: number,
): Job[] {
  const missing: Job[] = [];
  for (const j of jobs) {
    if (j.state !== "running") continue;
    if (!j.serverId) continue; // Request has not received its queued frame yet.
    if (serverIds.has(j.serverId)) continue; // server confirms it's alive
    if (now - j.lastProgressAt < RECONCILE_GRACE_MS) continue;
    missing.push(j);
  }
  return missing;
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
/** A detached job that vanished from the queue very likely FINISHED while
 * the page was away — never announce it as a generation failure. */
export const DETACHED_SETTLE_NOTE =
  "This ran while the page was away and has since finished or stopped on the server — check the Library for the result.";

/** Resolve the route for a job whose in-memory target died with its session
 * (API keys are never persisted): look the host back up in the registry so a
 * reloaded page reconciles against the machine that actually holds the job,
 * not the origin's unrelated queue. */
export function targetForJob(job: Job): Job["target"] {
  if (job.target) return job.target;
  if (!job.hostId || job.hostId === ORIGIN_HOST_ID) return null;
  const host = getHost(job.hostId);
  if (!host) return null;
  const target: StreamTarget = { baseUrl: host.url };
  if (host.apiKey) target.apiKey = host.apiKey;
  return target;
}

export function startQueueReconciler(
  jobs: Ref<Job[]>,
  failRunning: (id: string, error: string) => void,
  options: {
    intervalMs?: number;
    settleDetached?: (id: string, note: string) => void;
    resolveTarget?: (job: Job) => Job["target"];
  } = {},
): QueueReconcilerHandle {
  const intervalMs = options.intervalMs ?? RECONCILE_INTERVAL_MS;
  const settleDetached = options.settleDetached ?? failRunning;
  const resolveTarget = options.resolveTarget ?? targetForJob;
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
    const groups = new Map<string, { target: Job["target"]; jobs: Job[] }>();
    for (const job of candidates) {
      const key = job.hostId ?? "__origin__";
      const group = groups.get(key) ?? { target: resolveTarget(job), jobs: [] };
      group.jobs.push(job);
      groups.set(key, group);
    }
    const results = await Promise.allSettled(
      [...groups.values()].map(async (group) => {
        const listing = await fetchQueue(group.target ?? undefined);
        const known = new Set(listing.entries.map((e) => e.id));
        const missing = reconcileRound(group.jobs, known, Date.now());
        if (missing.length === 0) return;
        // A row leaves the queue for two reasons this loop cannot tell apart:
        // the job died, or it FINISHED. Absence decides neither — so ask the
        // host what it actually has. A print stamped with the job's own id is
        // proof it finished; the host's `durable` promise is proof it will.
        // Without either, the failure path stands: soft-settling a job that
        // genuinely died would retire its row from the strip and show no
        // output, leaving the user with silence instead of a wrong message.
        const produced = await listGalleryFrom(group.target ?? undefined)
          .then(
            (prints) =>
              new Set(
                prints
                  .map((print) => print.metadata?.job_id)
                  .filter((id): id is string => !!id),
              ),
          )
          .catch(() => new Set<string>());
        for (const job of missing) {
          const finished = !!job.serverId && produced.has(job.serverId);
          if (job.detached || job.durable === true || finished) {
            settleDetached(job.id, DETACHED_SETTLE_NOTE);
          } else {
            failRunning(job.id, "job not found on server — connection lost");
          }
        }
      }),
    );
    if (results.some((result) => result.status === "rejected")) {
      // A host is unreachable or returned 5xx. Never use another host's empty
      // queue to dead-letter its jobs; back off and try that exact route later.
      consecutiveFailures += 1;
    } else {
      consecutiveFailures = 0;
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

/** Wire the App-level generation stream to reconciliation without duplicating
 * its ownership boundary at the composition root. */
export function startGenerateQueueReconciler(
  stream: Pick<UseGenerateStream, "jobs" | "failRunning" | "settleDetached">,
  options: { intervalMs?: number } = {},
): QueueReconcilerHandle {
  return startQueueReconciler(stream.jobs, stream.failRunning, {
    ...options,
    settleDetached: stream.settleDetached,
  });
}
