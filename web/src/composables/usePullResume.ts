/*
 * Pull-and-resume for a missing-model generation on web — the browser half of
 * the desktop `pullResume` store (#1162). When Create routes a print that no
 * reachable machine can run because none of them has the model, the user is
 * offered the pull; this owns the "generate once it's ready" promise so the
 * page can navigate away without orphaning it.
 *
 * The which-job decision is shared with desktop (`@studio/lib/pullResume`);
 * only the plumbing differs. The origin machine already streams its downloads
 * into `useDownloads()`, so it is read from there; any other machine is polled
 * over its authenticated `/api/downloads`.
 */
import { ref, type Ref } from "vue";
import {
  pullResumeFailureMessage,
  resolvePullResumeOutcome,
  terminalPullJobIds,
  type PullResumeJob,
} from "@studio/lib/pullResume";
import { getHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { hostDownloads } from "../components/machines/hostClient";
import { toast } from "../lib/toasts";
import { useDownloads } from "./useDownloads";

const POLL_INTERVAL_MS = 2000;

export interface PendingPull {
  model: string;
  /** The enqueued pull's job id; null when the server didn't report one. */
  jobId: string | null;
  /** Registry host id the pull was started on (origin included). */
  hostId: string;
  hostLabel: string;
  /** Submit the frozen request on the frozen route. Runs once, when ready. */
  resume: () => void;
}

export interface UsePullResume {
  pending: Ref<PendingPull | null>;
  /**
   * `baseline` is the machine's already-terminal job ids, captured BEFORE the
   * download was started. Capturing it afterwards is a hang: a pull that
   * finishes inside that window lands in the baseline and is then excluded
   * forever. Pass `captureBaseline(hostId)`'s result.
   */
  arm: (next: PendingPull, baseline?: readonly string[]) => Promise<void>;
  captureBaseline: (hostId: string) => Promise<string[]>;
  cancel: (expected?: PendingPull) => void;
  /** Test seam: run one poll now instead of waiting for the interval. */
  check: () => Promise<void>;
}

const pending = ref<PendingPull | null>(null);
let seenTerminal: string[] = [];
let timer: ReturnType<typeof setInterval> | undefined;

async function jobsFor(hostId: string): Promise<PullResumeJob[]> {
  if (hostId === ORIGIN_HOST_ID) {
    const downloads = useDownloads();
    return [
      ...downloads.activeJobs.value,
      ...downloads.queued.value,
      ...downloads.history.value,
    ];
  }
  const entry = getHost(hostId);
  if (!entry) return [];
  try {
    const listing = await hostDownloads(entry);
    return [
      ...(listing.active_jobs ?? []),
      ...(listing.active ? [listing.active] : []),
      ...listing.queued,
      ...listing.history,
    ];
  } catch {
    // A blip is not evidence that the pull settled — keep waiting.
    return [];
  }
}

function stop(): void {
  if (timer !== undefined) clearInterval(timer);
  timer = undefined;
}

async function check(): Promise<void> {
  const watched = pending.value;
  if (!watched) {
    stop();
    return;
  }
  const outcome = resolvePullResumeOutcome(await jobsFor(watched.hostId), {
    model: watched.model,
    jobId: watched.jobId,
    seenTerminal,
  });
  if (outcome.kind === "waiting") return;
  // Re-read: an await boundary means the watch may have been replaced.
  if (pending.value !== watched) return;
  pending.value = null;
  stop();
  if (outcome.kind === "ready") {
    toast(
      "success",
      `${watched.model} is ready on ${watched.hostLabel} — generating`,
    );
    watched.resume();
    return;
  }
  toast("error", pullResumeFailureMessage(watched.model, outcome.job));
}

export function usePullResume(): UsePullResume {
  return {
    pending,
    async arm(next: PendingPull, baseline?: readonly string[]) {
      seenTerminal = [
        ...(baseline ?? terminalPullJobIds(await jobsFor(next.hostId))),
      ];
      pending.value = next;
      stop();
      timer = setInterval(() => void check(), POLL_INTERVAL_MS);
      await check();
    },
    async captureBaseline(hostId: string) {
      return terminalPullJobIds(await jobsFor(hostId));
    },
    cancel(expected?: PendingPull) {
      if (expected && pending.value !== expected) return;
      pending.value = null;
      stop();
    },
    check,
  };
}
