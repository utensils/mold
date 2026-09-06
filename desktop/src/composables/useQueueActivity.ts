import { computed, type ComputedRef } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import { compareNewestSubmitted } from "@studio/lib/activityOrder";
import {
  GENERATION_HISTORY_LIMIT,
  railOrder,
  useGenerationStore,
  type Job,
} from "../stores/generation";
import { useHostsStore } from "../stores/hosts";
import { useLiveActivityStore } from "../stores/liveActivity";

/**
 * One row of the queue, whichever authority it came from.
 *
 * There are two, not three: this client's own prints, and the server-owned
 * work `/api/activity` reports. Scene-by-scene authoring is retired, so no
 * client-side sequence row exists any more — a long clip the host has to
 * chain and stitch is still ONE print row carrying its stage counter.
 */
export type QueueRow =
  | { key: string; createdAtMs: number; kind: "shared"; shared: FleetActiveWork }
  | { key: string; createdAtMs: number; kind: "print"; print: Job };

export function jobRunning(job: Job): boolean {
  return job.status === "denoising" || job.status === "finishing" || job.status === "loading";
}

export function jobSettled(job: Job): boolean {
  return job.status === "complete" || job.status === "error";
}

/** Whether a row is being made right now (as opposed to waiting or done). */
export function rowRunning(row: QueueRow): boolean {
  if (row.kind === "print") return jobRunning(row.print);
  return row.shared.phase === "running" || row.shared.phase === "cancelling";
}

export function rowSettled(row: QueueRow): boolean {
  return row.kind === "print" && jobSettled(row.print);
}

export interface QueueActivity {
  /** Every row newest-first: recovered work, prints, then the retained
   * finished-print history. */
  rows: ComputedRef<QueueRow[]>;
  /** The row being made now, if any (the sidebar's active card). */
  active: ComputedRef<QueueRow | null>;
  activeCount: ComputedRef<number>;
  waitingCount: ComputedRef<number>;
  /** Everything in flight — the badge on the Queue destination. */
  liveCount: ComputedRef<number>;
}

/**
 * The queue as every shell surface sees it: it lives in the sidebar, and the
 * Queue view is the same list at full width. Merges this client's prints and
 * the fleet's recovered work into one newest-first timeline, deduplicating
 * rows this client already owns.
 */
export function useQueueActivity(): QueueActivity {
  const generation = useGenerationStore();
  const hosts = useHostsStore();
  const liveActivity = useLiveActivityStore();

  /** Every live job newest-first, then the recent finished prints. */
  const prints = computed<Job[]>(() => {
    const live = railOrder(generation.jobs.filter((j) => !jobSettled(j)));
    const done = generation.jobs.filter(jobSettled).slice(-GENERATION_HISTORY_LIMIT).reverse();
    return [...live, ...done];
  });

  /** Server-owned work that survives this client's restart, minus rows the
   * local print state already renders. */
  const shared = computed(() => {
    const primaryId = hosts.primaryHost?.id ?? "local";
    const local = new Set(
      generation.jobs.flatMap((job) =>
        job.id ? [`${job.hostId ?? primaryId}:generation:${job.id}`] : [],
      ),
    );
    // A long clip this client submitted is a chain job on the host, so
    // `/api/activity` reports it under the sequence key too. It is already on
    // screen as this client's own print row.
    for (const job of generation.jobs) {
      if (job.id) local.add(`${job.hostId ?? primaryId}:sequence:${job.id}`);
    }
    return liveActivity.rows.filter((row) => !local.has(row.key));
  });

  const rows = computed<QueueRow[]>(() =>
    [
      ...shared.value.map((work): QueueRow => ({
        key: `shared:${work.key}`,
        createdAtMs: work.created_at_unix_ms,
        kind: "shared",
        shared: work,
      })),
      ...prints.value.map((print): QueueRow => ({
        key: `print:${print.clientId}`,
        createdAtMs: print.submittedAtUnixMs,
        kind: "print",
        print,
      })),
    ].sort(compareNewestSubmitted),
  );

  const running = computed(() => rows.value.filter(rowRunning));
  const active = computed(() => running.value[0] ?? null);
  const waiting = computed(() => rows.value.filter((row) => !rowRunning(row) && !rowSettled(row)));

  return {
    rows,
    active,
    activeCount: computed(() => running.value.length),
    waitingCount: computed(() => waiting.value.length),
    liveCount: computed(() => running.value.length + waiting.value.length),
  };
}
