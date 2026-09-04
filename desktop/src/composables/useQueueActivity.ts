import { computed, type ComputedRef } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import {
  mergeActivity,
  partitionActivity,
  sequenceToVM,
  type ActivityJobVM,
} from "@studio/lib/activity";
import { compareNewestSubmitted } from "@studio/lib/activityOrder";
import { useChainJobsStore } from "../stores/chainJobs";
import {
  GENERATION_HISTORY_LIMIT,
  railOrder,
  useGenerationStore,
  type Job,
} from "../stores/generation";
import { useHostsStore } from "../stores/hosts";
import { useLiveActivityStore } from "../stores/liveActivity";

export type SequenceVM = ActivityJobVM & { kind: "sequence" };

/** One row of the queue, whichever authority it came from. */
export type QueueRow =
  | { key: string; createdAtMs: number; kind: "shared"; shared: FleetActiveWork }
  | { key: string; createdAtMs: number; kind: "sequence"; sequence: SequenceVM }
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
  if (row.kind === "sequence")
    return row.sequence.phase === "running" || row.sequence.phase === "finalizing";
  return row.shared.phase === "running" || row.shared.phase === "cancelling";
}

export function rowSettled(row: QueueRow): boolean {
  return row.kind === "print" && jobSettled(row.print);
}

export interface QueueActivity {
  /** Every row newest-first: recovered work, sequences, prints, then the
   * retained finished-print history. */
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
 * Queue view is the same list at full width. Merges this
 * client's prints, its sequences, and the fleet's recovered work into one
 * newest-first timeline, deduplicating rows this client already owns.
 */
export function useQueueActivity(): QueueActivity {
  const generation = useGenerationStore();
  const chains = useChainJobsStore();
  const hosts = useHostsStore();
  const liveActivity = useLiveActivityStore();

  /** Every live job newest-first, then the recent finished prints. */
  const prints = computed<Job[]>(() => {
    const live = railOrder(generation.jobs.filter((j) => !jobSettled(j)));
    const done = generation.jobs.filter(jobSettled).slice(-GENERATION_HISTORY_LIMIT).reverse();
    return [...live, ...done];
  });

  /** Live sequences only — settled ones already have two homes (the print in
   * My images, the job in History ▸ Sequences). */
  const sequences = computed(() =>
    partitionActivity(
      mergeActivity(
        [],
        chains.allJobs.map(({ hostId, job }) =>
          sequenceToVM(job, {
            hostId,
            hostLabel: hosts.all.find((h) => h.id === hostId)?.label ?? hostId,
          }),
        ),
      ),
    ).active.filter((vm): vm is SequenceVM => vm.kind === "sequence"),
  );

  /** Server-owned work that survives this client's restart, minus rows the
   * local print/sequence state already renders. */
  const shared = computed(() => {
    const primaryId = hosts.primaryHost?.id ?? "local";
    const local = new Set(
      generation.jobs.flatMap((job) =>
        job.id ? [`${job.hostId ?? primaryId}:generation:${job.id}`] : [],
      ),
    );
    for (const { hostId, job } of chains.allJobs) local.add(`${hostId}:sequence:${job.id}`);
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
      ...sequences.value.map((sequence): QueueRow => ({
        key: sequence.key,
        createdAtMs: sequence.createdAtMs,
        kind: "sequence",
        sequence,
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
