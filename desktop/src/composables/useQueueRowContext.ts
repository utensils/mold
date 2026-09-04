import { computed, type ComputedRef } from "vue";
import { predictedCompletionUnixMs } from "@studio/api/queuePlan";
import { buildQueueStatusIndex, queueStatusFor, queueWorkItemFor } from "@studio/lib/queuePosition";
import { useHostsStore } from "../stores/hosts";
import { useJobsStore } from "../stores/jobs";
import type { QueueRow } from "./useQueueActivity";
import type { QueueRowContext } from "../lib/queueRows";

export interface QueueRowContextReader {
  /** Everything the fleet says about one row beyond the job itself. */
  contextFor: ComputedRef<(row: QueueRow) => QueueRowContext>;
  /** Seconds until the last finish any host predicts, or null where none does. */
  totalEtaSeconds: ComputedRef<number | null>;
}

/**
 * The host-side half of a queue row: its live listing position and parked
 * reason, and the finish time the host's own planner predicts. Both the
 * sidebar rail and the Queue view read this, so the shell says exactly what
 * the Machines queue panel says about the same job.
 */
export function useQueueRowContext(): QueueRowContextReader {
  const hosts = useHostsStore();
  const jobs = useJobsStore();

  const statusIndex = computed(() =>
    buildQueueStatusIndex(
      Object.entries(jobs.queues).map(([hostId, snapshot]) => ({
        hostId,
        entries: snapshot?.entries ?? [],
        plan: snapshot?.plan ?? null,
        paused: snapshot?.paused,
      })),
    ),
  );

  /** The host and server id a row's queue entry lives under, when it has one. */
  function serverRef(row: QueueRow): { hostId: string; id: string } | null {
    if (row.kind === "print") {
      return row.print.id
        ? { hostId: row.print.hostId ?? hosts.primaryHost?.id ?? "local", id: row.print.id }
        : null;
    }
    if (row.kind === "shared" && row.shared.kind === "generation") {
      return { hostId: row.shared.hostId, id: row.shared.id };
    }
    return null;
  }

  const contextFor = computed(() => {
    const index = statusIndex.value;
    const queues = jobs.queues;
    const now = Date.now();
    return (row: QueueRow): QueueRowContext => {
      const ref = serverRef(row);
      if (!ref) return {};
      const finish = queueWorkItemFor(queues[ref.hostId]?.plan, ref.id)?.estimated_finish_unix_ms;
      return {
        wait: queueStatusFor(index, ref.hostId, ref.id),
        etaSeconds:
          typeof finish === "number" && Number.isFinite(finish)
            ? Math.max(0, Math.round((finish - now) / 1000))
            : null,
        queuePaused: queues[ref.hostId]?.paused === true,
      };
    };
  });

  const totalEtaSeconds = computed(() => {
    const now = Date.now();
    let last: number | null = null;
    for (const snapshot of Object.values(jobs.queues)) {
      const at = snapshot?.plan ? predictedCompletionUnixMs(snapshot.plan, now) : null;
      if (at !== null) last = last === null ? at : Math.max(last, at);
    }
    return last === null ? null : Math.round((last - now) / 1000);
  });

  return { contextFor, totalEtaSeconds };
}
