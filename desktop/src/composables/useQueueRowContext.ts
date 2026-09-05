import { computed, onScopeDispose, ref, watch, type ComputedRef } from "vue";
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

  /**
   * A 1s clock, running only while some host predicts a finish. An ETA is a
   * countdown, and `Date.now()` captured inside a computed only moves when the
   * queue store is reassigned — which is a 5s poll at best, and nothing at all
   * on a host that is quietly working, so "about 12s left" sat frozen.
   */
  const now = ref(Date.now());
  const predicting = computed(() =>
    Object.values(jobs.queues).some((snapshot) => snapshot?.plan != null),
  );
  let clock: ReturnType<typeof setInterval> | null = null;
  function stopClock() {
    if (clock !== null) clearInterval(clock);
    clock = null;
  }
  watch(
    predicting,
    (on) => {
      if (!on) return stopClock();
      if (clock !== null) return;
      now.value = Date.now();
      clock = setInterval(() => (now.value = Date.now()), 1000);
    },
    { immediate: true },
  );
  onScopeDispose(stopClock);

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
    const at = now.value;
    return (row: QueueRow): QueueRowContext => {
      const ref = serverRef(row);
      if (!ref) return {};
      const finish = queueWorkItemFor(queues[ref.hostId]?.plan, ref.id)?.estimated_finish_unix_ms;
      return {
        wait: queueStatusFor(index, ref.hostId, ref.id),
        etaSeconds:
          typeof finish === "number" && Number.isFinite(finish)
            ? Math.max(0, Math.round((finish - at) / 1000))
            : null,
        queuePaused: queues[ref.hostId]?.paused === true,
      };
    };
  });

  const totalEtaSeconds = computed(() => {
    const at = now.value;
    let last: number | null = null;
    for (const snapshot of Object.values(jobs.queues)) {
      const finish = snapshot?.plan ? predictedCompletionUnixMs(snapshot.plan, at) : null;
      if (finish !== null) last = last === null ? finish : Math.max(last, finish);
    }
    return last === null ? null : Math.round((last - at) / 1000);
  });

  return { contextFor, totalEtaSeconds };
}
