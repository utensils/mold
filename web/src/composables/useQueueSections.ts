import { computed, type Ref } from "vue";
import type { FleetActiveWork } from "@studio/api/activity";
import { QUEUE_SECTIONS, queueSection } from "@studio/lib/queueSections";
import {
  queueStatusFor,
  type QueueStatusIndex,
} from "@studio/lib/queuePosition";
import { ORIGIN_HOST_ID } from "../lib/hostRegistry";
import type { Job } from "./useGenerateStream";

/** Presentation over the deduplicated live sources; no second queue store. */
export function useQueueSections(
  jobs: Ref<Job[]>,
  rows: Ref<FleetActiveWork[]>,
  statuses: Ref<QueueStatusIndex | null>,
  hosts: Ref<Array<{ id: string; status: string }>>,
) {
  return computed(() =>
    QUEUE_SECTIONS.map((section) => {
      const local = jobs.value.filter((job) => {
        if (job.state !== "running" && !(job.state === "error" && job.error))
          return false;
        const status = queueStatusFor(
          statuses.value,
          job.hostId ?? ORIGIN_HOST_ID,
          job.serverId,
        );
        return (
          queueSection(
            job.state === "error"
              ? "error"
              : (status?.state ?? (job.workStarted ? "running" : "queued")),
            !hosts.value.some(
              (host) =>
                host.id === (job.hostId ?? ORIGIN_HOST_ID) &&
                host.status === "ready",
            ),
            job.holdError ? true : (status?.blockedReason ?? false),
          ) === section.id
        );
      });
      const shared = rows.value.filter((row) => {
        if (row.kind === "download") return false;
        const status = queueStatusFor(statuses.value, row.hostId, row.id);
        return (
          queueSection(
            status?.state ?? row.phase,
            row.stale,
            status?.blockedReason ?? false,
          ) === section.id
        );
      });
      return { ...section, local, shared, count: local.length + shared.length };
    }),
  );
}
