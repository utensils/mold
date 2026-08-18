import { defineStore } from "pinia";
import { listDevices, type DeviceInfo } from "@studio/api/devices";
import {
  parseQueueListing,
  predictedCompletionUnixMs,
  type QueuePlan,
} from "@studio/api/queuePlan";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { OutputMetadata, ServerStatus } from "../lib/api/types";
import { useGenerationStore } from "./generation";
import { useHostsStore, type HostView } from "./hosts";
import { useToastStore } from "./toasts";
import type { Job } from "./generation";

const POLL_INTERVAL_MS = 5_000;

/** One row of a host's `GET /api/queue` listing. */
export interface QueueEntry {
  id: string;
  model: string;
  /** `held` is additive: a journalled job the host parked after it exhausted
   * its replay or dispatch budget. It exists only in the durable queue, will
   * never start on its own, and is listed precisely so it is not invisible. */
  state: "queued" | "running" | "held";
  started_at_unix_ms: number;
  position: number;
  gpu?: number;
  target_gpu?: number;
  /** Whether the request pinned a seed — distinguishes an explicit seed 0
   * from "let the server pick" (additive; absent on older servers). */
  seed_pinned?: boolean | null;
  /** The submitted request's settings, metadata-shaped (additive; absent on
   * older servers; never carries image payloads). */
  metadata?: OutputMetadata | null;
  /** Why the host parked this job. Present only for `state: "held"`. */
  held_reason?: string | null;
}

/** Queue controls a host supports (from `GET /api/capabilities`). */
export interface HostQueueCaps {
  canPause: boolean;
  canCancelAll: boolean;
  /** Whether queued jobs can be reordered via `PATCH /api/queue/:id {position}`
   *  (older servers never report it → false). */
  canReorder: boolean;
  /** Additive capability; false/absent keeps older hosts' running rows read-only. */
  canCancelRunning?: boolean;
}

export interface HostQueueSnapshot {
  hostId: string;
  entries: QueueEntry[];
  /** null until the host reports it (older servers never do). */
  paused: boolean | null;
  caps: HostQueueCaps | null;
  /** Schedulable GPU ordinals from `/api/devices`, with `/api/status.gpus`
   * fallback for older servers. */
  gpuOrdinals: number[];
  devices?: DeviceInfo[] | null;
  plan?: QueuePlan | null;
  error: string | null;
}

/** A queue row joined with this app's own job (when it is ours). */
export interface EnrichedQueueEntry extends QueueEntry {
  mine: boolean;
  clientId: number | null;
}

/**
 * One row of the unified cross-host queue surface. Both the Machines queue
 * column and the Create activity strip render this exact shape so the two
 * surfaces never disagree about what is running or queued (G2).
 */
export interface QueueSurfaceRow {
  hostId: string;
  hostLabel: string;
  entry: EnrichedQueueEntry;
  /** Whether this host supports queued-job reordering (capability-gated). */
  canReorder: boolean;
  canCancelRunning: boolean;
}

/** Running first, then queued, then held — held work is parked, not in line,
 *  so it must never sort among rows that are actually waiting for a lane. */
function queueSurfaceRank(entry: EnrichedQueueEntry): number {
  if (entry.state === "running") return 0;
  return entry.state === "held" ? 2 : 1;
}

/**
 * Join a host's server-side queue listing with this app's jobs. An entry is
 * ours only when both the server id AND the host match — ids from different
 * hosts live in different id spaces.
 */
export function enrichQueueEntries(
  entries: QueueEntry[],
  hostId: string,
  localJobs: Job[],
  primaryHostId: string | null,
): EnrichedQueueEntry[] {
  return entries.map((entry) => {
    const owner = localJobs.find(
      (j) => j.id === entry.id && (j.hostId ?? primaryHostId) === hostId,
    );
    return { ...entry, mine: !!owner, clientId: owner?.clientId ?? null };
  });
}

/**
 * Server-queue mirror per connected host — the Jobs view's data source.
 * Unlike the generation store (this app's own jobs), this store sees the
 * WHOLE queue on every host, other clients' work included.
 */
export const useJobsStore = defineStore("jobs", {
  state: () => ({
    queues: {} as Record<string, HostQueueSnapshot>,
    requestGenerations: {} as Record<string, number>,
    pollTimer: null as ReturnType<typeof setInterval> | null,
    /** Views currently asking for the cross-host poll. Machines wants it while
     * it is open and Create wants it only while something is queued, so the
     * one that leaves first must not silence the other. */
    pollConsumers: 0,
  }),
  getters: {
    /**
     * Unified running+queued rows across every connected host, joined with
     * this app's own jobs. Running rows first, then by queue position. The
     * single source both the Machines queue column and the Create activity
     * strip render, so the two surfaces show identical state (G2).
     */
    queueSurface(state): QueueSurfaceRow[] {
      const hosts = useHostsStore();
      const generation = useGenerationStore();
      const primaryId = hosts.primaryHost?.id ?? "local";
      const rows: QueueSurfaceRow[] = [];
      for (const host of hosts.all) {
        const snap = state.queues[host.id];
        if (!snap) continue;
        const canReorder = snap.caps?.canReorder ?? false;
        const canCancelRunning = snap.caps?.canCancelRunning ?? false;
        for (const entry of enrichQueueEntries(snap.entries, host.id, generation.jobs, primaryId)) {
          rows.push({
            hostId: host.id,
            hostLabel: host.label,
            entry,
            canReorder,
            canCancelRunning,
          });
        }
      }
      return rows.sort(
        (a, b) =>
          queueSurfaceRank(a.entry) - queueSurfaceRank(b.entry) ||
          a.entry.position - b.entry.position,
      );
    },
  },
  actions: {
    targetFor(host: HostView): ApiTarget | null {
      return host.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    /** Snapshot ONE host's queue — the host detail page polls just its host. */
    async refreshHost(host: HostView) {
      const target = this.targetFor(host);
      if (!target || host.status !== "ready") return;
      const generation = (this.requestGenerations[host.id] ?? 0) + 1;
      this.requestGenerations[host.id] = generation;
      const hosts = useHostsStore();
      const isCurrent = () => {
        const current = hosts.all.find((candidate) => candidate.id === host.id);
        return (
          this.requestGenerations[host.id] === generation &&
          current?.baseUrl === host.baseUrl &&
          current.apiKey === host.apiKey
        );
      };
      const previous = this.queues[host.id];
      try {
        // Capabilities are fetched once per host and cached — they only
        // change across server upgrades.
        const caps =
          previous?.caps ??
          (await apiJsonTo<{
            queue?: {
              can_pause?: boolean;
              can_cancel_all?: boolean;
              can_reorder?: boolean;
              cooperative_cancellation?: boolean;
            };
          }>(target, "/api/capabilities").then(
            (c) => ({
              canPause: c.queue?.can_pause === true,
              canCancelAll: c.queue?.can_cancel_all === true,
              canReorder: c.queue?.can_reorder === true,
              canCancelRunning: c.queue?.cooperative_cancellation === true,
            }),
            () => null,
          ));
        const [listing, status, devices] = await Promise.all([
          apiJsonTo<unknown>(target, "/api/queue").then(parseQueueListing),
          apiJsonTo<ServerStatus & { queue_paused?: boolean }>(target, "/api/status"),
          listDevices(target).then(
            (snapshot) => snapshot.devices,
            () => null,
          ),
        ]);
        if (!isCurrent()) return;
        this.queues[host.id] = {
          hostId: host.id,
          entries: (listing.entries ?? [])
            .filter(
              (entry) =>
                entry.state === "queued" ||
                entry.state === "running" ||
                // Held rows exist only in the journal: dropping them here is
                // what would make the one job guaranteed never to run also the
                // one nobody can see or clear.
                entry.state === "held",
            )
            .map((entry) => {
              const { target_gpu: targetGpu, ...rest } = entry;
              const local = rest as QueueEntry;
              if (targetGpu != null) local.target_gpu = targetGpu;
              return local;
            }),
          plan: listing.plan,
          paused: status.queue_paused ?? null,
          caps,
          gpuOrdinals:
            devices !== null
              ? devices
                  .filter((device) => device.schedulable && device.ordinal !== null)
                  .map((device) => device.ordinal as number)
              : (status.gpus ?? [])
                  .filter((gpu) => gpu.state !== "degraded")
                  .map((gpu) => gpu.ordinal),
          devices,
          error: null,
        };
        const telemetry = hosts.telemetry[host.id];
        if (telemetry) {
          telemetry.predictedCompletionMs =
            listing.plan == null ? null : predictedCompletionUnixMs(listing.plan);
        }
      } catch (err) {
        if (!isCurrent()) return;
        this.queues[host.id] = {
          hostId: host.id,
          entries: previous?.entries ?? [],
          paused: previous?.paused ?? null,
          caps: previous?.caps ?? null,
          gpuOrdinals: previous?.gpuOrdinals ?? [],
          devices: previous?.devices ?? null,
          plan: previous?.plan ?? null,
          error: String(err),
        };
      }
    },
    async refresh() {
      const hosts = useHostsStore();
      await Promise.all(hosts.all.map((host) => this.refreshHost(host)));
      // Hosts that disconnected drop out of the map.
      const live = new Set(hosts.all.map((h) => h.id));
      for (const id of Object.keys(this.queues)) {
        if (!live.has(id)) {
          delete this.queues[id];
          delete this.requestGenerations[id];
        }
      }
    },
    async pause(hostId: string) {
      await this.queueControl(hostId, "/api/queue/pause", "POST");
      const q = this.queues[hostId];
      if (q) q.paused = true;
    },
    async resume(hostId: string) {
      await this.queueControl(hostId, "/api/queue/resume", "POST");
      const q = this.queues[hostId];
      if (q) q.paused = false;
    },
    /** Cancel every queued job on the host (running jobs finish). */
    async cancelAll(hostId: string) {
      await this.queueControl(hostId, "/api/queue", "DELETE");
      void this.refresh();
    },
    /** Cancel one job on a host directly (used for other clients' jobs). */
    async cancelJob(hostId: string, jobId: string) {
      await this.queueControl(hostId, `/api/queue/${encodeURIComponent(jobId)}`, "DELETE");
      void this.refresh();
    },
    /**
     * Move a queued job to another GPU lane on its OWNING host via
     * `PATCH /api/queue/:id`. Never optimistic: whatever the outcome, the
     * queue re-syncs from the server afterwards (the job may have started
     * between render and PATCH — that race answers 409 here).
     */
    async reassignGpu(hostId: string, jobId: string, targetGpu: number): Promise<boolean> {
      const toasts = useToastStore();
      try {
        await this.queueControl(hostId, `/api/queue/${encodeURIComponent(jobId)}`, "PATCH", {
          target_gpu: targetGpu,
        });
        toasts.push(`Moved to GPU ${targetGpu}`);
        return true;
      } catch (err) {
        const status = (err as { status?: number } | null)?.status ?? 0;
        const message =
          status === 404
            ? "That job is no longer queued."
            : status === 409
              ? "Job already started — lane changes only apply to queued jobs."
              : status === 422
                ? `GPU ${targetGpu} is not available on this host.`
                : String(err);
        toasts.push(message, "error");
        return false;
      } finally {
        // Server truth over local reordering, on success and failure alike.
        await this.refresh();
      }
    },
    /**
     * Move a queued job up or down its host's queue via `PATCH /api/queue/:id`
     * with a 0-based `position`. Never optimistic — the queue re-syncs from the
     * server afterwards whether the move succeeds or the job started mid-flight.
     */
    async reorderQueued(hostId: string, jobId: string, position: number): Promise<boolean> {
      const toasts = useToastStore();
      try {
        await this.queueControl(hostId, `/api/queue/${encodeURIComponent(jobId)}`, "PATCH", {
          position,
        });
        return true;
      } catch (err) {
        const status = (err as { status?: number } | null)?.status ?? 0;
        const message =
          status === 404
            ? "That job is no longer queued."
            : status === 409
              ? "Job already started — only queued jobs can be reordered."
              : String(err);
        toasts.push(message, "error");
        return false;
      } finally {
        await this.refresh();
      }
    },
    async queueControl(
      hostId: string,
      path: string,
      method: "POST" | "DELETE" | "PATCH",
      body?: unknown,
    ) {
      const hosts = useHostsStore();
      const host = hosts.all.find((h) => h.id === hostId);
      const target = host ? this.targetFor(host) : null;
      if (!target) throw new Error("Host is not connected.");
      const init: RequestInit = { method };
      if (body !== undefined) {
        init.body = JSON.stringify(body);
        init.headers = { "Content-Type": "application/json" };
      }
      await apiFetchTo(target, path, init);
    },
    startPolling() {
      this.pollConsumers += 1;
      if (this.pollTimer) return;
      void this.refresh();
      this.pollTimer = setInterval(() => void this.refresh(), POLL_INTERVAL_MS);
    },
    stopPolling() {
      this.pollConsumers = Math.max(0, this.pollConsumers - 1);
      if (this.pollConsumers > 0) return;
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = null;
    },
  },
});
