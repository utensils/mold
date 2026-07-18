import { defineStore } from "pinia";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { OutputMetadata, ServerStatus } from "../lib/api/types";
import { useHostsStore, type HostView } from "./hosts";
import { useToastStore } from "./toasts";
import type { Job } from "./generation";

const POLL_INTERVAL_MS = 5_000;

/** One row of a host's `GET /api/queue` listing. */
export interface QueueEntry {
  id: string;
  model: string;
  state: "queued" | "running";
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
}

/** Queue controls a host supports (from `GET /api/capabilities`). */
export interface HostQueueCaps {
  canPause: boolean;
  canCancelAll: boolean;
}

export interface HostQueueSnapshot {
  hostId: string;
  entries: QueueEntry[];
  /** null until the host reports it (older servers never do). */
  paused: boolean | null;
  caps: HostQueueCaps | null;
  /** GPU worker ordinals from `/api/status.gpus`; empty when unreported. */
  gpuOrdinals: number[];
  error: string | null;
}

/** A queue row joined with this app's own job (when it is ours). */
export interface EnrichedQueueEntry extends QueueEntry {
  mine: boolean;
  clientId: number | null;
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
    pollTimer: null as ReturnType<typeof setInterval> | null,
  }),
  actions: {
    targetFor(host: HostView): ApiTarget | null {
      return host.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    /** Snapshot ONE host's queue — the host detail page polls just its host. */
    async refreshHost(host: HostView) {
      const target = this.targetFor(host);
      if (!target || host.status !== "ready") return;
      const previous = this.queues[host.id];
      try {
        // Capabilities are fetched once per host and cached — they only
        // change across server upgrades.
        const caps =
          previous?.caps ??
          (await apiJsonTo<{ queue?: { can_pause?: boolean; can_cancel_all?: boolean } }>(
            target,
            "/api/capabilities",
          ).then(
            (c) => ({
              canPause: c.queue?.can_pause === true,
              canCancelAll: c.queue?.can_cancel_all === true,
            }),
            () => null,
          ));
        const [listing, status] = await Promise.all([
          apiJsonTo<{ entries: QueueEntry[] }>(target, "/api/queue"),
          apiJsonTo<ServerStatus & { queue_paused?: boolean }>(target, "/api/status"),
        ]);
        this.queues[host.id] = {
          hostId: host.id,
          entries: listing.entries,
          paused: status.queue_paused ?? null,
          caps,
          gpuOrdinals: (status.gpus ?? []).map((g) => g.ordinal),
          error: null,
        };
      } catch (err) {
        this.queues[host.id] = {
          hostId: host.id,
          entries: previous?.entries ?? [],
          paused: previous?.paused ?? null,
          caps: previous?.caps ?? null,
          gpuOrdinals: previous?.gpuOrdinals ?? [],
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
        if (!live.has(id)) delete this.queues[id];
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
      if (this.pollTimer) return;
      void this.refresh();
      this.pollTimer = setInterval(() => void this.refresh(), POLL_INTERVAL_MS);
    },
    stopPolling() {
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = null;
    },
  },
});
