import { defineStore } from "pinia";
import { listDevices, type DeviceInfo } from "@studio/api/devices";
import {
  getQueueJob,
  mergeQueueEntries,
  parseQueueListing,
  predictedCompletionUnixMs,
  queueListingPath,
  queuePageRequestForCapacity,
  type QueuePlan,
  type QueueJobAuthority,
  retryQueueJobRecoveringAmbiguity,
} from "@studio/api/queuePlan";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { OutputMetadata, ServerStatus } from "../lib/api/types";
import { useGenerationStore } from "./generation";
import { useHostsStore, type HostView } from "./hosts";
import { useToastStore } from "./toasts";
import type { Job } from "./generation";
import { compareNewestQueueEntry } from "@studio/lib/activityOrder";

const POLL_INTERVAL_MS = 5_000;

/** One row of a host's `GET /api/queue` listing. */
export interface QueueEntry {
  id: string;
  model: string;
  /** `held` is additive: a journalled job the host parked after it exhausted
   * its replay or dispatch budget. It exists only in the durable queue, will
   * never start on its own, and is listed precisely so it is not invisible. */
  state: "queued" | "running" | "paused" | "held";
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
  error?: string | null;
  retryable?: boolean | null;
  batch_id?: string | null;
  client_batch_id?: string | null;
  batch_index?: number | null;
  durable?: boolean | null;
  replayed?: boolean | null;
  dispatch_attempts?: number | null;
}

/** Queue controls a host supports (from `GET /api/capabilities`). */
export interface HostQueueCaps {
  canPause: boolean;
  canPauseJob?: boolean;
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
  /** The server-authorized page size and opaque continuation. Absent/null on
   * legacy hosts, whose in-memory queue was already capacity-bounded. */
  pageLimit?: number | null;
  nextCursor?: string | null;
  /** Explicitly loaded durable tail. Hot polls refresh the capacity-sized
   * live head and retain this user-requested snapshot without rescanning it. */
  tailEntries?: QueueEntry[];
  continued?: boolean;
  loadingMore?: boolean;
  loadMoreError?: string | null;
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

function visibleQueueEntries(entries: readonly import("@studio/api/queuePlan").QueueEntry[]) {
  return entries
    .filter(
      (entry) =>
        entry.state === "queued" ||
        entry.state === "running" ||
        entry.state === "paused" ||
        entry.state === "held",
    )
    .map((entry) => {
      const { target_gpu: targetGpu, ...rest } = entry;
      const local = rest as QueueEntry;
      if (targetGpu != null) local.target_gpu = targetGpu;
      return local;
    });
}

function distinctQueueEntries(entries: readonly QueueEntry[]): QueueEntry[] {
  const seen = new Set<string>();
  return entries.filter(({ id }) => {
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
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
  ownsDurableBatch: (clientBatchId: string | null | undefined) => boolean = () => false,
): EnrichedQueueEntry[] {
  return entries.map((entry) => {
    const owner = localJobs.find(
      (j) => j.id === entry.id && (j.hostId ?? primaryHostId) === hostId,
    );
    return {
      ...entry,
      mine: !!owner || ownsDurableBatch(entry.client_batch_id),
      clientId: owner?.clientId ?? null,
    };
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
    pollTimer: null as ReturnType<typeof setTimeout> | null,
    pollRunning: false,
    /** Views currently asking for the cross-host poll. Machines wants it while
     * it is open and Create wants it only while something is queued, so the
     * one that leaves first must not silence the other. */
    pollConsumers: 0,
  }),
  getters: {
    /**
     * Unified running+queued rows across every connected host, joined with
     * this app's own jobs. Newest submissions render first while scheduler
     * position remains untouched for queue actions. The
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
        for (const entry of enrichQueueEntries(
          snap.entries,
          host.id,
          generation.jobs,
          primaryId,
          (clientBatchId) => generation.ownsDurableBatch(clientBatchId),
        )) {
          rows.push({
            hostId: host.id,
            hostLabel: host.label,
            entry,
            canReorder,
            canCancelRunning,
          });
        }
      }
      return rows.sort((a, b) => compareNewestQueueEntry(a.entry, b.entry));
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
              can_pause_job?: boolean;
              can_cancel_all?: boolean;
              can_reorder?: boolean;
              cooperative_cancellation?: boolean;
            };
          }>(target, "/api/capabilities").then(
            (c) => ({
              canPause: c.queue?.can_pause === true,
              canPauseJob: c.queue?.can_pause_job === true,
              canCancelAll: c.queue?.can_cancel_all === true,
              canReorder: c.queue?.can_reorder === true,
              canCancelRunning: c.queue?.cooperative_cancellation === true,
            }),
            () => null,
          ));
        const statusRequest = apiJsonTo<ServerStatus & { queue_paused?: boolean }>(
          target,
          "/api/status",
        );
        const [listing, status, devices] = await Promise.all([
          statusRequest.then((current) =>
            apiJsonTo<unknown>(
              target,
              queueListingPath(queuePageRequestForCapacity(current.queue_capacity)),
            ).then(parseQueueListing),
          ),
          statusRequest,
          listDevices(target).then(
            (snapshot) => snapshot.devices,
            () => null,
          ),
        ]);
        if (!isCurrent()) return;
        // Continuation rows are an explicit snapshot, not live authority. A
        // bounded head refresh cannot prove that older rows still exist (an
        // external client may have cancelled one), so discard that snapshot
        // and re-arm its cursor rather than rendering ghost jobs forever.
        this.queues[host.id] = {
          hostId: host.id,
          entries: visibleQueueEntries(
            mergeQueueEntries(listing.entries, listing.live_only_entries ?? []),
          ),
          plan: listing.plan,
          pageLimit: listing.page?.limit ?? null,
          nextCursor: listing.page?.next_cursor ?? null,
          tailEntries: [],
          continued: false,
          loadingMore: false,
          loadMoreError: null,
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
          pageLimit: previous?.pageLimit ?? null,
          nextCursor: previous?.nextCursor ?? null,
          tailEntries: previous?.tailEntries ?? [],
          continued: previous?.continued ?? false,
          loadingMore: false,
          loadMoreError: previous?.loadMoreError ?? null,
          error: String(err),
        };
      }
    },
    async loadMoreHost(hostId: string) {
      const snapshot = this.queues[hostId];
      const cursor = snapshot?.nextCursor;
      const limit = snapshot?.pageLimit;
      if (!snapshot || !cursor || !limit || snapshot.loadingMore) return;
      const hosts = useHostsStore();
      const host = hosts.all.find((candidate) => candidate.id === hostId);
      const target = host ? this.targetFor(host) : null;
      if (!host || !target || host.status !== "ready") return;
      const authority = snapshot;
      snapshot.loadingMore = true;
      snapshot.loadMoreError = null;
      try {
        const listing = await apiJsonTo<unknown>(target, queueListingPath({ limit, cursor })).then(
          parseQueueListing,
        );
        const currentHost = hosts.all.find((candidate) => candidate.id === hostId);
        const current = this.queues[hostId];
        if (
          current !== authority ||
          current.nextCursor !== cursor ||
          currentHost?.baseUrl !== host.baseUrl ||
          currentHost.apiKey !== host.apiKey
        )
          return;
        if (!listing.page) {
          current.entries = visibleQueueEntries(listing.entries);
          current.pageLimit = null;
          current.nextCursor = null;
          current.tailEntries = [];
          current.continued = false;
          return;
        }
        const tail = distinctQueueEntries([
          ...(current.tailEntries ?? []),
          ...visibleQueueEntries(listing.entries),
        ]);
        current.tailEntries = tail;
        current.entries = distinctQueueEntries([
          ...current.entries,
          ...tail,
          ...visibleQueueEntries(listing.live_only_entries ?? []),
        ]);
        current.nextCursor = listing.page.next_cursor ?? null;
        current.continued = true;
        current.plan = listing.plan ?? current.plan ?? null;
      } catch (error) {
        const current = this.queues[hostId];
        if (
          current === authority &&
          current.nextCursor === cursor &&
          hosts.all.find((candidate) => candidate.id === hostId)?.baseUrl === host.baseUrl &&
          hosts.all.find((candidate) => candidate.id === hostId)?.apiKey === host.apiKey
        )
          current.loadMoreError = String(error);
      } finally {
        const current = this.queues[hostId];
        if (current === authority) current.loadingMore = false;
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
      await this.refresh();
    },
    /** Pause or resume one waiting row without changing host-wide dispatch. */
    async setJobPaused(hostId: string, jobId: string, paused: boolean) {
      await this.queueControl(
        hostId,
        `/api/queue/${encodeURIComponent(jobId)}/${paused ? "pause" : "resume"}`,
        "POST",
      );
      const snapshot = this.queues[hostId];
      const apply = (entries: QueueEntry[] | undefined) => {
        const entry = entries?.find(({ id }) => id === jobId);
        if (entry) entry.state = paused ? "paused" : "queued";
      };
      apply(snapshot?.entries);
      apply(snapshot?.tailEntries);
      void this.refresh();
    },
    /** Cancel every queued job on the host (running jobs finish). */
    async cancelAll(hostId: string) {
      await this.queueControl(hostId, "/api/queue", "DELETE");
      void this.refresh();
    },
    /** Cancel one job on a host directly (used for other clients' jobs). */
    async cancelJob(hostId: string, jobId: string) {
      await this.queueControl(hostId, `/api/queue/${encodeURIComponent(jobId)}`, "DELETE");
      const snapshot = this.queues[hostId];
      if (snapshot) {
        snapshot.entries = snapshot.entries.filter(({ id }) => id !== jobId);
        snapshot.tailEntries = snapshot.tailEntries?.filter(({ id }) => id !== jobId) ?? [];
      }
      void this.refresh();
    },
    /** Read the selected row's persisted request. Queue polling stays
     * payload-free; this explicit path powers details and Reuse settings. */
    async queueJob(hostId: string, jobId: string): Promise<QueueEntry> {
      const hosts = useHostsStore();
      const host = hosts.all.find((candidate) => candidate.id === hostId);
      const target = host ? this.targetFor(host) : null;
      if (!host || !target || host.status !== "ready") {
        throw new Error("The selected machine is not connected.");
      }
      return (await getQueueJob(target, jobId)).job as QueueEntry;
    },
    /** Retry a server-authorized durable hold even after this app restarted.
     * Authority lives on the row and host, not in an ephemeral clientId. */
    async retryJob(hostId: string, entry: QueueEntry): Promise<void> {
      const hosts = useHostsStore();
      const host = hosts.all.find((candidate) => candidate.id === hostId);
      const target = host ? this.targetFor(host) : null;
      if (!host || !target || host.status !== "ready") {
        throw new Error("The selected machine identity is unavailable.");
      }
      if (
        entry.state !== "held" ||
        entry.retryable !== true ||
        !entry.batch_id ||
        !entry.client_batch_id
      ) {
        throw new Error("This held generation is not retryable.");
      }
      const instanceId =
        host.instanceId ??
        (await apiJsonTo<{ instance_id?: string }>(target, "/api/status")).instance_id ??
        null;
      if (!instanceId) throw new Error("The selected machine identity is unavailable.");
      const authority: QueueJobAuthority = {
        instanceId,
        batchId: entry.batch_id,
        clientBatchId: entry.client_batch_id,
        jobId: entry.id,
      };
      const outcome = await retryQueueJobRecoveringAmbiguity(target, authority);
      if (outcome.kind === "uncertain") throw new Error(outcome.error);
      await this.refreshHost(host);
      void useGenerationStore().reconcileDurableHost(hostId);
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
      if (this.pollRunning || this.pollTimer) return;
      void this.runPoll();
    },
    async runPoll() {
      if (this.pollRunning || this.pollConsumers === 0) return;
      this.pollRunning = true;
      try {
        await this.refresh();
      } finally {
        this.pollRunning = false;
        if (this.pollConsumers > 0 && !this.pollTimer) {
          this.pollTimer = setTimeout(() => {
            this.pollTimer = null;
            void this.runPoll();
          }, POLL_INTERVAL_MS);
        }
      }
    },
    stopPolling() {
      this.pollConsumers = Math.max(0, this.pollConsumers - 1);
      if (this.pollConsumers > 0) return;
      if (this.pollTimer) clearTimeout(this.pollTimer);
      this.pollTimer = null;
    },
  },
});
