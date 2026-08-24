import { defineStore } from "pinia";
import { fetchServerCapabilities } from "../lib/api/serverCapabilities";
import { sseStream } from "../lib/api/sse";
import type { ServerEvent } from "../lib/api/types";
import { useGalleryStore } from "./gallery";
import { useGenerationStore } from "./generation";
import { useHostsStore } from "./hosts";
import { useJobsStore } from "./jobs";

/** Old-server fallback: refetch cadence while the queue is non-empty. */
const POLL_INTERVAL_MS = 5_000;
const authoritativeRefreshes = new WeakMap<object, Promise<void>>();

/**
 * App-wide subscriber to `GET /api/events` — one SSE connection that keeps
 * the gallery live while generations run anywhere (this window, another
 * client, the queue). Servers without the endpoint (capability probe says
 * so) fall back to polling the gallery while jobs are pending; probing
 * first matters because `sseStream` with `retry: true` would hammer a 404
 * forever.
 */
export const useEventsStore = defineStore("events", {
  state: () => ({
    subscribed: false,
    /** True when the connected server streams `/api/events`. */
    live: false,
    abort: null as AbortController | null,
    sharedHostId: null as string | null,
    pollTimer: null as ReturnType<typeof setInterval> | null,
    refreshScheduled: false,
    refreshAgain: false,
    refreshEpoch: 0,
  }),
  actions: {
    /** Subscribe (or start the fallback poller). Idempotent. */
    async subscribe() {
      if (this.subscribed) return;
      this.subscribed = true;
      let available = false;
      try {
        const caps = await fetchServerCapabilities();
        available = caps.events?.available === true;
      } catch {
        // Capabilities unreachable — treat as an old server; the poller
        // below degrades to a no-op when the connection is down anyway.
      }
      if (!this.subscribed) return; // unsubscribed while probing
      this.live = available;
      if (available) this.openStream();
      else this.startPolling();
    },
    unsubscribe() {
      this.abort?.abort();
      this.abort = null;
      if (this.sharedHostId) useGenerationStore().detachSharedDurableEventHost(this.sharedHostId);
      this.sharedHostId = null;
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = null;
      this.subscribed = false;
      this.live = false;
      this.refreshEpoch += 1;
      this.refreshScheduled = false;
      this.refreshAgain = false;
    },
    /** Re-probe after a connection change (new host may differ in caps). */
    async resubscribe() {
      this.unsubscribe();
      await this.subscribe();
    },
    openStream() {
      const primary = useHostsStore().primaryHost;
      if (!primary?.baseUrl) {
        this.live = false;
        this.startPolling();
        return;
      }
      const abort = new AbortController();
      this.abort = abort;
      this.sharedHostId = primary.id;
      useGenerationStore().attachSharedDurableEventHost(primary.id);
      void sseStream("/api/events", {
        target: { baseUrl: primary.baseUrl, apiKey: primary.apiKey },
        signal: abort.signal,
        retry: true,
        terminalHttpStatuses: [401, 403, 404],
        onOpen: () => {
          this.refreshAuthoritativePrimary();
        },
        onEvent: (event, data) => {
          useGenerationStore().onDurableEvent(primary.id, event, data);
          if (event !== "event" && event !== "message") return;
          try {
            this.apply(JSON.parse(data) as ServerEvent);
          } catch {
            /* skip malformed frame */
          }
        },
        onClose: () => {
          if (!abort.signal.aborted) useGenerationStore().onDurableEventClose(primary.id);
        },
      });
    },
    apply(ev: ServerEvent) {
      const gallery = useGalleryStore();
      switch (ev.type) {
        case "gallery_added":
          gallery.applyAdded(ev);
          break;
        case "gallery_removed":
          gallery.applyRemoved(ev.filename);
          break;
        // Library organization (titles / favorites / tags / collections /
        // trash) — primary-only like the two frames above.
        case "gallery_updated":
          gallery.applyUpdated(ev);
          break;
        case "gallery_trashed":
          gallery.applyTrashed(ev.filename);
          break;
        case "gallery_restored":
          gallery.applyRestored(ev);
          break;
        case "gallery_collections_changed":
          gallery.applyCollectionsChanged();
          break;
        // Queue pause state broadcast — another client (or this one) toggled
        // the primary host's queue; keep the Jobs view chip in sync live.
        case "queue_paused":
        case "queue_resumed": {
          const primary = useHostsStore().primaryHost;
          const queue = primary ? useJobsStore().queues[primary.id] : undefined;
          if (queue) queue.paused = ev.type === "queue_paused";
          break;
        }
        case "queue_plan_changed": {
          const primary = useHostsStore().primaryHost;
          const queue = primary ? useJobsStore().queues[primary.id] : undefined;
          if (queue && (!queue.plan || queue.plan.plan_version < ev.plan.plan_version)) {
            queue.plan = ev.plan;
          }
          this.refreshAuthoritativePrimary();
          break;
        }
        case "device_state_changed": {
          // This frame is deliberately a lean invalidation, not an
          // authoritative inventory. Refetch against the exact primary.
          this.refreshAuthoritativePrimary();
          break;
        }
        // job_* frames: the generation store tracks its own jobs via their
        // per-job streams; queue-wide UI can subscribe here later.
        default:
          break;
      }
    },
    refreshAuthoritativePrimary() {
      this.refreshAgain = true;
      if (this.refreshScheduled || authoritativeRefreshes.has(this)) return;
      this.refreshScheduled = true;
      const epoch = this.refreshEpoch;
      queueMicrotask(() => {
        this.refreshScheduled = false;
        if (epoch !== this.refreshEpoch || authoritativeRefreshes.has(this)) return;
        const refresh = (async () => {
          do {
            this.refreshAgain = false;
            const primary = useHostsStore().primaryHost;
            if (primary)
              await useJobsStore()
                .refreshHost(primary)
                .catch(() => undefined);
          } while (this.refreshAgain && epoch === this.refreshEpoch);
        })().finally(() => {
          authoritativeRefreshes.delete(this);
          // A resubscribe advances the epoch while the old host's read can
          // still be in flight. Preserve an invalidation raised by the new
          // stream and start it only after the old wave releases single-flight.
          if (this.refreshAgain) this.refreshAuthoritativePrimary();
        });
        authoritativeRefreshes.set(this, refresh);
      });
    },
    /**
     * Old-server fallback: while any generation is pending, refetch the
     * primary host's gallery bucket every few seconds, plus once more when
     * the queue drains so the last print always lands.
     */
    startPolling() {
      let wasPending = false;
      const tick = () => {
        const generation = useGenerationStore();
        const gallery = useGalleryStore();
        const primaryId = useHostsStore().primaryHost?.id ?? null;
        const bucket = primaryId ? gallery.buckets[primaryId] : undefined;
        const pending = generation.pending.length > 0;
        const shouldFetch = pending || wasPending;
        wasPending = pending;
        if (!shouldFetch || !primaryId || !bucket?.loaded || bucket.loading) return;
        void gallery.fetchBucket(primaryId);
      };
      this.pollTimer = setInterval(tick, POLL_INTERVAL_MS);
    },
  },
});
