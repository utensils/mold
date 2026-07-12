import { defineStore } from "pinia";
import { fetchServerCapabilities } from "../lib/api/serverCapabilities";
import { sseStream } from "../lib/api/sse";
import type { ServerEvent } from "../lib/api/types";
import { useGalleryStore } from "./gallery";
import { useGenerationStore } from "./generation";

/** Old-server fallback: refetch cadence while the queue is non-empty. */
const POLL_INTERVAL_MS = 5_000;

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
    pollTimer: null as ReturnType<typeof setInterval> | null,
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
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = null;
      this.subscribed = false;
      this.live = false;
    },
    /** Re-probe after a connection change (new host may differ in caps). */
    async resubscribe() {
      this.unsubscribe();
      await this.subscribe();
    },
    openStream() {
      const abort = new AbortController();
      this.abort = abort;
      void sseStream("/api/events", {
        signal: abort.signal,
        retry: true,
        onEvent: (_event, data) => {
          try {
            this.apply(JSON.parse(data) as ServerEvent);
          } catch {
            /* skip malformed frame */
          }
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
        // job_* frames: the generation store tracks its own jobs via their
        // per-job streams; queue-wide UI can subscribe here later.
        default:
          break;
      }
    },
    /**
     * Old-server fallback: while any generation is pending, refetch the
     * engine gallery every few seconds, plus once more when the queue
     * drains so the last print always lands.
     */
    startPolling() {
      let wasPending = false;
      const tick = () => {
        const generation = useGenerationStore();
        const gallery = useGalleryStore();
        const pending = generation.pending.length > 0;
        const shouldFetch =
          gallery.loaded && gallery.source === "engine" && (pending || wasPending);
        wasPending = pending;
        if (shouldFetch && !gallery.loading) void gallery.fetch();
      };
      this.pollTimer = setInterval(tick, POLL_INTERVAL_MS);
    },
  },
});
