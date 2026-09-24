import { defineStore } from "pinia";
import { watch } from "vue";
import { fetchServerCapabilities } from "../lib/api/serverCapabilities";
import { sseStream } from "../lib/api/sse";
import type { ServerEvent } from "../lib/api/types";
import { useGalleryStore } from "./gallery";
import { useGenerationStore } from "./generation";
import { useHostsStore } from "./hosts";
import { useJobsStore } from "./jobs";
import { useLandedPrintsStore } from "./landedPrints";

/** Old-server fallback: refetch cadence while the queue is non-empty. */
const POLL_INTERVAL_MS = 5_000;
const authoritativeRefreshes = new WeakMap<object, Promise<void>>();

type HostStream = {
  abort: AbortController;
  /** The machine this stream was opened against, as `hostIdentity` spells it. */
  identity: string;
};

type Subscription = {
  /** One live `/api/events` per ready machine, keyed by host id. */
  streams: Map<string, HostStream>;
  /**
   * host id → the identity that answered 401/403/404. `sseStream` throws on a
   * terminal status rather than retrying, so reopening it on the next fleet
   * change would be a connection per change forever. Cleared the moment that
   * machine's address, key or server identity changes, which is a new machine
   * to try.
   */
  silent: Map<string, string>;
  /** Stops the `hosts.all` watch that opens and closes them. */
  stopWatch: (() => void) | null;
};

/** Everything about a machine that decides whether its stream must be redone. */
function hostIdentity(host: {
  baseUrl: string | null;
  apiKey: string | null;
  instanceId: string | null;
}): string {
  return JSON.stringify([host.baseUrl, host.apiKey, host.instanceId]);
}

/*
 * Kept off the store's reactive state on purpose: an `AbortController` behind a
 * reactive proxy hands `fetch` a proxied `signal`, and the WeakMap is keyed by
 * the store instance so a fresh Pinia starts clean (the same shape
 * `authoritativeRefreshes` already uses).
 */
const subscriptions = new WeakMap<object, Subscription>();

function subscriptionFor(store: object): Subscription {
  let sub = subscriptions.get(store);
  if (!sub) {
    sub = { streams: new Map(), silent: new Map(), stopWatch: null };
    subscriptions.set(store, sub);
  }
  return sub;
}

/**
 * App-wide subscriber to `GET /api/events` — one SSE connection per READY
 * machine, so the gallery stays live while generations run anywhere (this
 * window, another client, the queue) and the Dock badge can count prints that
 * land on any of them while the app is away.
 *
 * Only the primary's frames change app state: the gallery store holds the
 * primary's bucket and the queue chips read the primary's queue. A secondary's
 * frames feed the durable job tracker and the landed-print count and nothing
 * else. The capability probe asks the PRIMARY and decides only whether the
 * old-server gallery poller runs, which is a primary-only fallback; the
 * streams themselves open for every ready machine, because a machine without
 * the endpoint answers 404 and `terminalHttpStatuses` closes it once rather
 * than letting `retry: true` hammer it.
 */
export const useEventsStore = defineStore("events", {
  state: () => ({
    subscribed: false,
    /** True when the connected server streams `/api/events`. */
    live: false,
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
      // Every ready machine gets a stream whatever the PRIMARY answered: the
      // probe asks one machine, and reading its answer as the fleet's silenced
      // every modern remote behind one primary that predates the endpoint. A
      // machine without `/api/events` answers 404, which `terminalHttpStatuses`
      // closes once instead of retrying.
      this.openStreams();
      // The old-server fallback refetches the PRIMARY's gallery bucket, so the
      // primary's own answer is exactly the question it settles.
      if (!available) this.startPolling();
    },
    unsubscribe() {
      const sub = subscriptionFor(this);
      sub.stopWatch?.();
      sub.stopWatch = null;
      for (const hostId of [...sub.streams.keys()]) this.closeHostStream(hostId);
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
    /**
     * Open the fleet's streams and keep them matched to it. A machine that
     * goes away, changes address or rotates its key has its stream closed and
     * a fresh one opened; the watch is what bounds `retry: true` against a
     * host nobody is connected to any more.
     */
    openStreams() {
      const hosts = useHostsStore();
      this.syncHostStreams();
      const sub = subscriptionFor(this);
      sub.stopWatch?.();
      sub.stopWatch = watch(
        // JSON, so no id, address or key can alias another tuple — and no
        // control characters in the source, which would make git call this
        // file binary and hide it from every diff and every search.
        () =>
          JSON.stringify(
            hosts.all.map((host) => [
              host.id,
              host.status,
              host.baseUrl,
              host.apiKey,
              host.instanceId,
            ]),
          ),
        () => this.syncHostStreams(),
      );
    },
    syncHostStreams() {
      const sub = subscriptionFor(this);
      const ready = useHostsStore().all.filter(
        (host) => host.status === "ready" && Boolean(host.baseUrl),
      );
      const wanted = new Map(ready.map((host) => [host.id, hostIdentity(host)]));
      for (const [hostId, stream] of [...sub.streams]) {
        const unchanged = wanted.get(hostId) === stream.identity && !stream.abort.signal.aborted;
        if (!unchanged) this.closeHostStream(hostId);
      }
      // A machine that changed in any way is a new machine to try, so its
      // "no event stream" note goes with the old identity.
      for (const [hostId, identity] of [...sub.silent]) {
        if (wanted.get(hostId) !== identity) sub.silent.delete(hostId);
      }
      for (const host of ready) {
        const identity = hostIdentity(host);
        if (sub.streams.has(host.id) || sub.silent.get(host.id) === identity) continue;
        this.openHostStream(host.id, host.baseUrl!, host.apiKey ?? null, identity);
      }
    },
    closeHostStream(hostId: string) {
      const streams = subscriptionFor(this).streams;
      const stream = streams.get(hostId);
      if (!stream) return;
      streams.delete(hostId);
      stream.abort.abort();
      // Hand the machine back: `ensureDurableHostStream` stands down while the
      // shared subscription claims it, so a missed detach strands every
      // durable job on that machine with no events at all.
      useGenerationStore().detachSharedDurableEventHost(hostId);
    },
    openHostStream(hostId: string, baseUrl: string, apiKey: string | null, identity: string) {
      const abort = new AbortController();
      subscriptionFor(this).streams.set(hostId, { abort, identity });
      useGenerationStore().attachSharedDurableEventHost(hostId);
      void sseStream("/api/events", {
        target: { baseUrl, apiKey },
        signal: abort.signal,
        retry: true,
        terminalHttpStatuses: [401, 403, 404],
        onOpen: () => {
          if (this.isPrimary(hostId)) this.refreshAuthoritativePrimary();
        },
        onEvent: (event, data) => {
          useGenerationStore().onDurableEvent(hostId, event, data);
          if (event !== "event" && event !== "message") return;
          let frame: ServerEvent;
          try {
            frame = JSON.parse(data) as ServerEvent;
          } catch {
            return; /* skip malformed frame */
          }
          // Fleet-wide: a print is a print whichever machine made it, and the
          // badge is the only consumer that cares about the others. A print
          // that was trashed or deleted — including one a machine published
          // and then trashed because Save every result was off — never landed.
          if (frame.type === "gallery_added" && !frame.imported)
            useLandedPrintsStore().noteLanded(hostId, frame.filename);
          else if (frame.type === "gallery_trashed" || frame.type === "gallery_removed")
            useLandedPrintsStore().forgetLanded(frame.filename);
          if (this.isPrimary(hostId)) this.apply(frame);
        },
        onClose: () => {
          if (abort.signal.aborted) return;
          // `retry: true` means reaching here at all is the end of this
          // machine's stream, not a blip: a terminal status threw. Remember
          // the identity that went quiet so the next fleet change does not
          // walk straight back into it, and hand the machine back so the
          // durable tracker can open its own.
          const sub = subscriptionFor(this);
          if (sub.streams.get(hostId)?.abort === abort) {
            sub.silent.set(hostId, identity);
            this.closeHostStream(hostId);
          }
          useGenerationStore().onDurableEventClose(hostId);
        },
      });
    },
    isPrimary(hostId: string): boolean {
      return useHostsStore().primaryHost?.id === hostId;
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
