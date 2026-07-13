import { defineStore } from "pinia";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import {
  evictHostMedia,
  evictMedia,
  galleryMediaPath,
  type GallerySource,
} from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { useConnectionStore } from "./connection";
import { useHostsStore, type HostView } from "./hosts";
import type { GalleryImage, ServerEvent } from "../lib/api/types";

/** Collapse a burst of row-less gallery_added events into one refetch. */
const REFETCH_DEBOUNCE_MS = 500;
let refetchTimer: ReturnType<typeof setTimeout> | null = null;

/** One host's (or this Mac's) slice of the unified gallery. */
export interface GalleryBucket {
  items: GalleryImage[];
  loading: boolean;
  error: string | null;
  loaded: boolean;
}

/** A bucket that should exist right now. */
export interface GallerySourceRef {
  key: string;
  label: string;
}

/** One print in the merged, date-sorted grid. */
export interface MergedPrint {
  item: GalleryImage;
  sourceKey: string;
  hostLabel: string;
}

export interface GalleryChip {
  key: string;
  label: string;
  count: number;
}

const emptyBucket = (): GalleryBucket => ({
  items: [],
  loading: false,
  error: null,
  loaded: false,
});

/**
 * Unified multi-host gallery: one bucket per origin, merged into a single
 * date-sorted grid. Buckets are keyed by "local" (This Mac via native IPC)
 * or a host id from the hosts store; API targets are always resolved at
 * call time from the hosts store — never cached in state.
 */
export const useGalleryStore = defineStore("gallery", {
  state: () => ({
    /** Keyed by "local" (This Mac via IPC) or a host id. */
    buckets: {} as Record<string, GalleryBucket>,
    /** Session-only chip filter: "all" or a bucket key. */
    filter: "all" as string,
  }),
  getters: {
    /**
     * Which buckets should exist right now: one per ready host (the primary
     * included — its id is "local" when the primary is the built-in or
     * external engine). A separate "local" (This Mac via IPC) bucket exists
     * ONLY when the primary is remote: when the primary is the built-in or
     * external engine, its bucket IS this Mac's gallery — listing both
     * would show every print twice.
     */
    sources(): GallerySourceRef[] {
      const conn = useConnectionStore();
      const hosts = useHostsStore();
      const refs: GallerySourceRef[] = [];
      if (conn.mode === "remote") refs.push({ key: "local", label: "This Mac" });
      for (const host of hosts.all) {
        if (host.status !== "ready") continue;
        refs.push({ key: host.id, label: host.label });
      }
      return refs;
    },
    /** Every loaded print across buckets, newest first. */
    merged(): MergedPrint[] {
      const rows: MergedPrint[] = [];
      for (const source of this.sources) {
        const bucket = this.buckets[source.key];
        if (!bucket) continue;
        for (const item of bucket.items) {
          rows.push({ item, sourceKey: source.key, hostLabel: source.label });
        }
      }
      return rows.sort((a, b) => b.item.timestamp - a.item.timestamp);
    },
    /** `merged` with the chip filter applied (an unknown filter = all). */
    filtered(): MergedPrint[] {
      if (this.filter === "all") return this.merged;
      if (!this.sources.some((s) => s.key === this.filter)) return this.merged;
      return this.merged.filter((e) => e.sourceKey === this.filter);
    },
    /** Per-source chips for the gallery header (HostFilterChips adds All). */
    chipCounts(): GalleryChip[] {
      return this.sources.map((s) => ({
        key: s.key,
        label: s.label,
        count: this.buckets[s.key]?.items.length ?? 0,
      }));
    },
    /** Bytes across the filtered set — the header stat. */
    totalBytes(): number {
      return this.filtered.reduce((sum, e) => sum + (e.item.size_bytes ?? 0), 0);
    },
    /** True once every current source's bucket has settled (or errored). */
    loaded(): boolean {
      return (
        this.sources.length > 0 &&
        this.sources.every((s) => {
          const bucket = this.buckets[s.key];
          return !!bucket && !bucket.loading && (bucket.loaded || bucket.error !== null);
        })
      );
    },
    /** First bucket error, labeled — the header's error line. */
    firstError(): string | null {
      for (const source of this.sources) {
        const error = this.buckets[source.key]?.error;
        if (error) return `${source.label}: ${error}`;
      }
      return null;
    },
    /** Flat compatibility view for consumers that predate buckets. */
    items(): GalleryImage[] {
      return this.merged.map((e) => e.item);
    },
  },
  actions: {
    /** The live host behind a bucket key, if any (resolved at call time). */
    hostFor(sourceKey: string): HostView | null {
      return useHostsStore().all.find((h) => h.id === sourceKey) ?? null;
    },
    /** How a bucket's media is addressed: API paths vs mold-local files. */
    mediaSourceOf(sourceKey: string): GallerySource {
      return this.hostFor(sourceKey) ? "host" : "local";
    },
    /** Auth target for a bucket's host; null for the This-Mac IPC bucket. */
    targetOf(sourceKey: string): ApiTarget | null {
      const host = this.hostFor(sourceKey);
      return host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    ensureBucket(key: string): GalleryBucket {
      if (!this.buckets[key]) this.buckets[key] = emptyBucket();
      return this.buckets[key]!;
    },
    /** Drop buckets whose source disappeared; their cached media goes too. */
    syncBuckets() {
      const keys = new Set(this.sources.map((s) => s.key));
      for (const key of Object.keys(this.buckets)) {
        if (keys.has(key)) continue;
        delete this.buckets[key];
        evictHostMedia(key);
      }
      if (this.filter !== "all" && !keys.has(this.filter)) this.filter = "all";
    },
    /** Fetch one bucket. Errors land on the bucket, never on siblings. */
    async fetchBucket(key: string) {
      const bucket = this.ensureBucket(key);
      if (bucket.loading) return;
      bucket.loading = true;
      bucket.error = null;
      try {
        // Resolve the origin at call time: a key backed by a live host reads
        // that host's /api/gallery; the bare "local" key is this Mac via IPC.
        const target = this.targetOf(key);
        let items: GalleryImage[];
        if (target) items = await apiJsonTo<GalleryImage[]>(target, "/api/gallery");
        else if (key === "local") items = await ipc.localGalleryList();
        else throw new Error("Host is not connected.");
        // Prints that vanished out-of-band (deleted by another client) must
        // release their cached blob URLs — the media cache only evicts on
        // explicit remove()/host teardown otherwise.
        const source = this.mediaSourceOf(key);
        const next = new Set(items.map((i) => i.filename));
        for (const old of bucket.items) {
          if (next.has(old.filename)) continue;
          evictMedia(galleryMediaPath(old.filename, source, true), key);
          evictMedia(galleryMediaPath(old.filename, source), key);
        }
        // Newest first, like a print drawer.
        bucket.items = items.sort((a, b) => b.timestamp - a.timestamp);
        bucket.loaded = true;
      } catch (err) {
        bucket.error = String(err);
      } finally {
        bucket.loading = false;
      }
    },
    /** (Re)fetch every current source. */
    async fetchAll() {
      this.syncBuckets();
      await Promise.all(this.sources.map((s) => this.fetchBucket(s.key)));
    },
    /**
     * Gallery-view poll: refetch every non-primary bucket (the primary stays
     * live over SSE / the events fallback poller). Sources that appeared
     * mid-session get their first fetch here too.
     */
    async pollExtras() {
      this.syncBuckets();
      const primaryId = useHostsStore().primaryHost?.id ?? null;
      await Promise.all(
        this.sources.filter((s) => s.key !== primaryId).map((s) => this.fetchBucket(s.key)),
      );
    },
    /**
     * Refetch one bucket after a routed job lands there. Only already-loaded
     * buckets refresh — a background completion must not force-load a
     * gallery bucket the user never opened.
     */
    async refreshHost(hostId: string) {
      const bucket = this.buckets[hostId];
      if (!bucket?.loaded || bucket.loading) return;
      await this.fetchBucket(hostId);
    },
    /** Delete a print where it lives, evicting only that origin's media. */
    async remove(sourceKey: string, filename: string) {
      const target = this.targetOf(sourceKey);
      if (target) {
        await apiFetchTo(target, `/api/gallery/image/${encodeURIComponent(filename)}`, {
          method: "DELETE",
        });
      } else if (sourceKey === "local") {
        await ipc.localGalleryDelete(filename);
      } else {
        throw new Error("Host is not connected.");
      }
      this.evictItemMedia(sourceKey, filename);
      const bucket = this.buckets[sourceKey];
      if (bucket) bucket.items = bucket.items.filter((i) => i.filename !== filename);
    },
    evictItemMedia(sourceKey: string, filename: string) {
      const source = this.mediaSourceOf(sourceKey);
      evictMedia(galleryMediaPath(filename, source, true), sourceKey);
      evictMedia(galleryMediaPath(filename, source), sourceKey);
    },
    /**
     * A `gallery_added` frame from the primary connection's `GET
     * /api/events` — SSE stays primary-only, so the row lands in the
     * primary host's bucket. When the event carries the row, insert in
     * place; otherwise (server DB disabled) debounce a full refetch.
     */
    applyAdded(ev: Extract<ServerEvent, { type: "gallery_added" }>) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      if (!ev.image) {
        if (refetchTimer) clearTimeout(refetchTimer);
        refetchTimer = setTimeout(() => {
          refetchTimer = null;
          // Re-check at fire time: the primary may have changed, or a fetch
          // may already be in flight — re-debounce rather than racing it.
          const nowKey = useHostsStore().primaryHost?.id ?? null;
          const nowBucket = nowKey ? this.buckets[nowKey] : undefined;
          if (!nowKey || !nowBucket?.loaded) return;
          if (nowBucket.loading) {
            this.applyAdded(ev);
            return;
          }
          void this.fetchBucket(nowKey);
        }, REFETCH_DEBOUNCE_MS);
        return;
      }
      if (bucket.items.some((i) => i.filename === ev.filename)) return;
      const image = ev.image;
      const at = bucket.items.findIndex((i) => i.timestamp < image.timestamp);
      if (at === -1) bucket.items.push(image);
      else bucket.items.splice(at, 0, image);
    },
    /** A `gallery_removed` frame — drop the primary's tile and its media. */
    applyRemoved(filename: string) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      if (!bucket.items.some((i) => i.filename === filename)) return;
      this.evictItemMedia(key, filename);
      bucket.items = bucket.items.filter((i) => i.filename !== filename);
    },
  },
});
