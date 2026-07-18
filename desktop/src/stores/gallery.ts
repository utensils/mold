import { defineStore } from "pinia";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import {
  evictHostMedia,
  evictMedia,
  galleryMediaPath,
  isVideoItem,
  type GallerySource,
} from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";
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
  /** Every gallery bucket that contains this filename. */
  availableOn: GallerySourceRef[];
}

export interface GalleryChip {
  key: string;
  label: string;
  count: number;
}

/** Media-kind chip filter: everything, stills only, or video only. */
export type GalleryKindFilter = "all" | "image" | "video";

/** Per-kind counts over the host-chip-filtered set (kind chip labels). */
export interface GalleryKindCounts {
  all: number;
  image: number;
  video: number;
}

const emptyBucket = (): GalleryBucket => ({
  items: [],
  loading: false,
  error: null,
  loaded: false,
});

/**
 * Seed used for cross-host identity. Video files embed no `mold:parameters`,
 * so rows for old locally-mirrored videos were synthesized with `seed: 0` —
 * but the desktop's own auto-save filenames encode the real seed
 * (`mold-<model>-<seed>-<epochMs>[-role].<ext>`), so synthetic rows recover
 * it from the name. Non-synthetic rows trust their recorded metadata.
 */
export function identitySeed(item: GalleryImage): number | null {
  if (item.metadata_synthetic) {
    // A synthesized row's recorded seed is a placeholder 0 ("unknown") — it
    // must never act as a real seed. Only the auto-save filename pattern
    // yields a trustworthy seed; otherwise the row opts out of identity.
    const match = /-(\d+)-(\d+)(?:-(?:original|upscaled))?\.[a-z0-9]+$/i.exec(item.filename);
    return match ? Number(match[1]) : null;
  }
  return item.metadata?.seed ?? null;
}

/** Filename-style slug of a model name — matches the auto-save filename
 *  vocabulary, so synthesized rows (model recovered from the name) compare
 *  equal to origin rows carrying the real `model:tag`. */
function modelIdentitySlug(model: string | undefined): string {
  return (model ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Cross-host identity beyond the filename: mirrored copies of one print are
 * byte-identical, so seed + exact byte size + model pins them together even
 * when an old auto-save invented its own filename or a video copy
 * synthesized its metadata (seed and model survive in the filename either
 * way). Rows missing seed or size opt out of identity matching entirely.
 */
export function printIdentity(item: GalleryImage): string | null {
  const size = item.size_bytes;
  const seed = identitySeed(item);
  if (!size || seed == null) return null;
  return `${seed}:${size}:${modelIdentitySlug(item.metadata?.model)}`;
}

/** Identity matches only count as one print when the rows were written
 *  around the same time: mirrors land within seconds of their origin, while
 *  a genuine re-generation that happens to reuse a seed (and byte length)
 *  lands much later and must stay a separate print. */
export const IDENTITY_WINDOW_SECS = 3600;

export function withinIdentityWindow(a: GalleryImage, b: GalleryImage): boolean {
  return Math.abs(a.timestamp - b.timestamp) <= IDENTITY_WINDOW_SECS;
}

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
    /** Session-only media-kind chip: all / image / video. */
    mediaKind: "all" as GalleryKindFilter,
    /** Session-only text query over filename/model/prompt. The view owns
     *  debouncing; the store only holds the settled value. */
    query: "" as string,
  }),
  getters: {
    /**
     * Which buckets should exist right now: This device first, then one per
     * ready host. The built-in engine is the internal primary (host id
     * "local"), and its `/api/gallery` already covers IPC-saved files (a
     * saved remote print writes a best-effort DB row into the same mold.db),
     * so there is no separate This-Mac IPC bucket while it's ready. When the
     * local server is NOT ready (failed start, mid-restart), the "local"
     * bucket stays listed and falls back to native IPC — this device's
     * prints must never vanish with the engine.
     */
    sources(): GallerySourceRef[] {
      const hosts = useHostsStore();
      const refs: GallerySourceRef[] = [];
      const keys = new Set<string>();
      const add = (source: GallerySourceRef) => {
        if (keys.has(source.key)) return;
        keys.add(source.key);
        refs.push(source);
      };
      const local = hosts.all.find((host) => host.id === "local");
      add({ key: "local", label: local?.label ?? PLATFORM_UI.deviceLabel });
      for (const host of hosts.all) {
        if (host.status !== "ready") continue;
        add({ key: host.id, label: host.label });
      }
      return refs;
    },
    /**
     * Every loaded print across buckets, newest first. Matching filenames are
     * one logical print in the All view: saved remote results retain their
     * filename when copied locally, so the filename is the cross-host identity
     * available in the gallery wire contract. Prefer the local copy for media
     * and actions, while retaining every location for display.
     */
    merged(): MergedPrint[] {
      const byFilename = new Map<string, MergedPrint>();
      // Second-level identity (seed + byte size) collapses copies whose
      // filenames diverged — auto-saves from before the server shipped its
      // gallery filename on the complete event minted their own names.
      const byIdentity = new Map<string, MergedPrint>();
      const prints: MergedPrint[] = [];
      for (const source of this.sources) {
        const bucket = this.buckets[source.key];
        if (!bucket) continue;
        for (const item of bucket.items) {
          const identity = printIdentity(item);
          let existing = byFilename.get(item.filename);
          if (!existing && identity) {
            const candidate = byIdentity.get(identity);
            if (candidate && withinIdentityWindow(candidate.item, item)) existing = candidate;
          }
          if (!existing) {
            const print: MergedPrint = {
              item,
              sourceKey: source.key,
              hostLabel: source.label,
              availableOn: [source],
            };
            byFilename.set(item.filename, print);
            if (identity) byIdentity.set(identity, print);
            prints.push(print);
            continue;
          }
          // A copy under a different name still joins the print, and its
          // name is indexed too so further copies under either name merge.
          byFilename.set(item.filename, existing);
          if (identity && !byIdentity.has(identity)) byIdentity.set(identity, existing);
          if (!existing.availableOn.some((s) => s.key === source.key)) {
            existing.availableOn.push(source);
          }
          if (source.key === "local" && existing.sourceKey !== "local") {
            existing.item = item;
            existing.sourceKey = source.key;
            existing.hostLabel = source.label;
          }
        }
      }
      return prints.sort((a, b) => b.item.timestamp - a.item.timestamp);
    },
    /**
     * `merged` in All; an individual host remains its complete raw bucket so
     * a print represented by This Mac in All is still visible on a host chip.
     * An unknown filter falls back to All. This is the chip-only set —
     * consumers that must not inherit the Gallery view's kind/search
     * narrowing (History → Runs) read this instead of `filtered`.
     */
    hostFiltered(): MergedPrint[] {
      if (this.filter === "all") return this.merged;
      const source = this.sources.find((s) => s.key === this.filter);
      if (!source) return this.merged;
      return (this.buckets[source.key]?.items ?? [])
        .map((item) => ({
          item,
          sourceKey: source.key,
          hostLabel: source.label,
          availableOn: [source],
        }))
        .sort((a, b) => b.item.timestamp - a.item.timestamp);
    },
    /**
     * True when this print already lives in this Mac's gallery — by filename
     * or by byte identity — whatever source the given tile came from. Host
     * -chip tiles carry only their own bucket in `availableOn`, so the local
     * bucket is probed directly.
     */
    existsLocally(): (entry: MergedPrint) => boolean {
      return (entry) => {
        if (entry.sourceKey === "local") return true;
        if (entry.availableOn.some((s) => s.key === "local")) return true;
        const identity = printIdentity(entry.item);
        return (this.buckets["local"]?.items ?? []).some(
          (item) =>
            item.filename === entry.item.filename ||
            (identity !== null &&
              printIdentity(item) === identity &&
              withinIdentityWindow(item, entry.item)),
        );
      };
    },
    /** What the Gallery grid renders: host chip → media kind → text query. */
    filtered(): MergedPrint[] {
      let entries = this.hostFiltered;
      if (this.mediaKind !== "all") {
        const wantVideo = this.mediaKind === "video";
        entries = entries.filter((e) => isVideoItem(e.item) === wantVideo);
      }
      const q = this.query.trim().toLowerCase();
      if (q) {
        entries = entries.filter(
          (e) =>
            e.item.filename.toLowerCase().includes(q) ||
            e.item.metadata.model.toLowerCase().includes(q) ||
            e.item.metadata.prompt.toLowerCase().includes(q),
        );
      }
      return entries;
    },
    /** Per-kind counts for the All/Images/Video chips. Computed over the
     *  host-chip-filtered set only, so chip labels stay stable while the
     *  kind chip or search narrows the grid. */
    kindCounts(): GalleryKindCounts {
      const entries = this.hostFiltered;
      let video = 0;
      for (const e of entries) if (isVideoItem(e.item)) video++;
      return { all: entries.length, image: entries.length - video, video };
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
    /** The live host behind a bucket key, if any (resolved at call time).
     *  The "local" key only counts as host-backed while the local server is
     *  READY — an errored local host can still expose a stale baseUrl, and
     *  routing there would land errors instead of falling back to IPC. */
    hostFor(sourceKey: string): HostView | null {
      const host = useHostsStore().all.find((h) => h.id === sourceKey) ?? null;
      if (sourceKey === "local" && host?.status !== "ready") return null;
      return host;
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
    /**
     * Bulk delete. Each print is deleted exactly like the single-delete
     * path — `remove()` per item, routed to that item's represented origin
     * (IPC for the host-less This-Mac bucket, authed HTTP for a host) — so
     * bulk semantics can never drift from the tile's Delete action.
     * Origins whose delete failed are refetched to reconverge with the
     * server (a lost response may still have deleted server-side).
     */
    async removeMany(
      items: Array<{ sourceKey: string; filename: string }>,
    ): Promise<{ deleted: number; failed: number }> {
      const results = await Promise.allSettled(
        items.map((i) => this.remove(i.sourceKey, i.filename)),
      );
      let deleted = 0;
      const failedOrigins = new Set<string>();
      results.forEach((r, idx) => {
        if (r.status === "fulfilled") deleted++;
        else failedOrigins.add(items[idx]!.sourceKey);
      });
      await Promise.all([...failedOrigins].map((key) => this.refreshHost(key)));
      return { deleted, failed: results.length - deleted };
    },
    evictItemMedia(sourceKey: string, filename: string) {
      const source = this.mediaSourceOf(sourceKey);
      evictMedia(galleryMediaPath(filename, source, true), sourceKey);
      evictMedia(galleryMediaPath(filename, source), sourceKey);
      // Consumers that render primary-gallery media without a cacheKey
      // (ImagePickerModal, StageCard) cache under the "primary" default —
      // a primary-bucket delete must clear those slots too.
      if (sourceKey === (useHostsStore().primaryHost?.id ?? null)) {
        evictMedia(galleryMediaPath(filename, source, true));
        evictMedia(galleryMediaPath(filename, source));
      }
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
