import { defineStore } from "pinia";
import { markRaw, toRaw } from "vue";
import { apiFetchTo, conditionalApiJsonTo, type ApiTarget } from "../lib/api/client";
import { isTransportFailure } from "../lib/api/errors";
import {
  evictHostMedia,
  evictMedia,
  galleryMediaPath,
  isAudioItem,
  isVideoItem,
  type GallerySource,
} from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";
import { useHostsStore, type HostView } from "./hosts";
import type { Collection, GalleryImage, ServerEvent, TagCount } from "../lib/api/types";
import {
  GALLERY_IDENTITY_WINDOW_SECS,
  galleryPrintIdentity,
} from "@studio/lib/galleryPrintIdentity";
import {
  collectionSlug as slugOfCollection,
  collectionSlugResolver,
  mergeCollectionsAcrossHosts,
  normalizeTagName,
  planOrganizationFanout,
  sortTags,
  tagKey,
  unionOrganization,
  visibleTagCounts,
  type MergedCollection,
  type OrganizationFanoutOp,
  type OrganizationMutation,
  type OrganizationTarget,
  type OrganizationUnion,
} from "@studio/lib/libraryOrganization";
import {
  createCollection as createCollectionOn,
  deleteManyForever,
  deleteCollection as deleteCollectionOn,
  emptyTrash as emptyTrashOn,
  listCollections,
  listTags,
  listTrash,
  mutateGalleryBulk,
  organizeGallery,
  patchGalleryImage,
  restoreTrashed,
  sweepTrash as sweepTrashOn,
  trashMany,
  updateCollection,
  updateCollectionHidden,
} from "@studio/api/galleryOrganization";
import {
  enqueueGalleryMutation,
  galleryBulkRequest,
  listGalleryMutations,
  removeGalleryMutation,
  updateGalleryMutationFailure,
} from "@studio/lib/galleryMutationOutbox";
import { createUuid } from "@studio/lib/id";

export type {
  MergedCollection,
  OrganizationMutation,
  OrganizationUnion,
} from "@studio/lib/libraryOrganization";

/** Collapse a burst of row-less gallery_added events into one refetch. */
const REFETCH_DEBOUNCE_MS = 500;
let refetchTimer: ReturnType<typeof setTimeout> | null = null;

/** Library header scope (V3 "Shelf"): the grid, the collections shelf, or
 *  the trash. Never a sixth workspace — it lives inside Library. */
export type LibraryScope = "prints" | "collections" | "trash";

/** One host's collections listing. */
export interface CollectionsBucket {
  items: Collection[];
  loading: boolean;
  error: string | null;
  loaded: boolean;
}

/** One host's tag counts. */
export interface TagsBucket {
  items: TagCount[];
  loaded: boolean;
}

/** A physical copy of a logical print: which bucket holds which row. */
export interface GalleryCopy {
  sourceKey: string;
  item: GalleryImage;
}

/** Per-host trash retention, for the Trash banner (`trashRetentionSummary`). */
export interface RetentionHostEntry {
  key: string;
  label: string;
  retentionDays: number;
}

/** Outcome of a fan-out mutation over every copy of the selected prints. */
export interface FanoutResult {
  /** Per-host operations that succeeded. */
  applied: number;
  /** Per-host operations that failed. */
  failed: number;
  /** Bucket keys whose operation failed (already refetched). */
  failedHosts: string[];
  /** First failure message, for the toast. */
  error: string | null;
}

/** One host's (or this Mac's) slice of the unified gallery. */
export interface GalleryBucket {
  items: GalleryImage[];
  loading: boolean;
  error: string | null;
  loaded: boolean;
  /** Authority that produced this snapshot. `null` means the lifecycle lock
   *  proved the server was Off and native filesystem access was used. */
  authorityTarget?: ApiTarget | null;
  authorityResolved?: boolean;
}

/** A bucket that should exist right now. */
export interface GallerySourceRef {
  key: string;
  label: string;
}

export interface GalleryLocation {
  sourceKey: string;
  filename: string;
}

/** One print in the merged, date-sorted grid. */
export interface MergedPrint {
  item: GalleryImage;
  sourceKey: string;
  hostLabel: string;
  /** Every gallery bucket that contains this filename. */
  availableOn: GallerySourceRef[];
  /** Every physical row this logical print merged (one per bucket) — the
   *  input to `organizationOf`. Optional so single-bucket views and tests
   *  that build entries by hand keep working. */
  copies?: GalleryCopy[];
}

export interface GalleryChip {
  key: string;
  label: string;
  count: number;
}

/** Media-kind chip filter: everything, stills only, or video only. */
export type GalleryKindFilter = "all" | "image" | "video" | "audio";

/** Per-kind counts over the host-chip-filtered set (kind chip labels). */
export interface GalleryKindCounts {
  all: number;
  image: number;
  video: number;
  audio: number;
}

const emptyBucket = (): GalleryBucket => ({
  items: [],
  loading: false,
  error: null,
  loaded: false,
  authorityTarget: null,
  authorityResolved: false,
});

/**
 * Seed used for cross-host identity. Video files embed no `mold:parameters`,
 * so rows for old locally-mirrored videos were synthesized with `seed: 0` —
 * but the desktop's own auto-save filenames encode the real seed
 * (`mold-<model>-<seed>-<epochMs>[-role].<ext>`), so synthetic rows recover
 * it from the name. Non-synthetic rows trust their recorded metadata.
 */
/**
 * Cross-host identity beyond the filename: mirrored copies of one print are
 * byte-identical, so seed + exact byte size + model pins them together even
 * when an old auto-save invented its own filename or a video copy
 * synthesized its metadata (seed and model survive in the filename either
 * way). Rows missing seed or size opt out of identity matching entirely.
 */
export const printIdentity = galleryPrintIdentity;

/** Identity matches only count as one print when the rows were written
 *  around the same time: mirrors land within seconds of their origin, while
 *  a genuine re-generation that happens to reuse a seed (and byte length)
 *  lands much later and must stay a separate print. */
export const IDENTITY_WINDOW_SECS = GALLERY_IDENTITY_WINDOW_SECS;

export function withinIdentityWindow(a: GalleryImage, b: GalleryImage): boolean {
  return Math.abs(a.timestamp - b.timestamp) <= IDENTITY_WINDOW_SECS;
}

/**
 * Merge every bucket into one newest-first list of logical prints. Matching
 * filenames are one print (saved remote results retain their filename when
 * copied locally); seed + byte size (`printIdentity`) collapses copies whose
 * filenames diverged. The local copy is preferred for media and actions
 * while every location is retained for display. Shared by the live grid and
 * the trash so the two can never disagree on what "one print" means.
 */
function mergeBuckets(
  sources: GallerySourceRef[],
  buckets: Record<string, GalleryBucket>,
  pendingDeletions: Set<string>,
): MergedPrint[] {
  const byFilename = new Map<string, MergedPrint>();
  const byIdentity = new Map<string, MergedPrint>();
  const prints: MergedPrint[] = [];
  for (const source of sources) {
    const bucket = buckets[source.key];
    if (!bucket) continue;
    for (const item of bucket.items) {
      if (pendingDeletions.has(`${source.key}::${item.filename}`)) continue;
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
          copies: [{ sourceKey: source.key, item }],
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
      existing.copies!.push({ sourceKey: source.key, item });
      if (source.key === "local" && existing.sourceKey !== "local") {
        existing.item = item;
        existing.sourceKey = source.key;
        existing.hostLabel = source.label;
      }
    }
  }
  return prints.sort((a, b) => b.item.timestamp - a.item.timestamp);
}

/** Every copy's filename → its merged print, for O(1) union lookups. */
function indexCopies(prints: MergedPrint[]): Map<string, MergedPrint> {
  const index = new Map<string, MergedPrint>();
  for (const print of prints) {
    for (const copy of print.copies ?? []) index.set(copy.item.filename, print);
    index.set(print.item.filename, print);
  }
  return index;
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

/**
 * Gallery rows are immutable snapshots: every change replaces the row object
 * (`replaceRow` / `patchRow`), never a field on it. That lets the rows skip
 * Vue's deep reactivity — a 1 000-print bucket of 60-field metadata objects
 * would otherwise be a few thousand proxies whose get traps sit on every
 * merge, identity, and layout pass. Arrays stay reactive; the rows inside do
 * not. Apply at every site that puts a row into a bucket.
 */
function rawRow(image: GalleryImage): GalleryImage {
  return markRaw(image);
}
function rawRows(images: GalleryImage[]): GalleryImage[] {
  for (const image of images) markRaw(image);
  return images;
}

/** Replace a bucket row in place (same index) so the grid keeps its order. */
function replaceRow(bucket: GalleryBucket | undefined, image: GalleryImage): boolean {
  if (!bucket) return false;
  const at = bucket.items.findIndex((i) => i.filename === image.filename);
  if (at === -1) return false;
  bucket.items.splice(at, 1, rawRow(image));
  return true;
}

/** Insert newest-first, deduped by filename. */
function insertRow(bucket: GalleryBucket, image: GalleryImage) {
  if (bucket.items.some((i) => i.filename === image.filename)) return;
  const at = bucket.items.findIndex((i) => i.timestamp < image.timestamp);
  if (at === -1) bucket.items.push(rawRow(image));
  else bucket.items.splice(at, 0, rawRow(image));
}

/** Trash order: newest-DELETED first (`trashed_at`, matching the server's
 *  `view=trash` contract), with the creation `timestamp` only as a fallback
 *  for rows that never recorded a deletion time. */
const trashOrderKey = (i: GalleryImage) => i.trashed_at ?? i.timestamp;

/** Insert into a trash bucket in newest-deleted-first order. */
function insertTrashRow(bucket: GalleryBucket, image: GalleryImage) {
  if (bucket.items.some((i) => i.filename === image.filename)) return;
  const at = bucket.items.findIndex((i) => trashOrderKey(i) < trashOrderKey(image));
  if (at === -1) bucket.items.push(rawRow(image));
  else bucket.items.splice(at, 0, rawRow(image));
}

function takeRow(bucket: GalleryBucket | undefined, filename: string): GalleryImage | null {
  if (!bucket) return null;
  const at = bucket.items.findIndex((i) => i.filename === filename);
  if (at === -1) return null;
  return bucket.items.splice(at, 1)[0] ?? null;
}

const nowSecs = () => Math.floor(Date.now() / 1000);

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
    /** Filenames the Library grid has already shown to the user. A print not
     *  in this set (once the Library has been opened at least once) wears a
     *  NEW badge — "developed since your last Library visit". Session-scoped;
     *  the view snapshots it on open, then calls `markLibrarySeen`. */
    seenFilenames: new Set<string>(),
    /** Whether the Library has been opened this session — nothing is NEW on the
     *  very first visit (that visit only establishes the baseline). */
    libraryVisited: false,
    /** Prints optimistically removed from the grid, pending commit or undo
     *  (§08 G12). Keyed `${sourceKey}::${filename}`; excluded from every view
     *  the moment delete is pressed, restored by undo, and only DELETEd on
     *  commit. */
    pendingDeletions: new Set<string>(),
    /** Library header scope: Prints grid, Collections shelf, or Trash. */
    scope: "prints" as LibraryScope,
    /** ♥ chip — only favorites (union over every copy). */
    favoritesOnly: false,
    /** Tag chips (tag keys, AND). */
    tagFilter: [] as string[],
    /** Collections drill-in: the open collection's slug, or null (the shelf). */
    collectionSlug: null as string | null,
    /** Per-host trashed prints (`GET /api/gallery?view=trash`). Same keys
     *  as `buckets`; fetched on demand by the Trash scope. */
    trashBuckets: {} as Record<string, GalleryBucket>,
    /** This Mac's OFFLINE `.trash/` retention, read from the last native
     *  trash listing while the lifecycle proved the server Off; null while
     *  a running server's capability snapshot is the authority. Feeds the
     *  Trash banner so it never claims no machine keeps a trash while
     *  offline trash items are displayed. */
    localOfflineTrashRetentionDays: null as number | null,
    /** Per-host collections listings, merged by slug in `mergedCollections`. */
    collectionsByHost: {} as Record<string, CollectionsBucket>,
    /** Per-host tag counts, merged by case-insensitive name in `mergedTags`. */
    tagsByHost: {} as Record<string, TagsBucket>,
    /** Organization edits retained in IndexedDB for unreachable hosts. */
    pendingOrganizationMutations: 0,
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
      // Second-level identity (seed + byte size) collapses copies whose
      // filenames diverged — auto-saves from before the server shipped its
      // gallery filename on the complete event minted their own names.
      return mergeBuckets(this.sources, this.buckets, this.pendingDeletions);
    },
    /** Every live copy's filename → its merged print. */
    mergedIndex(): Map<string, MergedPrint> {
      return indexCopies(this.merged);
    },
    /** The trash, merged across hosts with the very same identity rules —
     *  but ordered newest-DELETED first (`trashed_at` desc, `timestamp`
     *  fallback), matching the server's `view=trash` contract. */
    trashMerged(): MergedPrint[] {
      return mergeBuckets(this.sources, this.trashBuckets, this.pendingDeletions).sort(
        (a, b) => trashOrderKey(b.item) - trashOrderKey(a.item),
      );
    },
    trashIndex(): Map<string, MergedPrint> {
      return indexCopies(this.trashMerged);
    },
    /** Per-bucket filename + identity indexes over the live rows (pending
     *  deletions included, exactly like the buckets themselves). Rebuilt once
     *  per bucket change; read by every copy lookup so the grid never scans. */
    bucketIndex(): Map<string, BucketIndex> {
      return indexBuckets(this.buckets);
    },
    trashBucketIndex(): Map<string, BucketIndex> {
      return indexBuckets(this.trashBuckets);
    },
    /**
     * Every copy filename → its logical print's organization union, computed
     * ONCE per logical print per data change. This is the single place
     * `unionOrganization` runs over the grid: the filter chips, the shelf
     * counts, the kind counts, `filtered`, and every tile read this map, so
     * one gallery mutation costs one union pass rather than one per getter
     * per tile. Live prints are indexed first so they win over trashed twins.
     */
    organizationIndex(): Map<string, OrganizationUnion> {
      const resolve = this.collectionResolver;
      const index = new Map<string, OrganizationUnion>();
      const add = (prints: MergedPrint[]) => {
        for (const print of prints) {
          const copies = print.copies ?? [{ sourceKey: print.sourceKey, item: print.item }];
          const org = unionOrganization(
            copies.map((copy) => ({ hostId: copy.sourceKey, item: copy.item })),
            { localHostId: "local", resolveCollectionSlug: resolve },
          );
          if (!index.has(print.item.filename)) index.set(print.item.filename, org);
          for (const copy of copies) {
            if (!index.has(copy.item.filename)) index.set(copy.item.filename, org);
          }
        }
      };
      add(this.merged);
      add(this.trashMerged);
      return index;
    },
    /** Logical prints in the trash across every host. */
    trashCount(): number {
      return this.trashMerged.length;
    },
    /** Trash grid: host chip → kind → query, like `filtered`. */
    trashFiltered(): MergedPrint[] {
      let entries = this.trashMerged;
      if (this.filter !== "all" && this.sources.some((s) => s.key === this.filter)) {
        entries = entries.filter((e) => e.availableOn.some((s) => s.key === this.filter));
      }
      return this.narrowByKindAndQuery(entries);
    },
    // ── Organization capability (from /api/capabilities) ──────────────────
    /** Titles / favorites / tags / collections can be edited on this bucket's
     *  host. The "local" key is the built-in engine (host id "local"); with
     *  the server Off there is no capability snapshot, so the native IPC
     *  bucket reads as not organize-capable. */
    organizeCapable(): (hostKey: string) => boolean {
      const caps = useHostsStore().capabilities;
      return (hostKey) => caps[hostKey]?.gallery?.organize === true;
    },
    /** Delete moves to the trash on this bucket's host (else hard-deletes). */
    trashCapable(): (hostKey: string) => boolean {
      const caps = useHostsStore().capabilities;
      return (hostKey) => caps[hostKey]?.gallery?.trash?.enabled === true;
    },
    anyOrganizeCapable(): boolean {
      return this.sources.some((s) => this.organizeCapable(s.key));
    },
    anyTrashCapable(): boolean {
      return this.sources.some((s) => this.trashCapable(s.key));
    },
    /** Per-host retention for the Trash banner; trash-capable hosts only,
     *  This device first (it sets the sentence). With the local server Off
     *  there is no capability snapshot, but the offline `.trash/` is still
     *  in use — the native trash listing reports this Mac's configured
     *  retention, so This device stays represented rather than the banner
     *  claiming no machine keeps a trash. */
    retentionByHost(): RetentionHostEntry[] {
      const hosts = useHostsStore();
      const caps = hosts.capabilities;
      // `hostFor("local")` is an action; getters only see state + getters,
      // so the "engine not ready" check is inlined here.
      const localEngineReady = hosts.all.some((h) => h.id === "local" && h.status === "ready");
      const out: RetentionHostEntry[] = [];
      for (const source of this.sources) {
        const trash = caps[source.key]?.gallery?.trash;
        if (trash?.enabled) {
          out.push({ key: source.key, label: source.label, retentionDays: trash.retention_days });
          continue;
        }
        if (
          source.key === "local" &&
          this.localOfflineTrashRetentionDays !== null &&
          !localEngineReady
        ) {
          out.push({
            key: source.key,
            label: source.label,
            retentionDays: this.localOfflineTrashRetentionDays,
          });
        }
      }
      return out;
    },
    // ── Collections + tags ────────────────────────────────────────────────
    /** Collections across every loaded host, merged by slug (name-sorted). */
    mergedCollections(): MergedCollection[] {
      return mergeCollectionsAcrossHosts(
        this.sources
          .filter((s) => this.collectionsByHost[s.key]?.loaded)
          .map((s) => ({
            hostId: s.key,
            hostLabel: s.label,
            collections: this.collectionsByHost[s.key]!.items,
          })),
      );
    },
    /** `(hostKey, collectionId) → slug` over every loaded listing. */
    collectionResolver(): (hostKey: string, collectionId: string) => string | null | undefined {
      return collectionSlugResolver(
        Object.entries(this.collectionsByHost)
          .filter(([, bucket]) => bucket.loaded)
          .map(([hostId, bucket]) => ({ hostId, collections: bucket.items })),
      );
    },
    /** Tags across every loaded host, merged case-insensitively with counts
     *  summed, count-desc then name. */
    mergedTags(): TagCount[] {
      const byKey = new Map<string, TagCount>();
      for (const source of this.sources) {
        const bucket = this.tagsByHost[source.key];
        if (!bucket?.loaded) continue;
        for (const tag of bucket.items) {
          const name = normalizeTagName(tag.name);
          if (!name) continue;
          const key = name.toLowerCase();
          const existing = byKey.get(key);
          if (existing) existing.count += tag.count;
          else byKey.set(key, { name, count: tag.count });
        }
      }
      return sortTags([...byKey.values()]);
    },
    /**
     * One logical print's organization across every physical copy (union:
     * any-favorite, tag union, collections resolved to slugs, local title
     * preferred). Live copies win over trashed ones; an entry that matches
     * no merged print (hand-built, or mid-refetch) reads its own row.
     */
    organizationOf(): (entry: MergedPrint) => OrganizationUnion {
      const resolve = this.collectionResolver;
      const index = this.organizationIndex;
      return (entry) => {
        const indexed = index.get(entry.item.filename);
        if (indexed) return indexed;
        // Hand-built or mid-refetch entry that matches no merged print: read
        // its own copies (the only path that still unions per call).
        const copies = entry.copies ?? [{ sourceKey: entry.sourceKey, item: entry.item }];
        return unionOrganization(
          copies.map((copy) => ({ hostId: copy.sourceKey, item: copy.item })),
          { localHostId: "local", resolveCollectionSlug: resolve },
        );
      };
    },
    /** Excludes members of hidden collections only from the default Prints scope. */
    visibleInDefaultLibrary(): (entry: MergedPrint) => boolean {
      const hiddenSlugs = new Set(
        this.mergedCollections
          .filter((collection) => collection.hidden)
          .map((collection) => collection.slug),
      );
      const organizationOf = this.organizationOf;
      return (entry) => !organizationOf(entry).collections.some((slug) => hiddenSlugs.has(slug));
    },
    /** Logical prints available to default filter-chip counts. */
    basePrints(): MergedPrint[] {
      return this.scope === "prints"
        ? this.merged.filter(this.visibleInDefaultLibrary)
        : this.merged;
    },
    /** Header/chip count before host, kind, search, and organization narrowing. */
    basePrintCount(): number {
      return this.basePrints.length;
    },
    /** Exact tag counts over the same logical prints as the default grid. */
    filterChipTags(): TagCount[] {
      const visible = this.basePrints;
      const excluded =
        this.scope === "prints"
          ? this.merged.filter((entry) => !this.visibleInDefaultLibrary(entry))
          : [];
      return visibleTagCounts(
        this.mergedTags,
        visible.map(this.organizationOf),
        excluded.map(this.organizationOf),
      );
    },
    /** Logical prints per collection slug, over the merged live grid — the
     *  count a shelf card shows (a mirrored print counts once). */
    collectionCounts(): (slug: string) => number {
      const counts = new Map<string, number>();
      for (const entry of this.merged) {
        for (const slug of this.organizationOf(entry).collections) {
          counts.set(slug, (counts.get(slug) ?? 0) + 1);
        }
      }
      return (slug) => counts.get(slug) ?? 0;
    },
    /** Kind + text narrowing shared by the live grid and the trash. */
    narrowByKindAndQuery(): (entries: MergedPrint[]) => MergedPrint[] {
      const mediaKind = this.mediaKind;
      const q = this.query.trim().toLowerCase();
      return (input) => {
        let entries = input;
        if (mediaKind !== "all") {
          // Three disjoint kinds, not a video/not-video split: an audio print
          // has no frames and must not fall into the Images bucket.
          entries = entries.filter((e) => {
            if (isAudioItem(e.item)) return mediaKind === "audio";
            if (isVideoItem(e.item)) return mediaKind === "video";
            return mediaKind === "image";
          });
        }
        if (q) {
          entries = entries.filter(
            (e) =>
              e.item.filename.toLowerCase().includes(q) ||
              e.item.metadata.model.toLowerCase().includes(q) ||
              e.item.metadata.prompt.toLowerCase().includes(q) ||
              (e.item.title ?? "").toLowerCase().includes(q),
          );
        }
        return entries;
      };
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
        .filter((item) => !this.pendingDeletions.has(`${source.key}::${item.filename}`))
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
      const local = this.bucketIndex.get("local");
      return (entry) => {
        if (entry.sourceKey === "local") return true;
        if (entry.availableOn.some((s) => s.key === "local")) return true;
        if (!local) return false;
        if (local.byFilename.has(entry.item.filename)) return true;
        const identity = printIdentity(entry.item);
        if (identity === null) return false;
        return (local.byIdentity.get(identity) ?? []).some((row) =>
          withinIdentityWindow(row.item, entry.item),
        );
      };
    },
    /**
     * What the Gallery grid renders: host chip → media kind → text query →
     * ♥ favorites → tag chips (AND over the union tags) → the open
     * collection (Collections scope drill-in only). Filtering preserves the
     * gallery's newest-first generation order; collection membership order is
     * deliberately irrelevant because it records when prints were filed.
     */
    filtered(): MergedPrint[] {
      let entries = this.narrowByKindAndQuery(this.hostFiltered);
      const wantsFavorites = this.favoritesOnly;
      const tagKeys = this.tagFilter.map((t) => tagKey(t)).filter((k) => k.length > 0);
      const slug = this.scope === "collections" ? this.collectionSlug : null;
      if (this.scope === "prints") entries = entries.filter(this.visibleInDefaultLibrary);
      if (!wantsFavorites && tagKeys.length === 0 && !slug) return entries;
      const organizationOf = this.organizationOf;
      entries = entries.filter((entry) => {
        const org = organizationOf(entry);
        if (wantsFavorites && !org.favorite) return false;
        if (tagKeys.length > 0) {
          const have = new Set(org.tags.map((t) => t.toLowerCase()));
          if (!tagKeys.every((k) => have.has(k))) return false;
        }
        if (slug && !org.collections.includes(slug)) return false;
        return true;
      });
      return entries;
    },
    /** Per-kind counts for the All/Images/Video/Audio chips. Computed over
     *  the host-chip-filtered set only, so chip labels stay stable while the
     *  kind chip or search narrows the grid. */
    kindCounts(): GalleryKindCounts {
      const entries =
        this.scope === "prints"
          ? this.hostFiltered.filter(this.visibleInDefaultLibrary)
          : this.hostFiltered;
      let video = 0;
      let audio = 0;
      for (const e of entries) {
        if (isAudioItem(e.item)) audio++;
        else if (isVideoItem(e.item)) video++;
      }
      return {
        all: entries.length,
        image: entries.length - video - audio,
        video,
        audio,
      };
    },
    /** Per-source chips for the gallery header (HostFilterChips adds All). */
    chipCounts(): GalleryChip[] {
      const hiddenSlugs = new Set(
        this.mergedCollections
          .filter((collection) => collection.hidden)
          .map((collection) => collection.slug),
      );
      const organization = this.organizationIndex;
      const hidesInDefault = (item: GalleryImage) =>
        (organization.get(item.filename)?.collections ?? []).some((slug) => hiddenSlugs.has(slug));
      return this.sources.map((source) => {
        const items = this.buckets[source.key]?.items ?? [];
        const count =
          this.scope === "prints" && hiddenSlugs.size > 0
            ? items.filter((item) => !hidesInDefault(item)).length
            : items.length;
        return { key: source.key, label: source.label, count };
      });
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
    /**
     * Prints developed since the last Library visit — drives the Library nav
     * badge (§08 G11). Zero until the Library has been opened once (that visit
     * only establishes the baseline), then counts every merged print not yet
     * marked seen. Re-opening Library calls `markLibrarySeen`, resetting it.
     */
    newCount(): number {
      if (!this.libraryVisited) return 0;
      let n = 0;
      for (const entry of this.merged) {
        if (!this.seenFilenames.has(entry.item.filename)) n++;
      }
      return n;
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
      return this.targetOf(sourceKey) ? "host" : "local";
    },
    /** Auth target for a bucket's host; null for the This-Mac IPC bucket. */
    targetOf(sourceKey: string): ApiTarget | null {
      const bucket = this.buckets[sourceKey];
      if (sourceKey === "local" && bucket?.authorityResolved) {
        if (bucket.authorityTarget === undefined) {
          throw new Error("The local gallery snapshot is missing its authority target.");
        }
        return bucket.authorityTarget;
      }
      const host = this.hostFor(sourceKey);
      return host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    ensureBucket(key: string): GalleryBucket {
      if (!this.buckets[key]) this.buckets[key] = emptyBucket();
      return this.buckets[key]!;
    },
    /** Record every currently-known print as seen. Called by the Library view
     *  on open (after its prints load) so the NEW badges shown this visit are
     *  gone next time. The view snapshots the pre-visit set first, so marking
     *  seen here never erases the badges the user is looking at right now. */
    markLibrarySeen() {
      for (const entry of this.merged) this.seenFilenames.add(entry.item.filename);
      this.libraryVisited = true;
    },
    /** Drop buckets whose source disappeared; their cached media goes too. */
    syncBuckets() {
      const keys = new Set(this.sources.map((s) => s.key));
      for (const key of Object.keys(this.buckets)) {
        if (keys.has(key)) continue;
        delete this.buckets[key];
        evictHostMedia(key);
      }
      for (const key of Object.keys(this.trashBuckets)) {
        if (!keys.has(key)) delete this.trashBuckets[key];
      }
      for (const key of Object.keys(this.collectionsByHost)) {
        if (!keys.has(key)) delete this.collectionsByHost[key];
      }
      for (const key of Object.keys(this.tagsByHost)) {
        if (!keys.has(key)) delete this.tagsByHost[key];
      }
      if (this.filter !== "all" && !keys.has(this.filter)) this.filter = "all";
    },
    /** Fetch one bucket. Errors land on the bucket, never on siblings. */
    async fetchBucket(key: string) {
      const bucket = this.ensureBucket(key);
      if (bucket.loading) return;
      bucket.loading = true;
      bucket.error = null;
      const previousSource = this.mediaSourceOf(key);
      try {
        // Resolve the origin at call time: a key backed by a live host reads
        // that host's /api/gallery; the bare "local" key is this Mac via IPC.
        let items: GalleryImage[];
        let authorityTarget: ApiTarget | null;
        if (key === "local") {
          const snapshot = await ipc.localGalleryList();
          items = snapshot.images;
          authorityTarget = snapshot.target;
        } else {
          const target = this.targetOf(key);
          if (!target) throw new Error("Host is not connected.");
          items = await conditionalApiJsonTo<GalleryImage[]>(target, "/api/gallery");
          authorityTarget = target;
        }
        const authorityChanged =
          bucket.authorityResolved &&
          (bucket.authorityTarget?.baseUrl !== authorityTarget?.baseUrl ||
            bucket.authorityTarget?.apiKey !== authorityTarget?.apiKey);
        // A 304 hands back the very array already in the bucket. Nothing
        // changed, so nothing downstream (merge, indexes, layout, every
        // tile) may be invalidated — assigning the same rows again would
        // re-run all of it on every 15 s poll of every host. A changed
        // authority still falls through: its cached media must be evicted.
        if (bucket.loaded && !authorityChanged && toRaw(bucket.items) === items) return;
        // Prints that vanished out-of-band (deleted by another client) must
        // release their cached blob URLs — the media cache only evicts on
        // explicit remove()/host teardown otherwise.
        const next = new Set(items.map((i) => i.filename));
        for (const old of bucket.items) {
          if (!authorityChanged && next.has(old.filename)) continue;
          evictMedia(galleryMediaPath(old.filename, previousSource, true), key);
          evictMedia(galleryMediaPath(old.filename, previousSource), key);
        }
        bucket.authorityTarget = authorityTarget;
        bucket.authorityResolved = true;
        // Newest first, like a print drawer.
        bucket.items = rawRows(items.sort((a, b) => b.timestamp - a.timestamp));
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
    /**
     * Every currently loaded device copy of one logical print. Exact filenames
     * cover modern mirrors; seed + byte-size + model covers legacy auto-saves
     * that minted a different filename.
     */
    locationsOf(entry: MergedPrint): GalleryLocation[] {
      return locationsIn(this.bucketIndex, entry);
    },
    /** Every loaded trashed copy of one logical print. */
    trashLocationsOf(entry: MergedPrint): GalleryLocation[] {
      return locationsIn(this.trashBucketIndex, entry);
    },
    /** Live and trashed copies together — organization edits reach both. */
    allLocationsOf(entry: MergedPrint): GalleryLocation[] {
      const unique = new Map<string, GalleryLocation>();
      for (const location of [...this.locationsOf(entry), ...this.trashLocationsOf(entry)]) {
        unique.set(`${location.sourceKey}::${location.filename}`, location);
      }
      return [...unique.values()];
    },
    /** Optimistically hide a print from every view, pending commit or undo. */
    beginDelete(sourceKey: string, filename: string) {
      this.pendingDeletions.add(`${sourceKey}::${filename}`);
    },
    /** Hide every known copy and return the concrete locations held for undo. */
    beginDeleteEverywhere(entry: MergedPrint): GalleryLocation[] {
      const locations = this.locationsOf(entry);
      for (const location of locations) {
        this.beginDelete(location.sourceKey, location.filename);
      }
      return locations;
    },
    /** Undo an optimistic delete — no server call; the print returns to the grid. */
    cancelDelete(sourceKey: string, filename: string) {
      this.pendingDeletions.delete(`${sourceKey}::${filename}`);
    },
    cancelDeleteEverywhere(locations: GalleryLocation[]) {
      for (const location of locations) {
        this.cancelDelete(location.sourceKey, location.filename);
      }
    },
    /**
     * Commit an optimistic delete: run the real DELETE, then drop the pending
     * mark. A failed DELETE restores the print (the bucket row survived and the
     * mark clears) and rethrows so the caller can surface the error.
     */
    async commitDelete(sourceKey: string, filename: string) {
      const key = `${sourceKey}::${filename}`;
      if (!this.pendingDeletions.has(key)) return; // already undone/committed
      try {
        await this.remove(sourceKey, filename);
      } finally {
        this.pendingDeletions.delete(key);
      }
      await this.refreshTrash([sourceKey]);
    },
    async commitDeleteEverywhere(
      locations: GalleryLocation[],
    ): Promise<{ deleted: number; failed: number; error: string | null }> {
      const results = await Promise.allSettled(
        locations.map(async (location) => {
          try {
            await this.remove(location.sourceKey, location.filename);
          } finally {
            this.pendingDeletions.delete(`${location.sourceKey}::${location.filename}`);
          }
        }),
      );
      const failedOrigins = new Set<string>();
      results.forEach((result, index) => {
        if (result.status === "rejected") failedOrigins.add(locations[index]!.sourceKey);
      });
      await Promise.all([...failedOrigins].map((key) => this.refreshHost(key)));
      await this.refreshTrash(locations.map((l) => l.sourceKey));
      const failed = results.filter((result) => result.status === "rejected").length;
      const rejection = results.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      return {
        deleted: results.length - failed,
        failed,
        error: rejection
          ? rejection.reason instanceof Error
            ? rejection.reason.message
            : String(rejection.reason)
          : null,
      };
    },
    /**
     * Delete a print where it lives. On a trash-capable host the plain
     * DELETE moves the print to that host's trash (the row hops into the
     * trash bucket and its cached media survives — thumbnails do too); on an
     * older host it is gone for good. `permanent` bypasses (or purges from)
     * the trash on every host. The host-less This-device bucket routes
     * through the native commands, which make the same decision offline.
     */
    async remove(sourceKey: string, filename: string, options: { permanent?: boolean } = {}) {
      const permanent = options.permanent === true;
      const target = this.targetOf(sourceKey);
      // A print deleted forever leaves the persistent thumbnail cache too; a
      // trashed one keeps its tile (the Trash grid still shows it).
      if (permanent) this.forgetThumbnailFor(sourceKey, this.rowOf(sourceKey, filename));
      if (target) {
        const suffix = permanent ? "?permanent=true" : "";
        await apiFetchTo(target, `/api/gallery/image/${encodeURIComponent(filename)}${suffix}`, {
          method: "DELETE",
        });
      } else if (sourceKey === "local") {
        if (permanent) await ipc.localGalleryDeleteForever(filename);
        else await ipc.localGalleryDelete(filename);
      } else {
        throw new Error("Host is not connected.");
      }
      this.applyRemovalToBuckets(sourceKey, filename, permanent);
    },
    /**
     * Drop a permanently deleted print's tiles from the persistent native
     * cache, keyed by the SAME effective version the tile was filed under
     * (`media_version`, else the `timestamp:size` fallback older hosts get).
     * Every permanent path — direct delete, bulk delete forever, Empty
     * trash — must call this, or the bytes outlive the print until LRU.
     */
    forgetThumbnailFor(sourceKey: string, row: GalleryImage | null) {
      if (!row) return;
      const version = row.media_version ?? `${row.timestamp}:${row.size_bytes ?? "unknown"}`;
      const target = this.targetOf(sourceKey);
      // Deferred so a bridge that is absent (browser shell, older native
      // build) fails inside the promise rather than out of the delete path.
      void Promise.resolve()
        .then(() =>
          ipc.forgetGalleryThumbnail(sourceKey, target?.baseUrl ?? null, row.filename, version),
        )
        .catch(() => {});
    },
    applyRemovalToBuckets(sourceKey: string, filename: string, permanent: boolean) {
      const row = takeRow(this.buckets[sourceKey], filename);
      if (!permanent && this.trashCapable(sourceKey)) {
        const trash = this.trashBuckets[sourceKey];
        if (row && trash?.loaded) {
          const retention = useHostsStore().capabilities[sourceKey]?.gallery?.trash?.retention_days;
          const trashedAt = nowSecs();
          insertTrashRow(trash, {
            ...row,
            trashed_at: trashedAt,
            purge_at: retention && retention > 0 ? trashedAt + retention * 86_400 : null,
          });
        }
        return;
      }
      takeRow(this.trashBuckets[sourceKey], filename);
      this.evictItemMedia(sourceKey, filename);
    },
    async removeHostMany(sourceKey: string, filenames: string[], permanent = false) {
      const target = this.targetOf(sourceKey);
      const bulk = useHostsStore().capabilities[sourceKey]?.gallery?.bulk_mutations === true;
      if (target && bulk && (permanent || this.trashCapable(sourceKey))) {
        try {
          if (permanent) await deleteManyForever(target, filenames);
          else await trashMany(target, filenames);
          for (const filename of filenames) {
            if (permanent) this.forgetThumbnailFor(sourceKey, this.rowOf(sourceKey, filename));
            this.applyRemovalToBuckets(sourceKey, filename, permanent);
          }
          return { deleted: [...filenames], failed: [] as string[] };
        } catch {
          return { deleted: [] as string[], failed: [...filenames] };
        }
      }
      const results = await Promise.allSettled(
        filenames.map((filename) => this.remove(sourceKey, filename, { permanent })),
      );
      return {
        deleted: filenames.filter((_, index) => results[index]?.status === "fulfilled"),
        failed: filenames.filter((_, index) => results[index]?.status === "rejected"),
      };
    },
    /**
     * Bulk delete. Current hosts receive one request per origin; legacy and
     * native-only origins preserve the per-file path.
     * Origins whose delete failed are refetched to reconverge with the
     * server (a lost response may still have deleted server-side).
     */
    async removeMany(
      items: Array<{ sourceKey: string; filename: string }>,
    ): Promise<{ deleted: number; failed: number }> {
      const groups = new Map<string, string[]>();
      for (const item of items) {
        const names = groups.get(item.sourceKey) ?? [];
        names.push(item.filename);
        groups.set(item.sourceKey, names);
      }
      const grouped = [...groups];
      const results = await Promise.all(
        grouped.map(([sourceKey, filenames]) => this.removeHostMany(sourceKey, filenames)),
      );
      let deleted = 0;
      const failedOrigins = new Set<string>();
      results.forEach((r, idx) => {
        const [sourceKey] = grouped[idx]!;
        deleted += r.deleted.length;
        if (r.failed.length > 0) failedOrigins.add(sourceKey);
      });
      await Promise.all([...failedOrigins].map((key) => this.refreshHost(key)));
      await this.refreshTrash(items.map((i) => i.sourceKey));
      return { deleted, failed: items.length - deleted };
    },
    /**
     * Delete each selected logical print from every device bucket that
     * currently contains a matching copy. Concrete locations are de-duplicated
     * so selecting two visible twins never sends the same DELETE twice.
     */
    async removeEntriesEverywhere(
      entries: MergedPrint[],
    ): Promise<{ deletedPrints: number; failedPrints: number; deletedCopies: number }> {
      const groups = entries.map((entry) => this.locationsOf(entry));
      const unique = new Map<string, GalleryLocation>();
      for (const group of groups) {
        for (const location of group) {
          unique.set(`${location.sourceKey}::${location.filename}`, location);
        }
      }
      const locations = [...unique.values()];
      const byHost = new Map<string, GalleryLocation[]>();
      for (const location of locations) {
        const host = byHost.get(location.sourceKey) ?? [];
        host.push(location);
        byHost.set(location.sourceKey, host);
      }
      const hostGroups = [...byHost];
      const results = await Promise.all(
        hostGroups.map(([sourceKey, group]) =>
          this.removeHostMany(
            sourceKey,
            group.map((location) => location.filename),
          ),
        ),
      );
      const failedKeys = new Set<string>();
      const failedOrigins = new Set<string>();
      results.forEach((result, index) => {
        if (result.failed.length === 0) return;
        const [sourceKey] = hostGroups[index]!;
        failedOrigins.add(sourceKey);
        for (const filename of result.failed) {
          failedKeys.add(`${sourceKey}::${filename}`);
        }
      });
      await Promise.all([...failedOrigins].map((key) => this.refreshHost(key)));
      await this.refreshTrash(locations.map((l) => l.sourceKey));
      const failedPrints = groups.filter((group) =>
        group.some((location) => failedKeys.has(`${location.sourceKey}::${location.filename}`)),
      ).length;
      return {
        deletedPrints: entries.length - failedPrints,
        failedPrints,
        deletedCopies: locations.length - failedKeys.size,
      };
    },
    evictItemMedia(sourceKey: string, filename: string) {
      const source = this.mediaSourceOf(sourceKey);
      evictMedia(galleryMediaPath(filename, source, true), sourceKey);
      evictMedia(galleryMediaPath(filename, source), sourceKey);
      // Consumers that render primary-gallery media without a cacheKey
      // (StageCard) cache under the "primary" default — a primary-bucket
      // delete must clear those slots too.
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
        this.schedulePrimaryRefetch();
        return;
      }
      insertRow(bucket, ev.image);
    },
    /** Debounced refetch of the primary's live bucket for row-less frames. */
    schedulePrimaryRefetch() {
      if (refetchTimer) clearTimeout(refetchTimer);
      refetchTimer = setTimeout(() => {
        refetchTimer = null;
        // Re-check at fire time: the primary may have changed, or a fetch
        // may already be in flight — re-debounce rather than racing it.
        const nowKey = useHostsStore().primaryHost?.id ?? null;
        const nowBucket = nowKey ? this.buckets[nowKey] : undefined;
        if (!nowKey || !nowBucket?.loaded) return;
        if (nowBucket.loading) {
          this.schedulePrimaryRefetch();
          return;
        }
        void this.fetchBucket(nowKey);
      }, REFETCH_DEBOUNCE_MS);
    },
    /**
     * A `gallery_updated` frame — a primary row's title / favorite / tags /
     * collections changed. Replace the row in place (live or trash bucket);
     * without a row, refetch like `applyAdded`.
     */
    applyUpdated(ev: Extract<ServerEvent, { type: "gallery_updated" }>) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      const image = ev.image ?? null;
      if (!image) {
        this.schedulePrimaryRefetch();
        return;
      }
      if (replaceRow(bucket, image)) return;
      if (replaceRow(this.trashBuckets[key], image)) return;
      // Unknown filename: the row was added out-of-band (or is trashed and
      // the trash never loaded). A live row lands in the grid; a trashed
      // one is only inserted when the trash is loaded.
      if (image.trashed_at != null) {
        const trash = this.trashBuckets[key];
        if (trash?.loaded) insertTrashRow(trash, image);
        return;
      }
      insertRow(bucket, image);
    },
    /** A `gallery_trashed` frame — the primary's row hops into its trash. */
    applyTrashed(filename: string) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      const row = takeRow(bucket, filename);
      const trash = this.trashBuckets[key];
      if (!trash?.loaded) return;
      if (trash.items.some((i) => i.filename === filename)) return;
      if (!row) return; // trashed before we ever listed it; the next trash fetch has it
      const retention = useHostsStore().capabilities[key]?.gallery?.trash?.retention_days;
      const trashedAt = nowSecs();
      insertTrashRow(trash, {
        ...row,
        trashed_at: trashedAt,
        purge_at: retention && retention > 0 ? trashedAt + retention * 86_400 : null,
      });
    },
    /** A `gallery_restored` frame — back from the trash into the live grid. */
    applyRestored(ev: Extract<ServerEvent, { type: "gallery_restored" }>) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      const trashed = takeRow(this.trashBuckets[key], ev.filename);
      const image = ev.image ?? null;
      if (image) {
        insertRow(bucket, { ...image, trashed_at: null, purge_at: null });
        return;
      }
      if (trashed) {
        insertRow(bucket, { ...trashed, trashed_at: null, purge_at: null });
        return;
      }
      this.schedulePrimaryRefetch();
    },
    /** A `gallery_collections_changed` frame — refetch the primary's listing. */
    applyCollectionsChanged() {
      const key = useHostsStore().primaryHost?.id ?? null;
      if (!key || !this.collectionsByHost[key]?.loaded) return;
      void this.fetchCollections(key);
    },
    /** A `gallery_removed` frame — drop the primary's tile (live or trashed:
     *  a purge reuses this frame) and its media. */
    applyRemoved(filename: string) {
      const key = useHostsStore().primaryHost?.id ?? null;
      const bucket = key ? this.buckets[key] : undefined;
      if (!key || !bucket?.loaded) return;
      const live = takeRow(bucket, filename);
      const trashed = takeRow(this.trashBuckets[key], filename);
      if (!live && !trashed) return;
      this.evictItemMedia(key, filename);
    },

    // ── Organization: fetch ──────────────────────────────────────────────
    ensureCollectionsBucket(key: string): CollectionsBucket {
      if (!this.collectionsByHost[key]) {
        this.collectionsByHost[key] = { items: [], loading: false, error: null, loaded: false };
      }
      return this.collectionsByHost[key]!;
    },
    ensureTagsBucket(key: string): TagsBucket {
      if (!this.tagsByHost[key]) this.tagsByHost[key] = { items: [], loaded: false };
      return this.tagsByHost[key]!;
    },
    ensureTrashBucket(key: string): GalleryBucket {
      if (!this.trashBuckets[key]) this.trashBuckets[key] = emptyBucket();
      return this.trashBuckets[key]!;
    },
    /** Fetch one host's collections; non-capable hosts settle empty. */
    async fetchCollections(hostKey?: string) {
      if (hostKey === undefined) {
        await Promise.all(this.sources.map((s) => this.fetchCollections(s.key)));
        return;
      }
      const bucket = this.ensureCollectionsBucket(hostKey);
      if (bucket.loading) return;
      const target = this.targetOf(hostKey);
      if (!target || !this.organizeCapable(hostKey)) {
        bucket.items = [];
        bucket.error = null;
        bucket.loaded = true;
        return;
      }
      bucket.loading = true;
      bucket.error = null;
      try {
        bucket.items = await listCollections(target);
        bucket.loaded = true;
      } catch (err) {
        bucket.error = String(err);
      } finally {
        bucket.loading = false;
      }
    },
    /** Fetch one host's tag counts; non-capable hosts settle empty. */
    async fetchTags(hostKey?: string) {
      if (hostKey === undefined) {
        await Promise.all(this.sources.map((s) => this.fetchTags(s.key)));
        return;
      }
      const bucket = this.ensureTagsBucket(hostKey);
      const target = this.targetOf(hostKey);
      if (!target || !this.organizeCapable(hostKey)) {
        bucket.items = [];
        bucket.loaded = true;
        return;
      }
      try {
        bucket.items = await listTags(target);
        bucket.loaded = true;
      } catch {
        // Tag chips are advisory — keep the last listing.
      }
    },
    /**
     * Fetch one host's trash. The bare "local" key reads through the native
     * command (authenticated HTTP while the local server runs, the `.trash/`
     * listing when the lifecycle proves it Off); hosts that do not advertise
     * a trash settle empty rather than misreading `/api/gallery`.
     */
    async fetchTrash(hostKey?: string) {
      if (hostKey === undefined) {
        await Promise.all(this.sources.map((s) => this.fetchTrash(s.key)));
        return;
      }
      const bucket = this.ensureTrashBucket(hostKey);
      if (bucket.loading) return;
      bucket.loading = true;
      bucket.error = null;
      try {
        let items: GalleryImage[] = [];
        let authorityTarget: ApiTarget | null = null;
        if (hostKey === "local" && (this.trashCapable("local") || !this.hostFor("local"))) {
          const snapshot = await ipc.localGalleryTrashList();
          items = snapshot.images;
          authorityTarget = snapshot.target;
          // The offline listing carries this Mac's retention (read through
          // the same Config overlay the sweeper uses) so the Trash banner
          // can name This device while no capability snapshot exists
          // (server Off ⇒ no /api/capabilities). A native listing missing
          // the field but holding trashed prints falls back to the config
          // default of 30 days; a browser-shell stub (no field, no items)
          // claims nothing — there is no native trash to represent.
          this.localOfflineTrashRetentionDays =
            snapshot.target === null &&
            (snapshot.retentionDays !== undefined || snapshot.images.length > 0)
              ? (snapshot.retentionDays ?? 30)
              : null;
        } else {
          const target = this.targetOf(hostKey);
          if (target && this.trashCapable(hostKey)) {
            items = await listTrash<GalleryImage>(target);
            authorityTarget = target;
          }
        }
        bucket.authorityTarget = authorityTarget;
        bucket.authorityResolved = true;
        bucket.items = rawRows(items.sort((a, b) => trashOrderKey(b) - trashOrderKey(a)));
        bucket.loaded = true;
      } catch (err) {
        bucket.error = String(err);
      } finally {
        bucket.loading = false;
      }
    },
    /** Refetch already-loaded trash buckets after a mutation landed there. */
    async refreshTrash(hostKeys: string[]) {
      const keys = [...new Set(hostKeys)].filter((key) => this.trashBuckets[key]?.loaded);
      await Promise.all(keys.map((key) => this.fetchTrash(key)));
    },
    /** Collections, tags, and trash for every current source. */
    async fetchOrganization() {
      this.syncBuckets();
      await Promise.all([this.fetchCollections(), this.fetchTags(), this.fetchTrash()]);
      await this.flushOrganizationOutbox();
    },
    // ── Organization: mutate ─────────────────────────────────────────────
    /** Organize-capable per-host targets for a set of copies. Hosts that
     *  cannot organize (old servers, the offline IPC bucket) are skipped,
     *  never failed — the union still shows the edit from the hosts that
     *  took it. */
    organizeTargetsFor(entries: MergedPrint[]): OrganizationTarget[] {
      const unique = new Map<string, OrganizationTarget>();
      for (const entry of entries) {
        for (const location of this.allLocationsOf(entry)) {
          if (!this.organizeCapable(location.sourceKey) || !this.targetOf(location.sourceKey)) {
            continue;
          }
          unique.set(`${location.sourceKey}::${location.filename}`, {
            hostId: location.sourceKey,
            filename: location.filename,
          });
        }
      }
      return [...unique.values()];
    },
    /** Find the bucket row (live or trash) a mutation should update. */
    rowOf(hostKey: string, filename: string): GalleryImage | null {
      return (
        this.bucketIndex.get(hostKey)?.byFilename.get(filename)?.item ??
        this.trashBucketIndex.get(hostKey)?.byFilename.get(filename)?.item ??
        null
      );
    },
    /**
     * Replace one row (live or trash) with a patched copy. Rows are raw,
     * immutable snapshots (see `rawRow`), so an optimistic organization edit
     * must swap the object rather than assign a field — a field write would
     * neither notify the grid nor invalidate the organization index.
     */
    patchRow(hostKey: string, filename: string, patch: Partial<GalleryImage>): GalleryImage | null {
      for (const bucket of [this.buckets[hostKey], this.trashBuckets[hostKey]]) {
        if (!bucket) continue;
        const at = bucket.items.findIndex((i) => i.filename === filename);
        if (at === -1) continue;
        const next = rawRow({ ...bucket.items[at]!, ...patch });
        bucket.items.splice(at, 1, next);
        return next;
      }
      return null;
    },
    /** Apply a server-echoed row in place of whichever bucket holds it. */
    absorbRow(hostKey: string, image: GalleryImage) {
      if (replaceRow(this.buckets[hostKey], image)) return;
      replaceRow(this.trashBuckets[hostKey], image);
    },
    /** Adjust a host's tag counts after an add/remove landed. */
    adjustTagCounts(hostKey: string, tag: string, delta: number) {
      const bucket = this.ensureTagsBucket(hostKey);
      const key = tagKey(tag);
      const at = bucket.items.findIndex((t) => tagKey(t.name) === key);
      if (at === -1) {
        if (delta > 0) bucket.items.push({ name: normalizeTagName(tag), count: delta });
        return;
      }
      const next = bucket.items[at]!.count + delta;
      if (next <= 0) bucket.items.splice(at, 1);
      else bucket.items[at]!.count = next;
    },
    /** Collection on a host by slug, creating it by name when `name` is given
     *  and none exists. */
    async ensureCollectionOn(
      hostKey: string,
      slug: string,
      name: string | null,
    ): Promise<Collection | null> {
      const bucket = this.ensureCollectionsBucket(hostKey);
      const existing = bucket.items.find((c) => (c.slug || slugOfCollection(c.name)) === slug);
      if (existing) return existing;
      if (!name) return null;
      const target = this.targetOf(hostKey);
      if (!target) throw new Error("Host is not connected.");
      const created = await createCollectionOn(target, { name });
      bucket.items.push(created);
      bucket.loaded = true;
      return created;
    },
    /** Run one per-host fan-out op, updating that host's rows optimistically. */
    async runOrganizationOp(op: OrganizationFanoutOp) {
      const target = this.targetOf(op.hostId);
      if (!target) throw new Error("Host is not connected.");
      switch (op.kind) {
        case "setTitle": {
          const bulk = galleryBulkRequest(createUuid(), op);
          if (bulk && useHostsStore().capabilities[op.hostId]?.gallery?.bulk_mutations === true) {
            await mutateGalleryBulk(target, bulk);
            for (const filename of op.filenames) {
              this.patchRow(op.hostId, filename, { title: op.title });
            }
            return;
          }
          for (const filename of op.filenames) {
            const echoed = await patchGalleryImage<GalleryImage>(target, filename, {
              title: op.title ?? "",
            });
            if (echoed) this.absorbRow(op.hostId, echoed);
            else this.patchRow(op.hostId, filename, { title: op.title });
          }
          return;
        }
        case "setFavorite": {
          await organizeGallery(target, { filenames: op.filenames, favorite: op.favorite });
          for (const filename of op.filenames) {
            this.patchRow(op.hostId, filename, { favorite: op.favorite });
          }
          return;
        }
        case "addTags": {
          const tags = op.tags.map(normalizeTagName).filter((t) => t.length > 0);
          if (tags.length === 0) return;
          await organizeGallery(target, { filenames: op.filenames, add_tags: tags });
          for (const filename of op.filenames) {
            const row = this.rowOf(op.hostId, filename);
            if (!row) continue;
            const have = new Set((row.tags ?? []).map(tagKey));
            const next = [...(row.tags ?? [])];
            for (const tag of tags) {
              if (have.has(tagKey(tag))) continue;
              have.add(tagKey(tag));
              next.push(tag);
              this.adjustTagCounts(op.hostId, tag, 1);
            }
            if (next.length !== (row.tags ?? []).length) {
              this.patchRow(op.hostId, filename, { tags: next });
            }
          }
          return;
        }
        case "removeTags": {
          const keys = new Set(op.tags.map(tagKey).filter((k) => k.length > 0));
          if (keys.size === 0) return;
          await organizeGallery(target, { filenames: op.filenames, remove_tags: op.tags });
          for (const filename of op.filenames) {
            const row = this.rowOf(op.hostId, filename);
            if (!row?.tags) continue;
            const kept: string[] = [];
            for (const tag of row.tags) {
              if (keys.has(tagKey(tag))) this.adjustTagCounts(op.hostId, tag, -1);
              else kept.push(tag);
            }
            this.patchRow(op.hostId, filename, { tags: kept });
          }
          return;
        }
        case "addToCollection": {
          const collection = await this.ensureCollectionOn(
            op.hostId,
            op.ensureCollection.slug,
            op.ensureCollection.name,
          );
          if (!collection) return;
          await organizeGallery(target, {
            filenames: op.filenames,
            add_to_collections: [collection.id],
          });
          for (const filename of op.filenames) {
            const row = this.rowOf(op.hostId, filename);
            if (!row) continue;
            if ((row.collections ?? []).includes(collection.id)) continue;
            this.patchRow(op.hostId, filename, {
              collections: [...(row.collections ?? []), collection.id],
            });
            collection.count += 1;
          }
          return;
        }
        case "removeFromCollection": {
          const collection = await this.ensureCollectionOn(op.hostId, op.slug, null);
          if (!collection) return; // this host never had it
          await organizeGallery(target, {
            filenames: op.filenames,
            remove_from_collections: [collection.id],
          });
          for (const filename of op.filenames) {
            const row = this.rowOf(op.hostId, filename);
            if (!row?.collections?.includes(collection.id)) continue;
            this.patchRow(op.hostId, filename, {
              collections: row.collections.filter((id) => id !== collection.id),
            });
            collection.count = Math.max(0, collection.count - 1);
          }
          return;
        }
        case "trash":
        case "restore":
        case "deleteForever":
          throw new Error(`${op.kind} is not an organize mutation; use the trash actions.`);
      }
    },
    async retainOrganizationOp(op: OrganizationFanoutOp, operationId = createUuid()) {
      const host = this.hostFor(op.hostId);
      if (!host || host.kind !== "remote") throw new Error("Host is not connected.");
      await enqueueGalleryMutation({
        id: operationId,
        hostId: host.id,
        hostInstanceId: host.instanceId,
        hostName: host.label,
        op,
      });
      this.pendingOrganizationMutations = (await listGalleryMutations()).length;
    },
    async flushOrganizationOutbox() {
      const queued = await listGalleryMutations();
      for (const item of queued) {
        const host = useHostsStore().all.find((candidate) => candidate.id === item.hostId);
        if (!host || host.status !== "ready") continue;
        if (item.hostInstanceId && item.hostInstanceId !== (host.instanceId ?? null)) {
          await updateGalleryMutationFailure(
            item.id,
            "Host identity changed; the retained edit was not sent.",
          );
          continue;
        }
        try {
          const target = this.targetOf(item.hostId);
          const bulk = galleryBulkRequest(item.id, item.op);
          if (
            target &&
            bulk &&
            useHostsStore().capabilities[item.hostId]?.gallery?.bulk_mutations
          ) {
            await mutateGalleryBulk(target, bulk);
          } else {
            await this.runOrganizationOp(item.op);
          }
          await removeGalleryMutation(item.id);
        } catch (error) {
          await updateGalleryMutationFailure(item.id, errorMessage(error));
        }
      }
      this.pendingOrganizationMutations = (await listGalleryMutations()).length;
    },
    /**
     * One organization mutation over every copy of the given prints: plan
     * per host (`planOrganizationFanout`), run each host's op, refetch the
     * origins that failed so the grid reconverges with the server.
     */
    async organizeMany(
      entries: MergedPrint[],
      mutation: OrganizationMutation,
    ): Promise<FanoutResult> {
      const ops = planOrganizationFanout(this.organizeTargetsFor(entries), mutation);
      const queuedHosts = new Set<string>();
      const results = await Promise.allSettled(
        ops.map(async (op) => {
          try {
            await this.runOrganizationOp(op);
          } catch (error) {
            if (isTransportFailure(error) && this.hostFor(op.hostId)?.kind === "remote") {
              await this.retainOrganizationOp(op);
              queuedHosts.add(op.hostId);
              return;
            }
            throw error;
          }
        }),
      );
      const failedHosts: string[] = [];
      let error: string | null = null;
      results.forEach((result, index) => {
        if (result.status !== "rejected") return;
        failedHosts.push(ops[index]!.hostId);
        if (error === null) error = errorMessage(result.reason);
      });
      await Promise.all(
        failedHosts.map((key) =>
          Promise.all([
            this.refreshHost(key),
            this.refreshTrash([key]),
            this.fetchCollections(key),
          ]),
        ),
      );
      return {
        applied: ops.length - failedHosts.length,
        failed: failedHosts.length,
        failedHosts,
        error:
          error ??
          (queuedHosts.size > 0
            ? `${queuedHosts.size} host edit${queuedHosts.size === 1 ? " was" : "s were"} retained and will sync when reachable.`
            : null),
      };
    },
    setTitle(entry: MergedPrint, title: string | null): Promise<FanoutResult> {
      const value = title?.trim() ?? "";
      return this.organizeMany([entry], { kind: "setTitle", title: value.length ? value : null });
    },
    setFavorite(entry: MergedPrint | MergedPrint[], value: boolean): Promise<FanoutResult> {
      const entries = Array.isArray(entry) ? entry : [entry];
      return this.organizeMany(entries, { kind: "setFavorite", favorite: value });
    },
    addTags(entry: MergedPrint | MergedPrint[], names: string[]): Promise<FanoutResult> {
      const entries = Array.isArray(entry) ? entry : [entry];
      return this.organizeMany(entries, { kind: "addTags", tags: names });
    },
    removeTags(entry: MergedPrint | MergedPrint[], names: string[]): Promise<FanoutResult> {
      const entries = Array.isArray(entry) ? entry : [entry];
      return this.organizeMany(entries, { kind: "removeTags", tags: names });
    },
    /** Add prints to a collection by slug (an existing one) or by name (a
     *  new one, created on every host that lacks the slug). */
    addToCollection(
      entries: MergedPrint | MergedPrint[],
      collection: { slug?: string; name: string },
    ): Promise<FanoutResult> {
      const list = Array.isArray(entries) ? entries : [entries];
      const mutation: OrganizationMutation = {
        kind: "addToCollection",
        name: collection.name,
        ...(collection.slug ? { slug: collection.slug } : {}),
      };
      return this.organizeMany(list, mutation);
    },
    removeFromCollection(
      entries: MergedPrint | MergedPrint[],
      slug: string,
    ): Promise<FanoutResult> {
      const list = Array.isArray(entries) ? entries : [entries];
      return this.organizeMany(list, { kind: "removeFromCollection", slug });
    },

    // ── Collections CRUD ─────────────────────────────────────────────────
    /** Hosts a collection action fans out to by default: every source that
     *  can organize right now. */
    organizeCapableKeys(): string[] {
      return this.sources
        .filter((s) => this.organizeCapable(s.key) && this.targetOf(s.key))
        .map((s) => s.key);
    },
    /** Create a collection (by name) on the given hosts — all organize-capable
     *  ones by default; hosts already holding the slug are left alone. */
    async createCollection(
      name: string,
      hostKeys?: string[],
    ): Promise<FanoutResult & { slug: string }> {
      const trimmed = name.trim();
      const slug = slugOfCollection(trimmed);
      if (!slug)
        return { slug, applied: 0, failed: 0, failedHosts: [], error: "Name a collection." };
      const keys = [...new Set(hostKeys ?? this.organizeCapableKeys())];
      const results = await Promise.allSettled(
        keys.map((key) => this.ensureCollectionOn(key, slug, trimmed)),
      );
      return { slug, ...this.settleHosts(keys, results) };
    },
    /** Rename on every host holding the slug (the slug may change with it). */
    async renameCollection(slug: string, name: string): Promise<FanoutResult> {
      const trimmed = name.trim();
      if (!trimmed) return { applied: 0, failed: 0, failedHosts: [], error: "Name a collection." };
      const merged = this.mergedCollections.find((c) => c.slug === slug);
      if (!merged)
        return { applied: 0, failed: 0, failedHosts: [], error: "Collection not found." };
      const keys = merged.hosts.map((h) => h.hostId);
      const results = await Promise.allSettled(
        merged.hosts.map(async (host) => {
          const target = this.targetOf(host.hostId);
          if (!target) throw new Error("Host is not connected.");
          const updated = await updateCollection(target, host.id, { name: trimmed });
          const bucket = this.ensureCollectionsBucket(host.hostId);
          const at = bucket.items.findIndex((c) => c.id === host.id);
          if (at === -1) bucket.items.push(updated);
          else bucket.items.splice(at, 1, updated);
        }),
      );
      const outcome = this.settleHosts(keys, results);
      if (this.collectionSlug === slug && outcome.failed === 0) {
        this.collectionSlug = slugOfCollection(trimmed) || slug;
      }
      return outcome;
    },
    /** Hide/show the collection on every host holding the slug. */
    async setCollectionHidden(slug: string, hidden: boolean): Promise<FanoutResult> {
      const merged = this.mergedCollections.find((collection) => collection.slug === slug);
      if (!merged)
        return { applied: 0, failed: 0, failedHosts: [], error: "Collection not found." };
      const keys = merged.hosts.map((host) => host.hostId);
      const results = await Promise.allSettled(
        merged.hosts.map(async (host) => {
          const target = this.targetOf(host.hostId);
          if (!target) throw new Error("Host is not connected.");
          const updated = await updateCollectionHidden(target, host.id, hidden);
          const bucket = this.ensureCollectionsBucket(host.hostId);
          const at = bucket.items.findIndex((collection) => collection.id === host.id);
          if (at === -1) bucket.items.push(updated);
          else bucket.items.splice(at, 1, updated);
        }),
      );
      return this.settleHosts(keys, results);
    },
    /** Delete the collection on every host (never its prints). */
    async deleteCollection(slug: string): Promise<FanoutResult> {
      const merged = this.mergedCollections.find((c) => c.slug === slug);
      if (!merged)
        return { applied: 0, failed: 0, failedHosts: [], error: "Collection not found." };
      const keys = merged.hosts.map((h) => h.hostId);
      const results = await Promise.allSettled(
        merged.hosts.map(async (host) => {
          const target = this.targetOf(host.hostId);
          if (!target) throw new Error("Host is not connected.");
          await deleteCollectionOn(target, host.id);
          const bucket = this.ensureCollectionsBucket(host.hostId);
          bucket.items = bucket.items.filter((c) => c.id !== host.id);
          for (const store of [this.buckets[host.hostId], this.trashBuckets[host.hostId]]) {
            if (!store) continue;
            let touched = false;
            const next = store.items.map((row) => {
              if (!row.collections?.includes(host.id)) return row;
              touched = true;
              return rawRow({
                ...row,
                collections: row.collections.filter((id) => id !== host.id),
              });
            });
            if (touched) store.items = next;
          }
        }),
      );
      const outcome = this.settleHosts(keys, results);
      if (this.collectionSlug === slug && outcome.failed === 0) this.collectionSlug = null;
      return outcome;
    },
    /** Set the cover on every host that holds both the slug and a copy. */
    async setCollectionCover(slug: string, entry: MergedPrint): Promise<FanoutResult> {
      const merged = this.mergedCollections.find((c) => c.slug === slug);
      if (!merged)
        return { applied: 0, failed: 0, failedHosts: [], error: "Collection not found." };
      const copies = this.allLocationsOf(entry);
      const plan = merged.hosts
        .map((host) => ({
          host,
          copy: copies.find((c) => c.sourceKey === host.hostId) ?? null,
        }))
        .filter(
          (p): p is { host: (typeof merged.hosts)[number]; copy: GalleryLocation } =>
            p.copy !== null,
        );
      const keys = plan.map((p) => p.host.hostId);
      const results = await Promise.allSettled(
        plan.map(async ({ host, copy }) => {
          const target = this.targetOf(host.hostId);
          if (!target) throw new Error("Host is not connected.");
          const updated = await updateCollection(target, host.id, {
            cover_filename: copy.filename,
          });
          const bucket = this.ensureCollectionsBucket(host.hostId);
          const at = bucket.items.findIndex((c) => c.id === host.id);
          if (at === -1) bucket.items.push(updated);
          else bucket.items.splice(at, 1, updated);
        }),
      );
      return this.settleHosts(keys, results);
    },
    /** Turn per-host settled promises into a `FanoutResult`, refetching the
     *  collections listing of every host that failed. */
    settleHosts(keys: string[], results: PromiseSettledResult<unknown>[]): FanoutResult {
      const failedHosts: string[] = [];
      let error: string | null = null;
      results.forEach((result, index) => {
        if (result.status !== "rejected") return;
        failedHosts.push(keys[index]!);
        if (error === null) error = errorMessage(result.reason);
      });
      for (const key of failedHosts) void this.fetchCollections(key);
      return {
        applied: keys.length - failedHosts.length,
        failed: failedHosts.length,
        failedHosts,
        error,
      };
    },

    // ── Trash ────────────────────────────────────────────────────────────
    /** Bring trashed prints back, on every host holding a trashed copy. A
     *  409 (the live filename is taken again) is reported, never retried. */
    async restore(entries: MergedPrint[]): Promise<FanoutResult & { restored: number }> {
      const byHost = new Map<string, string[]>();
      for (const entry of entries) {
        for (const location of this.trashLocationsOf(entry)) {
          const list = byHost.get(location.sourceKey) ?? [];
          if (!list.includes(location.filename)) list.push(location.filename);
          byHost.set(location.sourceKey, list);
        }
      }
      const keys = [...byHost.keys()];
      let restored = 0;
      const results = await Promise.allSettled(
        keys.map(async (key) => {
          const filenames = byHost.get(key)!;
          const target = this.targetOf(key);
          if (target) await restoreTrashed(target, filenames);
          else if (key === "local") {
            for (const filename of filenames) await ipc.localGalleryRestore(filename);
          } else throw new Error("Host is not connected.");
          const live = this.buckets[key];
          for (const filename of filenames) {
            const row = takeRow(this.trashBuckets[key], filename);
            if (row && live?.loaded) insertRow(live, { ...row, trashed_at: null, purge_at: null });
            restored += 1;
          }
        }),
      );
      const failedHosts: string[] = [];
      let error: string | null = null;
      results.forEach((result, index) => {
        if (result.status !== "rejected") return;
        failedHosts.push(keys[index]!);
        if (error === null) error = errorMessage(result.reason);
      });
      await Promise.all(
        failedHosts.map((key) => Promise.all([this.refreshHost(key), this.refreshTrash([key])])),
      );
      return {
        applied: keys.length - failedHosts.length,
        failed: failedHosts.length,
        failedHosts,
        error,
        restored,
      };
    },
    /** Hard-delete every copy (live or trashed) of the given prints. */
    async deleteForever(entries: MergedPrint[]): Promise<{
      deletedPrints: number;
      failedPrints: number;
      deletedCopies: number;
      error: string | null;
    }> {
      const groups = entries.map((entry) => this.allLocationsOf(entry));
      const unique = new Map<string, GalleryLocation>();
      for (const group of groups) {
        for (const location of group)
          unique.set(`${location.sourceKey}::${location.filename}`, location);
      }
      const locations = [...unique.values()];
      const results = await Promise.allSettled(
        locations.map((location) =>
          this.remove(location.sourceKey, location.filename, { permanent: true }),
        ),
      );
      const failedKeys = new Set<string>();
      const failedOrigins = new Set<string>();
      let error: string | null = null;
      results.forEach((result, index) => {
        if (result.status === "fulfilled") return;
        const location = locations[index]!;
        failedKeys.add(`${location.sourceKey}::${location.filename}`);
        failedOrigins.add(location.sourceKey);
        if (error === null) error = errorMessage(result.reason);
      });
      await Promise.all(
        [...failedOrigins].map((key) =>
          Promise.all([this.refreshHost(key), this.refreshTrash([key])]),
        ),
      );
      const failedPrints = groups.filter((group) =>
        group.some((location) => failedKeys.has(`${location.sourceKey}::${location.filename}`)),
      ).length;
      return {
        deletedPrints: entries.length - failedPrints,
        failedPrints,
        deletedCopies: results.length - failedKeys.size,
        error,
      };
    },
    /** Purge the trash on the given hosts (every trash-capable one by default). */
    async emptyTrash(hostKeys?: string[]): Promise<FanoutResult & { purged: number }> {
      const keys =
        hostKeys ??
        this.sources
          .filter((s) => this.trashCapable(s.key) || (s.key === "local" && !this.hostFor("local")))
          .map((s) => s.key);
      let purged = 0;
      const results = await Promise.allSettled(
        keys.map(async (key) => {
          const target = this.targetOf(key);
          const trash = this.trashBuckets[key];
          if (target) {
            const result = await emptyTrashOn(target);
            purged += result.purged;
          } else if (key === "local") {
            // Offline: the native command purges one file at a time.
            for (const row of trash?.items ?? []) {
              await ipc.localGalleryDeleteForever(row.filename);
              purged += 1;
            }
          } else throw new Error("Host is not connected.");
          for (const row of trash?.items ?? []) {
            this.evictItemMedia(key, row.filename);
            this.forgetThumbnailFor(key, row);
          }
          if (trash) trash.items = [];
        }),
      );
      const failedHosts: string[] = [];
      let error: string | null = null;
      results.forEach((result, index) => {
        if (result.status !== "rejected") return;
        failedHosts.push(keys[index]!);
        if (error === null) error = errorMessage(result.reason);
      });
      await this.refreshTrash(failedHosts);
      return {
        applied: keys.length - failedHosts.length,
        failed: failedHosts.length,
        failedHosts,
        error,
        purged,
      };
    },
    /** Purge what has passed its retention on one host, then refetch. */
    async sweepTrash(hostKey: string): Promise<{ purged: number; remaining: number }> {
      const target = this.targetOf(hostKey);
      if (!target) throw new Error("Host is not connected.");
      const result = await sweepTrashOn(target);
      await this.refreshTrash([hostKey]);
      return result;
    },
  },
});

/**
 * One bucket's rows indexed for O(1) identity lookups. `byIdentity` holds
 * every row sharing a seed:size:model identity so the caller can still apply
 * the per-pair timestamp window (`sameLogicalGalleryPrint`'s contract).
 */
export interface IndexedRow {
  item: GalleryImage;
  /** Position in the bucket, so lookups can reproduce bucket order. */
  ordinal: number;
}

export interface BucketIndex {
  byFilename: Map<string, IndexedRow>;
  byIdentity: Map<string, IndexedRow[]>;
}

export function indexBucketRows(items: readonly GalleryImage[]): BucketIndex {
  const byFilename = new Map<string, IndexedRow>();
  const byIdentity = new Map<string, IndexedRow[]>();
  items.forEach((item, ordinal) => {
    const row = { item, ordinal };
    if (!byFilename.has(item.filename)) byFilename.set(item.filename, row);
    const identity = printIdentity(item);
    if (identity === null) return;
    const twins = byIdentity.get(identity);
    if (twins) twins.push(row);
    else byIdentity.set(identity, [row]);
  });
  return { byFilename, byIdentity };
}

function indexBuckets(buckets: Record<string, GalleryBucket>): Map<string, BucketIndex> {
  const index = new Map<string, BucketIndex>();
  for (const [sourceKey, bucket] of Object.entries(buckets)) {
    index.set(sourceKey, indexBucketRows(bucket.items));
  }
  return index;
}

/**
 * Every copy of a logical print inside one bucket map — the same answer the
 * old full scan gave (`sameLogicalGalleryPrint` per row: exact filename, or
 * identity inside the timestamp window), read off the per-bucket index so a
 * grid of N tiles costs O(N), never O(N²). Matches are emitted in BUCKET
 * order (by ordinal), exactly as the scan did — cover selection takes the
 * first copy per host, so a legacy-named twin listed before the canonical
 * row must still come first.
 */
function locationsIn(index: Map<string, BucketIndex>, entry: MergedPrint): GalleryLocation[] {
  const locations: GalleryLocation[] = [];
  const identity = printIdentity(entry.item);
  for (const [sourceKey, bucket] of index) {
    const matches: IndexedRow[] = [];
    const exact = bucket.byFilename.get(entry.item.filename);
    if (exact) matches.push(exact);
    if (identity !== null) {
      for (const twin of bucket.byIdentity.get(identity) ?? []) {
        if (twin !== exact && withinIdentityWindow(twin.item, entry.item)) matches.push(twin);
      }
    }
    if (matches.length > 1) matches.sort((a, b) => a.ordinal - b.ordinal);
    for (const match of matches) locations.push({ sourceKey, filename: match.item.filename });
  }
  return locations;
}
