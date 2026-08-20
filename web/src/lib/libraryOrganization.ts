/*
 * Web-side Library organization plumbing — titles, favorites, tags,
 * collections, and the trash across every host in the registry (D1/D2).
 *
 * The arithmetic (slugs, unions, fan-out planning, retention copy) lives in
 * `@studio/lib/libraryOrganization`; this module binds it to the web's host
 * registry: it fetches each host's capabilities, collections, tags, and trash
 * listing, merges them, filters the logical grid, and executes mutations by
 * fanning out to every physical copy through the studio explicit-target
 * helpers (`x-api-key` header, never a key in a URL).
 *
 * Capability gating is per host: a host whose `/api/capabilities` lacks
 * `gallery.organize` / `gallery.trash` (older server, `MOLD_DB_DISABLE=1`,
 * or unreachable) contributes no organization and keeps hard-delete wording.
 */
import {
  createCollection,
  deleteCollection,
  deleteGalleryImageForever,
  emptyTrash,
  listCollections,
  listTags,
  organizeGallery,
  patchGalleryImage,
  restoreTrashed,
  setCollectionItems,
  trashMany,
  updateCollection,
} from "@studio/api/galleryOrganization";
import type { ApiTarget } from "@studio/api/client";
import {
  collectionSlug,
  collectionSlugResolver,
  mergeCollectionsAcrossHosts,
  planOrganizationFanout,
  sortTags,
  tagKey,
  titleSlug,
  type CollectionSlugResolver,
  type MergedCollection,
  type OrganizationMutation,
  type RetentionHost,
} from "@studio/lib/libraryOrganization";
import {
  hostApiTarget,
  hostCapabilities,
  hostGallery,
  type HostCapabilities,
} from "../components/machines/hostClient";
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";
import type { HostGalleryImage } from "./multiHostGallery";
import type { Collection, GalleryImage, TagCount } from "../types";

// ── Per-host snapshot ───────────────────────────────────────────────────────

export interface HostOrganizationSnapshot {
  hostId: string;
  hostLabel: string;
  /** `gallery.organize` — titles / favorites / tags / collections editable.
   * `null` = the `/api/capabilities` probe FAILED, so support is unknown —
   * deliberately distinguishable from an answered `organize: false`, because
   * unknown hosts stay in the mutation fan-out (codex review). */
  organize: boolean | null;
  /** `gallery.trash` — DELETE moves to the trash; `null` = permanent delete. */
  trash: { enabled: boolean; retentionDays: number } | null;
  collections: Collection[];
  tags: TagCount[];
  /** `GET /api/gallery?view=trash` rows, host-tagged. */
  trashed: HostGalleryImage[];
  /** True only when the trash listing itself was fetched successfully this
   * refresh. A failed listing degrades to `trashed: []`, and pending local
   * shadows must not be cleared on that non-evidence (codex review). */
  trashListingOk: boolean;
}

export interface OrganizationFetchers {
  capabilities: (
    host: HostEntry,
    signal?: AbortSignal,
  ) => Promise<HostCapabilities>;
  collections: (
    target: ApiTarget,
    signal?: AbortSignal,
  ) => Promise<Collection[]>;
  tags: (target: ApiTarget, signal?: AbortSignal) => Promise<TagCount[]>;
  trash: (host: HostEntry, signal?: AbortSignal) => Promise<GalleryImage[]>;
}

// Bound lazily so a partially mocked hostClient (tests) is only touched when
// a probe actually runs, never at module load.
const defaultFetchers: OrganizationFetchers = {
  capabilities: (host, signal) => hostCapabilities(host, signal),
  collections: (target, signal) => listCollections(target, signal),
  tags: (target, signal) => listTags(target, signal),
  trash: (host, signal) => hostGallery(host, signal, "trash"),
};

function emptySnapshot(host: HostEntry): HostOrganizationSnapshot {
  return {
    hostId: host.id,
    hostLabel: host.name,
    organize: null,
    trash: null,
    collections: [],
    tags: [],
    trashed: [],
    trashListingOk: false,
  };
}

async function fetchHostSnapshot(
  host: HostEntry,
  fetchers: OrganizationFetchers,
  signal?: AbortSignal,
): Promise<HostOrganizationSnapshot> {
  const snapshot = emptySnapshot(host);
  const caps = await fetchers.capabilities(host, signal).catch(() => null);
  const gallery = caps?.gallery;
  // A failed probe is UNKNOWN (null), never "answered organize: false".
  snapshot.organize = caps ? gallery?.organize === true : null;
  snapshot.trash = gallery?.trash
    ? {
        enabled: gallery.trash.enabled,
        retentionDays: gallery.trash.retention_days,
      }
    : null;
  const target = hostApiTarget(host);
  let trashListingOk = false;
  const [collections, tags, trashed] = await Promise.all([
    snapshot.organize === true
      ? fetchers.collections(target, signal).catch(() => [] as Collection[])
      : Promise.resolve([] as Collection[]),
    snapshot.organize === true
      ? fetchers.tags(target, signal).catch(() => [] as TagCount[])
      : Promise.resolve([] as TagCount[]),
    snapshot.trash?.enabled
      ? fetchers
          .trash(host, signal)
          .then((rows) => {
            trashListingOk = true;
            return rows;
          })
          .catch(() => [] as GalleryImage[])
      : Promise.resolve([] as GalleryImage[]),
  ]);
  snapshot.trashListingOk = trashListingOk;
  snapshot.collections = collections;
  snapshot.tags = tags;
  snapshot.trashed = trashed.map((item) => ({
    ...item,
    hostId: host.id,
    hostLabel: host.name,
  }));
  return snapshot;
}

/** Every host's organization state, in registry order. A host that fails
 * any probe degrades to "no organization" rather than failing the merge. */
export function fetchOrganization(
  hosts: readonly HostEntry[],
  fetchers: OrganizationFetchers = defaultFetchers,
  signal?: AbortSignal,
): Promise<HostOrganizationSnapshot[]> {
  return Promise.all(
    hosts.map((host) => fetchHostSnapshot(host, fetchers, signal)),
  );
}

// ── Merging ─────────────────────────────────────────────────────────────────

export function snapshotFor(
  snapshots: readonly HostOrganizationSnapshot[],
  hostId: string,
): HostOrganizationSnapshot | null {
  return snapshots.find((s) => s.hostId === hostId) ?? null;
}

export function anyHostOrganizes(
  snapshots: readonly HostOrganizationSnapshot[],
): boolean {
  return snapshots.some((s) => s.organize === true);
}

export function anyHostTrashes(
  snapshots: readonly HostOrganizationSnapshot[],
): boolean {
  return snapshots.some((s) => s.trash?.enabled);
}

export function hostTrashes(
  snapshots: readonly HostOrganizationSnapshot[],
  hostId: string,
): boolean {
  return snapshotFor(snapshots, hostId)?.trash?.enabled === true;
}

export function hostOrganizes(
  snapshots: readonly HostOrganizationSnapshot[],
  hostId: string,
): boolean {
  return snapshotFor(snapshots, hostId)?.organize === true;
}

export function mergedCollections(
  snapshots: readonly HostOrganizationSnapshot[],
): MergedCollection[] {
  return mergeCollectionsAcrossHosts(
    snapshots.map((s) => ({
      hostId: s.hostId,
      hostLabel: s.hostLabel,
      collections: s.collections,
    })),
  );
}

export function collectionResolver(
  snapshots: readonly HostOrganizationSnapshot[],
): CollectionSlugResolver {
  return collectionSlugResolver(
    snapshots.map((s) => ({ hostId: s.hostId, collections: s.collections })),
  );
}

/** Tags merged across hosts by case-insensitive name (first-seen casing),
 * counts summed, sorted count-desc then name. */
export function mergedTags(
  snapshots: readonly HostOrganizationSnapshot[],
): TagCount[] {
  const byKey = new Map<string, TagCount>();
  for (const snapshot of snapshots) {
    for (const tag of snapshot.tags) {
      const key = tagKey(tag.name);
      if (!key) continue;
      const existing = byKey.get(key);
      if (existing) existing.count += tag.count;
      else byKey.set(key, { name: tag.name, count: tag.count });
    }
  }
  return sortTags([...byKey.values()]);
}

/** Retention banner input: trash-capable hosts, origin first. */
export function retentionHosts(
  snapshots: readonly HostOrganizationSnapshot[],
): RetentionHost[] {
  const capable = snapshots.filter((s) => s.trash?.enabled);
  capable.sort((a, b) =>
    a.hostId === ORIGIN_HOST_ID ? -1 : b.hostId === ORIGIN_HOST_ID ? 1 : 0,
  );
  return capable.map((s) => ({
    label: s.hostLabel,
    retentionDays: s.trash?.retentionDays ?? 0,
  }));
}

// ── Filtering ───────────────────────────────────────────────────────────────

export interface OrganizationFilter {
  favoritesOnly?: boolean;
  /** Every active tag must be present (AND), case-insensitive. */
  tags?: readonly string[];
  /** Collection slug the print must belong to. */
  collectionSlug?: string | null;
}

function entryTagKeys(entry: GalleryImage): Set<string> {
  return new Set((entry.tags ?? []).map((tag) => tagKey(tag)));
}

function entryCollectionSlugs(entry: HostGalleryImage): readonly string[] {
  return entry.organization?.collections ?? [];
}

export function entryMatchesOrganization(
  entry: HostGalleryImage,
  filter: OrganizationFilter,
): boolean {
  if (filter.favoritesOnly && !entry.favorite) return false;
  if (filter.tags && filter.tags.length > 0) {
    const keys = entryTagKeys(entry);
    for (const tag of filter.tags) {
      if (!keys.has(tagKey(tag))) return false;
    }
  }
  if (filter.collectionSlug) {
    if (!entryCollectionSlugs(entry).includes(filter.collectionSlug))
      return false;
  }
  return true;
}

export function filterByOrganization<T extends HostGalleryImage>(
  entries: readonly T[],
  filter: OrganizationFilter,
): T[] {
  return entries.filter((entry) => entryMatchesOrganization(entry, filter));
}

/** Search over filename, model, prompt, title, and tags. */
export function entryMatchesSearch(
  entry: GalleryImage,
  query: string,
): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (entry.filename.toLowerCase().includes(q)) return true;
  if (entry.metadata.model.toLowerCase().includes(q)) return true;
  if (entry.metadata.prompt?.toLowerCase().includes(q)) return true;
  const title = entry.title ?? entry.metadata.title;
  if (title && title.toLowerCase().includes(q)) return true;
  return (entry.tags ?? []).some((tag) => tag.toLowerCase().includes(q));
}

// ── Collection cards ────────────────────────────────────────────────────────

export interface CollectionCard {
  slug: string;
  name: string;
  /** Logical prints in the collection (counted from the merged grid). */
  count: number;
  hostLabels: string[];
  /** Latest `updated_at` across hosts (unix secs), `null` when unknown. */
  updatedAt: number | null;
  /** Up to four prints for the cover mosaic — the explicit cover first. */
  covers: HostGalleryImage[];
  merged: MergedCollection;
}

export function collectionCards(
  merged: readonly MergedCollection[],
  entries: readonly HostGalleryImage[],
  snapshots: readonly HostOrganizationSnapshot[],
  rawEntries: readonly HostGalleryImage[] = entries,
): CollectionCard[] {
  return merged.map((collection) => {
    const members = entries.filter((entry) =>
      entryCollectionSlugs(entry).includes(collection.slug),
    );
    const hostLabels = collection.hosts.map(
      (host) => snapshotFor(snapshots, host.hostId)?.hostLabel ?? host.hostId,
    );
    let updatedAt: number | null = null;
    for (const host of collection.hosts) {
      const row = snapshotFor(snapshots, host.hostId)?.collections.find(
        (c) => c.id === host.id,
      );
      if (row && (updatedAt === null || row.updated_at > updatedAt))
        updatedAt = row.updated_at;
    }
    const covers: HostGalleryImage[] = [];
    if (collection.cover) {
      const cover =
        rawEntries.find(
          (entry) =>
            entry.hostId === collection.cover?.hostId &&
            entry.filename === collection.cover.filename,
        ) ?? null;
      if (cover) covers.push(cover);
    }
    for (const member of members) {
      if (covers.length >= 4) break;
      if (
        !covers.some(
          (c) => c.hostId === member.hostId && c.filename === member.filename,
        )
      )
        covers.push(member);
    }
    return {
      slug: collection.slug,
      name: collection.name,
      count: members.length,
      hostLabels,
      updatedAt,
      covers,
      merged: collection,
    };
  });
}

// ── Misc helpers ────────────────────────────────────────────────────────────

/** Suggested download filename: the title's slug when one exists. */
export function downloadFilename(
  title: string | null | undefined,
  filename: string,
): string {
  const slug = title ? titleSlug(title) : null;
  if (!slug) return filename;
  const dot = filename.lastIndexOf(".");
  return dot > 0 ? `${slug}${filename.slice(dot)}` : slug;
}

// ── Mutations (fan-out) ─────────────────────────────────────────────────────

export interface FanoutFailure {
  hostId: string;
  error: string;
}

export interface FanoutResult {
  /** Hosts whose op succeeded. */
  ok: string[];
  failed: FanoutFailure[];
}

export type HostLookup = (hostId: string) => HostEntry | null | undefined;

function errMsg(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function fanout<T extends { hostId: string }>(
  ops: readonly T[],
  hostById: HostLookup,
  run: (op: T, host: HostEntry, target: ApiTarget) => Promise<void>,
): Promise<FanoutResult> {
  const result: FanoutResult = { ok: [], failed: [] };
  await Promise.all(
    ops.map(async (op) => {
      const host = hostById(op.hostId);
      if (!host) {
        result.failed.push({
          hostId: op.hostId,
          error: "That host isn't connected anymore.",
        });
        return;
      }
      try {
        await run(op, host, hostApiTarget(host));
        result.ok.push(op.hostId);
      } catch (error) {
        result.failed.push({ hostId: op.hostId, error: errMsg(error) });
      }
    }),
  );
  return result;
}

export interface MutationContext {
  hostById: HostLookup;
  snapshots: readonly HostOrganizationSnapshot[];
}

/** Resolve a host's collection id for a slug from the current snapshots. */
export function hostCollectionId(
  snapshots: readonly HostOrganizationSnapshot[],
  hostId: string,
  slug: string,
): string | null {
  const snapshot = snapshotFor(snapshots, hostId);
  if (!snapshot) return null;
  const match = snapshot.collections.find(
    (c) => (c.slug || collectionSlug(c.name)) === slug,
  );
  return match?.id ?? null;
}

/**
 * Apply one organization mutation to every physical copy of a logical print
 * (or of many), one request per host. Collection adds create the collection
 * by name on a host that lacks it (D7); collection removes skip hosts that
 * never had it.
 */
export async function applyOrganizationMutation(
  copies: readonly HostGalleryImage[],
  mutation: OrganizationMutation,
  context: MutationContext,
): Promise<FanoutResult> {
  // A host whose snapshot ANSWERED `gallery.organize: false` (older build,
  // MOLD_DB_DISABLE) contributes no organization state and would 404/501
  // every edit — skip its copies instead of surfacing partial failures for
  // prints that are defined as unorganizable there (codex review). A host
  // with no snapshot, or whose capability probe failed (`organize: null`),
  // stays in: unknown is not incapable, and its edits must either land or
  // surface as that host's own failure — never silently skip (codex review).
  const incapable = new Set(
    context.snapshots.filter((s) => s.organize === false).map((s) => s.hostId),
  );
  const ops = planOrganizationFanout(
    copies
      .filter((copy) => !incapable.has(copy.hostId))
      .map((copy) => ({ hostId: copy.hostId, filename: copy.filename })),
    mutation,
  );
  return fanout(ops, context.hostById, async (op, _host, target) => {
    switch (op.kind) {
      case "setTitle":
        await Promise.all(
          op.filenames.map((filename) =>
            patchGalleryImage(target, filename, { title: op.title ?? "" }),
          ),
        );
        return;
      case "setFavorite":
        await organizeGallery(target, {
          filenames: op.filenames,
          favorite: op.favorite,
        });
        return;
      case "addTags":
        await organizeGallery(target, {
          filenames: op.filenames,
          add_tags: op.tags,
        });
        return;
      case "removeTags":
        await organizeGallery(target, {
          filenames: op.filenames,
          remove_tags: op.tags,
        });
        return;
      case "addToCollection": {
        let id = hostCollectionId(
          context.snapshots,
          op.hostId,
          op.ensureCollection.slug,
        );
        if (!id) {
          const created = await createCollection(target, {
            name: op.ensureCollection.name,
          });
          id = created.id;
        }
        await setCollectionItems(target, id, {
          add: op.filenames,
          remove: [],
        });
        return;
      }
      case "removeFromCollection": {
        const id = hostCollectionId(context.snapshots, op.hostId, op.slug);
        if (!id) return;
        await setCollectionItems(target, id, {
          add: [],
          remove: op.filenames,
        });
        return;
      }
      case "trash":
        await trashMany(target, op.filenames);
        return;
      case "restore":
        await restoreTrashed(target, op.filenames);
        return;
      case "deleteForever":
        await Promise.all(
          op.filenames.map((filename) =>
            deleteGalleryImageForever(target, filename),
          ),
        );
        return;
    }
  });
}

/** Create a collection on one host (the primary by default). */
export function createCollectionOn(
  host: HostEntry,
  name: string,
): Promise<Collection> {
  return createCollection(hostApiTarget(host), { name });
}

/** Rename every host's copy of a merged collection. */
export function renameCollectionEverywhere(
  collection: MergedCollection,
  name: string,
  hostById: HostLookup,
): Promise<FanoutResult> {
  return fanout(collection.hosts, hostById, (host, _entry, target) =>
    updateCollection(target, host.id, { name }).then(() => undefined),
  );
}

/** Delete every host's copy of a merged collection — never its prints. */
export function deleteCollectionEverywhere(
  collection: MergedCollection,
  hostById: HostLookup,
): Promise<FanoutResult> {
  return fanout(collection.hosts, hostById, (host, _entry, target) =>
    deleteCollection(target, host.id),
  );
}

/** Set the cover on the host that holds that copy. */
export function setCollectionCover(
  collection: MergedCollection,
  cover: { hostId: string; filename: string },
  hostById: HostLookup,
): Promise<FanoutResult> {
  const ops = collection.hosts.filter((host) => host.hostId === cover.hostId);
  return fanout(ops, hostById, (host, _entry, target) =>
    updateCollection(target, host.id, {
      cover_filename: cover.filename,
    }).then(() => undefined),
  );
}

/**
 * Purge the trash on every trash-capable host. Deliberately not limited to
 * hosts whose snapshot already lists trashed rows: a print committed to the
 * trash inside the last poll interval exists only as a pending shadow entry,
 * and skipping its host would make Empty trash silently do nothing while the
 * button showed a positive count (codex review). `DELETE /api/gallery/trash`
 * on an already-empty host is a cheap no-op.
 */
export function emptyTrashEverywhere(
  snapshots: readonly HostOrganizationSnapshot[],
  hostById: HostLookup,
): Promise<FanoutResult> {
  const ops = snapshots.filter((s) => s.trash?.enabled);
  return fanout(ops, hostById, (_op, _host, target) =>
    emptyTrash(target).then(() => undefined),
  );
}
