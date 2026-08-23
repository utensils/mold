/**
 * iPhone-side Library organization state helpers — scopes, chip filters,
 * cross-host collection/tag merges, per-host capability gating, and the
 * mutation fan-out the Library tab and viewer info sheet run.
 *
 * Pure functions over the shared studio contracts
 * (`@studio/lib/libraryOrganization`, `@studio/api/galleryOrganization`), so
 * `MobileApp.vue` stays a thin orchestrator and every rule here is unit
 * testable without mounting the shell.
 */

import type { ApiTarget } from "@studio/api/client";
import {
  createCollection,
  deleteGalleryImageForever,
  deleteManyForever,
  organizeGallery,
  patchGalleryImage,
  restoreTrashed,
  setCollectionItems,
  trashGalleryImage,
  trashMany,
} from "@studio/api/galleryOrganization";
import type {
  Collection,
  GalleryCapabilitiesWire,
  GalleryOrganizationFields,
  TagCount,
} from "@studio/lib/api/galleryOrganization";
import {
  groupLogicalGalleryPrints,
  type GalleryPrintIdentityInput,
} from "@studio/lib/galleryPrintIdentity";
import {
  collectionSlug,
  mergeCollectionsAcrossHosts,
  purgeCountdownFromPurgeAt,
  sortTags,
  tagKey,
  unionOrganization,
  validatePrintTitle,
  type CollectionSlugResolver,
  type MergedCollection,
  type OrganizationFanoutOp,
  type OrganizationUnion,
  type RetentionHost,
} from "@studio/lib/libraryOrganization";
import type { GalleryImage, OutputMetadata, ServerCapabilities } from "../lib/api/types";

// ── Types ───────────────────────────────────────────────────────────────────

/** A gallery entry as a current host serves it — the desktop wire type plus
 * the additive organization fields older hosts omit. */
export type MobileGalleryImage = GalleryImage & GalleryOrganizationFields;

export type MobileLibraryScope = "prints" | "collections" | "trash";

export const MOBILE_LIBRARY_SCOPES: readonly MobileLibraryScope[] = [
  "prints",
  "collections",
  "trash",
];

export const MOBILE_LIBRARY_SCOPE_LABELS: Record<MobileLibraryScope, string> = {
  prints: "Prints",
  collections: "Collections",
  trash: "Trash",
};

/** The title slot on `OutputMetadata` is additive; older desktop types omit it. */
export type MobileOutputMetadata = OutputMetadata & { title?: string | null };

export interface MobileLibraryHost {
  id: string;
  name: string;
}

export interface MobileLibraryFilters {
  favoritesOnly: boolean;
  /** `tagKey` of the active tag chip; null = every tag. */
  tag: string | null;
  hostId: string | null;
  /** Collection slug for the Collections drill-in; null outside it. */
  collectionSlug: string | null;
}

export const EMPTY_LIBRARY_FILTERS: MobileLibraryFilters = {
  favoritesOnly: false,
  tag: null,
  hostId: null,
  collectionSlug: null,
};

// ── Capability gating ───────────────────────────────────────────────────────

export interface MobileLibrarySupport {
  /** Hosts advertising `capabilities.gallery.organize`. */
  organizeHostIds: Set<string>;
  /** Hosts advertising `capabilities.gallery.trash.enabled`. */
  trashHostIds: Set<string>;
  /** Per-host `trash.retention_days` (0 = forever). */
  retentionDays: Record<string, number>;
  /** Any connected host supports titles / favorites / tags / collections. */
  organize: boolean;
  /** Any connected host supports the trash. */
  trash: boolean;
}

export function galleryCapabilitiesOf(
  capabilities: ServerCapabilities | null | undefined,
): GalleryCapabilitiesWire | null {
  const gallery = (capabilities as { gallery?: unknown } | null | undefined)?.gallery;
  if (!gallery || typeof gallery !== "object") return null;
  return gallery as GalleryCapabilitiesWire;
}

/** Gate every organization affordance on what the connected hosts advertise.
 * A host whose capabilities have not been read supports nothing yet. */
export function libraryOrganizationSupport(
  hosts: readonly MobileLibraryHost[],
  capabilities: Record<string, ServerCapabilities | null | undefined>,
): MobileLibrarySupport {
  const organizeHostIds = new Set<string>();
  const trashHostIds = new Set<string>();
  const retentionDays: Record<string, number> = {};
  for (const host of hosts) {
    const gallery = galleryCapabilitiesOf(capabilities[host.id]);
    if (!gallery) continue;
    if (gallery.organize === true) organizeHostIds.add(host.id);
    if (gallery.trash?.enabled === true) {
      trashHostIds.add(host.id);
      const days = gallery.trash.retention_days;
      retentionDays[host.id] = Number.isFinite(days) && days > 0 ? Math.floor(days) : 0;
    }
  }
  return {
    organizeHostIds,
    trashHostIds,
    retentionDays,
    organize: organizeHostIds.size > 0,
    trash: trashHostIds.size > 0,
  };
}

/** Which delete a Select-mode action performs: the trash only when EVERY
 * host holding a selected copy can trash; otherwise today's hard delete. */
export function selectionDeleteKind(
  hostIds: Iterable<string>,
  support: Pick<MobileLibrarySupport, "trashHostIds">,
): "trash" | "delete" {
  let any = false;
  for (const hostId of hostIds) {
    any = true;
    if (!support.trashHostIds.has(hostId)) return "delete";
  }
  return any ? "trash" : "delete";
}

export interface DeleteActionCopy {
  /** Status text in the action bar. */
  status: string;
  /** Label of the danger button. */
  button: string;
}

/** Two-tap wording for the Select-mode destructive action. */
export function deleteActionCopy(
  kind: "trash" | "delete" | "delete-forever",
  count: number,
  confirming: boolean,
  busy = false,
): DeleteActionCopy {
  if (busy) {
    return {
      status: `${count} selected`,
      button: kind === "trash" ? "Moving…" : kind === "delete-forever" ? "Deleting…" : "Deleting…",
    };
  }
  if (!confirming) {
    return {
      status: `${count} selected`,
      button: kind === "trash" ? "Trash" : kind === "delete-forever" ? "Delete forever" : "Delete",
    };
  }
  return {
    status:
      kind === "trash"
        ? `Move ${count} to trash?`
        : kind === "delete-forever"
          ? `Delete ${count} forever?`
          : `Delete ${count} everywhere?`,
    button: "Confirm",
  };
}

// ── Organization index ──────────────────────────────────────────────────────

export interface OrganizationCopyLike extends GalleryPrintIdentityInput, GalleryOrganizationFields {
  hostId: string;
  filename: string;
}

export const printOrganizationKey = (print: { hostId: string; filename: string }): string =>
  `${print.hostId}|${print.filename}`;

/**
 * One `OrganizationUnion` per physical copy key, computed over the logical
 * print group each copy belongs to — so the representative tile, the viewer,
 * and every sibling copy read the same merged title / ♥ / tags / collections.
 */
export function buildOrganizationIndex<T extends OrganizationCopyLike>(
  copies: readonly T[],
  resolveCollectionSlug: CollectionSlugResolver,
  localHostId: string | null = null,
): Map<string, OrganizationUnion> {
  const index = new Map<string, OrganizationUnion>();
  for (const group of groupLogicalGalleryPrints(copies)) {
    const union = unionOrganization(
      group.copies.map((copy) => ({ hostId: copy.hostId, item: copy })),
      { localHostId, resolveCollectionSlug },
    );
    for (const copy of group.copies) index.set(printOrganizationKey(copy), union);
  }
  return index;
}

/** Every physical copy in the same logical group as `print`. */
export function logicalCopiesOf<T extends OrganizationCopyLike>(
  copies: readonly T[],
  print: { hostId: string; filename: string },
): T[] {
  const key = printOrganizationKey(print);
  for (const group of groupLogicalGalleryPrints(copies)) {
    if (group.copies.some((copy) => printOrganizationKey(copy) === key)) return group.copies;
  }
  const exact = copies.find((copy) => printOrganizationKey(copy) === key);
  return exact ? [exact] : [];
}

/** Index every physical key to its logical group in one gallery pass. */
export function logicalCopyIndex<T extends OrganizationCopyLike>(
  copies: readonly T[],
): Map<string, readonly T[]> {
  const index = new Map<string, readonly T[]>();
  for (const group of groupLogicalGalleryPrints(copies)) {
    for (const copy of group.copies) index.set(printOrganizationKey(copy), group.copies);
  }
  return index;
}

// ── Filtering ───────────────────────────────────────────────────────────────

/** Client-side chip/scope filter over the logical representatives. */
export function filterLibraryPrints<T extends { hostId: string; filename: string }>(
  prints: readonly T[],
  filters: MobileLibraryFilters,
  organizationOf: (print: T) => OrganizationUnion | undefined,
  copiesOf: (print: T) => readonly { hostId: string }[] = (print) => [print],
  hiddenCollectionSlugs: ReadonlySet<string> = new Set(),
): T[] {
  return prints.filter((print) => {
    if (filters.hostId && !copiesOf(print).some((copy) => copy.hostId === filters.hostId)) {
      return false;
    }
    if (
      !filters.favoritesOnly &&
      !filters.tag &&
      !filters.collectionSlug &&
      hiddenCollectionSlugs.size === 0
    )
      return true;
    const organization = organizationOf(print);
    if (
      hiddenCollectionSlugs.size > 0 &&
      (organization?.collections ?? []).some((slug) => hiddenCollectionSlugs.has(slug))
    ) {
      return false;
    }
    if (filters.favoritesOnly && !organization?.favorite) return false;
    if (filters.tag && !(organization?.tags ?? []).some((tag) => tagKey(tag) === filters.tag)) {
      return false;
    }
    if (
      filters.collectionSlug &&
      !(organization?.collections ?? []).includes(filters.collectionSlug)
    ) {
      return false;
    }
    return true;
  });
}

// ── Tags across hosts ───────────────────────────────────────────────────────

/** Case-insensitive union of every host's tag counts (counts summed — an
 * upper bound when a print is mirrored, same caveat as collections). Pass
 * `hostIds` to scope the merge to the currently connected hosts so a
 * disconnected or forgotten machine's retained bucket leaves no ghost chips. */
export function mergeHostTags(
  perHost: Record<string, readonly TagCount[] | undefined>,
  hostIds?: readonly string[],
): TagCount[] {
  const buckets = hostIds ? hostIds.map((id) => perHost[id]) : Object.values(perHost);
  const byKey = new Map<string, TagCount>();
  for (const tags of buckets) {
    for (const tag of tags ?? []) {
      const key = tagKey(tag.name);
      if (!key) continue;
      const merged = byKey.get(key);
      if (merged) merged.count += tag.count;
      else byKey.set(key, { name: tag.name, count: tag.count });
    }
  }
  return sortTags([...byKey.values()]);
}

export interface TagChipPlan {
  visible: TagCount[];
  overflow: TagCount[];
}

export const TAG_CHIP_LIMIT = 8;

/** Top `limit` tags ride the chip row; the rest live behind "More…". The
 * active tag is always visible so the filter it names is never hidden. */
export function tagChipPlan(
  tags: readonly TagCount[],
  activeKey: string | null,
  limit = TAG_CHIP_LIMIT,
): TagChipPlan {
  const visible = tags.slice(0, limit);
  const overflow = tags.slice(limit);
  if (activeKey) {
    const index = overflow.findIndex((tag) => tagKey(tag.name) === activeKey);
    if (index >= 0) {
      const [active] = overflow.splice(index, 1);
      if (active) {
        const bumped = visible.pop();
        visible.push(active);
        if (bumped) overflow.unshift(bumped);
      }
    }
  }
  return { visible, overflow };
}

// ── Collections across hosts ────────────────────────────────────────────────

export interface MobileCollectionCard {
  slug: string;
  name: string;
  /** Sum of per-host counts (upper bound for mirrored prints). */
  count: number;
  hostIds: string[];
  /** "This Mac · plato" style label from the host names. */
  hostsLabel: string;
  cover: { hostId: string; filename: string } | null;
  hidden: boolean;
}

export function collectionCards(
  merged: readonly MergedCollection[],
  hostNames: Record<string, string>,
): MobileCollectionCard[] {
  return merged.map((collection) => {
    const hostIds = collection.hosts.map((host) => host.hostId);
    return {
      slug: collection.slug,
      name: collection.name,
      count: collection.count,
      hostIds,
      hostsLabel: hostIds.map((id) => hostNames[id] ?? id).join(" · "),
      cover: collection.cover,
      hidden: collection.hidden === true,
    };
  });
}

export function mergedCollectionsFor(
  perHost: Record<string, readonly Collection[] | undefined>,
  hosts: readonly MobileLibraryHost[],
): MergedCollection[] {
  return mergeCollectionsAcrossHosts(
    hosts.map((host) => ({
      hostId: host.id,
      hostLabel: host.name,
      collections: perHost[host.id] ?? [],
    })),
  );
}

/** Per-host collection for a merged slug (used by Rename / Delete fan-out). */
export function collectionOnHost(
  perHost: Record<string, readonly Collection[] | undefined>,
  hostId: string,
  slug: string,
): Collection | undefined {
  return (perHost[hostId] ?? []).find(
    (entry) => (entry.slug || collectionSlug(entry.name)) === slug,
  );
}

export interface CollectionNameValidation {
  ok: boolean;
  value: string;
  reason?: string;
}

export function validateCollectionName(raw: string): CollectionNameValidation {
  const value = raw.replace(/\s+/g, " ").trim();
  if (!value) return { ok: false, value, reason: "Name the collection first." };
  if (!collectionSlug(value)) {
    return { ok: false, value, reason: "Use at least one letter or number in the name." };
  }
  if (value.length > 80) return { ok: false, value, reason: "Names are at most 80 characters." };
  return { ok: true, value };
}

// ── Trash ───────────────────────────────────────────────────────────────────

/** Hosts that can trash, in Library order, for the retention banner. */
export function trashRetentionHosts(
  hosts: readonly MobileLibraryHost[],
  support: Pick<MobileLibrarySupport, "trashHostIds" | "retentionDays">,
): RetentionHost[] {
  return hosts
    .filter((host) => support.trashHostIds.has(host.id))
    .map((host) => ({ label: host.name, retentionDays: support.retentionDays[host.id] ?? 0 }));
}

export interface TrashSnapshotInput<T extends { hostId: string; timestamp: number }> {
  /** The previous merged trash listing (last good per-host reads). */
  previous: readonly T[];
  /** Copies read this pass. */
  refreshed: readonly T[];
  /** Hosts whose listing was actually read this pass. */
  refreshedHostIds: ReadonlySet<string>;
  /** Every currently connected trash-capable host. */
  trashCapableHostIds: ReadonlySet<string>;
  /** Hosts whose read rejected. */
  rejectedHosts: number;
  /** Trash-capable hosts skipped before the fetch (known offline). */
  skippedHosts: number;
}

export interface TrashSnapshotOutcome<T extends { hostId: string; timestamp: number }> {
  copies: T[];
  /** True only when every trash-capable host was read this pass. A partial
   * snapshot must never be authoritative — the scope stays retry-eligible. */
  complete: boolean;
  /** Rejected + skipped hosts, for the "N hosts unavailable" disclosure. */
  failedHosts: number;
}

/** Merge one trash refresh pass over the previous snapshot: refreshed hosts
 * are replaced, unread trash-capable hosts keep their prior copies, and
 * hosts that are no longer connected/trash-capable are dropped. */
export function mergeTrashSnapshot<T extends { hostId: string; timestamp: number }>(
  input: TrashSnapshotInput<T>,
): TrashSnapshotOutcome<T> {
  const retained = input.previous.filter(
    (copy) =>
      input.trashCapableHostIds.has(copy.hostId) && !input.refreshedHostIds.has(copy.hostId),
  );
  const failedHosts = input.rejectedHosts + input.skippedHosts;
  return {
    copies: [...retained, ...input.refreshed].sort((a, b) => b.timestamp - a.timestamp),
    complete: failedHosts === 0,
    failedHosts,
  };
}

/** Tile chip for a trashed print; null when the host keeps trash forever. */
export function purgeChipLabel(
  purgeAtSecs: number | null | undefined,
  nowMs: number,
): string | null {
  const countdown = purgeCountdownFromPurgeAt(purgeAtSecs, nowMs);
  return countdown.kind === "kept" ? null : countdown.label;
}

// ── Titles ──────────────────────────────────────────────────────────────────

/** The `title` a mobile-built `GenerateRequest` carries: validated, trimmed,
 * absent when blank. An invalid title is reported, never silently dropped. */
export function requestTitle(
  raw: string,
): { ok: true; title: string | null } | { ok: false; reason: string } {
  const result = validatePrintTitle(raw);
  return result.ok ? { ok: true, title: result.value } : result;
}

/** Title to restore into the Create form from a print's saved metadata. */
export function reusedPrintTitle(metadata: MobileOutputMetadata | null | undefined): string {
  const title = metadata?.title;
  return typeof title === "string" ? title.trim() : "";
}

// ── Fan-out execution ───────────────────────────────────────────────────────

export interface FanoutHost {
  id: string;
  name: string;
  target: ApiTarget;
  collections: readonly Collection[];
}

export interface FanoutFailure {
  hostId: string;
  hostName: string;
  error: unknown;
}

export interface FanoutResult {
  failures: FanoutFailure[];
  /** Collections created on the way (so the caller can refresh listings). */
  createdCollections: Array<{ hostId: string; collection: Collection }>;
  /** Host ids whose ops all succeeded. */
  succeededHostIds: string[];
}

export interface FanoutApi {
  patchGalleryImage: typeof patchGalleryImage;
  organizeGallery: typeof organizeGallery;
  createCollection: typeof createCollection;
  setCollectionItems: typeof setCollectionItems;
  trashMany: typeof trashMany;
  restoreTrashed: typeof restoreTrashed;
  deleteGalleryImageForever: typeof deleteGalleryImageForever;
  deleteManyForever?: typeof deleteManyForever;
  /** Hard delete for hosts without a trash (today's `DELETE`). */
  deleteGalleryImage: (target: ApiTarget, filename: string) => Promise<void>;
}

export const defaultFanoutApi: FanoutApi = {
  patchGalleryImage,
  organizeGallery,
  createCollection,
  setCollectionItems,
  trashMany,
  restoreTrashed,
  deleteGalleryImageForever,
  deleteManyForever,
  // A host without a trash hard-deletes on the plain `DELETE` it has always
  // answered; `?permanent=true` is never sent to a host that lacks the trash.
  deleteGalleryImage: (target, filename) => trashGalleryImage(target, filename),
};

/**
 * Run one planned fan-out (`planOrganizationFanout`) against the exact
 * Keychain-authenticated target of every host. Hosts are independent: one
 * failing host never blocks another, and every failure is returned so the
 * caller can name it inline.
 */
export async function runOrganizationFanout(
  ops: readonly OrganizationFanoutOp[],
  hosts: Record<string, FanoutHost | undefined>,
  api: FanoutApi = defaultFanoutApi,
  options: { trashHostIds?: ReadonlySet<string>; bulkHostIds?: ReadonlySet<string> } = {},
): Promise<FanoutResult> {
  const result: FanoutResult = { failures: [], createdCollections: [], succeededHostIds: [] };
  await Promise.all(
    ops.map(async (op) => {
      const host = hosts[op.hostId];
      if (!host) {
        result.failures.push({
          hostId: op.hostId,
          hostName: op.hostId,
          error: new Error("This host is no longer connected."),
        });
        return;
      }
      try {
        await runHostOp(op, host, api, result, options.trashHostIds, options.bulkHostIds);
        result.succeededHostIds.push(host.id);
      } catch (error) {
        result.failures.push({ hostId: host.id, hostName: host.name, error });
      }
    }),
  );
  return result;
}

async function runHostOp(
  op: OrganizationFanoutOp,
  host: FanoutHost,
  api: FanoutApi,
  result: FanoutResult,
  trashHostIds: ReadonlySet<string> | undefined,
  bulkHostIds: ReadonlySet<string> | undefined,
): Promise<void> {
  switch (op.kind) {
    case "setTitle":
      for (const filename of op.filenames) {
        await api.patchGalleryImage(host.target, filename, { title: op.title ?? "" });
      }
      return;
    case "setFavorite":
      await api.organizeGallery(host.target, { filenames: op.filenames, favorite: op.favorite });
      return;
    case "addTags":
      await api.organizeGallery(host.target, { filenames: op.filenames, add_tags: op.tags });
      return;
    case "removeTags":
      await api.organizeGallery(host.target, { filenames: op.filenames, remove_tags: op.tags });
      return;
    case "addToCollection": {
      let collection = host.collections.find(
        (entry) => (entry.slug || collectionSlug(entry.name)) === op.ensureCollection.slug,
      );
      if (!collection) {
        collection = await api.createCollection(host.target, { name: op.ensureCollection.name });
        result.createdCollections.push({ hostId: host.id, collection });
      }
      await api.setCollectionItems(host.target, collection.id, { add: op.filenames, remove: [] });
      return;
    }
    case "removeFromCollection": {
      const collection = host.collections.find(
        (entry) => (entry.slug || collectionSlug(entry.name)) === op.slug,
      );
      if (!collection) return;
      await api.setCollectionItems(host.target, collection.id, { add: [], remove: op.filenames });
      return;
    }
    case "trash":
      if (trashHostIds && !trashHostIds.has(host.id)) {
        for (const filename of op.filenames) await api.deleteGalleryImage(host.target, filename);
      } else {
        await api.trashMany(host.target, op.filenames);
      }
      return;
    case "restore":
      await api.restoreTrashed(host.target, op.filenames);
      return;
    case "deleteForever":
      if (bulkHostIds?.has(host.id) && api.deleteManyForever) {
        await api.deleteManyForever(host.target, op.filenames);
      } else {
        for (const filename of op.filenames) {
          await api.deleteGalleryImageForever(host.target, filename);
        }
      }
      return;
  }
}

/** Inline banner copy for a partially failed fan-out. */
export function fanoutFailureMessage(
  action: string,
  failures: readonly FanoutFailure[],
  describe: (error: unknown, hostName: string) => string,
): string {
  if (failures.length === 0) return "";
  const named = failures.map((failure) => failure.hostName);
  const hosts = named.length === 1 ? named[0] : `${named.length} hosts (${named.join(", ")})`;
  const first = failures[0]!;
  return `Couldn’t ${action} on ${hosts}. ${describe(first.error, first.hostName)}`;
}
