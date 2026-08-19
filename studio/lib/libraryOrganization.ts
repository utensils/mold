/**
 * Shared Library organization logic — titles, favorites, tags, collections,
 * and the trash — for web, desktop, and iPhone.
 *
 * Organization state lives per host in that host's `mold.db`; the browser
 * merges it across hosts exactly like the gallery (D1/D2): collections join
 * by slug, tags by case-insensitive name, and every mutation fans out to every
 * physical copy of a logical print. This module owns that arithmetic so no
 * surface grows its own policy. Browser-safe: no DOM, no shell imports.
 */

import type {
  Collection,
  GalleryOrganizationFields,
  TagCount,
} from "./api/galleryOrganization";

// ── Slugs ───────────────────────────────────────────────────────────────────

/**
 * Slug algorithm shared with `mold_core::title_slug` (Rust). Keep the two in
 * lock-step — `libraryOrganization.test.ts` carries the parity fixture table.
 *
 * 1. ASCII-lowercase (`A-Z` → `a-z`; non-ASCII letters are NOT folded).
 * 2. Every char outside `[a-z0-9]` becomes `-`.
 * 3. Collapse runs of `-`; trim leading and trailing `-`.
 * 4. Cut to `maxLen` chars, then trim any trailing `-` the cut exposed.
 * 5. Empty ⇒ `""` (callers map that to `null` / `None`).
 */
function slugify(input: string, maxLen: number): string {
  const lowered = input.replace(/[A-Z]/g, (char) => char.toLowerCase());
  let slug = lowered.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (slug.length > maxLen) {
    slug = slug.slice(0, maxLen).replace(/-+$/g, "");
  }
  return slug;
}

export const TITLE_SLUG_MAX_LEN = 40;
export const COLLECTION_SLUG_MAX_LEN = 80;

/** Filename slug for a print title (`mold-{model}-{ts}~{slug}.{ext}`).
 * Mirrors `mold_core::title_slug`; `null` when nothing survives. */
export function titleSlug(title: string): string | null {
  const slug = slugify(title, TITLE_SLUG_MAX_LEN);
  return slug.length === 0 ? null : slug;
}

/** Cross-host merge key for a collection name. Same algorithm as
 * `titleSlug` with an 80-char cap; `""` when nothing survives. */
export function collectionSlug(name: string): string {
  return slugify(name, COLLECTION_SLUG_MAX_LEN);
}

// ── Tags ────────────────────────────────────────────────────────────────────

// Control characters (C0 + DEL); never legal in a title or tag.
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;
const CONTROL_CHARS_ALL = /[\u0000-\u001f\u007f]/g;

/** Display form of a tag as typed: trimmed, inner whitespace collapsed, a
 * leading `#` dropped, control characters removed. Case is preserved —
 * the server stores tags `COLLATE NOCASE`, so compare with `tagKey`. */
export function normalizeTagName(raw: string): string {
  return raw
    .replace(CONTROL_CHARS_ALL, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^#+\s*/, "")
    .trim();
}

/** Case-insensitive merge key for a tag (tags merge across hosts by it). */
export function tagKey(name: string): string {
  return normalizeTagName(name).toLowerCase();
}

// ── Titles ──────────────────────────────────────────────────────────────────

export const PRINT_TITLE_MAX_LEN = 120;

export type PrintTitleValidation =
  { ok: true; value: string | null } | { ok: false; reason: string };

/** Mirrors `mold_core::validate_print_title`: trims, rejects control
 * characters, caps at 120 chars; empty means "no title". */
export function validatePrintTitle(raw: string): PrintTitleValidation {
  const value = raw.trim();
  if (value.length === 0) return { ok: true, value: null };
  if (CONTROL_CHARS.test(value)) {
    return { ok: false, reason: "Titles cannot contain control characters." };
  }
  if (value.length > PRINT_TITLE_MAX_LEN) {
    return {
      ok: false,
      reason: `Titles are at most ${PRINT_TITLE_MAX_LEN} characters.`,
    };
  }
  return { ok: true, value };
}

export interface DisplayTitleInput {
  title?: string | null;
  metadata?: { prompt?: string | null } | null;
  filename: string;
}

function truncateLabel(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, Math.max(1, maxLen - 1));
  const lastSpace = cut.search(/\s+\S*$/);
  const trimmed =
    lastSpace > maxLen / 2 ? cut.slice(0, lastSpace) : cut.trimEnd();
  return `${trimmed}…`;
}

/** Filename without its extension (the `~slug` suffix is kept: it is the
 * only hint a title ever existed for legacy rows). */
export function filenameStem(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot > 0 ? filename.slice(0, dot) : filename;
}

/** What the Library shows for a print: title ?? prompt excerpt ?? filename
 * stem. The raw filename is demoted to a detail line everywhere (D5). */
export function displayTitle(entry: DisplayTitleInput, maxLen = 48): string {
  const title = entry.title?.trim();
  if (title) return truncateLabel(title, maxLen);
  const prompt = entry.metadata?.prompt?.replace(/\s+/g, " ").trim();
  if (prompt) return truncateLabel(prompt, maxLen);
  return filenameStem(entry.filename);
}

// ── Collections across hosts ────────────────────────────────────────────────

export interface HostCollections {
  hostId: string;
  hostLabel?: string;
  collections: readonly Collection[];
}

export interface MergedCollectionHost {
  hostId: string;
  /** That host's `Collection.id`. */
  id: string;
  count: number;
}

export interface MergedCollection {
  slug: string;
  /** First-seen display name. */
  name: string;
  /**
   * Sum of the per-host counts — an UPPER BOUND on the number of logical
   * prints, since the same print mirrored on two hosts counts twice and
   * nothing here can join copies. Render it as "N prints" only when a
   * single host holds the collection; otherwise prefer the grouped count
   * the Library computes from its own logical-print groups.
   */
  count: number;
  hosts: MergedCollectionHost[];
  /** First non-null cover across hosts, in input order. */
  cover: { hostId: string; filename: string } | null;
}

function resolvedCollectionSlug(entry: Collection): string {
  return entry.slug && entry.slug.length > 0
    ? entry.slug
    : collectionSlug(entry.name);
}

/** Merge every host's collections by slug. Returns the shelf sorted by name
 * (`sortCollections`). */
export function mergeCollectionsAcrossHosts(
  perHost: readonly HostCollections[],
): MergedCollection[] {
  const bySlug = new Map<string, MergedCollection>();
  for (const host of perHost) {
    for (const entry of host.collections) {
      const slug = resolvedCollectionSlug(entry);
      if (!slug) continue;
      let merged = bySlug.get(slug);
      if (!merged) {
        merged = { slug, name: entry.name, count: 0, hosts: [], cover: null };
        bySlug.set(slug, merged);
      }
      merged.count += entry.count;
      merged.hosts.push({
        hostId: host.hostId,
        id: entry.id,
        count: entry.count,
      });
      if (!merged.cover && entry.cover_filename) {
        merged.cover = { hostId: host.hostId, filename: entry.cover_filename };
      }
    }
  }
  return sortCollections([...bySlug.values()]);
}

export type CollectionSlugResolver = (
  hostId: string,
  collectionId: string,
) => string | null | undefined;

/** Build the `(hostId, collectionId) → slug` resolver `unionOrganization`
 * needs from the per-host collection listings. */
export function collectionSlugResolver(
  perHost: readonly Pick<HostCollections, "hostId" | "collections">[],
): CollectionSlugResolver {
  const map = new Map<string, string>();
  for (const host of perHost) {
    for (const entry of host.collections) {
      const slug = resolvedCollectionSlug(entry);
      if (slug) map.set(`${host.hostId} ${entry.id}`, slug);
    }
  }
  return (hostId, collectionId) => map.get(`${hostId} ${collectionId}`);
}

// ── Union across copies ─────────────────────────────────────────────────────

export interface OrganizationCopy<
  T extends GalleryOrganizationFields = GalleryOrganizationFields,
> {
  hostId: string;
  item: T;
}

export interface OrganizationUnion {
  /** Local copy's title if set, else the first non-empty title. */
  title: string | null;
  /** True when ANY copy is a favorite. */
  favorite: boolean;
  /** Case-insensitive union, first-seen casing, sorted for display. */
  tags: string[];
  /** Collection slugs (sorted) every copy's memberships resolve to. */
  collections: string[];
  /** Earliest trash stamp (unix secs) across copies; null when live. */
  trashedAt: number | null;
  /** Earliest purge stamp (unix secs); null when kept forever / live. */
  purgeAt: number | null;
  /** Memberships whose collection id the resolver did not know — the host's
   * collection listing has not loaded, or it is stale. Never silently
   * dropped into `collections`. */
  unresolvedCollectionIds: Array<{ hostId: string; id: string }>;
}

export interface UnionOrganizationOptions {
  localHostId?: string | null;
  resolveCollectionSlug: CollectionSlugResolver;
}

function minStamp(
  a: number | null,
  b: number | null | undefined,
): number | null {
  if (b == null || !Number.isFinite(b)) return a;
  return a === null ? b : Math.min(a, b);
}

/** Read one logical print's organization across every physical copy. */
export function unionOrganization(
  copies: readonly OrganizationCopy[],
  options: UnionOrganizationOptions,
): OrganizationUnion {
  let title: string | null = null;
  const local = options.localHostId
    ? copies.find((copy) => copy.hostId === options.localHostId)
    : undefined;
  const localTitle = local?.item.title?.trim();
  if (localTitle) title = localTitle;
  else {
    for (const copy of copies) {
      const candidate = copy.item.title?.trim();
      if (candidate) {
        title = candidate;
        break;
      }
    }
  }

  let favorite = false;
  const tagsByKey = new Map<string, string>();
  const slugs = new Set<string>();
  const unresolvedCollectionIds: Array<{ hostId: string; id: string }> = [];
  let trashedAt: number | null = null;
  let purgeAt: number | null = null;

  for (const copy of copies) {
    if (copy.item.favorite) favorite = true;
    for (const raw of copy.item.tags ?? []) {
      const name = normalizeTagName(raw);
      if (!name) continue;
      const key = name.toLowerCase();
      if (!tagsByKey.has(key)) tagsByKey.set(key, name);
    }
    for (const id of copy.item.collections ?? []) {
      const slug = options.resolveCollectionSlug(copy.hostId, id);
      if (slug) slugs.add(slug);
      else unresolvedCollectionIds.push({ hostId: copy.hostId, id });
    }
    trashedAt = minStamp(trashedAt, copy.item.trashed_at);
    purgeAt = minStamp(purgeAt, copy.item.purge_at);
  }

  return {
    title,
    favorite,
    tags: [...tagsByKey.values()].sort(compareNames),
    collections: [...slugs].sort(compareNames),
    trashedAt,
    purgeAt,
    unresolvedCollectionIds,
  };
}

// ── Fan-out planning ────────────────────────────────────────────────────────

export interface OrganizationTarget {
  hostId: string;
  filename: string;
}

export type OrganizationMutation =
  | { kind: "setTitle"; title: string | null }
  | { kind: "setFavorite"; favorite: boolean }
  | { kind: "addTags"; tags: string[] }
  | { kind: "removeTags"; tags: string[] }
  | { kind: "addToCollection"; name: string; slug?: string }
  | { kind: "removeFromCollection"; slug: string }
  | { kind: "trash" }
  | { kind: "restore" }
  | { kind: "deleteForever" };

interface FanoutBase {
  hostId: string;
  /** Distinct filenames on that host, first-seen order. */
  filenames: string[];
}

export type OrganizationFanoutOp = FanoutBase &
  (
    | { kind: "setTitle"; title: string | null }
    | { kind: "setFavorite"; favorite: boolean }
    | { kind: "addTags"; tags: string[] }
    | { kind: "removeTags"; tags: string[] }
    | {
        kind: "addToCollection";
        /** Create this collection by name on the host when it lacks one
         * with this slug, then add `filenames` to it. */
        ensureCollection: { name: string; slug: string };
      }
    | { kind: "removeFromCollection"; slug: string }
    | { kind: "trash" }
    | { kind: "restore" }
    | { kind: "deleteForever" }
  );

/** Turn one mutation over many physical copies into one op per host. */
export function planOrganizationFanout(
  copies: readonly OrganizationTarget[],
  mutation: OrganizationMutation,
): OrganizationFanoutOp[] {
  const byHost = new Map<string, string[]>();
  for (const copy of copies) {
    const list = byHost.get(copy.hostId) ?? [];
    if (!list.includes(copy.filename)) list.push(copy.filename);
    byHost.set(copy.hostId, list);
  }
  const ops: OrganizationFanoutOp[] = [];
  for (const [hostId, filenames] of byHost) {
    const base: FanoutBase = { hostId, filenames };
    switch (mutation.kind) {
      case "setTitle":
        ops.push({ ...base, kind: "setTitle", title: mutation.title });
        break;
      case "setFavorite":
        ops.push({ ...base, kind: "setFavorite", favorite: mutation.favorite });
        break;
      case "addTags":
        ops.push({ ...base, kind: "addTags", tags: [...mutation.tags] });
        break;
      case "removeTags":
        ops.push({ ...base, kind: "removeTags", tags: [...mutation.tags] });
        break;
      case "addToCollection":
        ops.push({
          ...base,
          kind: "addToCollection",
          ensureCollection: {
            name: mutation.name,
            slug: mutation.slug ?? collectionSlug(mutation.name),
          },
        });
        break;
      case "removeFromCollection":
        ops.push({
          ...base,
          kind: "removeFromCollection",
          slug: mutation.slug,
        });
        break;
      case "trash":
      case "restore":
      case "deleteForever":
        ops.push({ ...base, kind: mutation.kind });
        break;
    }
  }
  return ops;
}

// ── Trash retention ─────────────────────────────────────────────────────────

/** Settings choices for `gallery.trash_retention_days`; `0` = forever. */
export const RETENTION_OPTIONS: readonly number[] = [1, 7, 30, 90, 365, 0];

const DAY_SECS = 86_400;

export function retentionLabel(days: number): string {
  if (!Number.isFinite(days) || days <= 0) return "Forever";
  if (days % 365 === 0) {
    const years = days / 365;
    return years === 1 ? "1 year" : `${years} years`;
  }
  return days === 1 ? "1 day" : `${days} days`;
}

export type PurgeCountdown =
  | { kind: "purges"; days: number; label: string }
  | { kind: "today"; label: string }
  | { kind: "kept"; label: string };

const KEPT: PurgeCountdown = {
  kind: "kept",
  label: "Kept until you empty the trash",
};

/** Countdown from the server's own `purge_at` (unix secs). */
export function purgeCountdownFromPurgeAt(
  purgeAtSecs: number | null | undefined,
  nowMs: number,
): PurgeCountdown {
  if (purgeAtSecs == null || !Number.isFinite(purgeAtSecs)) return KEPT;
  const remainingMs = purgeAtSecs * 1000 - nowMs;
  const days = Math.ceil(remainingMs / (DAY_SECS * 1000));
  if (days <= 0) return { kind: "today", label: "Purges today" };
  return { kind: "purges", days, label: `Purges in ${days} d` };
}

/** Countdown derived from the trash stamp and the host's retention. */
export function purgeCountdown(
  trashedAtSecs: number | null | undefined,
  retentionDays: number,
  nowMs: number,
): PurgeCountdown {
  if (trashedAtSecs == null || !Number.isFinite(trashedAtSecs)) return KEPT;
  if (!Number.isFinite(retentionDays) || retentionDays <= 0) return KEPT;
  return purgeCountdownFromPurgeAt(
    trashedAtSecs + retentionDays * DAY_SECS,
    nowMs,
  );
}

export interface RetentionHost {
  label: string;
  retentionDays: number;
}

export interface RetentionSummarySegment {
  text: string;
  /** Render in the mono face — it is a number with a unit. */
  mono: boolean;
}

export interface RetentionSummary {
  text: string;
  segments: RetentionSummarySegment[];
}

function retentionDaysMono(days: number): string {
  return `${days} d`;
}

/**
 * Banner copy for the Trash view. The first host is the primary (This
 * device / origin) and sets the sentence; hosts that differ are named:
 * "Prints stay in the trash 30 d before purge · plato keeps 7 d".
 */
export function trashRetentionSummary(
  hosts: readonly RetentionHost[],
): RetentionSummary {
  if (hosts.length === 0) return { text: "", segments: [] };
  const [primary, ...rest] = hosts as [RetentionHost, ...RetentionHost[]];
  const segments: RetentionSummarySegment[] = [];
  const primaryDays = normalizeRetention(primary.retentionDays);
  if (primaryDays === 0) {
    segments.push({
      text: "Prints stay in the trash until you empty it",
      mono: false,
    });
  } else {
    segments.push(
      { text: "Prints stay in the trash ", mono: false },
      { text: retentionDaysMono(primaryDays), mono: true },
      { text: " before purge", mono: false },
    );
  }
  for (const host of rest) {
    const days = normalizeRetention(host.retentionDays);
    if (days === primaryDays) continue;
    if (days === 0) {
      segments.push({
        text: ` · ${host.label} keeps trash forever`,
        mono: false,
      });
    } else {
      segments.push(
        { text: ` · ${host.label} keeps `, mono: false },
        { text: retentionDaysMono(days), mono: true },
      );
    }
  }
  return { text: segments.map((segment) => segment.text).join(""), segments };
}

function normalizeRetention(days: number): number {
  return Number.isFinite(days) && days > 0 ? Math.floor(days) : 0;
}

// ── Sorting ─────────────────────────────────────────────────────────────────

function compareNames(a: string, b: string): number {
  return (
    a.localeCompare(b, undefined, { sensitivity: "base" }) ||
    (a < b ? -1 : a > b ? 1 : 0)
  );
}

/** Name ascending, case-insensitive; returns a new array. */
export function sortCollections<T extends { name: string }>(
  list: readonly T[],
): T[] {
  return [...list].sort((a, b) => compareNames(a.name, b.name));
}

/** Count descending, then name ascending; returns a new array. */
export function sortTags<T extends TagCount>(list: readonly T[]): T[] {
  return [...list].sort(
    (a, b) => b.count - a.count || compareNames(a.name, b.name),
  );
}
