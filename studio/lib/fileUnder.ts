/**
 * "File under" — Create-time Library organization, shared by web, desktop,
 * and iPhone.
 *
 * The Create form grows one capability-gated group with two rows:
 *
 * - **Tags** — a dashed, removable GHOST chip derived from the print title's
 *   slug, then the real chips the user typed, then an `Add tag…` field with
 *   suggestions from the host's own tag counts.
 * - **Collection** — auto pre-selected (visible and clearable, NEVER
 *   auto-created) when the title's slug equals an existing collection's slug,
 *   otherwise whatever the user picked.
 *
 * At submit both materialize into the additive wire fields
 * `GenerateRequest.tags?: string[]` and
 * `GenerateRequest.collection?: { id?, name? }`. Clients always send
 * `{ name }` and let the server get-or-create by slug, so one request works
 * against every host in a fleet without the browser learning host-local
 * collection ids.
 *
 * Slug and tag normalization are NOT restated here: `titleSlug`,
 * `collectionSlug`, `normalizeTagName`, `tagKey`, and `sortTags` come from
 * `libraryOrganization`, which is already pinned to `mold_core` by its own
 * parity fixtures. Rust remains the authority for every limit mirrored below.
 *
 * Framework-free and browser-safe: no Vue, no Pinia, no DOM, no shell
 * imports. Every state helper returns a NEW state object so a surface can
 * hold it in `ref()`, `reactive()`, or plain component state.
 */

import type { TagCount } from "./api/galleryOrganization";
import {
  collectionSlug,
  normalizeTagName,
  sortTags,
  tagKey,
  titleSlug,
} from "./libraryOrganization";

// ── Limits (mirrors of the Rust request validation) ─────────────────────────

/** Most tags one `GenerateRequest` may carry. Mirrors Rust's
 * `MAX_REQUEST_TAGS`; the server rejects a longer list outright. */
export const MAX_REQUEST_TAGS = 20;

/** Longest tag name a request may carry, in Unicode scalar values (Rust
 * counts `.chars()`, so 64 emoji are 64 characters, not 128). */
export const REQUEST_TAG_MAX_LEN = 64;

/** Web's `localStorage` key for the "tag prints with their title" toggle.
 * Desktop keeps the preference in its own settings store and iPhone in
 * `mold.mobile.settings.v1`, so neither needs a key of its own. */
export const AUTO_TAG_SETTING_WEB = "mold.create.autoTagTitle.v1";

// Control characters (C0 + DEL). Rejected rather than stripped so a pasted
// tag is reported instead of silently altered.
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

// ── State ───────────────────────────────────────────────────────────────────

/** An explicit collection choice. `id` is the host-local `Collection.id`
 * when the user picked an existing row; a brand-new name carries none. */
export interface FileUnderCollectionPick {
  id?: string;
  name: string;
}

/**
 * The group's draft state. Deliberately NOT the resolved answer: the ghost
 * tag and the title match are derived from the live title every time, so a
 * title edit moves them without any watcher.
 */
export interface FileUnderState {
  /** The user removed the dashed ghost chip; it stays gone for this draft. */
  ghostRemoved: boolean;
  /** Tags the user typed, in the order they typed them (display case). */
  manualTags: string[];
  /** The explicit collection choice, when there is one. */
  picked: FileUnderCollectionPick | null;
  /** True once the user touched the row — picking OR clearing. An explicit
   * pick always beats the title match. */
  pickedExplicitly: boolean;
  /**
   * The title slug whose auto-match the user cleared. This is what makes a
   * clear STICK: the title keeps re-deriving the same slug on every
   * keystroke, so a boolean would be re-offered immediately. A genuinely
   * different slug is a new offer and re-matches.
   */
  clearedMatchSlug: string | null;
}

/** A fresh draft — also what ⌘N / "new print" resets to. */
export function emptyFileUnderState(): FileUnderState {
  return {
    ghostRemoved: false,
    manualTags: [],
    picked: null,
    pickedExplicitly: false,
    clearedMatchSlug: null,
  };
}

// ── Ghost tag ───────────────────────────────────────────────────────────────

/**
 * The auto-derived tag shown as a dashed removable chip: the print title's
 * slug. `null` when auto-tagging is off, the print is untitled, or the title
 * has nothing sluggable in it (`"日本語"`, `"!!!"`).
 */
export function deriveGhostTag(
  title: string | null | undefined,
  autoTagEnabled: boolean,
): string | null {
  if (!autoTagEnabled) return null;
  const raw = title?.trim();
  if (!raw) return null;
  return titleSlug(raw);
}

// ── Collection matching ─────────────────────────────────────────────────────

/** The shape `matchCollection` needs: anything with a name, optionally the
 * server's own slug and id (`Collection`, `MergedCollection`, …). */
export interface FileUnderCollectionLike {
  id?: string;
  name: string;
  slug?: string | null;
}

function resolvedSlug(entry: FileUnderCollectionLike): string {
  const slug = entry.slug?.trim();
  return slug && slug.length > 0 ? slug : collectionSlug(entry.name);
}

/**
 * The collection whose slug equals the title's slug, or `null`. Pure: the
 * caller decides how to render it (the Collection row shows the name plus
 * "· matched to title") and NOTHING here creates a collection.
 */
export function matchCollection<T extends FileUnderCollectionLike>(
  title: string | null | undefined,
  collections: readonly T[],
): T | null {
  const slug = title ? titleSlug(title.trim()) : null;
  if (!slug) return null;
  for (const entry of collections) {
    if (resolvedSlug(entry) === slug) return entry;
  }
  return null;
}

// ── Tag helpers ─────────────────────────────────────────────────────────────

/**
 * The tags this print would be filed under: the ghost tag first unless it was
 * removed, then the manual tags in typing order — every name through
 * `normalizeTagName`, deduped case-insensitively (first casing wins), empties
 * dropped.
 */
export function effectiveTags(
  state: FileUnderState,
  title: string | null | undefined,
  autoTagEnabled: boolean,
): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const push = (raw: string) => {
    const name = normalizeTagName(raw);
    if (!name) return;
    const key = name.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(name);
  };
  const ghost = state.ghostRemoved
    ? null
    : deriveGhostTag(title, autoTagEnabled);
  if (ghost) push(ghost);
  for (const tag of state.manualTags) push(tag);
  return out;
}

/** Add a typed tag. Blank and case-insensitively duplicate names are
 * ignored, so the caller can wire this straight to Enter. */
export function addTag(state: FileUnderState, raw: string): FileUnderState {
  const name = normalizeTagName(raw);
  if (!name) return { ...state, manualTags: [...state.manualTags] };
  const key = tagKey(name);
  const exists = state.manualTags.some((tag) => tagKey(tag) === key);
  return {
    ...state,
    manualTags: exists ? [...state.manualTags] : [...state.manualTags, name],
  };
}

/**
 * Remove one chip. Removing the ghost chip retires the ghost (and any manual
 * tag identical to it, so it does not reappear in its place); removing any
 * other chip drops it from the manual list. Case-insensitive throughout.
 */
export function removeTag(
  state: FileUnderState,
  name: string,
  title: string | null | undefined,
  autoTagEnabled: boolean,
): FileUnderState {
  const key = tagKey(name);
  const ghost = deriveGhostTag(title, autoTagEnabled);
  const isGhost = ghost !== null && tagKey(ghost) === key;
  return {
    ...state,
    ghostRemoved: state.ghostRemoved || isGhost,
    manualTags: state.manualTags.filter((tag) => tagKey(tag) !== key),
  };
}

/** Undo a ghost removal (the chip is offered again for this draft). */
export function restoreGhostTag(state: FileUnderState): FileUnderState {
  return { ...state, ghostRemoved: false, manualTags: [...state.manualTags] };
}

// ── Collection row ──────────────────────────────────────────────────────────

/** What the Collection row is showing. `source` drives the copy: a `title`
 * choice is the auto pre-selection ("· matched to title"). */
export interface EffectiveCollection {
  /** Host-local id, only when the choice came from an existing row. */
  id?: string;
  name: string;
  /** Cross-host merge key for the name. */
  slug: string;
  source: "picked" | "title";
}

/** Record an explicit choice — it outranks the title match from here on. */
export function pickCollection(
  state: FileUnderState,
  pick: FileUnderCollectionPick,
): FileUnderState {
  return {
    ...state,
    manualTags: [...state.manualTags],
    picked: pick,
    pickedExplicitly: true,
  };
}

/**
 * Clear the row. The current title's slug is remembered so the auto-match
 * does not immediately re-offer itself while the user keeps editing a title
 * that slugs the same; a different slug is a new offer.
 */
export function clearCollection(
  state: FileUnderState,
  title: string | null | undefined,
): FileUnderState {
  const slug = title ? titleSlug(title.trim()) : null;
  return {
    ...state,
    manualTags: [...state.manualTags],
    picked: null,
    pickedExplicitly: false,
    clearedMatchSlug: slug ?? state.clearedMatchSlug,
  };
}

/**
 * The collection this print would be filed under: an explicit pick wins,
 * otherwise the title match unless the user cleared that slug. `null` when
 * the row is empty — which is also "create nothing".
 */
export function effectiveCollection<T extends FileUnderCollectionLike>(
  state: FileUnderState,
  title: string | null | undefined,
  collections: readonly T[],
): EffectiveCollection | null {
  const picked = state.pickedExplicitly ? state.picked : null;
  if (picked) {
    const name = picked.name.trim();
    if (name) {
      return {
        ...(picked.id !== undefined ? { id: picked.id } : {}),
        name,
        slug: collectionSlug(name),
        source: "picked",
      };
    }
  }
  const slug = title ? titleSlug(title.trim()) : null;
  if (!slug || state.clearedMatchSlug === slug) return null;
  const match = matchCollection(title, collections);
  if (!match) return null;
  return {
    ...(match.id !== undefined ? { id: match.id } : {}),
    name: match.name,
    slug: resolvedSlug(match),
    source: "title",
  };
}

// ── Wire fields ─────────────────────────────────────────────────────────────

/** The additive slice of `GenerateRequest` this group owns. Both fields are
 * ABSENT — never `[]`, never `null` — when nothing is filed. */
export interface FileUnderRequestFields {
  tags?: string[];
  /** Clients send the name only; the server get-or-creates by slug, so one
   * request files correctly on any host in the fleet. */
  collection?: { name: string };
}

/**
 * Materialize the group into request fields. Tags the server would reject are
 * dropped and the list is clamped to `MAX_REQUEST_TAGS` — the input gate
 * (`validateNewTag`) already prevents both, this is the last-resort fence so
 * one bad chip can never fail the whole generation.
 */
export function buildFileUnderRequestFields<T extends FileUnderCollectionLike>(
  state: FileUnderState,
  title: string | null | undefined,
  autoTagEnabled: boolean,
  collections: readonly T[],
): FileUnderRequestFields {
  const fields: FileUnderRequestFields = {};
  const tags = effectiveTags(state, title, autoTagEnabled)
    .filter((tag) => validateRequestTag(tag) === null)
    .slice(0, MAX_REQUEST_TAGS);
  if (tags.length > 0) fields.tags = tags;
  const collection = effectiveCollection(state, title, collections);
  if (collection) fields.collection = { name: collection.name };
  return fields;
}

// ── Capability gate ─────────────────────────────────────────────────────────

/**
 * Whether the host can file prints at all — read from its
 * `/api/capabilities.gallery.organize`. POSITIVE knowledge only: an older
 * server, `MOLD_DB_DISABLE=1`, and a capability snapshot that has not been
 * read yet all answer `false`, and the whole group hides — exactly how the V3
 * Library gates its organization controls.
 *
 * The parameter is `unknown` on purpose: every surface has its own
 * `ServerCapabilities` shape, and reading defensively (as the mobile
 * `galleryCapabilitiesOf` already does) beats forcing them to converge.
 */
export function fileUnderAvailable(capabilities: unknown): boolean {
  const gallery = (capabilities as { gallery?: unknown } | null | undefined)
    ?.gallery;
  if (!gallery || typeof gallery !== "object") return false;
  return (gallery as { organize?: unknown }).organize === true;
}

// ── Validation (Rust is the authority) ──────────────────────────────────────

/**
 * Mirror of the Rust request-tag validation: control characters rejected,
 * 1..=`REQUEST_TAG_MAX_LEN` characters once normalized. Returns the message
 * to show, or `null` when the tag is fine. Reject at the input — never at
 * submit, where the user has already lost the context.
 */
export function validateRequestTag(name: string): string | null {
  if (CONTROL_CHARS.test(name)) {
    return "Tags cannot contain control characters.";
  }
  const normalized = normalizeTagName(name);
  if (normalized.length === 0) return "Enter a tag name.";
  // Count Unicode scalar values like Rust's `.chars().count()`.
  if (Array.from(normalized).length > REQUEST_TAG_MAX_LEN) {
    return `Tags are at most ${REQUEST_TAG_MAX_LEN} characters.`;
  }
  return null;
}

/**
 * What the `Add tag…` field checks before accepting an entry: the per-tag
 * rules plus this print's own context — a case-insensitive duplicate of a
 * chip already on the row, and the per-request cap.
 */
export function validateNewTag(
  raw: string,
  active: readonly string[],
): string | null {
  const perTag = validateRequestTag(raw);
  if (perTag) return perTag;
  const key = tagKey(raw);
  if (active.some((tag) => tagKey(tag) === key)) {
    return "That tag is already on this print.";
  }
  if (active.length >= MAX_REQUEST_TAGS) {
    return `A print can carry at most ${MAX_REQUEST_TAGS} tags.`;
  }
  return null;
}

// ── Download name ───────────────────────────────────────────────────────────

export interface DownloadFileNameInput {
  /** The print's title; absent / unsluggable drops the segment. */
  title?: string | null;
  /** Resolved model id — `flux-dev`, `flux-dev:q4`, `cv:12345`. */
  model: string;
  /** u64 seeds exceed `Number.MAX_SAFE_INTEGER`, so a string is accepted. */
  seed?: number | bigint | string | null;
  /** Extension, with or without its leading dot. */
  ext?: string | null;
}

function seedSegment(seed: DownloadFileNameInput["seed"]): string | null {
  if (seed === null || seed === undefined) return null;
  if (typeof seed === "bigint") return `s${seed.toString()}`;
  if (typeof seed === "number") {
    if (!Number.isFinite(seed)) return null;
    return `s${Math.trunc(seed).toString()}`;
  }
  const trimmed = seed.trim();
  return /^-?\d+$/.test(trimmed) ? `s${trimmed}` : null;
}

/**
 * The name a Save / Download hands to the file dialog:
 * `{title-slug}__{model}__s{seed}.{ext}`, falling back to
 * `{model}__s{seed}.{ext}` when the print is untitled or its title has
 * nothing sluggable in it. Mirrors `mold-core`'s `print_title`
 * `download_file_name` — the parity table lives in `fileUnder.test.ts`.
 *
 * The model runs through the same slug algorithm at the 80-character
 * collection cap (compact ids run past 40), so `flux-dev:q4` and `cv:12345`
 * cannot smuggle a `:` into a filename. A segment that slugs to nothing is
 * dropped rather than written as `undefined`; if nothing survives at all the
 * stem is `print`. The gallery filename itself never changes.
 */
export function downloadFileName(input: DownloadFileNameInput): string {
  const segments: string[] = [];
  const title = input.title?.trim();
  const slug = title ? titleSlug(title) : null;
  if (slug) segments.push(slug);
  const model = collectionSlug(input.model ?? "");
  if (model) segments.push(model);
  const seed = seedSegment(input.seed);
  if (seed) segments.push(seed);
  const stem = segments.length > 0 ? segments.join("__") : "print";
  const ext = (input.ext ?? "").trim().replace(/^\.+/, "").toLowerCase();
  return ext ? `${stem}.${ext}` : stem;
}

// ── Suggestions ─────────────────────────────────────────────────────────────

/**
 * Autocomplete for `Add tag…`, fed by the host's own `GET /api/gallery/tags`
 * counts. Prefix matches come before substring matches, each group by count
 * descending then name (`sortTags`); tags already on the print are excluded;
 * an empty query lists everything by count so the field is useful before the
 * first keystroke. The query is normalized like a tag, so a typed `#` and
 * stray whitespace still match.
 */
export function suggestTags<T extends TagCount>(
  existing: readonly T[],
  query: string,
  active: readonly string[],
): T[] {
  const taken = new Set(active.map((tag) => tagKey(tag)));
  const candidates = existing.filter((tag) => !taken.has(tagKey(tag.name)));
  const needle = tagKey(query);
  if (!needle) return sortTags(candidates);
  const prefix: T[] = [];
  const substring: T[] = [];
  for (const tag of candidates) {
    const key = tagKey(tag.name);
    if (key.startsWith(needle)) prefix.push(tag);
    else if (key.includes(needle)) substring.push(tag);
  }
  return [...sortTags(prefix), ...sortTags(substring)];
}
