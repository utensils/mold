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
 * Slug logic is NOT restated here: `titleSlug`, `collectionSlug`, `slugify`,
 * and `sortTags` come from `libraryOrganization`, which is already pinned to
 * `mold_core` by its own parity fixtures. Rust remains the authority for
 * every limit and rule mirrored below.
 *
 * Tag normalization is the one deliberate exception. The Library's
 * `normalizeTagName` also strips a leading `#` — a display affordance the V3
 * tag editors chose — but `mold_core::organization::normalize_tag_name` does
 * NOT: `#blue` is the literal tag `#blue` there. A request tag is storage,
 * not display, so `normalizeRequestTag` below mirrors Rust exactly and the
 * `#` affordance is offered separately as `stripTagHash`, for a surface to
 * apply to TYPED input before calling `addTag` (never to a suggestion the
 * host reported, or picking `#blue` would file a different tag called
 * `blue`). Searching is the exception `suggestTags` owns: it strips the
 * QUERY itself, because a habit-typed `#gra` should still find `grain`, and
 * the names it returns stay verbatim.
 *
 * Framework-free and browser-safe: no Vue, no Pinia, no DOM, no shell
 * imports. Every state helper returns a NEW state object so a surface can
 * hold it in `ref()`, `reactive()`, or plain component state.
 */

import type { TagCount } from "./api/galleryOrganization";
import {
  collectionSlug,
  slugify,
  sortTags,
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

/**
 * Cap for the model component of a download name. Mirrors Rust's
 * `DOWNLOAD_MODEL_SLUG_MAX_LEN` - deliberately its OWN constant rather than
 * the collection-slug cap it happens to equal, because the reason is a
 * filesystem budget (`title(40) + model + s{20 digits} + ext` must stay under
 * 255 bytes), not a collection rule. A long `hf:` path really is cut here.
 */
export const DOWNLOAD_MODEL_SLUG_MAX_LEN = 80;

/** Stem when nothing about a print slugs to anything usable. Mirrors Rust's
 * `DOWNLOAD_FALLBACK_STEM` - unreachable there, where the seed is a `u64`;
 * reachable here, where a gallery entry can be seedless. */
export const DOWNLOAD_FALLBACK_STEM = "print";

// Exactly what Rust refuses: `is_control() && !is_whitespace()`. Tab, LF, VT,
// FF, CR, and U+0085 NEL are whitespace, so they collapse into the tag's
// inner spacing instead of failing it - refusing a pasted tab would reject a
// tag the server accepts. NEL is the reason this range is split rather than
// a flat U+007F-U+009F: it is a control character Rust admits.
const REJECTED_CONTROL_CHARS =
  /[\u0000-\u0008\u000e-\u001f\u007f-\u0084\u0086-\u009f]/;

// Whitespace runs, mirroring Rust's `split_whitespace` (the White_Space
// property). JS's `\s` covers all of it except U+0085 NEL.
const WHITESPACE_RUN = /[\s\u0085]+/g;

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
 * Display form of a request tag, mirroring
 * `mold_core::organization::normalize_tag_name`: whitespace runs (tabs and
 * newlines included) collapse to single spaces and the edges are trimmed.
 * Case is preserved — the server stores tags `COLLATE NOCASE`, so compare
 * with `requestTagKey`. A leading `#` is NOT stripped: see `stripTagHash`.
 */
export function normalizeRequestTag(raw: string): string {
  return raw.replace(WHITESPACE_RUN, " ").trim();
}

/** Case-insensitive merge key for a request tag; mirrors the `to_lowercase`
 * fold in Rust's `normalize_request_tags`. */
export function requestTagKey(raw: string): string {
  return normalizeRequestTag(raw).toLowerCase();
}

/**
 * The `#` affordance, offered as its own step: people type `#kodak` out of
 * habit, and Rust would file that as the literal tag `#kodak`. A surface
 * applies this to what the user TYPED, before `addTag` — never to a tag the
 * host reported, where stripping it would file a different tag.
 */
export function stripTagHash(raw: string): string {
  return raw.replace(/^\s*#+\s*/, "");
}

/**
 * The tags this print would be filed under: the ghost tag first unless it was
 * removed, then the manual tags in typing order — every name through
 * `normalizeRequestTag`, deduped case-insensitively (first casing wins),
 * empties dropped.
 */
export function effectiveTags(
  state: FileUnderState,
  title: string | null | undefined,
  autoTagEnabled: boolean,
): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const push = (raw: string) => {
    const name = normalizeRequestTag(raw);
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
  const name = normalizeRequestTag(raw);
  if (!name) return { ...state, manualTags: [...state.manualTags] };
  const key = requestTagKey(name);
  const exists = state.manualTags.some((tag) => requestTagKey(tag) === key);
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
  const key = requestTagKey(name);
  const ghost = deriveGhostTag(title, autoTagEnabled);
  const isGhost = ghost !== null && requestTagKey(ghost) === key;
  return {
    ...state,
    ghostRemoved: state.ghostRemoved || isGhost,
    manualTags: state.manualTags.filter((tag) => requestTagKey(tag) !== key),
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
  if (REJECTED_CONTROL_CHARS.test(name)) {
    return "Tags cannot contain control characters.";
  }
  const normalized = normalizeRequestTag(name);
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
  const key = requestTagKey(raw);
  if (active.some((tag) => requestTagKey(tag) === key)) {
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
  /**
   * The print's seed. A `number` is only used when it is exactly
   * representable — `metadata.seed` is a JSON double, so a u64 above
   * `Number.MAX_SAFE_INTEGER` has already been rounded by the time it gets
   * here and is dropped rather than written down (see `seedSegment`). A
   * digits-only `string` is passed through verbatim, which is how an
   * exact-seed wire would supply one.
   */
  seed?: number | string | null;
  /** Extension, with or without its leading dot. */
  ext?: string | null;
}

/**
 * `s{seed}`, or `null` when this side cannot state the seed EXACTLY.
 *
 * Rust never faces this: `OutputMetadata.seed` is a `u64`. The browser reads
 * that same value out of JSON as a double, so any seed above
 * `Number.MAX_SAFE_INTEGER` — which is most randomly generated ones — has
 * ALREADY been rounded before it reaches this function. Writing it into the
 * name would produce a plausible, wrong identifier: a file claiming a seed
 * that never rendered it. Omitting the segment is the honest answer and
 * yields exactly the shape an absent seed does.
 *
 * A digits-only string is trusted verbatim, so an exact-seed wire (or a
 * caller that kept the raw JSON text) can still name the file precisely.
 */
function seedSegment(seed: DownloadFileNameInput["seed"]): string | null {
  if (seed === null || seed === undefined) return null;
  if (typeof seed === "number") {
    if (!Number.isSafeInteger(seed) || seed < 0) return null;
    return `s${seed.toString()}`;
  }
  const trimmed = seed.trim();
  // Digits only: u64 seeds are never negative and never exponential.
  return /^\d+$/.test(trimmed) ? `s${trimmed}` : null;
}

/**
 * The name a Save / Download hands to the file dialog:
 * `{title-slug}__{model}__s{seed}.{ext}`, falling back to
 * `{model}__s{seed}.{ext}` when the print is untitled or its title has
 * nothing sluggable in it. Mirrors `mold-core`'s `print_title`
 * `download_file_name` — the parity table lives in `fileUnder.test.ts`.
 *
 * The model runs through the same slug algorithm at
 * `DOWNLOAD_MODEL_SLUG_MAX_LEN`, so `flux-dev:q4` and `cv:12345` cannot
 * smuggle a `:` into a filename. A segment that contributes nothing is
 * dropped rather than written as `undefined`; if nothing survives at all the
 * stem is `DOWNLOAD_FALLBACK_STEM`. The gallery filename itself never
 * changes.
 *
 * One component can legally differ from Rust, and only in what each side is
 * able to know: Rust always holds the exact `u64` seed, while a browser
 * holding `metadata.seed` as a JSON double may hold a rounded one. This
 * function omits the seed segment whenever it cannot be exact rather than
 * naming a print after a seed that never rendered it.
 */
export function downloadFileName(input: DownloadFileNameInput): string {
  const segments: string[] = [];
  const title = input.title?.trim();
  const slug = title ? titleSlug(title) : null;
  if (slug) segments.push(slug);
  const model = slugify(input.model ?? "", DOWNLOAD_MODEL_SLUG_MAX_LEN);
  if (model) segments.push(model);
  const seed = seedSegment(input.seed);
  if (seed) segments.push(seed);
  const stem =
    segments.length > 0 ? segments.join("__") : DOWNLOAD_FALLBACK_STEM;
  // Rust: `ext.trim().trim_start_matches('.').trim()`, then lowercased — the
  // second trim is what makes `". Png "` agree with `"png"`.
  const ext = (input.ext ?? "").trim().replace(/^\.+/, "").trim().toLowerCase();
  return ext ? `${stem}.${ext}` : stem;
}

// ── Suggestions ─────────────────────────────────────────────────────────────

/**
 * Autocomplete for `Add tag…`, fed by the host's own `GET /api/gallery/tags`
 * counts. Prefix matches come before substring matches, each group by count
 * descending then name (`sortTags`); tags already on the print are excluded;
 * an empty query lists everything by count so the field is useful before the
 * first keystroke.
 *
 * The QUERY — and only the query — goes through `stripTagHash` as well as the
 * tag normalization, because searching is the one place the `#` affordance is
 * free: people type `#gra` out of habit, and matching that literally hides the
 * plain `grain` they were reaching for. The host's own names are still matched
 * and RETURNED verbatim, so a literal `#grain` comes back as `#grain` (a
 * substring match on the stripped needle) and picking it files that exact tag.
 * The strip is idempotent, so a surface that already applies it before calling
 * gets the same answer.
 */
export function suggestTags<T extends TagCount>(
  existing: readonly T[],
  query: string,
  active: readonly string[],
): T[] {
  const taken = new Set(active.map((tag) => requestTagKey(tag)));
  const candidates = existing.filter(
    (tag) => !taken.has(requestTagKey(tag.name)),
  );
  const needle = requestTagKey(stripTagHash(query));
  if (!needle) return sortTags(candidates);
  const prefix: T[] = [];
  const substring: T[] = [];
  for (const tag of candidates) {
    const key = requestTagKey(tag.name);
    if (key.startsWith(needle)) prefix.push(tag);
    else if (key.includes(needle)) substring.push(tag);
  }
  return [...sortTags(prefix), ...sortTags(substring)];
}
