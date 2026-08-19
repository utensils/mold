/**
 * Library organization wire types shared by every surface — titles,
 * favorites, tags, collections, and the trash.
 *
 * These mirror the additive Rust types in `mold-core` (`types.rs`) exactly:
 * `GalleryImage`'s organization fields, `Collection`, `TagCount`,
 * `GalleryTrashCapabilities`, the request/result bodies of the
 * `/api/gallery/*` organization routes, and the additive `ServerEvent`
 * variants. Per-surface `types.ts` files extend their own `GalleryImage`
 * with `GalleryOrganizationFields` rather than restating the shape.
 *
 * Every field here is additive on the server side: older hosts omit them, so
 * readers must treat a missing field as "unknown / not organized" and never
 * as an incompatible host. Browser-safe: no DOM, no shell imports.
 */

/** Additive organization fields carried by every `/api/gallery` entry. */
export interface GalleryOrganizationFields {
  /** User-authored title; `null`/absent when the print is untitled. */
  title?: string | null;
  /** Tag names as stored (display case); compare case-insensitively. */
  tags?: string[];
  favorite?: boolean;
  /** Collection ids (host-local `Collection.id`, NOT slugs). */
  collections?: string[];
  /** Unix seconds the print was moved to the trash; absent when live. */
  trashed_at?: number | null;
  /** Unix seconds the sweeper will purge it; absent when kept forever. */
  purge_at?: number | null;
}

/** Minimal gallery entry shape the organization routes hand back. Surfaces
 * narrow it to their own `GalleryImage` via the generic parameters below. */
export interface GalleryEntryWire extends GalleryOrganizationFields {
  filename: string;
  timestamp?: number;
  metadata?: Record<string, unknown> | null;
  [extra: string]: unknown;
}

/** `GET /api/gallery/collections` row (mirrors `mold_core::Collection`). */
export interface Collection {
  id: string;
  name: string;
  /** Normalized name; cross-host merge key (`collectionSlug`). */
  slug: string;
  description: string | null;
  cover_filename: string | null;
  /** Number of prints on this host in the collection. */
  count: number;
  /** Unix seconds. */
  created_at: number;
  /** Unix seconds. */
  updated_at: number;
}

/** `GET /api/gallery/tags` row (mirrors `mold_core::TagCount`). */
export interface TagCount {
  name: string;
  count: number;
}

/** `GET /api/capabilities` → `gallery.trash`. */
export interface GalleryTrashCapabilities {
  enabled: boolean;
  /** `0` = keep forever. */
  retention_days: number;
}

/** `GET /api/capabilities` → `gallery`, with the additive organization
 * fields. Older hosts return only `can_delete`. */
export interface GalleryCapabilitiesWire {
  can_delete: boolean;
  trash?: GalleryTrashCapabilities | null;
  /** Titles / favorites / tags / collections are available. */
  organize?: boolean;
}

/** `PATCH /api/gallery/image/:filename` body. Every field is optional; an
 * absent field leaves that aspect untouched. */
export interface GalleryPatchRequest {
  /** Empty string clears the title. */
  title?: string | null;
  favorite?: boolean | null;
  /** Replace the whole tag set. */
  tags?: string[] | null;
  add_tags?: string[] | null;
  remove_tags?: string[] | null;
}

/** `POST /api/gallery/organize` body — one bulk mutation over many prints. */
export interface GalleryOrganizeRequest {
  filenames: string[];
  favorite?: boolean | null;
  add_tags?: string[] | null;
  remove_tags?: string[] | null;
  /** Collection ids. */
  add_to_collections?: string[] | null;
  /** Collection ids. */
  remove_from_collections?: string[] | null;
}

/** `POST /api/gallery/collections` body. */
export interface CollectionCreateRequest {
  name: string;
  description?: string | null;
}

/** `PATCH /api/gallery/collections/:id` body. */
export interface CollectionUpdateRequest {
  name?: string | null;
  description?: string | null;
  cover_filename?: string | null;
}

/** `PUT /api/gallery/collections/:id/items` body — filenames. */
export interface CollectionItemsRequest {
  add: string[];
  remove: string[];
}

/** `POST /api/gallery/trash` and `POST /api/gallery/trash/restore` body. */
export interface TrashFilenamesRequest {
  filenames: string[];
}

/** `POST /api/gallery/trash/sweep` result. */
export interface TrashSweepResult {
  purged: number;
  remaining: number;
}

/** `DELETE /api/gallery/trash` result. */
export interface EmptyTrashResult {
  purged: number;
}

/** `PATCH /api/gallery/tags/:name` body. */
export interface TagRenameRequest {
  name: string;
}

/** `GET /api/gallery?view=` values. Absent = `library`. */
export type GalleryView = "library" | "trash";

// ── SSE ─────────────────────────────────────────────────────────────────────

/** A print's organization fields changed (title / favorite / tags /
 * collections). The server omits `image` (Rust `Option::None` is skipped,
 * never `null`) when the row is not carried — bulk organize, tag renames —
 * which means "refetch `/api/gallery`"; always test with a nullish check. */
export interface GalleryUpdatedEvent<TImage = GalleryEntryWire> {
  type: "gallery_updated";
  filename: string;
  image?: TImage | null;
}

/** A print moved to the trash. */
export interface GalleryTrashedEvent {
  type: "gallery_trashed";
  filename: string;
}

/** A print came back from the trash. `image: null` = refetch. */
export interface GalleryRestoredEvent<TImage = GalleryEntryWire> {
  type: "gallery_restored";
  filename: string;
  /** Omitted (not `null`) when the server could not enrich the row. */
  image?: TImage | null;
}

/** Collections were created / renamed / deleted / re-covered; refetch
 * `GET /api/gallery/collections`. Purges reuse `gallery_removed`. */
export interface GalleryCollectionsChangedEvent {
  type: "gallery_collections_changed";
}

export type GalleryOrganizationEvent<TImage = GalleryEntryWire> =
  | GalleryUpdatedEvent<TImage>
  | GalleryTrashedEvent
  | GalleryRestoredEvent<TImage>
  | GalleryCollectionsChangedEvent;

export const GALLERY_ORGANIZATION_EVENT_TYPES = [
  "gallery_updated",
  "gallery_trashed",
  "gallery_restored",
  "gallery_collections_changed",
] as const satisfies readonly GalleryOrganizationEvent["type"][];

export function isGalleryOrganizationEvent<TImage = GalleryEntryWire>(
  value: unknown,
): value is GalleryOrganizationEvent<TImage> {
  if (typeof value !== "object" || value === null) return false;
  const type = (value as { type?: unknown }).type;
  if (
    typeof type !== "string" ||
    !(GALLERY_ORGANIZATION_EVENT_TYPES as readonly string[]).includes(type)
  ) {
    return false;
  }
  if (type === "gallery_collections_changed") return true;
  return typeof (value as { filename?: unknown }).filename === "string";
}
