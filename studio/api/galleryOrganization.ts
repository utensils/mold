/**
 * Explicit-target HTTP helpers for the Library organization routes — titles,
 * favorites, tags, collections, and the trash. Every call names its host via
 * `ApiTarget` (multi-host Library fan-out) and sends the key as a header,
 * never in a URL. Wire shapes live in `../lib/api/galleryOrganization`.
 */

import type {
  Collection,
  CollectionCreateRequest,
  CollectionItemsRequest,
  CollectionUpdateRequest,
  EmptyTrashResult,
  GalleryEntryWire,
  GalleryOrganizeRequest,
  GalleryPatchRequest,
  GalleryView,
  TagCount,
  TagRenameRequest,
  TrashFilenamesRequest,
  TrashSweepResult,
} from "../lib/api/galleryOrganization";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "./client";

const JSON_HEADERS = { "content-type": "application/json" } as const;

function encodePath(segment: string): string {
  return encodeURIComponent(segment);
}

/** Parse a JSON body when the server sent one; `null` on 204 / empty. */
async function jsonOrNull<T>(response: Response): Promise<T | null> {
  if (response.status === 204) return null;
  const text = await response.text();
  if (text.trim().length === 0) return null;
  return JSON.parse(text) as T;
}

/** Accept either a bare array or `{ [key]: [...] }` for list endpoints. */
function unwrapList<T>(value: unknown, key: string, what: string): T[] {
  if (Array.isArray(value)) return value as T[];
  if (typeof value === "object" && value !== null) {
    const nested = (value as Record<string, unknown>)[key];
    if (Array.isArray(nested)) return nested as T[];
  }
  throw new Error(`This Mold host returned an unreadable ${what} listing.`);
}

// ── Per-print ───────────────────────────────────────────────────────────────

/** `PATCH /api/gallery/image/:filename` — title / favorite / tags. Resolves
 * to the updated entry when the server echoes it, else `null`. */
export async function patchGalleryImage<TImage = GalleryEntryWire>(
  target: ApiTarget,
  filename: string,
  body: GalleryPatchRequest,
): Promise<TImage | null> {
  const response = await apiFetchTo(
    target,
    `/api/gallery/image/${encodePath(filename)}`,
    { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) },
  );
  return jsonOrNull<TImage>(response);
}

/** `POST /api/gallery/organize` — one bulk mutation over many prints. */
export async function organizeGallery(
  target: ApiTarget,
  body: GalleryOrganizeRequest,
): Promise<void> {
  await apiFetchTo(target, "/api/gallery/organize", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

// ── Collections ─────────────────────────────────────────────────────────────

export async function listCollections(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<Collection[]> {
  const value = await apiJsonTo<unknown>(target, "/api/gallery/collections", {
    signal: signal ?? null,
  });
  return unwrapList<Collection>(value, "collections", "collections");
}

export function createCollection(
  target: ApiTarget,
  body: CollectionCreateRequest,
): Promise<Collection> {
  return apiJsonTo<Collection>(target, "/api/gallery/collections", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export function updateCollection(
  target: ApiTarget,
  id: string,
  body: CollectionUpdateRequest,
): Promise<Collection> {
  return apiJsonTo<Collection>(
    target,
    `/api/gallery/collections/${encodePath(id)}`,
    { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) },
  );
}

/**
 * Update collection visibility and verify the server actually understood it.
 * Older organizing servers ignore unknown PATCH fields while still returning
 * 200, so a successful response alone cannot prove that hiding took effect.
 */
export async function updateCollectionHidden(
  target: ApiTarget,
  id: string,
  hidden: boolean,
): Promise<Collection> {
  const updated = await updateCollection(target, id, { hidden });
  const matches = hidden ? updated.hidden === true : updated.hidden !== true;
  if (!matches) {
    throw new Error("This host does not support hidden collections.");
  }
  return updated;
}

/** Deletes the collection only — never the prints in it (D7). */
export async function deleteCollection(
  target: ApiTarget,
  id: string,
): Promise<void> {
  await apiFetchTo(target, `/api/gallery/collections/${encodePath(id)}`, {
    method: "DELETE",
  });
}

/** `PUT /api/gallery/collections/:id/items {add, remove}` (filenames).
 * Resolves to the collection when the server echoes it, else `null`. */
export async function setCollectionItems(
  target: ApiTarget,
  id: string,
  body: CollectionItemsRequest,
): Promise<Collection | null> {
  const response = await apiFetchTo(
    target,
    `/api/gallery/collections/${encodePath(id)}/items`,
    { method: "PUT", headers: JSON_HEADERS, body: JSON.stringify(body) },
  );
  return jsonOrNull<Collection>(response);
}

// ── Tags ────────────────────────────────────────────────────────────────────

export async function listTags(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<TagCount[]> {
  const value = await apiJsonTo<unknown>(target, "/api/gallery/tags", {
    signal: signal ?? null,
  });
  return unwrapList<TagCount>(value, "tags", "tags");
}

export async function renameTag(
  target: ApiTarget,
  name: string,
  newName: string,
): Promise<void> {
  const body: TagRenameRequest = { name: newName };
  await apiFetchTo(target, `/api/gallery/tags/${encodePath(name)}`, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export async function deleteTag(
  target: ApiTarget,
  name: string,
): Promise<void> {
  await apiFetchTo(target, `/api/gallery/tags/${encodePath(name)}`, {
    method: "DELETE",
  });
}

// ── Trash ───────────────────────────────────────────────────────────────────

/** `DELETE /api/gallery/image/:filename` — moves the print to the trash on
 * a trash-capable host (older hosts hard-delete; check capabilities). */
export async function trashGalleryImage(
  target: ApiTarget,
  filename: string,
): Promise<void> {
  await apiFetchTo(target, `/api/gallery/image/${encodePath(filename)}`, {
    method: "DELETE",
  });
}

/** `DELETE /api/gallery/image/:filename?permanent=true` — bypasses the
 * trash. Destructive; gate behind the shared confirm dialog. */
export async function deleteGalleryImageForever(
  target: ApiTarget,
  filename: string,
): Promise<void> {
  await apiFetchTo(
    target,
    `/api/gallery/image/${encodePath(filename)}?permanent=true`,
    { method: "DELETE" },
  );
}

/** `POST /api/gallery/trash {filenames}` — bulk move to trash. */
export async function trashMany(
  target: ApiTarget,
  filenames: string[],
): Promise<void> {
  const body: TrashFilenamesRequest = { filenames };
  await apiFetchTo(target, "/api/gallery/trash", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

/** `POST /api/gallery/trash/restore {filenames}`. A restore whose live
 * filename is taken again answers 409 (`ApiError.status`). */
export async function restoreTrashed(
  target: ApiTarget,
  filenames: string[],
): Promise<void> {
  const body: TrashFilenamesRequest = { filenames };
  await apiFetchTo(target, "/api/gallery/trash/restore", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

/** `DELETE /api/gallery/trash` — purge everything in the trash now. */
export function emptyTrash(target: ApiTarget): Promise<EmptyTrashResult> {
  return apiJsonTo<EmptyTrashResult>(target, "/api/gallery/trash", {
    method: "DELETE",
  });
}

/** `POST /api/gallery/trash/sweep` — purge what has passed its retention. */
export function sweepTrash(target: ApiTarget): Promise<TrashSweepResult> {
  return apiJsonTo<TrashSweepResult>(target, "/api/gallery/trash/sweep", {
    method: "POST",
  });
}

/** `GET /api/gallery?view=trash` — the trashed entries only. */
export async function listTrash<TImage = GalleryEntryWire>(
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<TImage[]> {
  const view: GalleryView = "trash";
  const value = await apiJsonTo<unknown>(target, `/api/gallery?view=${view}`, {
    signal: signal ?? null,
  });
  if (!Array.isArray(value)) {
    throw new Error("This Mold host returned an unreadable trash listing.");
  }
  return value as TImage[];
}
