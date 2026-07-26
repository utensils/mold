/**
 * Shared web/desktop Library thumbnail sizing contract. Both surfaces persist
 * the same pixel target, then feed it into their own grid layout engine.
 */
export const GALLERY_THUMBNAIL_SIZE_STORAGE_KEY =
  "mold.gallery.thumbnailSize.v1";
export const GALLERY_THUMBNAIL_SIZE_MIN = 120;
export const GALLERY_THUMBNAIL_SIZE_MAX = 360;
export const GALLERY_THUMBNAIL_SIZE_STEP = 10;
export const GALLERY_THUMBNAIL_SIZE_DEFAULT = 220;

export interface GalleryThumbnailStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function browserStorage(): GalleryThumbnailStorage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function normalizeGalleryThumbnailSize(value: number): number {
  if (!Number.isFinite(value)) return GALLERY_THUMBNAIL_SIZE_DEFAULT;
  const stepped =
    Math.round(value / GALLERY_THUMBNAIL_SIZE_STEP) *
    GALLERY_THUMBNAIL_SIZE_STEP;
  return Math.min(
    GALLERY_THUMBNAIL_SIZE_MAX,
    Math.max(GALLERY_THUMBNAIL_SIZE_MIN, stepped),
  );
}

export function loadGalleryThumbnailSize(
  storage: GalleryThumbnailStorage | null = browserStorage(),
): number {
  if (!storage) return GALLERY_THUMBNAIL_SIZE_DEFAULT;
  try {
    const saved = storage.getItem(GALLERY_THUMBNAIL_SIZE_STORAGE_KEY);
    if (saved === null || saved.trim() === "")
      return GALLERY_THUMBNAIL_SIZE_DEFAULT;
    const parsed = Number(saved);
    return Number.isFinite(parsed)
      ? normalizeGalleryThumbnailSize(parsed)
      : GALLERY_THUMBNAIL_SIZE_DEFAULT;
  } catch {
    return GALLERY_THUMBNAIL_SIZE_DEFAULT;
  }
}

export function saveGalleryThumbnailSize(
  value: number,
  storage: GalleryThumbnailStorage | null = browserStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(
      GALLERY_THUMBNAIL_SIZE_STORAGE_KEY,
      String(normalizeGalleryThumbnailSize(value)),
    );
  } catch {
    // Storage may be disabled; sizing still works for the current visit.
  }
}
