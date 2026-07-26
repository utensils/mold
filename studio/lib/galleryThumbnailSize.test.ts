import { describe, expect, it } from "vitest";
import {
  GALLERY_THUMBNAIL_SIZE_DEFAULT,
  GALLERY_THUMBNAIL_SIZE_MAX,
  GALLERY_THUMBNAIL_SIZE_MIN,
  GALLERY_THUMBNAIL_SIZE_STORAGE_KEY,
  loadGalleryThumbnailSize,
  saveGalleryThumbnailSize,
} from "./galleryThumbnailSize";

function storage(seed: Record<string, string> = {}) {
  const values = new Map(Object.entries(seed));
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    value: (key: string) => values.get(key) ?? null,
  };
}

describe("gallery thumbnail size persistence", () => {
  it("uses the shared default when no preference exists", () => {
    expect(loadGalleryThumbnailSize(storage())).toBe(
      GALLERY_THUMBNAIL_SIZE_DEFAULT,
    );
  });

  it("loads a saved size and clamps stale values to the supported range", () => {
    expect(
      loadGalleryThumbnailSize(
        storage({ [GALLERY_THUMBNAIL_SIZE_STORAGE_KEY]: "280" }),
      ),
    ).toBe(280);
    expect(
      loadGalleryThumbnailSize(
        storage({ [GALLERY_THUMBNAIL_SIZE_STORAGE_KEY]: "999" }),
      ),
    ).toBe(GALLERY_THUMBNAIL_SIZE_MAX);
    expect(
      loadGalleryThumbnailSize(
        storage({ [GALLERY_THUMBNAIL_SIZE_STORAGE_KEY]: "10" }),
      ),
    ).toBe(GALLERY_THUMBNAIL_SIZE_MIN);
  });

  it("falls back for malformed values and saves normalized pixel sizes", () => {
    const target = storage({
      [GALLERY_THUMBNAIL_SIZE_STORAGE_KEY]: "not-a-number",
    });
    expect(loadGalleryThumbnailSize(target)).toBe(
      GALLERY_THUMBNAIL_SIZE_DEFAULT,
    );

    saveGalleryThumbnailSize(287, target);
    expect(target.value(GALLERY_THUMBNAIL_SIZE_STORAGE_KEY)).toBe("290");
  });
});
