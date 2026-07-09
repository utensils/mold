import { defineStore } from "pinia";
import { apiFetch, apiJson } from "../lib/api/client";
import { evictMedia, mediaPath, thumbnailPath } from "../lib/gallery/media";
import type { GalleryImage } from "../lib/api/types";

export const useGalleryStore = defineStore("gallery", {
  state: () => ({
    items: [] as GalleryImage[],
    loading: false,
    error: null as string | null,
    loaded: false,
  }),
  actions: {
    async fetch() {
      this.loading = true;
      this.error = null;
      try {
        const items = await apiJson<GalleryImage[]>("/api/gallery");
        // Newest first, like a print drawer.
        this.items = items.sort((a, b) => b.timestamp - a.timestamp);
        this.loaded = true;
      } catch (err) {
        this.error = String(err);
      } finally {
        this.loading = false;
      }
    },
    async remove(filename: string) {
      await apiFetch(`/api/gallery/image/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      evictMedia(thumbnailPath(filename));
      evictMedia(mediaPath(filename));
      this.items = this.items.filter((i) => i.filename !== filename);
    },
  },
});
