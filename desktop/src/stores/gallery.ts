import { defineStore } from "pinia";
import { apiFetch, apiJson } from "../lib/api/client";
import { evictMedia, galleryMediaPath, type GallerySource } from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import type { GalleryImage } from "../lib/api/types";

export const useGalleryStore = defineStore("gallery", {
  state: () => ({
    items: [] as GalleryImage[],
    loading: false,
    error: null as string | null,
    loaded: false,
    source: "engine" as GallerySource,
  }),
  actions: {
    async fetch(source?: GallerySource) {
      const target = source ?? this.source;
      this.loading = true;
      this.error = null;
      this.source = target;
      try {
        const items =
          target === "local"
            ? await ipc.localGalleryList()
            : await apiJson<GalleryImage[]>("/api/gallery");
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
      if (this.source === "local") {
        await ipc.localGalleryDelete(filename);
      } else {
        await apiFetch(`/api/gallery/image/${encodeURIComponent(filename)}`, {
          method: "DELETE",
        });
      }
      evictMedia(galleryMediaPath(filename, this.source, true));
      evictMedia(galleryMediaPath(filename, this.source));
      this.items = this.items.filter((i) => i.filename !== filename);
    },
  },
});
