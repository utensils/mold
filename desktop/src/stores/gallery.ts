import { defineStore } from "pinia";
import { apiFetch, apiJson } from "../lib/api/client";
import { evictMedia, galleryMediaPath, type GallerySource } from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import type { GalleryImage, ServerEvent } from "../lib/api/types";

/** Collapse a burst of row-less gallery_added events into one refetch. */
const REFETCH_DEBOUNCE_MS = 500;
let refetchTimer: ReturnType<typeof setTimeout> | null = null;

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
    /**
     * A `gallery_added` frame from `GET /api/events`. Only applies while
     * showing the engine gallery — the local (This Mac) source is unrelated
     * to server prints. When the event carries the row, insert in place;
     * otherwise (server DB disabled) debounce a full refetch.
     */
    applyAdded(ev: Extract<ServerEvent, { type: "gallery_added" }>) {
      if (this.source !== "engine" || !this.loaded) return;
      if (!ev.image) {
        if (refetchTimer) clearTimeout(refetchTimer);
        refetchTimer = setTimeout(() => {
          refetchTimer = null;
          void this.fetch();
        }, REFETCH_DEBOUNCE_MS);
        return;
      }
      if (this.items.some((i) => i.filename === ev.filename)) return;
      const image = ev.image;
      const at = this.items.findIndex((i) => i.timestamp < image.timestamp);
      if (at === -1) this.items.push(image);
      else this.items.splice(at, 0, image);
    },
    /** A `gallery_removed` frame — drop the tile and its cached media. */
    applyRemoved(filename: string) {
      if (this.source !== "engine" || !this.loaded) return;
      if (!this.items.some((i) => i.filename === filename)) return;
      evictMedia(galleryMediaPath(filename, this.source, true));
      evictMedia(galleryMediaPath(filename, this.source));
      this.items = this.items.filter((i) => i.filename !== filename);
    },
  },
});
