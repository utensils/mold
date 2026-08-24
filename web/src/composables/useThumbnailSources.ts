/*
 * Host-aware thumbnail URLs for gallery tiles. A merged entry carries the
 * host it came from: resolving every tile against the origin 404s every
 * remote print, because the file lives on that machine. Keyless hosts
 * (including the origin) resolve synchronously to a direct URL; an
 * authenticated host needs a ticketed fetch, so its tile fills in once the
 * blob resolves. Shared by the Library grid and the Collections shelf.
 */
import { onBeforeUnmount, reactive } from "vue";
import {
  galleryThumbnailScheduler,
  type ThumbnailHandle,
} from "@studio/lib/thumbnailScheduler";
import { thumbnailUrl } from "../api";
import { getHost } from "../lib/hostRegistry";
import { resolveThumbnailSrc } from "../lib/galleryMedia";
import { printKey } from "../lib/multiHostGallery";
import type { GalleryImage } from "../types";

export function useThumbnailSources() {
  const remoteSrc = reactive(new Map<string, string>());
  // Keys with a resolve genuinely in flight. Success settles into
  // `remoteSrc`; failure clears the key so the next render retries — a
  // transient host error must not blank the tile forever (codex review).
  const requested = new Map<string, ThumbnailHandle<string>>();

  function srcFor(entry: GalleryImage): string {
    const id = (entry as { hostId?: string }).hostId;
    const host = id ? getHost(id) : null;
    if (!host) return thumbnailUrl(entry.filename);
    const key = printKey(entry as { hostId?: string; filename: string });
    const resolved = remoteSrc.get(key);
    if (resolved !== undefined) return resolved;
    if (!requested.has(key)) {
      const mediaVersion = `${entry.timestamp}:${entry.size_bytes ?? "unknown"}`;
      const handle = galleryThumbnailScheduler.schedule({
        key: `${host.id}|${entry.filename}|${mediaVersion}`,
        hostKey: host.id,
        priority: "visible",
        run: (signal) =>
          resolveThumbnailSrc(host, entry.filename, { signal, mediaVersion }),
      });
      requested.set(key, handle);
      void handle.promise
        .then((url) => {
          remoteSrc.set(key, url);
          requested.delete(key);
        })
        .catch(() => {
          // Leave no settled entry: the tile shows nothing now, and the
          // next gallery refresh re-requests once the host recovers.
          if (requested.get(key) === handle) requested.delete(key);
        });
    }
    return "";
  }

  onBeforeUnmount(() => {
    for (const handle of requested.values()) handle.cancel();
    requested.clear();
  });

  return { srcFor };
}
