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
  type ThumbnailPriority,
} from "@studio/lib/thumbnailScheduler";
import { thumbnailUrl } from "../api";
import { getHost } from "../lib/hostRegistry";
import { resolveThumbnailSrc } from "../lib/galleryMedia";
import { printKey } from "../lib/multiHostGallery";
import type { GalleryImage } from "../types";

export function useThumbnailSources(maxResolvedSources = 320) {
  const remoteSrc = reactive(new Map<string, string>());
  // Recency must stay non-reactive: srcFor runs during render, and mutating a
  // reactive Map on a cache hit recursively invalidates that same render.
  const recency = new Map<string, true>();
  // Keys with a resolve genuinely in flight. Success settles into
  // `remoteSrc`; failure clears the key so the next render retries — a
  // transient host error must not blank the tile forever (codex review).
  const requested = new Map<string, ThumbnailHandle<string>>();

  /** `priority` is `visible` for on-screen tiles and `near` for the overscan
   *  band; a queued request is promoted (never demoted) when a tile scrolls
   *  into view, so a fast scroll never starves what is actually shown. */
  function srcFor(
    entry: GalleryImage,
    priority: ThumbnailPriority = "visible",
  ): string {
    const id = (entry as { hostId?: string }).hostId;
    const host = id ? getHost(id) : null;
    if (!host) return thumbnailUrl(entry.filename);
    const print = printKey(entry as { hostId?: string; filename: string });
    const mediaVersion =
      entry.media_version ??
      `${entry.timestamp}:${entry.size_bytes ?? "unknown"}`;
    // A filename is not a physical-media identity: a restored or overwritten
    // print can keep its path while its bytes change. Keep the version in both
    // local maps so a gallery refresh cannot short-circuit to a stale blob URL.
    const key = `${print}|${mediaVersion}`;
    const resolved = remoteSrc.get(key);
    // srcFor runs during Vue render. Reading is safe; mutating the reactive
    // map here would recursively invalidate the component rendering it.
    if (resolved !== undefined) {
      recency.delete(key);
      recency.set(key, true);
      return resolved;
    }
    const pending = requested.get(key);
    if (pending) {
      pending.setPriority(priority);
    } else {
      const handle = galleryThumbnailScheduler.schedule({
        key: `${host.id}|${entry.filename}|${mediaVersion}`,
        hostKey: host.id,
        priority,
        run: (signal) =>
          resolveThumbnailSrc(host, entry.filename, { signal, mediaVersion }),
      });
      requested.set(key, handle);
      void handle.promise
        .then((url) => {
          remoteSrc.set(key, url);
          recency.delete(key);
          recency.set(key, true);
          while (recency.size > maxResolvedSources) {
            const oldest = recency.keys().next().value;
            if (oldest === undefined) break;
            recency.delete(oldest);
            remoteSrc.delete(oldest);
          }
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
    recency.clear();
  });

  return { srcFor };
}
