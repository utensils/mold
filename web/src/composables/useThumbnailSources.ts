/*
 * Host-aware thumbnail URLs for gallery tiles. A merged entry carries the
 * host it came from: resolving every tile against the origin 404s every
 * remote print, because the file lives on that machine. Keyless hosts
 * (including the origin) resolve synchronously to a direct URL; an
 * authenticated host needs a ticketed fetch, so its tile fills in once the
 * blob resolves. Shared by the Library grid and the Collections shelf.
 */
import { ref } from "vue";
import { thumbnailUrl } from "../api";
import { getHost } from "../lib/hostRegistry";
import { resolveThumbnailSrc } from "../lib/galleryMedia";
import { printKey } from "../lib/multiHostGallery";
import type { GalleryImage } from "../types";

export function useThumbnailSources() {
  const remoteSrc = ref(new Map<string, string>());
  // Keys with a resolve in flight (or already settled, even to ""), so a
  // re-render never re-issues the same ticket request.
  const requested = new Set<string>();

  function srcFor(entry: GalleryImage): string {
    const id = (entry as { hostId?: string }).hostId;
    const host = id ? getHost(id) : null;
    if (!host) return thumbnailUrl(entry.filename);
    const key = printKey(entry as { hostId?: string; filename: string });
    const resolved = remoteSrc.value.get(key);
    if (resolved !== undefined) return resolved;
    if (!requested.has(key)) {
      requested.add(key);
      void resolveThumbnailSrc(host, entry.filename)
        .then((url) => {
          remoteSrc.value = new Map(remoteSrc.value).set(key, url);
        })
        .catch(() => {
          /* a tile that can't resolve keeps the browser's broken-image state */
        });
    }
    return "";
  }

  return { srcFor };
}
