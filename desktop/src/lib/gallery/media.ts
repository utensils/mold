import { apiFetch } from "../api/client";

/**
 * Gallery media sits behind X-Api-Key auth, and <img>/<video> cannot send
 * headers — so media is fetched with auth and served to the DOM as object
 * URLs, cached per path. (A tokenized media URL is a candidate upstream
 * improvement to let <video> stream with Range instead of full-buffering.)
 */
const cache = new Map<string, Promise<string>>();

export function authedMediaUrl(path: string): Promise<string> {
  let url = cache.get(path);
  if (!url) {
    url = apiFetch(path)
      .then((r) => r.blob())
      .then((b) => URL.createObjectURL(b));
    cache.set(path, url);
    url.catch(() => cache.delete(path));
  }
  return url;
}

export function evictMedia(path: string): void {
  const cached = cache.get(path);
  cache.delete(path);
  void cached?.then((u) => URL.revokeObjectURL(u)).catch(() => {});
}

export const thumbnailPath = (filename: string) =>
  `/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
export const mediaPath = (filename: string) => `/api/gallery/image/${encodeURIComponent(filename)}`;
