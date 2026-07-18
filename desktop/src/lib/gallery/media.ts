import { apiFetch, apiFetchTo, type ApiTarget } from "../api/client";
import type { GalleryImage } from "../api/types";

/**
 * Gallery media sits behind X-Api-Key auth, and <img>/<video> cannot send
 * headers — so media is fetched with auth and served to the DOM as object
 * URLs. Entries are cached per (origin, path): the same API path on two
 * hosts is two different images, so the cache key is prefixed with an
 * origin `cacheKey` ("primary" = the app's primary connection). (A
 * tokenized media URL is a candidate upstream improvement to let <video>
 * stream with Range instead of full-buffering.)
 */
const cache = new Map<string, Promise<string>>();

export interface AuthedMediaOptions {
  /** Explicit host to fetch from; defaults to the primary connection. */
  target?: ApiTarget;
  /** Cache bucket, usually the origin host id; defaults to "primary". */
  cacheKey?: string;
}

const keyOf = (path: string, cacheKey?: string) => `${cacheKey ?? "primary"}|${path}`;

export function authedMediaUrl(path: string, opts: AuthedMediaOptions = {}): Promise<string> {
  if (path.startsWith("mold-local:")) return Promise.resolve(path);
  const key = keyOf(path, opts.cacheKey);
  let url = cache.get(key);
  if (!url) {
    const target = opts.target;
    url = (target ? apiFetchTo(target, path) : apiFetch(path))
      .then((r) => r.blob())
      .then((b) => URL.createObjectURL(b));
    cache.set(key, url);
    url.catch(() => cache.delete(key));
  }
  return url;
}

export function evictMedia(path: string, cacheKey?: string): void {
  const key = keyOf(path, cacheKey);
  const cached = cache.get(key);
  cache.delete(key);
  void cached?.then((u) => URL.revokeObjectURL(u)).catch(() => {});
}

/** Drop every cached blob belonging to one origin (host bucket dropped). */
export function evictHostMedia(cacheKey: string): void {
  const prefix = `${cacheKey}|`;
  for (const [key, cached] of [...cache]) {
    if (!key.startsWith(prefix)) continue;
    cache.delete(key);
    void cached.then((u) => URL.revokeObjectURL(u)).catch(() => {});
  }
}

export const thumbnailPath = (filename: string) =>
  `/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
export const mediaPath = (filename: string) => `/api/gallery/image/${encodeURIComponent(filename)}`;

/**
 * Where a gallery item's bytes live: a mold server ("host" — API paths,
 * fetched with that host's auth) or this Mac's output dir ("local" — served
 * over the restricted native protocol).
 */
export type GallerySource = "host" | "local";

export const localMediaPath = (filename: string) =>
  `mold-local://localhost/${encodeURIComponent(filename)}`;

export const galleryMediaPath = (filename: string, source: GallerySource, thumbnail = false) =>
  source === "local"
    ? localMediaPath(filename)
    : thumbnail
      ? thumbnailPath(filename)
      : mediaPath(filename);

/**
 * Whether a gallery print is a video. The single source of truth shared by
 * the grid's ▶ badge, the store's Images/Video kind filter, and the chip
 * counts — so they can never disagree.
 */
export const isVideoItem = (item: GalleryImage): boolean =>
  item.format === "mp4" || item.filename.endsWith(".mp4") || !!item.metadata.video_frames;
