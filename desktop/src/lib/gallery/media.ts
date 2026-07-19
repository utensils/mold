import { ApiError, apiFetch, apiFetchTo, currentTarget, type ApiTarget } from "../api/client";
import type { GalleryImage } from "../api/types";

/**
 * Gallery media sits behind X-Api-Key auth, and <img>/<video> cannot send
 * headers — so media is fetched with auth and served to the DOM as object
 * URLs. Entries are cached per (origin, path): the same API path on two
 * hosts is two different images, so the cache key is prefixed with an
 * origin `cacheKey` ("primary" = the app's primary connection). Video
 * viewers should use `streamableMediaUrl` instead, which preserves Range
 * requests through a short-lived read-only ticket.
 */
const cache = new Map<string, Promise<string>>();

export interface AuthedMediaOptions {
  /** Explicit host to fetch from; defaults to the primary connection. */
  target?: ApiTarget;
  /** Cache bucket, usually the origin host id; defaults to "primary". */
  cacheKey?: string;
}

interface GalleryMediaTicket {
  token: string | null;
  expires_at: number | null;
  auth_required?: boolean;
}

export interface StreamableMediaOptions extends AuthedMediaOptions {
  /** Older hosts have no streaming-ticket endpoint. Images may safely fall
   * back to a blob; videos must not buffer an unbounded file on iPhone. */
  allowLegacyBlob?: boolean;
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

/**
 * Return a URL an `<img>` or `<video>` can load directly. Headerless hosts use
 * their ordinary media URL, preserving HTTP Range requests. Authenticated
 * hosts exchange the API key for a short-lived, read-only gallery ticket so
 * WebKit can stream and seek without putting the long-lived key in the URL.
 */
export async function streamableMediaUrl(
  path: string,
  opts: StreamableMediaOptions = {},
): Promise<string> {
  if (path.startsWith("mold-local:")) return path;
  const target = opts.target ?? currentTarget();
  const directUrl = `${target.baseUrl.replace(/\/$/, "")}${path}`;
  if (!target.apiKey) return directUrl;

  try {
    const response = await apiFetchTo(target, "/api/gallery/media-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const ticket = (await response.json()) as GalleryMediaTicket;
    // A key can remain in Keychain after a host disables authentication. New
    // hosts make that state explicit so the media element can use its normal
    // direct URL instead of treating the harmless stale key as a failure.
    if (ticket.auth_required === false) return directUrl;
    if (!ticket.token || !Number.isSafeInteger(ticket.expires_at)) {
      throw new Error("The host returned an invalid gallery media ticket.");
    }
    const url = new URL(directUrl);
    url.searchParams.set("media_token", ticket.token);
    url.searchParams.set("expires", String(ticket.expires_at));
    return url.toString();
  } catch (error) {
    const endpointMissing =
      error instanceof ApiError && (error.status === 404 || error.status === 405);
    // Older auth-disabled hosts have no ticket endpoint. A headerless HEAD is
    // bounded and distinguishes them from older authenticated hosts without
    // ever buffering the underlying video.
    if (endpointMissing) {
      try {
        const probe = await fetch(directUrl, { method: "HEAD" });
        if (probe.ok) return directUrl;
      } catch {
        // Fall through to the bounded image fallback / upgrade guidance.
      }
    }
    if (endpointMissing && opts.allowLegacyBlob) return authedMediaUrl(path, opts);
    if (endpointMissing) {
      throw new Error("Update this Mold host to stream videos on iPhone.");
    }
    throw error;
  }
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
