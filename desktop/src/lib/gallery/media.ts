import { ApiError, apiFetch, apiFetchTo, currentTarget, type ApiTarget } from "../api/client";
import type { GalleryImage } from "../api/types";
import { inTauri, ipc } from "../ipc";

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
  /** The caller's media kind (`isVideoItem`); video never buffers natively.
   * Absent, the filename extension decides. */
  video?: boolean;
}

const keyOf = (path: string, target: ApiTarget, cacheKey?: string) =>
  `${cacheKey ?? "primary"}|${path}|${JSON.stringify([target.baseUrl, target.apiKey])}`;

export function authedMediaUrl(path: string, opts: AuthedMediaOptions = {}): Promise<string> {
  if (path.startsWith("mold-local:")) return Promise.resolve(path);
  const target = opts.target;
  const effectiveTarget = target ?? currentTarget();
  // Target identity is part of the cache authority. A reconnect may retain
  // the host bucket and path while changing URL or credentials; an in-flight
  // object URL from the old route must never satisfy the new one.
  const key = keyOf(path, effectiveTarget, opts.cacheKey);
  let url = cache.get(key);
  if (!url) {
    const thumbnailPrefix = "/api/gallery/thumbnail/";
    const encodedFilename = path.startsWith(thumbnailPrefix)
      ? path.slice(thumbnailPrefix.length)
      : null;
    const nativeThumbnail = async (): Promise<Blob | null> => {
      if (!target || !inTauri() || encodedFilename === null) return null;
      let filename: string;
      try {
        filename = decodeURIComponent(encodedFilename);
      } catch {
        return null;
      }
      const media = await ipc.fetchGalleryThumbnail(target, filename);
      if (!media) return null;
      const bytes = Uint8Array.from(atob(media.base64), (character) => character.charCodeAt(0));
      return new Blob([bytes], { type: media.contentType });
    };
    url = nativeThumbnail()
      .then(
        async (native) =>
          native ?? (await (target ? apiFetchTo(target, path) : apiFetch(path))).blob(),
      )
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

const GALLERY_IMAGE_PREFIX = "/api/gallery/image/";

/** The gallery filename behind a `/api/gallery/image/<encoded>` path, or null. */
export function galleryFilenameOfPath(path: string): string | null {
  if (!path.startsWith(GALLERY_IMAGE_PREFIX)) return null;
  const encoded = path.slice(GALLERY_IMAGE_PREFIX.length);
  if (!encoded || encoded.includes("/") || encoded.includes("?")) return null;
  try {
    return decodeURIComponent(encoded);
  } catch {
    return null;
  }
}

const MEDIA_MIME_BY_EXTENSION: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  apng: "image/apng",
  mp4: "video/mp4",
  webm: "video/webm",
  wav: "audio/wav",
};

/** MIME type for a gallery filename — native byte fetches carry no header. */
export function mediaMimeType(filename: string): string {
  const extension = filename.toLowerCase().split(".").pop() ?? "";
  return MEDIA_MIME_BY_EXTENSION[extension] ?? "application/octet-stream";
}

/**
 * Full-size media for a media element, native-first. In the desktop app a
 * host-backed still or audio print is fetched by the Rust HTTP client and
 * served as an object URL, because an `<img>` pointed straight at the host
 * shares WebKit's per-host connection pool with every held-open generation
 * and download stream to that host — the same starvation that moved
 * thumbnails native in #1132. Video deliberately stays on
 * `streamableMediaUrl` (Range-friendly ticket/direct URL) so it can seek
 * without buffering the whole file. Outside Tauri, for `mold-local:` paths,
 * and when the native route refuses (too large, unreachable), stills fall
 * back to `streamableMediaUrl` as well.
 */
/**
 * Full-size object URLs are tens of MB each, so unlike thumbnails they live
 * in a small LRU rather than the session-lifetime cache: stepping through a
 * remote gallery of 4K prints must not accumulate gigabytes of blobs.
 * Insertion order is the recency order; a hit re-inserts.
 */
const FULL_SIZE_CACHE_ENTRIES = 8;
const fullSizeCache = new Map<string, Promise<string>>();

function rememberFullSize(key: string, url: Promise<string>): void {
  fullSizeCache.delete(key);
  fullSizeCache.set(key, url);
  while (fullSizeCache.size > FULL_SIZE_CACHE_ENTRIES) {
    const oldest = fullSizeCache.keys().next().value;
    if (oldest === undefined) break;
    const evicted = fullSizeCache.get(oldest);
    fullSizeCache.delete(oldest);
    void evicted?.then((u) => URL.revokeObjectURL(u)).catch(() => {});
  }
}

/** Bytes from the native IPC bridge: an `ArrayBuffer` on the custom-protocol
 * route, but a plain number array if Tauri ever falls back to postMessage. */
export const nativeBytes = (bytes: ArrayBuffer | ArrayLike<number>): Uint8Array<ArrayBuffer> =>
  bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : Uint8Array.from(bytes);

export async function fullSizeMediaUrl(
  path: string,
  opts: StreamableMediaOptions = {},
): Promise<string> {
  if (path.startsWith("mold-local:")) return path;
  const target = opts.target;
  const filename = galleryFilenameOfPath(path);
  const isVideo = opts.video ?? (filename !== null && mediaMimeType(filename).startsWith("video/"));
  if (target && filename !== null && inTauri() && !isVideo) {
    const key = keyOf(path, target, opts.cacheKey);
    let url = fullSizeCache.get(key);
    if (!url) {
      url = ipc.fetchGalleryMedia(target, filename).then((bytes) => {
        if (!bytes) throw new Error("Native gallery media is unavailable.");
        return URL.createObjectURL(
          new Blob([nativeBytes(bytes)], { type: mediaMimeType(filename) }),
        );
      });
      url.catch(() => {
        if (fullSizeCache.get(key) === url) fullSizeCache.delete(key);
      });
    }
    rememberFullSize(key, url);
    try {
      return await url;
    } catch {
      // Fall through to the webview's own route (ticketed or direct URL).
    }
  }
  return streamableMediaUrl(path, opts);
}

/**
 * Raw bytes for one host-backed gallery file (clipboard copy, source reuse),
 * native-first for the same pool reason as `fullSizeMediaUrl`; a refused
 * native read falls back to the webview's authenticated HTTP route.
 */
export async function fetchGalleryMediaBytes(path: string, target: ApiTarget): Promise<Uint8Array> {
  const filename = galleryFilenameOfPath(path);
  if (filename !== null && inTauri()) {
    try {
      const bytes = await ipc.fetchGalleryMedia(target, filename);
      if (bytes) return nativeBytes(bytes);
    } catch {
      // Fall through to the webview's authenticated HTTP route.
    }
  }
  return new Uint8Array(await (await apiFetchTo(target, path)).arrayBuffer());
}

function evictPrefix(prefix: string): void {
  for (const store of [cache, fullSizeCache]) {
    for (const [key, cached] of [...store]) {
      if (!key.startsWith(prefix)) continue;
      store.delete(key);
      void cached.then((u) => URL.revokeObjectURL(u)).catch(() => {});
    }
  }
}

export function evictMedia(path: string, cacheKey?: string): void {
  evictPrefix(`${cacheKey ?? "primary"}|${path}|`);
}

/** Drop every cached blob belonging to one origin (host bucket dropped). */
export function evictHostMedia(cacheKey: string): void {
  evictPrefix(`${cacheKey}|`);
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

/**
 * Whether a gallery print is an audio-only artifact (LTX-2 text-to-audio).
 * Deliberately separate from `isVideoItem`: audio has no frames, so every
 * `<video>` element, ▶ badge and frame-seeking path must keep treating it as
 * "not a video" while still getting its own transport.
 */
export const isAudioItem = (item: GalleryImage): boolean =>
  item.format === "wav" || item.filename.toLowerCase().endsWith(".wav");
