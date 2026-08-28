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
interface CachedObjectUrl {
  url: Promise<string>;
  /** Blob bytes retained by the object URL once the request settles. */
  bytes: number | null;
  settled: boolean;
}

// A viewport only needs a small working set. Keeping every thumbnail for the
// entire session made a long Library scroll retain hundreds of MB in WebKit's
// blob/graphics heaps. Both limits are intentional: entry count protects a
// gallery of tiny thumbnails, while bytes protect unusually large posters.
const THUMBNAIL_CACHE_ENTRIES = 512;
const THUMBNAIL_CACHE_BYTES = 64 * 1024 * 1024;
const cache = new Map<string, CachedObjectUrl>();
/** Sum of `bytes` over settled cache entries, kept incrementally: summing
 *  (or copying) the whole map per insert made every thumbnail load O(cache). */
let retainedBytes = 0;

function revokeCachedObjectUrl(entry: CachedObjectUrl): void {
  void entry.url.then((url) => URL.revokeObjectURL(url)).catch(() => {});
}

/** Record a settled entry's bytes exactly once. */
function settleThumbnail(entry: CachedObjectUrl, bytes: number): void {
  entry.bytes = bytes;
  entry.settled = true;
  retainedBytes += bytes;
}

function dropThumbnail(key: string, entry: CachedObjectUrl): void {
  cache.delete(key);
  if (entry.settled) retainedBytes -= entry.bytes ?? 0;
}

function trimThumbnailCache(): void {
  while (cache.size > THUMBNAIL_CACHE_ENTRIES || retainedBytes > THUMBNAIL_CACHE_BYTES) {
    // Never evict an unresolved promise: its caller has not received a usable
    // URL yet. Resolution re-enters this function, so a burst can exceed the
    // bound only while requests are actively in flight. Map iteration order
    // is recency order, so the first settled entry is the oldest.
    let oldest: [string, CachedObjectUrl] | null = null;
    for (const candidate of cache) {
      if (candidate[1].settled) {
        oldest = candidate;
        break;
      }
    }
    if (!oldest) break;
    dropThumbnail(oldest[0], oldest[1]);
    revokeCachedObjectUrl(oldest[1]);
  }
}

function rememberThumbnail(key: string, entry: CachedObjectUrl): void {
  cache.delete(key);
  cache.set(key, entry);
  trimThumbnailCache();
}

export interface AuthedMediaOptions {
  /** Explicit host to fetch from; defaults to the primary connection. */
  target?: ApiTarget;
  /** Cache bucket, usually the origin host id; defaults to "primary". */
  cacheKey?: string;
  /** Opaque content identity; timestamp:size until hosts advertise media_version. */
  mediaVersion?: string;
  /** Cancels queued/native transfer and skips decode/blob work when stale. */
  signal?: AbortSignal;
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

const keyOf = (path: string, target: ApiTarget, cacheKey?: string, mediaVersion?: string) =>
  `${cacheKey ?? "primary"}|${path}|${mediaVersion ?? "legacy"}|${JSON.stringify([target.baseUrl, target.apiKey])}`;

export function authedMediaUrl(path: string, opts: AuthedMediaOptions = {}): Promise<string> {
  if (path.startsWith("mold-local:")) return Promise.resolve(path);
  // A local thumbnail request that reached the blob route (no native cache
  // available) degrades to the full-size local file, as it always did.
  if (path.startsWith("mold-thumb:")) {
    const filename = thumbnailFilenameOfPath(path);
    return Promise.resolve(
      filename === null ? path : localMediaPath(filename, isTrashThumbnailPath(path)),
    );
  }
  const target = opts.target;
  const effectiveTarget = target ?? currentTarget();
  // Target identity is part of the cache authority. A reconnect may retain
  // the host bucket and path while changing URL or credentials; an in-flight
  // object URL from the old route must never satisfy the new one.
  const key = keyOf(path, effectiveTarget, opts.cacheKey, opts.mediaVersion);
  let cached = cache.get(key);
  if (!cached) {
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
      try {
        if (opts.signal?.aborted) throw new DOMException("Thumbnail cancelled", "AbortError");
        const requestId = crypto.randomUUID();
        const cancelNative = () => void ipc.cancelGalleryThumbnail(requestId).catch(() => {});
        opts.signal?.addEventListener("abort", cancelNative, { once: true });
        try {
          const media = await ipc.fetchGalleryThumbnail(target, filename, requestId);
          if (!media) return null;
          if (opts.signal?.aborted) throw new DOMException("Thumbnail cancelled", "AbortError");
          return new Blob([nativeBytes(media)], { type: "image/png" });
        } finally {
          opts.signal?.removeEventListener("abort", cancelNative);
        }
      } catch {
        if (opts.signal?.aborted) throw new DOMException("Thumbnail cancelled", "AbortError");
        // Native commands differ between the desktop and iPhone shells. A
        // missing/refused desktop-only bridge must preserve the authenticated
        // web fallback instead of making every mobile thumbnail unreadable.
        return null;
      }
    };
    const entry: CachedObjectUrl = { url: Promise.resolve(""), bytes: null, settled: false };
    entry.url = nativeThumbnail()
      .then(
        async (native) =>
          native ??
          (
            await (target
              ? opts.signal
                ? apiFetchTo(target, path, { signal: opts.signal })
                : apiFetchTo(target, path)
              : opts.signal
                ? apiFetch(path, { signal: opts.signal })
                : apiFetch(path))
          ).blob(),
      )
      .then((blob) => {
        const objectUrl = URL.createObjectURL(blob);
        // The host/path may have been invalidated while its request was in
        // flight. Do not leak an object URL that is no longer authoritative.
        if (cache.get(key) !== entry) {
          entry.settled = true;
          URL.revokeObjectURL(objectUrl);
        } else {
          settleThumbnail(entry, blob.size);
          trimThumbnailCache();
        }
        return objectUrl;
      });
    cached = entry;
    rememberThumbnail(key, entry);
    entry.url.catch(() => {
      if (cache.get(key) === entry) dropThumbnail(key, entry);
      entry.settled = true;
    });
  } else {
    // Map insertion order is recency order.
    rememberThumbnail(key, cached);
  }
  return cached.url;
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
  // Deleting the current entry during Map iteration is well-defined, so no
  // snapshot copy is needed (a refetch called this once per removed row).
  for (const [key, cached] of cache) {
    if (!key.startsWith(prefix)) continue;
    dropThumbnail(key, cached);
    revokeCachedObjectUrl(cached);
  }
  for (const [key, cached] of fullSizeCache) {
    if (!key.startsWith(prefix)) continue;
    fullSizeCache.delete(key);
    void cached.then((url) => URL.revokeObjectURL(url)).catch(() => {});
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

// ── Persistent native thumbnails (`mold-thumb:`) ────────────────────────────
//
// In the desktop app a tile is PREPARED (cache-first; a miss fetches from the
// host or renders this device's file, then lands on disk) and then DISPLAYED
// through the `mold-thumb://` protocol, which only reads that cache. JS holds
// no bytes, blobs, or object URLs for a tile: WebKit decodes off-thread and
// keeps its own bitmap cache, and a cold launch paints the grid from local
// files without touching any host. The blob route above stays as the
// fallback outside Tauri and for a print with no `media_version` (nothing
// safe to key a persistent entry on).

const THUMBNAIL_PROTOCOL_PREFIX = "mold-thumb://localhost/";
const THUMBNAIL_API_PREFIX = "/api/gallery/thumbnail/";

/** Retina displays get the 512 px tier; everything else 256. Two tiers only,
 *  so the cache never holds one entry per slider position. */
export function thumbnailTier(devicePixelRatio = globalThis.devicePixelRatio ?? 1): 256 | 512 {
  return devicePixelRatio >= 1.5 ? 512 : 256;
}

/** This device's tile REQUEST path while the server is Off. It names the
 *  print, not a size or version — `prepareNativeThumbnail` resolves those. */
export const localThumbnailPath = (filename: string, fromTrash = false) =>
  `${THUMBNAIL_PROTOCOL_PREFIX}local/${encodeURIComponent(filename)}${fromTrash ? "?view=trash" : ""}`;

/** A Trash-view local thumbnail request reads the `.trash/` copy first, like
 *  the `mold-local:` full-size route, so a trashed print is never shadowed
 *  by a NEW live file under the same name. */
export const isTrashThumbnailPath = (path: string) =>
  path.startsWith(THUMBNAIL_PROTOCOL_PREFIX) && path.endsWith("?view=trash");

/** Whether a media path asks for a thumbnail (host API or native local). */
export const isThumbnailPath = (path: string) =>
  path.startsWith(THUMBNAIL_API_PREFIX) || path.startsWith(THUMBNAIL_PROTOCOL_PREFIX);

/** The gallery filename behind either thumbnail path shape, or null. */
export function thumbnailFilenameOfPath(path: string): string | null {
  let encoded: string | null = null;
  if (path.startsWith(THUMBNAIL_API_PREFIX)) encoded = path.slice(THUMBNAIL_API_PREFIX.length);
  else if (path.startsWith(`${THUMBNAIL_PROTOCOL_PREFIX}local/`)) {
    encoded = path.slice(`${THUMBNAIL_PROTOCOL_PREFIX}local/`.length);
  }
  if (!encoded || encoded.includes("/")) return null;
  const query = encoded.indexOf("?");
  if (query !== -1) encoded = encoded.slice(0, query);
  try {
    return decodeURIComponent(encoded);
  } catch {
    return null;
  }
}

export interface NativeThumbnailRequest {
  path: string;
  /** Host to fetch from; null for this device with its server Off. */
  target: ApiTarget | null;
  cacheKey: string | null;
  mediaVersion: string | null;
  signal?: AbortSignal;
}

/**
 * Resolve a tile to a `mold-thumb://` URL through the native cache, or null
 * when this route does not apply (outside Tauri, or no content version to
 * key on) so the caller takes the blob route. A native refusal other than
 * cancellation also answers null — the fallback must keep the tile visible.
 */
export async function prepareNativeThumbnail(
  request: NativeThumbnailRequest,
): Promise<string | null> {
  if (!inTauri()) return null;
  const filename = thumbnailFilenameOfPath(request.path);
  if (filename === null || !request.mediaVersion) return null;
  const cacheKey = request.cacheKey ?? (request.target ? "primary" : "local");
  if (request.signal?.aborted) throw new DOMException("Thumbnail cancelled", "AbortError");
  const requestId = crypto.randomUUID();
  const cancelNative = () => void ipc.cancelGalleryThumbnail(requestId).catch(() => {});
  request.signal?.addEventListener("abort", cancelNative, { once: true });
  try {
    return await ipc.prepareGalleryThumbnail(
      request.target,
      cacheKey,
      filename,
      request.mediaVersion,
      thumbnailTier(),
      requestId,
      isTrashThumbnailPath(request.path),
    );
  } catch {
    if (request.signal?.aborted) throw new DOMException("Thumbnail cancelled", "AbortError");
    return null;
  } finally {
    request.signal?.removeEventListener("abort", cancelNative);
  }
}

/**
 * Where a gallery item's bytes live: a mold server ("host" — API paths,
 * fetched with that host's auth) or this Mac's output dir ("local" — served
 * over the restricted native protocol).
 */
export type GallerySource = "host" | "local";

/** `fromTrash` flips the native protocol's live-first resolution so a
 *  Trash-view row reads its `.trash/` bytes even when a NEW live file later
 *  landed under the same name. */
export const localMediaPath = (filename: string, fromTrash = false) =>
  `mold-local://localhost/${encodeURIComponent(filename)}${fromTrash ? "?view=trash" : ""}`;

/** A local THUMBNAIL is a native-cache request, never the full-size file: a
 *  grid of this device's prints with the server Off used to decode every
 *  full-resolution PNG (and mount a `<video>` per clip) in its tiles. */
export const galleryMediaPath = (
  filename: string,
  source: GallerySource,
  thumbnail = false,
  fromTrash = false,
) =>
  source === "local"
    ? thumbnail
      ? localThumbnailPath(filename, fromTrash)
      : localMediaPath(filename, fromTrash)
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
