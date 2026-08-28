/*
 * Multi-host gallery media (Task #22). `<img>`/`<video>` elements cannot send
 * an `x-api-key` header and a durable key must never enter a URL, so remote
 * gallery media is addressed one of two ways depending on the surface:
 *
 *   - Thumbnails (grid tiles) are small, bounded images. On an authenticated
 *     host they are blob-fetched with the key and served to the DOM as an
 *     object URL — the thumbnail route (`/api/gallery/thumbnail/:file`) is not
 *     coverable by a media ticket (the server only signs `/image/:file`).
 *   - Full-size media (lightbox image + video) uses the server's short-lived
 *     `POST /api/gallery/media-token` ticket, appended to the direct URL as a
 *     `media_token`/`expires` query pair so WebKit can Range-stream a video
 *     without buffering it and without the durable key in the URL. Hosts that
 *     lack the current endpoint are rejected with an upgrade message.
 *
 * The serving origin ("this server") is same-origin and keyless in the web
 * registry, so it always uses plain relative URLs — exactly today's behaviour.
 * Mirrors the desktop app's desktop/src/lib/gallery/media.ts contract.
 */
import {
  persistentThumbnailKey,
  persistentThumbnailStore,
  thumbnailRenditionQuery,
  thumbnailTier,
  type PersistentThumbnailStore,
} from "@studio/lib/thumbnailPersistentCache";
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";

/** Blob object-URL cache, keyed by `${hostId}|${path}`. Ticket URLs are not
 *  cached — they are cheap to mint and expire, and the lightbox renews on open. */
interface CachedThumbnail {
  url: Promise<string>;
  bytes: number | null;
  settled: boolean;
}

const THUMBNAIL_CACHE_ENTRIES = 512;
const THUMBNAIL_CACHE_BYTES = 64 * 1024 * 1024;
const cache = new Map<string, CachedThumbnail>();
/** Sum of settled entries' bytes, kept incrementally (summing the whole map
 *  per insert made every tile load O(cache)). */
let retainedBytes = 0;

/**
 * Authenticated hosts' tiles persist in the Cache API (secure contexts
 * only; `null` on a plain-http origin), so a reload paints the grid without
 * one request per tile. Resolved lazily and once per page.
 */
let persistent: PersistentThumbnailStore | null | undefined;
function persistentStore(): PersistentThumbnailStore | null {
  persistent ??= persistentThumbnailStore();
  return persistent;
}

const keyOf = (hostId: string, path: string) => `${hostId}|${path}`;

/**
 * The tile path. `?v=` carries the content version so a rewritten print is a
 * different URL to the browser's own HTTP cache (keyless hosts load this
 * directly); the rendition query asks a current server for the display's
 * tier as JPEG, which an older server ignores.
 */
export function thumbnailPath(filename: string, mediaVersion?: string): string {
  const path = `/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
  const params = [
    mediaVersion ? `v=${encodeURIComponent(mediaVersion)}` : null,
    thumbnailRenditionQuery(),
  ].filter((p): p is string => p !== null);
  return `${path}?${params.join("&")}`;
}

export function mediaPath(filename: string): string {
  return `/api/gallery/image/${encodeURIComponent(filename)}`;
}

/** URL prefix for a host's API. The origin is same-origin, so relative. */
export function hostMediaBase(host: HostEntry): string {
  return host.id === ORIGIN_HOST_ID ? "" : host.url.replace(/\/$/, "");
}

/** Direct (keyless) URL for a host's thumbnail — relative for the origin. */
export function directThumbnailUrl(
  host: HostEntry,
  filename: string,
  mediaVersion?: string,
): string {
  return `${hostMediaBase(host)}${thumbnailPath(filename, mediaVersion)}`;
}

/** Direct (keyless) URL for a host's full-size media — relative for the origin. */
export function directMediaUrl(host: HostEntry, filename: string): string {
  return `${hostMediaBase(host)}${mediaPath(filename)}`;
}

function authHeaders(host: HostEntry): Record<string, string> {
  return host.apiKey ? { "x-api-key": host.apiKey } : {};
}

/** A host needs authenticated (blob / ticket) media only when it carries a
 *  per-host API key. The keyless origin and keyless remotes load directly. */
export function needsAuthedMedia(host: HostEntry): boolean {
  return !!host.apiKey;
}

async function fetchAuthedObjectUrl(
  host: HostEntry,
  path: string,
  signal?: AbortSignal,
  persistentKey?: string,
): Promise<{ url: string; bytes: number }> {
  const store = persistentKey ? persistentStore() : null;
  if (store && persistentKey) {
    const stored = await store.get(persistentKey);
    if (stored) {
      if (signal?.aborted)
        throw new DOMException("Thumbnail cancelled", "AbortError");
      return { url: URL.createObjectURL(stored), bytes: stored.size };
    }
  }
  const res = await fetch(`${hostMediaBase(host)}${path}`, {
    headers: authHeaders(host),
    signal,
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  const blob = await res.blob();
  if (signal?.aborted)
    throw new DOMException("Thumbnail cancelled", "AbortError");
  if (store && persistentKey) void store.put(persistentKey, blob);
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

/** Fetch raw bytes for a gallery path with the host's key — used by
 *  "Use as source", which needs the actual blob rather than a display URL. */
export async function fetchGalleryBlob(
  host: HostEntry,
  filename: string,
): Promise<Blob> {
  const res = await fetch(`${hostMediaBase(host)}${mediaPath(filename)}`, {
    headers: authHeaders(host),
  });
  if (!res.ok) throw new Error(`GET media failed: ${res.status}`);
  return res.blob();
}

/** Fetch the exact gallery poster/waveform without first listing the gallery.
 * Durable completion hydration already owns the authoritative filename, so
 * this keeps video/audio recovery on the same O(1) artifact path as the
 * primary media. */
export async function fetchGalleryThumbnailBlob(
  host: HostEntry,
  filename: string,
): Promise<Blob> {
  // The exact artifact (a video poster / audio waveform), not a grid
  // rendition — no rendition query here.
  const path = `/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
  const res = await fetch(`${hostMediaBase(host)}${path}`, {
    headers: authHeaders(host),
  });
  if (!res.ok) throw new Error(`GET thumbnail failed: ${res.status}`);
  return res.blob();
}

/**
 * Grid-tile thumbnail source. Keyless hosts (origin included) use the plain
 * direct URL so the browser can lazy-load it; authenticated hosts blob-fetch
 * the small poster with the key and cache the resulting object URL per
 * (host, filename).
 */
export function resolveThumbnailSrc(
  host: HostEntry,
  filename: string,
  options: { signal?: AbortSignal; mediaVersion?: string } = {},
): Promise<string> {
  const path = thumbnailPath(filename, options.mediaVersion);
  if (!needsAuthedMedia(host)) {
    return Promise.resolve(`${hostMediaBase(host)}${path}`);
  }
  const key = keyOf(host.id, path);
  let cached = cache.get(key);
  if (!cached) {
    const entry: CachedThumbnail = {
      url: Promise.resolve(""),
      bytes: null,
      settled: false,
    };
    // Only a print with a content version may persist: a "legacy" key would
    // pin stale bytes across reloads forever.
    const persistentKey = options.mediaVersion
      ? persistentThumbnailKey(
          host.id,
          filename,
          options.mediaVersion,
          thumbnailTier(),
        )
      : undefined;
    entry.url = fetchAuthedObjectUrl(
      host,
      path,
      options.signal,
      persistentKey,
    ).then(({ url, bytes }) => {
      if (cache.get(key) !== entry) {
        entry.settled = true;
        URL.revokeObjectURL(url);
      } else {
        settleThumbnail(entry, bytes);
        trimThumbnailCache();
      }
      return url;
    });
    cached = entry;
    cache.set(key, entry);
    trimThumbnailCache();
    entry.url.catch(() => {
      if (cache.get(key) === entry) dropThumbnail(key, entry);
      entry.settled = true;
    });
  } else {
    cache.delete(key);
    cache.set(key, cached);
  }
  return cached.url;
}

function settleThumbnail(entry: CachedThumbnail, bytes: number): void {
  entry.bytes = bytes;
  entry.settled = true;
  retainedBytes += bytes;
}

function dropThumbnail(key: string, entry: CachedThumbnail): void {
  cache.delete(key);
  if (entry.settled) retainedBytes -= entry.bytes ?? 0;
}

function trimThumbnailCache(): void {
  while (
    cache.size > THUMBNAIL_CACHE_ENTRIES ||
    retainedBytes > THUMBNAIL_CACHE_BYTES
  ) {
    // Map order is recency order; the first settled entry is the oldest.
    let oldest: [string, CachedThumbnail] | null = null;
    for (const candidate of cache) {
      if (candidate[1].settled) {
        oldest = candidate;
        break;
      }
    }
    if (!oldest) break;
    dropThumbnail(oldest[0], oldest[1]);
    void oldest[1].url.then(revokeIfObjectUrl).catch(() => {});
  }
}

interface GalleryMediaTicket {
  token: string | null;
  expires_at: number | null;
  auth_required?: boolean;
}

/** Thrown when an authenticated host cannot issue the required media ticket. */
export class MediaUpgradeRequiredError extends Error {
  constructor() {
    super("Upgrade this Mold host before loading authenticated gallery media.");
    this.name = "MediaUpgradeRequiredError";
  }
}

/**
 * A URL an `<img>`/`<video>` can load directly for full-size media. Keyless
 * hosts use the plain direct URL (Range-friendly). Authenticated hosts exchange
 * the key for a short-lived read ticket appended to the URL.
 */
export async function resolveStreamableSrc(
  host: HostEntry,
  filename: string,
): Promise<string> {
  const path = mediaPath(filename);
  const directUrl = directMediaUrl(host, filename);
  if (!needsAuthedMedia(host)) return directUrl;

  const res = await fetch(`${hostMediaBase(host)}/api/gallery/media-token`, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeaders(host) },
    body: JSON.stringify({ path }),
  });
  if (res.status === 404 || res.status === 405) {
    throw new MediaUpgradeRequiredError();
  }
  if (!res.ok) throw new Error(`media-token failed: ${res.status}`);
  const ticket = (await res.json()) as GalleryMediaTicket;
  if (ticket.auth_required === false) return directUrl;
  if (!ticket.token || !Number.isSafeInteger(ticket.expires_at)) {
    throw new Error("host returned an invalid gallery media ticket");
  }
  const url = new URL(directUrl, hostMediaBase(host) || window.location.origin);
  url.searchParams.set("media_token", ticket.token);
  url.searchParams.set("expires", String(ticket.expires_at));
  return url.toString();
}

/** Drop every cached object URL for one host (e.g. host removed / refetched). */
export function evictHostMedia(hostId: string): void {
  const prefix = `${hostId}|`;
  // Deleting the current entry during Map iteration is well-defined.
  for (const [key, cached] of cache) {
    if (!key.startsWith(prefix)) continue;
    dropThumbnail(key, cached);
    void cached.url.then((u) => revokeIfObjectUrl(u)).catch(() => {});
  }
  void persistentStore()?.evictPrefix(prefix);
}

function revokeIfObjectUrl(url: string): void {
  if (url.startsWith("blob:")) URL.revokeObjectURL(url);
}

/** Test hook — clear the object-URL cache without touching the DOM. */
export function __resetGalleryMediaForTests(): void {
  for (const cached of cache.values()) {
    void cached.url.then(revokeIfObjectUrl).catch(() => {});
  }
  cache.clear();
  retainedBytes = 0;
  persistent = undefined;
}
