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
 *     without buffering it and without the durable key in the URL. Older hosts
 *     that lack the endpoint fall back to a bounded blob fetch for IMAGES only;
 *     a video on such a host surfaces an "upgrade to stream" state instead.
 *
 * The serving origin ("this server") is same-origin and keyless in the web
 * registry, so it always uses plain relative URLs — exactly today's behaviour.
 * Mirrors the desktop app's desktop/src/lib/gallery/media.ts contract.
 */
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";

/** Blob object-URL cache, keyed by `${hostId}|${path}`. Ticket URLs are not
 *  cached — they are cheap to mint and expire, and the lightbox renews on open. */
const cache = new Map<string, Promise<string>>();

const keyOf = (hostId: string, path: string) => `${hostId}|${path}`;

export function thumbnailPath(filename: string): string {
  return `/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
}

export function mediaPath(filename: string): string {
  return `/api/gallery/image/${encodeURIComponent(filename)}`;
}

/** URL prefix for a host's API. The origin is same-origin, so relative. */
export function hostMediaBase(host: HostEntry): string {
  return host.id === ORIGIN_HOST_ID ? "" : host.url.replace(/\/$/, "");
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
): Promise<string> {
  const res = await fetch(`${hostMediaBase(host)}${path}`, {
    headers: authHeaders(host),
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
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

/**
 * Grid-tile thumbnail source. Keyless hosts (origin included) use the plain
 * direct URL so the browser can lazy-load it; authenticated hosts blob-fetch
 * the small poster with the key and cache the resulting object URL per
 * (host, filename).
 */
export function resolveThumbnailSrc(
  host: HostEntry,
  filename: string,
): Promise<string> {
  const path = thumbnailPath(filename);
  if (!needsAuthedMedia(host)) {
    return Promise.resolve(`${hostMediaBase(host)}${path}`);
  }
  const key = keyOf(host.id, path);
  let url = cache.get(key);
  if (!url) {
    url = fetchAuthedObjectUrl(host, path);
    cache.set(key, url);
    url.catch(() => cache.delete(key));
  }
  return url;
}

interface GalleryMediaTicket {
  token: string | null;
  expires_at: number | null;
  auth_required?: boolean;
}

/** Thrown when a video lives on an authenticated host too old to issue a
 *  streaming ticket — the caller shows an upgrade prompt rather than buffering. */
export class MediaUpgradeRequiredError extends Error {
  constructor() {
    super("Connect a newer Mold host to stream this video.");
    this.name = "MediaUpgradeRequiredError";
  }
}

export interface StreamableOptions {
  /** Images may fall back to a bounded blob fetch on ticket-less hosts;
   *  videos must not (an unbounded buffer), so they pass this false. */
  allowLegacyBlob: boolean;
}

/**
 * A URL an `<img>`/`<video>` can load directly for full-size media. Keyless
 * hosts use the plain direct URL (Range-friendly). Authenticated hosts exchange
 * the key for a short-lived read ticket appended to the URL. Ticket-less legacy
 * hosts fall back to a blob for images; for videos they raise
 * `MediaUpgradeRequiredError`.
 */
export async function resolveStreamableSrc(
  host: HostEntry,
  filename: string,
  opts: StreamableOptions,
): Promise<string> {
  const path = mediaPath(filename);
  const directUrl = `${hostMediaBase(host)}${path}`;
  if (!needsAuthedMedia(host)) return directUrl;

  try {
    const res = await fetch(`${hostMediaBase(host)}/api/gallery/media-token`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authHeaders(host) },
      body: JSON.stringify({ path }),
    });
    if (res.status === 404 || res.status === 405) {
      return legacyStreamableFallback(host, filename, opts);
    }
    if (!res.ok) throw new Error(`media-token failed: ${res.status}`);
    const ticket = (await res.json()) as GalleryMediaTicket;
    // A key can linger after a host turns auth off; the server says so
    // explicitly so we use the plain URL instead of treating it as a failure.
    if (ticket.auth_required === false) return directUrl;
    if (!ticket.token || !Number.isSafeInteger(ticket.expires_at)) {
      throw new Error("host returned an invalid gallery media ticket");
    }
    const url = new URL(
      directUrl,
      hostMediaBase(host) || window.location.origin,
    );
    url.searchParams.set("media_token", ticket.token);
    url.searchParams.set("expires", String(ticket.expires_at));
    return url.toString();
  } catch (err) {
    if (err instanceof MediaUpgradeRequiredError) throw err;
    // Network / transport error reaching the token endpoint. Images can still
    // try the bounded blob; videos surface the upgrade state.
    return legacyStreamableFallback(host, filename, opts);
  }
}

async function legacyStreamableFallback(
  host: HostEntry,
  filename: string,
  opts: StreamableOptions,
): Promise<string> {
  if (!opts.allowLegacyBlob) throw new MediaUpgradeRequiredError();
  const path = mediaPath(filename);
  const key = keyOf(host.id, path);
  let url = cache.get(key);
  if (!url) {
    url = fetchAuthedObjectUrl(host, path);
    cache.set(key, url);
    url.catch(() => cache.delete(key));
  }
  return url;
}

/** Revoke and drop one cached object URL (host bucket + path). */
export function evictMedia(hostId: string, path: string): void {
  const key = keyOf(hostId, path);
  const cached = cache.get(key);
  cache.delete(key);
  void cached?.then((u) => revokeIfObjectUrl(u)).catch(() => {});
}

/** Drop every cached object URL for one host (e.g. host removed / refetched). */
export function evictHostMedia(hostId: string): void {
  const prefix = `${hostId}|`;
  for (const [key, cached] of [...cache]) {
    if (!key.startsWith(prefix)) continue;
    cache.delete(key);
    void cached.then((u) => revokeIfObjectUrl(u)).catch(() => {});
  }
}

function revokeIfObjectUrl(url: string): void {
  if (url.startsWith("blob:")) URL.revokeObjectURL(url);
}

/** Test hook — clear the object-URL cache without touching the DOM. */
export function __resetGalleryMediaForTests(): void {
  cache.clear();
}
