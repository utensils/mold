/**
 * Browser-side persistent thumbnail cache for surfaces without a native
 * disk cache (web, and the phone's WebView). Authenticated hosts cannot use
 * the browser's HTTP cache — the tile is fetched with an `x-api-key` header
 * and served as an object URL — so those bytes are kept in the Cache API
 * instead, keyed by host + filename + content version + rendition. A cold
 * load then paints the grid from the origin's storage without a request per
 * tile, matching the desktop's `thumbnail_cache.rs`.
 *
 * `CacheStorage` exists only in secure contexts (HTTPS, localhost, the app
 * shells' custom-scheme origins), so a plain-`http://` LAN origin gets
 * `null` here and keeps the in-memory blob cache. Every call is wrapped: a
 * quota error or a browser that refuses storage must never fail a tile.
 */

export const THUMBNAIL_STORE_NAME = "mold-thumbs-v1";
/** Bound on stored tiles; pruned oldest-first (insertion order) past this. */
export const THUMBNAIL_STORE_MAX_ENTRIES = 4_000;
const PRUNE_EVERY = 64;

/** The rendition a display should ask for: 512 px on retina, 256 otherwise.
 *  Two tiers only, so no cache holds one entry per slider position. */
export function thumbnailTier(devicePixelRatio = globalThis.devicePixelRatio ?? 1): 256 | 512 {
  return devicePixelRatio >= 1.5 ? 512 : 256;
}

/** Query string for the rendition (`?size=…&fmt=jpeg`); older servers ignore
 *  it and answer their 256 px PNG, which the caller still displays. */
export function thumbnailRenditionQuery(tier: 256 | 512 = thumbnailTier()): string {
  return `size=${tier}&fmt=jpeg`;
}

export interface PersistentThumbnailStore {
  get(key: string): Promise<Blob | null>;
  put(key: string, blob: Blob): Promise<void>;
  /** Drop every entry whose key starts with `prefix` (a removed host). */
  evictPrefix(prefix: string): Promise<void>;
}

/** Cache API request keys must be http(s) URLs; a reserved-TLD host keeps
 *  them from ever colliding with a real fetch. */
function requestUrlFor(key: string): string {
  return `https://mold-thumbs.invalid/${encodeURIComponent(key)}`;
}

function keyOfRequestUrl(url: string): string | null {
  const prefix = "https://mold-thumbs.invalid/";
  if (!url.startsWith(prefix)) return null;
  try {
    return decodeURIComponent(url.slice(prefix.length));
  } catch {
    return null;
  }
}

/**
 * The store for this origin, or null when the Cache API is unavailable.
 * `storage` is injectable for tests; production passes nothing and reads
 * `globalThis.caches`.
 */
export function persistentThumbnailStore(
  storage: CacheStorage | undefined = (globalThis as { caches?: CacheStorage }).caches,
  name = THUMBNAIL_STORE_NAME,
): PersistentThumbnailStore | null {
  if (!storage || typeof storage.open !== "function") return null;
  let opened: Promise<Cache | null> | null = null;
  const open = () => {
    opened ??= storage.open(name).catch(() => null);
    return opened;
  };
  let putsSincePrune = 0;
  return {
    async get(key) {
      try {
        const cache = await open();
        const hit = await cache?.match(requestUrlFor(key));
        return hit ? await hit.blob() : null;
      } catch {
        return null;
      }
    },
    async put(key, blob) {
      try {
        const cache = await open();
        if (!cache) return;
        await cache.put(
          requestUrlFor(key),
          new Response(blob, {
            headers: { "content-type": blob.type || "image/png" },
          }),
        );
        putsSincePrune += 1;
        if (putsSincePrune >= PRUNE_EVERY) {
          putsSincePrune = 0;
          const keys = await cache.keys();
          // Prune down to a headroom below the bound so the store never
          // exceeds it between two prune passes.
          const excess = keys.length - (THUMBNAIL_STORE_MAX_ENTRIES - PRUNE_EVERY);
          for (let i = 0; i < excess; i++) await cache.delete(keys[i]!);
        }
      } catch {
        // Quota or a storage-refusing browser: the tile still rendered.
      }
    },
    async evictPrefix(prefix) {
      try {
        const cache = await open();
        if (!cache) return;
        for (const request of await cache.keys()) {
          const key = keyOfRequestUrl(request.url);
          if (key !== null && key.startsWith(prefix)) await cache.delete(request);
        }
      } catch {
        // Best effort.
      }
    },
  };
}

/** One key shape for every surface: host, filename, content version, tier. */
export function persistentThumbnailKey(
  hostId: string,
  filename: string,
  mediaVersion: string,
  tier: 256 | 512,
): string {
  return `${hostId}|${filename}|${mediaVersion}|${tier}`;
}
