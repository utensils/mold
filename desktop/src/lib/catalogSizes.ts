/**
 * Lazy size resolution for catalog cards.
 *
 * HF search-summary rows arrive without `size_bytes` (the live proxy skips
 * the per-repo tree fetch on the hot search path). Cards resolve the real
 * size on demand via the single-entry endpoint `GET /api/catalog/{id}` —
 * the id goes RAW in the path (colons and slashes are part of the wildcard
 * route match, same as `startCatalogDownload`).
 *
 * Successes are memoized per id for the session so scrolling back through
 * results never refetches, and lookups are capped at 4 in flight so a page
 * of 24 HF rows doesn't burst two dozen upstream tree fetches at once.
 * Failures resolve as null ("no size line", never an error state) but are
 * NOT cached — a transient upstream hiccup shouldn't hide sizes for the
 * whole session, so a later render retries.
 */
import { fetchCatalogDetail } from "./api/catalog";
import type { CatalogEntry } from "./api/types";

const MAX_IN_FLIGHT = 4;

let cache = new Map<string, Promise<number | null>>();
let active = 0;
let queue: Array<() => void> = [];

function acquire(): Promise<void> {
  if (active < MAX_IN_FLIGHT) {
    active += 1;
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    queue.push(() => {
      active += 1;
      resolve();
    });
  });
}

function release(): void {
  active -= 1;
  queue.shift()?.();
}

async function fetchSizeBytes(id: string): Promise<number | null> {
  await acquire();
  try {
    // Read through the shared detail fetch (`lib/api/catalog.ts`) so the
    // drawer and size resolution share one `GET /api/catalog/:id` code path.
    const detail = await fetchCatalogDetail(id);
    return detail.size_bytes ?? null;
  } finally {
    release();
  }
}

/** Resolve an entry's weights size, fetching the single-entry detail when absent. */
export function resolveEntrySize(entry: CatalogEntry): Promise<number | null> {
  if (entry.size_bytes != null) return Promise.resolve(entry.size_bytes);
  const hit = cache.get(entry.id);
  if (hit) return hit;
  const pending = fetchSizeBytes(entry.id).catch(() => {
    // Concurrent callers share this in-flight promise, but the failure is
    // evicted so the next render retries instead of staying blank all
    // session.
    cache.delete(entry.id);
    return null;
  });
  cache.set(entry.id, pending);
  return pending;
}

/** Test hook — drops memoized results and any queued lookups. */
export function resetCatalogSizeCache(): void {
  cache = new Map();
  active = 0;
  queue = [];
}
