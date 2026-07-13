/**
 * Lazy size resolution for catalog cards.
 *
 * HF search-summary rows arrive without `size_bytes` (the live proxy skips
 * the per-repo tree fetch on the hot search path). Cards resolve the real
 * size on demand via the single-entry endpoint `GET /api/catalog/{id}` —
 * the id goes RAW in the path (colons and slashes are part of the wildcard
 * route match, same as `startCatalogDownload`).
 *
 * Results are memoized per id for the session so scrolling back through
 * results never refetches, and lookups are capped at 4 in flight so a page
 * of 24 HF rows doesn't burst two dozen upstream tree fetches at once.
 * Failures resolve (and stay cached) as null — an unknown size renders as
 * "no size line", never as an error state.
 */
import { apiJson } from "./api/client";
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
    const detail = await apiJson<CatalogEntry>(`/api/catalog/${id}`);
    return detail.size_bytes ?? null;
  } catch {
    return null;
  } finally {
    release();
  }
}

/** Resolve an entry's weights size, fetching the single-entry detail when absent. */
export function resolveEntrySize(entry: CatalogEntry): Promise<number | null> {
  if (entry.size_bytes != null) return Promise.resolve(entry.size_bytes);
  const hit = cache.get(entry.id);
  if (hit) return hit;
  const pending = fetchSizeBytes(entry.id);
  cache.set(entry.id, pending);
  return pending;
}

/** Test hook — drops memoized results and any queued lookups. */
export function resetCatalogSizeCache(): void {
  cache = new Map();
  active = 0;
  queue = [];
}
