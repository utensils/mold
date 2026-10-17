/*
 * The phone's per-host gallery replicas (`@studio/lib/galleryReplica`),
 * persisted to the IndexedDB gallery cache.
 *
 * One replica per host route — the cache key (server instance, else host
 * id) plus URL, credential and connection mode, so a re-keyed or re-pointed
 * machine never resumes another's cursor. A host advertising
 * `gallery.changes_feed` is pulled O(delta); any other host keeps its plain
 * `/api/gallery` listing. Persistence writes only after a pull that changed
 * something, behind the same host fence every other cache writer honours, so
 * an idle Library never rewrites its IndexedDB record and a host cleared
 * mid-pull is never resurrected.
 */
import {
  GalleryReplica,
  type GalleryReplicaPersistence,
  type PersistedGalleryReplica,
} from "@studio/lib/galleryReplica";
import { fetchGalleryChanges, galleryChangesFeedAdvertised } from "@studio/api/galleryChanges";
import { apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { ServerCapabilities } from "../lib/api/types";
import {
  captureCachedHostFence,
  clearCachedGalleryHosts,
  loadCachedGalleryReplica,
  storeCachedGalleryReplica,
  type CachedHostFence,
} from "./galleryCache";
import type { MobileGalleryImage } from "./libraryOrganization";

export interface MobileReplicaRoute {
  cacheKey: string;
  target: ApiTarget;
  capabilities: () => ServerCapabilities | null | undefined;
  /** The deadline the caller's reads run under (its own abort controller). */
  signal: AbortSignal;
}

interface Slot {
  identity: string;
  replica: GalleryReplica<MobileGalleryImage>;
  /** The latest caller's route: its deadline and cache fence. */
  route: MobileReplicaRoute;
  fence: CachedHostFence;
}

const slots = new Map<string, Slot>();

function slotFor(route: MobileReplicaRoute): Slot {
  const { cacheKey, target } = route;
  const identity = JSON.stringify([target.baseUrl, target.apiKey, target.connectionMode ?? null]);
  const existing = slots.get(cacheKey);
  if (existing?.identity === identity) {
    existing.route = route;
    return existing;
  }
  existing?.replica.dispose();
  const persistence: GalleryReplicaPersistence<MobileGalleryImage> = {
    load: async () => {
      const saved = await loadCachedGalleryReplica(cacheKey);
      if (!saved) return null;
      return {
        version: 1,
        cursor: saved.cursor,
        instanceId: saved.instanceId,
        rows: saved.rows as MobileGalleryImage[],
      } satisfies PersistedGalleryReplica<MobileGalleryImage>;
    },
    save: (state) => storeCachedGalleryReplica(cacheKey, state, slot.fence),
  };
  const replica = new GalleryReplica<MobileGalleryImage>(
    {
      feedSupported: () => galleryChangesFeedAdvertised(slot.route.capabilities() ?? undefined),
      changes: (since, limit) =>
        fetchGalleryChanges<MobileGalleryImage>(target, since, limit, {
          signal: slot.route.signal,
        }),
      legacyLibrary: () =>
        apiJsonTo<MobileGalleryImage[]>(target, "/api/gallery", { signal: slot.route.signal }),
    },
    { persistence },
  );
  const slot: Slot = { identity, replica, route, fence: captureCachedHostFence(cacheKey) };
  slots.set(cacheKey, slot);
  return slot;
}

/**
 * Pull a host's replica (O(delta) on a feed host) under a fresh cache fence:
 * the save at the end of this pull lands only if no clear, removal or
 * optimistic edit of that host's record happened meanwhile.
 */
export async function pullMobileGalleryReplica(
  route: MobileReplicaRoute,
): Promise<GalleryReplica<MobileGalleryImage>> {
  const slot = slotFor(route);
  slot.fence = captureCachedHostFence(route.cacheKey);
  await slot.replica.pull();
  return slot.replica;
}

/** A host's replica with its saved rows loaded — what a cold Library or the
 * arrivals observer paints before the network answers. */
export async function hydratedMobileGalleryReplica(
  route: MobileReplicaRoute,
): Promise<GalleryReplica<MobileGalleryImage>> {
  const slot = slotFor(route);
  await slot.replica.hydrate();
  return slot.replica;
}

/** Forget hosts' replicas AND their cached records (host removal, a new
 * instance behind the same machine). */
export async function clearMobileGalleryHosts(cacheKeys: readonly string[]): Promise<void> {
  for (const key of cacheKeys) {
    slots.get(key)?.replica.dispose();
    slots.delete(key);
  }
  await clearCachedGalleryHosts(cacheKeys);
}
