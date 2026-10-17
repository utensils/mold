/*
 * Desktop bindings for the shared gallery replica (`@studio/lib/galleryReplica`).
 *
 * Every gallery bucket owns one replica, keyed by the bucket key and the
 * authority it reads (URL + key + connection mode + instance for a remote;
 * "local" for this device). Remote buckets follow `GET /api/gallery/changes`
 * when the host advertises `gallery.changes_feed`, else the conditional
 * `/api/gallery`. This device's bucket rides the `local_gallery_changes` IPC
 * (the feed over the running engine's authenticated HTTP) and falls back to
 * `local_gallery_list` — the same singular authority rule as every other
 * native gallery path: an engine that is Off answers `offline`, never a feed.
 *
 * The registry lives per store instance (keyed on the RAW store, never the
 * Pinia proxy `this`, which differs per action call in a dev build).
 */
import {
  GalleryFeedUnavailableError,
  GalleryReplica,
  type GalleryReplicaTransport,
  type ReplicaDelta,
} from "@studio/lib/galleryReplica";
import {
  fetchGalleryChanges,
  galleryChangesFeedAdvertised,
  parseGalleryChangesPage,
} from "@studio/api/galleryChanges";
import { conditionalApiJsonTo, type ApiTarget } from "../api/client";
import { ipc } from "../ipc";
import type { GalleryImage, ServerCapabilities } from "../api/types";

export interface BucketReplicaSlot {
  identity: string;
  /** Unique per replica instance, so a stamp never survives a swap. */
  epoch: number;
  replica: GalleryReplica<GalleryImage>;
  /** The authority the last local read went through (`null` = Off). */
  authority: { target: ApiTarget | null };
  /** Deltas published since the bucket last applied this replica. */
  pending: ReplicaDelta<GalleryImage>[];
  /** The bucket stamp `pending` applies on top of. */
  pendingBase: string | null;
}

let nextEpoch = 1;
const registries = new WeakMap<object, Map<string, BucketReplicaSlot>>();

function registryFor(owner: object): Map<string, BucketReplicaSlot> {
  let registry = registries.get(owner);
  if (!registry) {
    registry = new Map();
    registries.set(owner, registry);
  }
  return registry;
}

/** The stamp a bucket records once it reflects `slot`'s replica. */
export function replicaStamp(slot: BucketReplicaSlot): string {
  return `${slot.epoch}:${slot.replica.revision}`;
}

function slotFor(
  owner: object,
  key: string,
  identity: string,
  transport: (authority: { target: ApiTarget | null }) => GalleryReplicaTransport<GalleryImage>,
): BucketReplicaSlot {
  const registry = registryFor(owner);
  const existing = registry.get(key);
  if (existing?.identity === identity) return existing;
  existing?.replica.dispose();
  const authority = { target: null as ApiTarget | null };
  const replica = new GalleryReplica<GalleryImage>(transport(authority));
  const slot: BucketReplicaSlot = {
    identity,
    epoch: nextEpoch++,
    replica,
    authority,
    pending: [],
    pendingBase: null,
  };
  replica.subscribe((delta) => slot.pending.push(delta));
  registry.set(key, slot);
  return slot;
}

export function remoteBucketReplica(
  owner: object,
  key: string,
  target: ApiTarget,
  instanceId: string | null,
  capabilities: () => ServerCapabilities | undefined,
): BucketReplicaSlot {
  const identity = JSON.stringify([
    target.baseUrl,
    target.apiKey,
    target.connectionMode ?? null,
    instanceId,
  ]);
  return slotFor(owner, key, identity, (authority) => {
    authority.target = target;
    return {
      feedSupported: () => galleryChangesFeedAdvertised(capabilities()),
      changes: (since, limit) => fetchGalleryChanges<GalleryImage>(target, since, limit),
      legacyLibrary: () => conditionalApiJsonTo<GalleryImage[]>(target, "/api/gallery"),
    };
  });
}

export function localBucketReplica(
  owner: object,
  capabilities: () => ServerCapabilities | undefined,
): BucketReplicaSlot {
  return slotFor(owner, "local", "local", (authority) => ({
    // Only the engine's own advertisement opens the feed: an unread
    // snapshot (boot, an engine that is Off) keeps the listing.
    feedSupported: () => galleryChangesFeedAdvertised(capabilities()),
    changes: async (since, limit) => {
      const answer = await ipc.localGalleryChanges(since, limit);
      authority.target = answer.target;
      if (answer.unavailable || answer.page === undefined)
        throw new GalleryFeedUnavailableError(answer.unavailable ?? "unsupported");
      return parseGalleryChangesPage<GalleryImage>(answer.page);
    },
    legacyLibrary: async () => {
      const snapshot = await ipc.localGalleryList();
      authority.target = snapshot.target;
      return snapshot.images;
    },
  }));
}

/** Drop the replicas of buckets that no longer exist. */
export function retainBucketReplicas(owner: object, keys: ReadonlySet<string>): void {
  const registry = registryFor(owner);
  for (const [key, slot] of registry) {
    if (keys.has(key)) continue;
    slot.replica.dispose();
    registry.delete(key);
  }
}

export function bucketReplicaSlot(owner: object, key: string): BucketReplicaSlot | undefined {
  return registryFor(owner).get(key);
}
