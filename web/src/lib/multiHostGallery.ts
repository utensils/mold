/*
 * Multi-host gallery merge. The web Gallery used to fetch only the origin's
 * /api/gallery while claiming "all hosts"; this reaches every host in the
 * registry (origin + stored, each with its own x-api-key), tags each print
 * with its owning host, and merges newest-first. A per-host failure degrades
 * to an "unreachable" note — the grid never blanks because one box is down.
 *
 * Physical copies are retained for host filters and delete-everywhere, while
 * the All view receives one representative per logical print regardless of
 * which pair of hosts contains the copies.
 */
import { groupLogicalGalleryPrints } from "@studio/lib/galleryPrintIdentity";
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";
import { hostGallery } from "../components/machines/hostClient";
import type { GalleryImage } from "../types";

export interface HostGalleryImage extends GalleryImage {
  /** Registry id of the host this print lives on. */
  hostId: string;
  /** Display label for the per-tile host badge. */
  hostLabel: string;
}

export interface MergedGallery {
  /** Logical prints for the All view, merged newest-first and deduplicated. */
  entries: HostGalleryImage[];
  /** Every physical copy for host filters and device-routed actions. */
  rawEntries: HostGalleryImage[];
  /** Hosts whose /api/gallery succeeded. */
  reachableHostIds: string[];
  /** Hosts that failed (network, authentication, or incompatible API) — surfaced as an
   * "unreachable" chip rather than a silent gap. */
  unreachableHostIds: string[];
  /** Non-origin hosts that were attempted — drives the honest count line
   * ("all hosts" vs "this server"). */
  remoteHostCount: number;
}

/**
 * Stable identity for one print: the pair of the host that holds it and its
 * filename. mold names outputs model+seed+timestamp, so two machines routinely
 * hold the same filename — keying selection, delete routing or lightbox lookup
 * on the filename alone makes those two prints the same print, which deletes
 * the wrong file on the wrong box. Host ids are URL slugs, so "|" can't collide.
 */
export function makePrintKey(hostId: string, filename: string): string {
  return `${hostId}|${filename}`;
}

/** `makePrintKey` for a gallery entry; an untagged entry is the origin's. */
export function printKey(entry: { hostId?: string; filename: string }): string {
  return makePrintKey(entry.hostId ?? ORIGIN_HOST_ID, entry.filename);
}

export type HostGalleryFetcher = (host: HostEntry) => Promise<GalleryImage[]>;

/** Fetch + merge every host's gallery. `fetcher` is injectable for tests. */
export async function fetchMergedGallery(
  hosts: HostEntry[],
  fetcher: HostGalleryFetcher = hostGallery,
): Promise<MergedGallery> {
  const reachableHostIds: string[] = [];
  const unreachableHostIds: string[] = [];
  const entries: HostGalleryImage[] = [];

  const results = await Promise.allSettled(hosts.map((h) => fetcher(h)));
  results.forEach((result, i) => {
    const h = hosts[i];
    if (result.status === "fulfilled") {
      reachableHostIds.push(h.id);
      for (const item of result.value) {
        entries.push({ ...item, hostId: h.id, hostLabel: h.name });
      }
    } else {
      unreachableHostIds.push(h.id);
    }
  });

  // Newest first across all hosts. The first copy becomes the representative,
  // so a newer reachable copy supplies media/actions when no local preference
  // exists (web has no special local filesystem bucket).
  entries.sort((a, b) => b.timestamp - a.timestamp);
  const rawEntries = entries;
  const deduplicated = groupLogicalGalleryPrints(rawEntries).map(
    (group) => group.representative,
  );

  return {
    entries: deduplicated,
    rawEntries,
    reachableHostIds,
    unreachableHostIds,
    remoteHostCount: hosts.filter((h) => h.id !== ORIGIN_HOST_ID).length,
  };
}
