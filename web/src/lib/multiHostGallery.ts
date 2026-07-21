/*
 * Multi-host gallery merge. The web Gallery used to fetch only the origin's
 * /api/gallery while claiming "all hosts"; this reaches every host in the
 * registry (origin + stored, each with its own x-api-key), tags each print
 * with its owning host, and merges newest-first. A per-host failure degrades
 * to an "unreachable" note — the grid never blanks because one box is down.
 *
 * Dedupe is intentionally minimal: the host registry already collapses one
 * box reached by several addresses to a single entry (instance-UUID dedupe),
 * so we don't double-count instances here. The desktop app additionally
 * collapses cross-host copies of the same print by filename / seed+byte-size,
 * but that only exists because desktop auto-saves remote outputs locally; web
 * doesn't, so that heavier collapse is deliberately deferred.
 */
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
  /** Every reachable host's prints, merged newest-first, host-tagged. */
  entries: HostGalleryImage[];
  /** Hosts whose /api/gallery succeeded. */
  reachableHostIds: string[];
  /** Hosts that failed (network / auth / older server) — surfaced as an
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

  // Newest first across all hosts.
  entries.sort((a, b) => b.timestamp - a.timestamp);

  return {
    entries,
    reachableHostIds,
    unreachableHostIds,
    remoteHostCount: hosts.filter((h) => h.id !== ORIGIN_HOST_ID).length,
  };
}
