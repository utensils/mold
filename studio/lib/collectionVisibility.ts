/** Shared privacy intentions. Persist routes and installation identities, never credentials. */
import type { ApiTarget } from "../api/client";
import { updateCollectionHidden } from "../api/galleryOrganization";
import type { Collection } from "./api/galleryOrganization";
import type { MergedCollection } from "./libraryOrganization";
export type CollectionAvailability = "present" | "absent" | "unavailable";
export function collectionAvailability(
  hosts: readonly { hostId: string }[],
  hostId: string,
  listingOk: boolean,
): CollectionAvailability {
  return !listingOk
    ? "unavailable"
    : hosts.some((h) => h.hostId === hostId)
      ? "present"
      : "absent";
}
export function scopedCollectionInventory(
  collections: readonly MergedCollection[],
  hostId: string | null,
): MergedCollection[] {
  if (!hostId || hostId === "all") return [...collections];
  return collections.map((c) => ({
    ...c,
    count: c.hosts.find((h) => h.hostId === hostId)?.count ?? 0,
    cover: c.cover?.hostId === hostId ? c.cover : null,
  }));
}
export interface VisibilityHost {
  hostId: string;
  target: ApiTarget;
  instanceId: string | null;
  collections: Collection[];
  listingOk: boolean;
  readRevision?: number;
}
interface Intent {
  slug: string;
  hidden: boolean;
  revision: number;
  routes: Record<string, { url: string; instanceId: string | null }>;
}
const KEY = "mold-collection-visibility-v1";
let intents: Record<string, Intent> | null = null;
let revision = Date.now();
const latestEdits = new Map<string, { hidden: boolean; revision: number }>();
let running: Promise<string[]> | null = null;
let writes: Promise<unknown> = Promise.resolve();
export function serializeCollectionVisibility<T>(
  run: () => Promise<T>,
): Promise<T> {
  const task = writes.then(run, run);
  writes = task.catch(() => undefined);
  return task;
}
export function collectionHiddenAfterRevision(
  slug: string,
  fallback: boolean,
  capturedRevision: number,
): boolean {
  const edit = latestEdits.get(slug) ?? read()[slug];
  return edit && edit.revision > capturedRevision
    ? edit.hidden
    : desiredCollectionHidden(slug, fallback);
}
function read(): Record<string, Intent> {
  if (intents) return intents;
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    intents = {};
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
      for (const [slug, value] of Object.entries(parsed)) {
        if (
          value &&
          typeof value === "object" &&
          typeof value.hidden === "boolean" &&
          typeof value.revision === "number" &&
          value.routes &&
          typeof value.routes === "object"
        )
          intents[slug] = value;
      }
  } catch {
    intents = {};
  }
  return intents!;
}
function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(read()));
  } catch {
    /* Session intent survives unavailable storage. */
  }
}
export function rememberCollectionVisibility(
  slug: string,
  hidden: boolean,
  hosts: readonly VisibilityHost[],
): void {
  read()[slug] = {
    slug,
    hidden,
    revision: ++revision,
    routes: Object.fromEntries(
      hosts.map((h) => [
        h.hostId,
        { url: h.target.baseUrl, instanceId: h.instanceId },
      ]),
    ),
  };
  latestEdits.set(slug, { hidden, revision });
  save();
}
export function collectionVisibilityRevision(): number {
  read();
  return revision;
}
export function protectCollectionVisibilityListing(
  rows: Collection[],
  readRevision: number,
): Collection[] {
  return rows.map((row) => {
    const edit = latestEdits.get(row.slug) ?? read()[row.slug];
    return edit && edit.revision > readRevision
      ? { ...row, hidden: edit.hidden }
      : row;
  });
}
export function desiredCollectionHidden(
  slug: string,
  fallback: boolean,
): boolean {
  return read()[slug]?.hidden ?? fallback;
}
export function resetCollectionVisibilityForTests() {
  intents = null;
  running = null;
  latestEdits.clear();
}
/** Serialized replay reads latest intent before every write. A later explicit edit wins. */
export async function reconcileCollectionVisibility(
  hosts: readonly VisibilityHost[],
  liveHosts: () => readonly VisibilityHost[] = () => hosts,
): Promise<string[]> {
  if (running) {
    await running;
    return reconcileCollectionVisibility(hosts, liveHosts);
  }
  running = serializeCollectionVisibility(async () => {
    const errors: string[] = [];
    const slugs = new Set([
      ...Object.keys(read()),
      ...hosts.flatMap((h) => h.collections.map((c) => c.slug)),
    ]);
    for (const slug of slugs) {
      const intent = read()[slug];
      const hidden =
        intent?.hidden ??
        hosts.some((h) =>
          h.collections.some((c) => c.slug === slug && c.hidden),
        );
      const confirmed = new Set(
        hosts
          .filter(
            (host) =>
              intent &&
              (host.readRevision ?? -1) >= intent.revision &&
              host.listingOk &&
              host.collections
                .filter((c) => c.slug === slug)
                .every((c) => (c.hidden === true) === intent.hidden),
          )
          .map((h) => h.hostId),
      );
      for (const host of hosts) {
        const live = liveHosts().find((h) => h.hostId === host.hostId);
        if (
          !live ||
          live.target.baseUrl !== host.target.baseUrl ||
          live.instanceId !== host.instanceId
        ) {
          errors.push(
            `${host.hostId}: The machine route or installation changed; visibility was not replayed.`,
          );
          continue;
        }
        if (!host.listingOk) {
          if (intent?.routes[host.hostId])
            errors.push(
              `${host.hostId}: Collection visibility is pending until this machine reconnects.`,
            );
          continue;
        }
        const row = host.collections.find((c) => c.slug === slug);
        if (!row) continue;
        const current = read()[slug];
        if (current && current.revision !== intent?.revision) break;
        const route = current?.routes[host.hostId];
        if (
          current &&
          (!route ||
            route.url !== host.target.baseUrl ||
            route.instanceId !== host.instanceId)
        ) {
          errors.push(
            `${host.hostId}: The machine route or installation changed; visibility was not replayed.`,
          );
          continue;
        }
        if (row.hidden === hidden || (!hidden && row.hidden !== true)) continue;
        try {
          const updated = await updateCollectionHidden(
            host.target,
            row.id,
            hidden,
          );
          const after = liveHosts().find((h) => h.hostId === host.hostId);
          if (
            after?.target.baseUrl === host.target.baseUrl &&
            after.instanceId === host.instanceId
          )
            Object.assign(row, updated);
        } catch (error) {
          errors.push(`${host.hostId}: ${String(error)}`);
        }
      }
      if (
        intent &&
        read()[slug]?.revision === intent.revision &&
        Object.entries(intent.routes).every(([id, route]) => {
          const host = hosts.find((h) => h.hostId === id);
          return (
            confirmed.has(id) &&
            host?.listingOk &&
            host.target.baseUrl === route.url &&
            host.instanceId === route.instanceId &&
            host.collections
              .filter((c) => c.slug === slug)
              .every((c) => (c.hidden === true) === intent.hidden)
          );
        })
      ) {
        delete read()[slug];
        save();
      }
    }
    return errors;
  });
  try {
    return await running;
  } finally {
    running = null;
  }
}
