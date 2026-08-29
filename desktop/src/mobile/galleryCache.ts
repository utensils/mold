import type { GalleryImage, ModelEntry, ServerCapabilities } from "../lib/api/types";
import type { MobileGalleryImage } from "./libraryOrganization";

const DB_NAME = "mold-mobile-gallery-cache";
const DB_VERSION = 5;
const GALLERIES_STORE = "galleries";
const MEDIA_STORE = "media";
const PRESENTATIONS_STORE = "host-presentations";
const STATS_STORE = "cache-stats";
const THUMBNAIL_STATS_ID = "thumbnail";
const MAX_PRINTS_PER_HOST = 4_000;
const MAX_THUMBNAIL_RECORDS = 4_000;
const MAX_THUMBNAIL_BYTES = 256 * 1024 * 1024;
export const MAX_CACHED_THUMBNAIL_BYTES = 2 * 1024 * 1024;
const TOUCH_INTERVAL_MS = 60 * 60 * 1_000;
const REPAIR_EVERY_WRITES = 64;
const mediaMutationVersions = new Map<string, number>();
const hostMutationVersions = new Map<string, number>();

export type MobileThumbnailTier = 256 | 512;
export type CachedGalleryMediaRef = {
  hostId: string;
  filename: string;
  mediaVersion: string;
  tier: MobileThumbnailTier;
};

export function createThumbnailRouteGenerationRegistry(): (
  hostId: string,
  target: { baseUrl: string; apiKey?: string | null },
) => number {
  const hosts = new Map<
    string,
    {
      nextGeneration: number;
      routes: Array<{ baseUrl: string; apiKey: string | null; generation: number }>;
    }
  >();
  return (hostId, target) => {
    let state = hosts.get(hostId);
    if (!state) {
      state = { nextGeneration: 1, routes: [] };
      hosts.set(hostId, state);
    }
    const apiKey = target.apiKey ?? null;
    const prior = state.routes.find(
      (route) => route.baseUrl === target.baseUrl && route.apiKey === apiKey,
    );
    if (prior) return prior.generation;
    const generation = state.nextGeneration++;
    state.routes.push({ baseUrl: target.baseUrl, apiKey, generation });
    return generation;
  };
}

type CachedMediaKey = [string, string, string, MobileThumbnailTier];

interface CachedGalleryRecord {
  hostId: string;
  updatedAt: number;
  prints: GalleryImage[];
}

/** Non-secret host state needed to interpret a saved print while its machine
 * is offline. The key is the same stable instance identity as gallery rows. */
export interface CachedHostPresentation {
  hostId: string;
  updatedAt: number;
  instanceId: string | null;
  serverVersion: string | null;
  models: ModelEntry[];
  capabilities: ServerCapabilities | null;
}

export interface CachedHostFence {
  hostId: string;
  version: number;
}

interface CachedMediaRecord extends CachedGalleryMediaRef {
  key: CachedMediaKey;
  cachedAt: number;
  bytes: ArrayBuffer;
  mimeType: string;
  size: number;
}

interface ThumbnailCacheStats {
  id: typeof THUMBNAIL_STATS_ID;
  count: number;
  bytes: number;
}

function mediaKey(ref: CachedGalleryMediaRef): CachedMediaKey {
  return [ref.hostId, ref.filename, ref.mediaVersion, ref.tier];
}

function mediaMutationKey(hostId: string, filename: string): string {
  return `${hostId}\u0000${filename}`;
}

function sameMediaKey(left: IDBValidKey, right: CachedMediaKey): boolean {
  return (
    Array.isArray(left) &&
    left.length === right.length &&
    left.every((part, index) => part === right[index])
  );
}

function finiteSize(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

let databaseFactory: IDBFactory | null = null;
let databasePromise: Promise<IDBDatabase | null> | null = null;
let openDatabase: IDBDatabase | null = null;
let mediaWriteTail: Promise<void> = Promise.resolve();
let successfulMediaWrites = 0;

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  if (databaseFactory !== indexedDB) {
    openDatabase?.close();
    openDatabase = null;
    databasePromise = null;
    databaseFactory = indexedDB;
    successfulMediaWrites = 0;
  }
  if (databasePromise) return databasePromise;
  databasePromise = new Promise((resolve) => {
    let settled = false;
    const finish = (db: IDBDatabase | null) => {
      if (settled) {
        db?.close();
        return;
      }
      settled = true;
      resolve(db);
    };
    let request: IDBOpenDBRequest;
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      finish(null);
      return;
    }
    request.onupgradeneeded = (event) => {
      const db = request.result;
      if (!db.objectStoreNames.contains(GALLERIES_STORE)) {
        db.createObjectStore(GALLERIES_STORE, { keyPath: "hostId" });
      }
      let media: IDBObjectStore;
      if (!db.objectStoreNames.contains(MEDIA_STORE)) {
        media = db.createObjectStore(MEDIA_STORE, { keyPath: "key" });
      } else {
        media = request.transaction!.objectStore(MEDIA_STORE);
      }
      if (!media.indexNames.contains("cachedAt")) media.createIndex("cachedAt", "cachedAt");
      if (!media.indexNames.contains("cachedAtSize")) {
        media.createIndex("cachedAtSize", ["cachedAt", "size"]);
      }
      if (!media.indexNames.contains("hostId")) media.createIndex("hostId", "hostId");
      if (!media.indexNames.contains("hostFile")) {
        media.createIndex("hostFile", ["hostId", "filename"]);
      }
      if (!db.objectStoreNames.contains(PRESENTATIONS_STORE)) {
        db.createObjectStore(PRESENTATIONS_STORE, { keyPath: "hostId" });
      }
      let stats: IDBObjectStore;
      if (!db.objectStoreNames.contains(STATS_STORE)) {
        stats = db.createObjectStore(STATS_STORE, { keyPath: "id" });
      } else {
        stats = request.transaction!.objectStore(STATS_STORE);
      }
      // v4 media keys had no content version or rendition. They cannot be
      // migrated honestly, so keep metadata/presentations and rebuild tiles.
      if ((event as IDBVersionChangeEvent).oldVersion < 5) media.clear();
      stats.put({ id: THUMBNAIL_STATS_ID, count: 0, bytes: 0 });
    };
    request.onsuccess = () => {
      openDatabase = request.result;
      openDatabase.onversionchange = () => {
        openDatabase?.close();
        openDatabase = null;
        databasePromise = null;
      };
      finish(request.result);
    };
    request.onerror = () => {
      databasePromise = null;
      finish(null);
    };
    request.onblocked = () => {
      databasePromise = null;
      finish(null);
    };
  });
  return databasePromise;
}

async function requestFromStore<T>(
  storeName: string,
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: T | null) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    try {
      const transaction = db.transaction(storeName, mode);
      const request = operation(transaction.objectStore(storeName));
      let value: T | null = null;
      request.onsuccess = () => {
        value = request.result ?? null;
      };
      request.onerror = () => finish(null);
      transaction.oncomplete = () => finish(value);
      transaction.onerror = () => finish(null);
      transaction.onabort = () => finish(null);
    } catch {
      finish(null);
    }
  });
}

function enqueueMediaWrite(operation: (db: IDBDatabase) => Promise<void>): Promise<void> {
  const task = mediaWriteTail.then(async () => {
    const db = await openDb();
    if (db) await operation(db);
  });
  mediaWriteTail = task.catch(() => {});
  return task.catch(() => {});
}

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return String.fromCharCode(...bytes.slice(start, end));
}

function sniffThumbnailBytes(bytes: Uint8Array): boolean {
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return true;
  }
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return true;
  if (ascii(bytes, 0, 4) === "GIF8") return true;
  if (ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WEBP") return true;
  const text = new TextDecoder().decode(bytes.slice(0, 256)).replace(/^\uFEFF|^[\s]+/u, "");
  return text.startsWith("<svg") || text.startsWith("<?xml");
}

function validStoredMedia(record: CachedMediaRecord | null): record is CachedMediaRecord {
  if (!record?.bytes || !record.mimeType?.startsWith("image/")) return false;
  const size = record.bytes.byteLength;
  if (size <= 0 || size > MAX_CACHED_THUMBNAIL_BYTES || size !== record.size) return false;
  return sniffThumbnailBytes(new Uint8Array(record.bytes, 0, Math.min(256, size)));
}

export async function validGalleryThumbnailBlob(blob: Blob): Promise<boolean> {
  if (!blob.type.startsWith("image/") || blob.size <= 0 || blob.size > MAX_CACHED_THUMBNAIL_BYTES) {
    return false;
  }
  const head = new Uint8Array(await blob.slice(0, 256).arrayBuffer());
  return sniffThumbnailBytes(head);
}

export async function loadCachedGallery(hostId: string): Promise<GalleryImage[]> {
  const record = await requestFromStore<CachedGalleryRecord>(GALLERIES_STORE, "readonly", (store) =>
    store.get(hostId),
  );
  return Array.isArray(record?.prints) ? record.prints : [];
}

export async function storeCachedGallery(
  hostId: string,
  prints: readonly GalleryImage[],
  fence: CachedHostFence = captureCachedHostFence(hostId),
): Promise<void> {
  const bounded = [...prints]
    .sort((left, right) => right.timestamp - left.timestamp)
    .slice(0, MAX_PRINTS_PER_HOST);
  await requestFromStore(GALLERIES_STORE, "readwrite", (store) =>
    fence.hostId === hostId && (hostMutationVersions.get(hostId) ?? 0) === fence.version
      ? store.put({ hostId, updatedAt: Date.now(), prints: bounded })
      : store.get(hostId),
  );
}

export async function loadCachedHostPresentation(
  hostId: string,
): Promise<CachedHostPresentation | null> {
  const record = await requestFromStore<CachedHostPresentation>(
    PRESENTATIONS_STORE,
    "readonly",
    (store) => store.get(hostId),
  );
  return record && Array.isArray(record.models) ? record : null;
}

export async function storeCachedHostPresentation(
  presentation: CachedHostPresentation,
  fence: CachedHostFence = captureCachedHostFence(presentation.hostId),
): Promise<void> {
  await requestFromStore(PRESENTATIONS_STORE, "readwrite", (store) =>
    fence.hostId === presentation.hostId &&
    (hostMutationVersions.get(presentation.hostId) ?? 0) === fence.version
      ? store.put(presentation)
      : store.get(presentation.hostId),
  );
}

/** Capture before a network refresh. A later clear/removal advances this
 * generation so the response cannot recreate a purged instance record. */
export function captureCachedHostFence(hostId: string): CachedHostFence {
  return { hostId, version: hostMutationVersions.get(hostId) ?? 0 };
}

export function isCachedHostFenceCurrent(fence: CachedHostFence): boolean {
  return (hostMutationVersions.get(fence.hostId) ?? 0) === fence.version;
}

function repairStatsInTransaction(media: IDBObjectStore, stats: IDBObjectStore): void {
  let retainedCount = 0;
  let retainedBytes = 0;
  const cursorRequest = media.index("cachedAtSize").openKeyCursor(null, "prev");
  cursorRequest.onsuccess = () => {
    const cursor = cursorRequest.result;
    if (!cursor) {
      stats.put({ id: THUMBNAIL_STATS_ID, count: retainedCount, bytes: retainedBytes });
      return;
    }
    const indexKey = cursor.key as [number, number];
    const size = finiteSize(indexKey[1]);
    if (retainedCount + 1 <= MAX_THUMBNAIL_RECORDS && retainedBytes + size <= MAX_THUMBNAIL_BYTES) {
      retainedCount += 1;
      retainedBytes += size;
    } else {
      cursor.delete();
    }
    cursor.continue();
  };
}

export function pruneCachedGalleryMedia(): Promise<void> {
  return enqueueMediaWrite(
    (db) =>
      new Promise((resolve) => {
        try {
          const transaction = db.transaction([MEDIA_STORE, STATS_STORE], "readwrite");
          repairStatsInTransaction(
            transaction.objectStore(MEDIA_STORE),
            transaction.objectStore(STATS_STORE),
          );
          transaction.oncomplete = () => resolve();
          transaction.onerror = transaction.onabort = () => resolve();
        } catch {
          resolve();
        }
      }),
  );
}

function touchCachedGalleryMedia(
  ref: CachedGalleryMediaRef,
  expectedCachedAt: number,
  hostVersion: number,
  mutationVersion: number,
): void {
  void enqueueMediaWrite(
    (db) =>
      new Promise((resolve) => {
        const mutationKey = mediaMutationKey(ref.hostId, ref.filename);
        if (
          (hostMutationVersions.get(ref.hostId) ?? 0) !== hostVersion ||
          (mediaMutationVersions.get(mutationKey) ?? 0) !== mutationVersion
        ) {
          resolve();
          return;
        }
        try {
          const transaction = db.transaction(MEDIA_STORE, "readwrite");
          const store = transaction.objectStore(MEDIA_STORE);
          const request = store.get(mediaKey(ref));
          request.onsuccess = () => {
            const record = request.result as CachedMediaRecord | undefined;
            if (
              record &&
              record.cachedAt === expectedCachedAt &&
              (hostMutationVersions.get(ref.hostId) ?? 0) === hostVersion &&
              (mediaMutationVersions.get(mutationKey) ?? 0) === mutationVersion
            ) {
              store.put({ ...record, cachedAt: Date.now() });
            }
          };
          transaction.oncomplete = () => resolve();
          transaction.onerror = transaction.onabort = () => resolve();
        } catch {
          resolve();
        }
      }),
  );
}

export async function loadCachedGalleryMedia(ref: CachedGalleryMediaRef): Promise<Blob | null> {
  const hostVersion = hostMutationVersions.get(ref.hostId) ?? 0;
  const mutationKey = mediaMutationKey(ref.hostId, ref.filename);
  const mutationVersion = mediaMutationVersions.get(mutationKey) ?? 0;
  const record = await requestFromStore<CachedMediaRecord>(MEDIA_STORE, "readonly", (store) =>
    store.get(mediaKey(ref)),
  );
  if (!validStoredMedia(record)) {
    if (record) void evictCachedGalleryMediaEntry(ref);
    return null;
  }
  if (Date.now() - record.cachedAt >= TOUCH_INTERVAL_MS) {
    touchCachedGalleryMedia(ref, record.cachedAt, hostVersion, mutationVersion);
  }
  return new Blob([record.bytes], { type: record.mimeType });
}

export async function probeCachedGalleryMedia(
  refs: readonly CachedGalleryMediaRef[],
): Promise<boolean[]> {
  const db = await openDb();
  if (!db || refs.length === 0) return refs.map(() => false);
  return new Promise((resolve) => {
    try {
      const transaction = db.transaction(MEDIA_STORE, "readonly");
      const store = transaction.objectStore(MEDIA_STORE);
      const hits = refs.map(() => false);
      refs.forEach((ref, index) => {
        const request = store.getKey(mediaKey(ref));
        request.onsuccess = () => {
          hits[index] = request.result !== undefined;
        };
      });
      transaction.oncomplete = () => resolve(hits);
      transaction.onerror = transaction.onabort = () => resolve(refs.map(() => false));
    } catch {
      resolve(refs.map(() => false));
    }
  });
}

function putCachedMediaRecord(db: IDBDatabase, record: CachedMediaRecord): Promise<boolean> {
  return new Promise((resolve) => {
    let stored = false;
    try {
      const transaction = db.transaction([MEDIA_STORE, STATS_STORE], "readwrite");
      const media = transaction.objectStore(MEDIA_STORE);
      const statsStore = transaction.objectStore(STATS_STORE);
      const oldRequest = media.get(record.key);
      const statsRequest = statsStore.get(THUMBNAIL_STATS_ID);
      let oldReady = false;
      let statsReady = false;
      let oldRecord: CachedMediaRecord | null = null;
      let stats: ThumbnailCacheStats = { id: THUMBNAIL_STATS_ID, count: 0, bytes: 0 };
      const begin = () => {
        if (!oldReady || !statsReady) return;
        let projectedCount = Math.max(0, stats.count) + (oldRecord ? 0 : 1);
        let projectedBytes = Math.max(0, stats.bytes) - finiteSize(oldRecord?.size) + record.size;
        media.put(record);
        const cursorRequest = media.index("cachedAtSize").openKeyCursor(null, "next");
        cursorRequest.onsuccess = () => {
          const cursor = cursorRequest.result;
          if (
            !cursor ||
            (projectedCount <= MAX_THUMBNAIL_RECORDS && projectedBytes <= MAX_THUMBNAIL_BYTES)
          ) {
            statsStore.put({
              id: THUMBNAIL_STATS_ID,
              count: projectedCount,
              bytes: projectedBytes,
            });
            stored = true;
            return;
          }
          const indexKey = cursor.key as [number, number];
          if (!sameMediaKey(cursor.primaryKey, record.key)) {
            cursor.delete();
            projectedCount = Math.max(0, projectedCount - 1);
            projectedBytes = Math.max(0, projectedBytes - finiteSize(indexKey[1]));
          }
          cursor.continue();
        };
      };
      oldRequest.onsuccess = () => {
        oldRecord = (oldRequest.result as CachedMediaRecord | undefined) ?? null;
        oldReady = true;
        begin();
      };
      statsRequest.onsuccess = () => {
        const saved = statsRequest.result as ThumbnailCacheStats | undefined;
        if (saved) stats = saved;
        statsReady = true;
        begin();
      };
      transaction.oncomplete = () => resolve(stored);
      transaction.onerror = transaction.onabort = () => resolve(false);
    } catch {
      resolve(false);
    }
  });
}

export async function storeCachedGalleryMedia(
  ref: CachedGalleryMediaRef,
  blob: Blob,
): Promise<void> {
  if (!(await validGalleryThumbnailBlob(blob))) return;
  const mutationKey = mediaMutationKey(ref.hostId, ref.filename);
  const version = mediaMutationVersions.get(mutationKey) ?? 0;
  const hostVersion = hostMutationVersions.get(ref.hostId) ?? 0;
  const bytes = await blob.arrayBuffer();
  let stored = false;
  await enqueueMediaWrite(async (db) => {
    if (
      (mediaMutationVersions.get(mutationKey) ?? 0) !== version ||
      (hostMutationVersions.get(ref.hostId) ?? 0) !== hostVersion
    ) {
      return;
    }
    stored = await putCachedMediaRecord(db, {
      ...ref,
      key: mediaKey(ref),
      cachedAt: Date.now(),
      bytes,
      mimeType: blob.type,
      size: bytes.byteLength,
    });
  });
  if (!stored) return;
  successfulMediaWrites += 1;
  if (successfulMediaWrites % REPAIR_EVERY_WRITES === 0) void pruneCachedGalleryMedia();
}

function deleteMediaByIndex(indexName: "hostId" | "hostFile", query: IDBValidKey): Promise<void> {
  return enqueueMediaWrite(
    (db) =>
      new Promise((resolve) => {
        try {
          const transaction = db.transaction([MEDIA_STORE, STATS_STORE], "readwrite");
          const media = transaction.objectStore(MEDIA_STORE);
          const stats = transaction.objectStore(STATS_STORE);
          const hostFile = indexName === "hostFile" ? (query as [string, string]) : null;
          const request = media.index("hostId").getAllKeys(hostFile?.[0] ?? query);
          request.onsuccess = () => {
            for (const primaryKey of request.result) {
              if (!hostFile || (Array.isArray(primaryKey) && primaryKey[1] === hostFile[1])) {
                media.delete(primaryKey);
              }
            }
            repairStatsInTransaction(media, stats);
          };
          transaction.oncomplete = () => resolve();
          transaction.onerror = transaction.onabort = () => resolve();
        } catch {
          resolve();
        }
      }),
  );
}

export function evictCachedGalleryMediaEntry(ref: CachedGalleryMediaRef): Promise<void> {
  const mutationKey = mediaMutationKey(ref.hostId, ref.filename);
  mediaMutationVersions.set(mutationKey, (mediaMutationVersions.get(mutationKey) ?? 0) + 1);
  return enqueueMediaWrite(
    (db) =>
      new Promise((resolve) => {
        try {
          const transaction = db.transaction(MEDIA_STORE, "readwrite");
          transaction.objectStore(MEDIA_STORE).delete(mediaKey(ref));
          transaction.oncomplete = () => {
            void pruneCachedGalleryMedia();
            resolve();
          };
          transaction.onerror = transaction.onabort = () => resolve();
        } catch {
          resolve();
        }
      }),
  );
}

/** Remove live gallery rows without dropping thumbnail renditions. Moving a
 * print to Trash uses this path so its cached tile survives there. */
export async function removeCachedGalleryRows(
  removed: readonly { hostId: string; filename: string }[],
): Promise<void> {
  for (const hostId of new Set(removed.map(({ hostId }) => hostId))) {
    hostMutationVersions.set(hostId, (hostMutationVersions.get(hostId) ?? 0) + 1);
  }
  const filenamesByHost = new Map<string, Set<string>>();
  for (const { hostId, filename } of removed) {
    const filenames = filenamesByHost.get(hostId) ?? new Set<string>();
    filenames.add(filename);
    filenamesByHost.set(hostId, filenames);
  }
  for (const [hostId, filenames] of filenamesByHost) {
    const prints = await loadCachedGallery(hostId);
    await storeCachedGallery(
      hostId,
      prints.filter((print) => !filenames.has(print.filename)),
    );
  }
}

export async function evictCachedGalleryMedia(
  removed: readonly { hostId: string; filename: string }[],
): Promise<void> {
  const unique = new Map<string, { hostId: string; filename: string }>();
  for (const ref of removed) unique.set(mediaMutationKey(ref.hostId, ref.filename), ref);
  for (const [key, { hostId, filename }] of unique) {
    mediaMutationVersions.set(key, (mediaMutationVersions.get(key) ?? 0) + 1);
    await deleteMediaByIndex("hostFile", [hostId, filename]);
  }
}

/** Hard-delete compatibility helper: remove both saved rows and every cached
 * rendition. Trash-capable callers must use `removeCachedGalleryRows`. */
export async function removeCachedGalleryPrints(
  removed: readonly { hostId: string; filename: string }[],
): Promise<void> {
  await removeCachedGalleryRows(removed);
  await evictCachedGalleryMedia(removed);
}

/**
 * Apply organization edits (title / favorite / tags / collections) to cached
 * rows so an offline reopen shows what the user last saw. Bumps the host
 * fence first, so a gallery listing fetched BEFORE the edit and still in
 * flight cannot overwrite the patched record with its stale copy.
 */
export async function patchCachedGalleryPrints(
  hostId: string,
  patches: readonly { filename: string; patch: Partial<MobileGalleryImage> }[],
): Promise<void> {
  if (patches.length === 0) return;
  hostMutationVersions.set(hostId, (hostMutationVersions.get(hostId) ?? 0) + 1);
  const byFilename = new Map(patches.map(({ filename, patch }) => [filename, patch]));
  const prints = await loadCachedGallery(hostId);
  let changed = false;
  const next = prints.map((print) => {
    const patch = byFilename.get(print.filename);
    if (!patch) return print;
    changed = true;
    return { ...print, ...patch };
  });
  if (changed) await storeCachedGallery(hostId, next);
}

export async function clearCachedGalleryHosts(hostIds: readonly string[]): Promise<void> {
  const ids = [...new Set(hostIds.filter(Boolean))];
  if (ids.length === 0) return;
  for (const hostId of ids) {
    hostMutationVersions.set(hostId, (hostMutationVersions.get(hostId) ?? 0) + 1);
  }
  for (const hostId of ids) {
    await Promise.all([
      requestFromStore(GALLERIES_STORE, "readwrite", (store) => store.delete(hostId)),
      requestFromStore(PRESENTATIONS_STORE, "readwrite", (store) => store.delete(hostId)),
    ]);
    await deleteMediaByIndex("hostId", hostId);
  }
}
