import type { GalleryImage, ModelEntry, ServerCapabilities } from "../lib/api/types";
import type { MobileGalleryImage } from "./libraryOrganization";

const DB_NAME = "mold-mobile-gallery-cache";
const DB_VERSION = 4;
const GALLERIES_STORE = "galleries";
const MEDIA_STORE = "media";
const PRESENTATIONS_STORE = "host-presentations";
const MAX_PRINTS_PER_HOST = 500;
const MAX_THUMBNAIL_RECORDS = 320;
const MAX_THUMBNAIL_BYTES = 64 * 1024 * 1024;
const mediaMutationVersions = new Map<string, number>();
const hostMutationVersions = new Map<string, number>();

type MediaKind = "thumbnail";

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

interface CachedMediaRecord {
  key: string;
  hostId: string;
  filename: string;
  kind: MediaKind;
  cachedAt: number;
  bytes: ArrayBuffer;
  mimeType: string;
  size: number;
}

function mediaKey(hostId: string, filename: string, kind: MediaKind): string {
  return `${hostId}\u0000${kind}\u0000${filename}`;
}

let databaseFactory: IDBFactory | null = null;
let databasePromise: Promise<IDBDatabase | null> | null = null;
let openDatabase: IDBDatabase | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  if (databaseFactory !== indexedDB) {
    openDatabase?.close();
    openDatabase = null;
    databasePromise = null;
    databaseFactory = indexedDB;
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
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(GALLERIES_STORE)) {
        db.createObjectStore(GALLERIES_STORE, { keyPath: "hostId" });
      }
      if (!db.objectStoreNames.contains(MEDIA_STORE)) {
        const media = db.createObjectStore(MEDIA_STORE, { keyPath: "key" });
        media.createIndex("cachedAt", "cachedAt");
        media.createIndex("hostId", "hostId");
      } else {
        const media = request.transaction!.objectStore(MEDIA_STORE);
        if (!media.indexNames.contains("hostId")) media.createIndex("hostId", "hostId");
      }
      if (!db.objectStoreNames.contains(PRESENTATIONS_STORE)) {
        db.createObjectStore(PRESENTATIONS_STORE, { keyPath: "hostId" });
      }
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

export async function loadCachedGallery(hostId: string): Promise<GalleryImage[]> {
  const record = await requestFromStore<CachedGalleryRecord>(GALLERIES_STORE, "readonly", (store) =>
    store.get(hostId),
  );
  return Array.isArray(record?.prints) ? record.prints : [];
}

export async function storeCachedGallery(
  hostId: string,
  prints: readonly GalleryImage[],
): Promise<void> {
  const hostVersion = hostMutationVersions.get(hostId) ?? 0;
  const bounded = [...prints]
    .sort((left, right) => right.timestamp - left.timestamp)
    .slice(0, MAX_PRINTS_PER_HOST);
  await requestFromStore(GALLERIES_STORE, "readwrite", (store) =>
    (hostMutationVersions.get(hostId) ?? 0) === hostVersion
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

export async function loadCachedGalleryMedia(
  hostId: string,
  filename: string,
  kind: MediaKind,
): Promise<Blob | null> {
  const record = await requestFromStore<CachedMediaRecord>(MEDIA_STORE, "readonly", (store) =>
    store.get(mediaKey(hostId, filename, kind)),
  );
  return record?.bytes ? new Blob([record.bytes], { type: record.mimeType }) : null;
}

export async function storeCachedGalleryMedia(
  hostId: string,
  filename: string,
  kind: MediaKind,
  blob: Blob,
): Promise<void> {
  if (!blob.size) return;
  const key = mediaKey(hostId, filename, kind);
  const version = mediaMutationVersions.get(key) ?? 0;
  const hostVersion = hostMutationVersions.get(hostId) ?? 0;
  const bytes = await blob.arrayBuffer();
  await requestFromStore(MEDIA_STORE, "readwrite", (store) =>
    (mediaMutationVersions.get(key) ?? 0) === version &&
    (hostMutationVersions.get(hostId) ?? 0) === hostVersion
      ? store.put({
          key,
          hostId,
          filename,
          kind,
          cachedAt: Date.now(),
          bytes,
          mimeType: blob.type,
          size: bytes.byteLength,
        })
      : store.get(key),
  );
}

export async function pruneCachedGalleryMedia(): Promise<void> {
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    try {
      const transaction = db.transaction(MEDIA_STORE, "readwrite");
      const store = transaction.objectStore(MEDIA_STORE);
      const request = store.index("cachedAt").openCursor(null, "prev");
      let retainedRecords = 0;
      let retainedBytes = 0;
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return;
        const record = cursor.value as CachedMediaRecord;
        retainedRecords += 1;
        retainedBytes += record.size ?? record.bytes?.byteLength ?? 0;
        if (retainedRecords > MAX_THUMBNAIL_RECORDS || retainedBytes > MAX_THUMBNAIL_BYTES) {
          cursor.delete();
        }
        cursor.continue();
      };
      request.onerror = finish;
      transaction.oncomplete = finish;
      transaction.onerror = finish;
      transaction.onabort = finish;
    } catch {
      finish();
    }
  });
}

export async function removeCachedGalleryPrints(
  removed: readonly { hostId: string; filename: string }[],
): Promise<void> {
  for (const hostId of new Set(removed.map(({ hostId }) => hostId))) {
    hostMutationVersions.set(hostId, (hostMutationVersions.get(hostId) ?? 0) + 1);
  }
  for (const { hostId, filename } of removed) {
    const key = mediaKey(hostId, filename, "thumbnail");
    mediaMutationVersions.set(key, (mediaMutationVersions.get(key) ?? 0) + 1);
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
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const transaction = db.transaction(MEDIA_STORE, "readwrite");
      const store = transaction.objectStore(MEDIA_STORE);
      for (const { hostId, filename } of removed) {
        store.delete(mediaKey(hostId, filename, "thumbnail"));
      }
      transaction.oncomplete = () => resolve();
      transaction.onerror = transaction.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
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
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    try {
      const transaction = db.transaction(
        [GALLERIES_STORE, MEDIA_STORE, PRESENTATIONS_STORE],
        "readwrite",
      );
      const galleries = transaction.objectStore(GALLERIES_STORE);
      const media = transaction.objectStore(MEDIA_STORE);
      const presentations = transaction.objectStore(PRESENTATIONS_STORE);
      const hostIndex = media.index("hostId");
      for (const hostId of ids) {
        galleries.delete(hostId);
        presentations.delete(hostId);
        const keys = hostIndex.getAllKeys(hostId);
        keys.onsuccess = () => {
          for (const key of keys.result) {
            if (typeof key === "string") {
              mediaMutationVersions.set(key, (mediaMutationVersions.get(key) ?? 0) + 1);
            }
            media.delete(key);
          }
        };
      }
      transaction.oncomplete = finish;
      transaction.onerror = finish;
      transaction.onabort = finish;
    } catch {
      finish();
    }
  });
}
