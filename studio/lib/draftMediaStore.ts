/**
 * Quota-safe media persistence for Create drafts.
 *
 * localStorage owns the small form metadata while IndexedDB owns base64 media.
 * The structural type keeps this helper shared by web, desktop, and iPhone.
 */
export interface DraftMediaRecord {
  draftId?: string;
  base64?: string | null;
}

const DB_NAME = "mold-generate-drafts";
const STORE_NAME = "media";
const DB_VERSION = 1;
const memory = new Map<string, DraftMediaRecord>();

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function recordsMatch(
  left: DraftMediaRecord,
  right: DraftMediaRecord,
): boolean {
  if (left.base64 !== right.base64) return false;
  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const keys = new Set([
    ...Object.keys(leftRecord),
    ...Object.keys(rightRecord),
  ]);
  for (const key of keys) {
    if (key === "base64") continue;
    if (JSON.stringify(leftRecord[key]) !== JSON.stringify(rightRecord[key])) {
      return false;
    }
  }
  return true;
}

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE_NAME)) {
        req.result.createObjectStore(STORE_NAME, { keyPath: "draftId" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, mode);
    const req = fn(tx.objectStore(STORE_NAME));
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => resolve(null);
    tx.oncomplete = () => db.close();
    tx.onerror = () => db.close();
  });
}

async function writeStore(
  fn: (store: IDBObjectStore) => IDBRequest,
): Promise<boolean> {
  const db = await openDb();
  if (!db) return false;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const req = fn(tx.objectStore(STORE_NAME));
    let settled = false;
    const finish = (result: boolean) => {
      if (settled) return;
      settled = true;
      db.close();
      resolve(result);
    };
    req.onerror = () => finish(false);
    tx.oncomplete = () => finish(true);
    tx.onerror = () => finish(false);
    tx.onabort = () => finish(false);
  });
}

export async function putDraftMedia<T extends DraftMediaRecord>(
  media: T,
): Promise<void> {
  if (!media.draftId || !media.base64) return;
  const existing = memory.get(media.draftId);
  if (existing && recordsMatch(existing, media)) return;
  const saved = clone(media);
  memory.set(media.draftId, saved);
  await withStore("readwrite", (store) => store.put(saved));
}

/**
 * Persist media durably, returning false when IndexedDB is unavailable or the
 * transaction fails. Templates use this stricter contract because reporting a
 * successful save and silently retaining bytes only in memory would create a
 * broken template after the next launch.
 */
export async function putDurableMedia<T extends DraftMediaRecord>(
  media: T,
): Promise<boolean> {
  if (!media.draftId || !media.base64) return false;
  const saved = clone(media);
  const persisted = await writeStore((store) => store.put(saved));
  if (persisted) memory.set(media.draftId, saved);
  return persisted;
}

export async function getDraftMedia<T extends DraftMediaRecord>(
  draftId: string,
): Promise<T | null> {
  const fromDb = await withStore<T>("readonly", (store) => store.get(draftId));
  if (fromDb) {
    memory.set(draftId, clone(fromDb));
    return fromDb;
  }
  const fromMemory = memory.get(draftId);
  return fromMemory ? (clone(fromMemory) as T) : null;
}

export async function deleteDraftMedia(draftId: string): Promise<void> {
  memory.delete(draftId);
  await withStore("readwrite", (store) => store.delete(draftId));
}

export function clearMemoryDraftsForTest() {
  memory.clear();
}
