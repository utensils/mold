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

export async function putDraftMedia<T extends DraftMediaRecord>(
  media: T,
): Promise<void> {
  if (!media.draftId || !media.base64) return;
  const saved = clone(media);
  memory.set(media.draftId, saved);
  await withStore("readwrite", (store) => store.put(saved));
}

export async function getDraftMedia<T extends DraftMediaRecord>(
  draftId: string,
): Promise<T | null> {
  const fromDb = await withStore<T>("readonly", (store) => store.get(draftId));
  if (fromDb) return fromDb;
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
