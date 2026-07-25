import type { SourceImageState, SourceMediaState } from "../types";
import { createUuid } from "@studio/lib/id";

type DraftMedia = SourceImageState | SourceMediaState;

const DB_NAME = "mold-generate-drafts";
const STORE_NAME = "media";
const DB_VERSION = 1;
const memory = new Map<string, DraftMedia>();

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME, { keyPath: "draftId" });
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

export function newDraftId(): string {
  return createUuid();
}

export async function putDraftMedia(media: DraftMedia): Promise<void> {
  if (!media.draftId || !media.base64) return;
  const saved = clone(media);
  memory.set(media.draftId, saved);
  await withStore("readwrite", (store) => store.put(saved));
}

export async function getDraftMedia<T extends DraftMedia>(
  draftId: string,
): Promise<T | null> {
  const fromDb = await withStore<T>("readonly", (store) => store.get(draftId));
  if (fromDb) return fromDb;
  const fromMemory = memory.get(draftId);
  return fromMemory ? (clone(fromMemory) as T) : null;
}

export function clearMemoryDraftsForTest() {
  memory.clear();
}
