/** Restart-safe, secret-free queue for organization edits targeting offline hosts. */

import type { OrganizationFanoutOp } from "./libraryOrganization";
import type { GalleryBulkMutationRequest } from "./api/galleryOrganization";
import { createUuid } from "./id";

const DB_NAME = "mold-gallery-mutation-outbox";
const DB_VERSION = 1;
const STORE = "operations";
const volatileFallback = new Map<string, QueuedGalleryMutation>();

export interface QueuedGalleryMutation {
  id: string;
  hostId: string;
  hostInstanceId: string | null;
  hostName: string;
  op: OrganizationFanoutOp;
  createdAt: number;
  /** Strict enqueue order, persisted so inverse edits replay in order. */
  sequence: number;
  attempts: number;
  lastError: string | null;
}

/** Translate a planned organization op to the server's single bulk wire call. */
export function galleryBulkRequest(
  operationId: string,
  op: OrganizationFanoutOp,
): GalleryBulkMutationRequest | null {
  const base = { operation_id: operationId, filenames: [...op.filenames] };
  switch (op.kind) {
    case "setTitle":
      return {
        ...base,
        filenames: [],
        titles: op.filenames.map((filename) => ({
          filename,
          title: op.title ?? "",
        })),
      };
    case "setFavorite":
      return { ...base, favorite: op.favorite };
    case "addTags":
      return { ...base, add_tags: [...op.tags] };
    case "removeTags":
      return { ...base, remove_tags: [...op.tags] };
    case "addToCollection":
      return { ...base, add_to_collection: { name: op.ensureCollection.name } };
    case "removeFromCollection":
      return { ...base, remove_from_collection_slug: op.slug };
    case "trash":
    case "restore":
    case "deleteForever":
      return null;
  }
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE))
        db.createObjectStore(STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Could not open gallery outbox."));
  });
}

async function transaction<T>(
  mode: IDBTransactionMode,
  run: (
    store: IDBObjectStore,
    resolve: (value: T) => void,
    reject: (reason: unknown) => void,
  ) => void,
): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    let result: T | undefined;
    let hasResult = false;
    let failed = false;
    const succeed = (value: T) => {
      result = value;
      hasResult = true;
    };
    const fail = (reason: unknown) => {
      if (failed) return;
      failed = true;
      try {
        tx.abort();
      } catch {
        // The request may already have aborted the transaction.
      }
      reject(reason);
    };
    tx.oncomplete = () => {
      if (!failed && hasResult) resolve(result as T);
    };
    tx.onabort = () =>
      fail(tx.error ?? new Error("Gallery outbox transaction aborted."));
    tx.onerror = () =>
      fail(tx.error ?? new Error("Gallery outbox transaction failed."));
    run(tx.objectStore(STORE), succeed, fail);
  }).finally(() => db.close());
}

let enqueueTail: Promise<void> = Promise.resolve();

export function enqueueGalleryMutation(
  input: Omit<
    QueuedGalleryMutation,
    "id" | "createdAt" | "sequence" | "attempts" | "lastError"
  > & {
    id?: string;
  },
): Promise<QueuedGalleryMutation> {
  const pending = enqueueTail.then(async () => {
    const existing = await listGalleryMutations();
    const createdAt = Date.now();
    const sequence = Math.max(
      createdAt * 1000,
      existing.reduce(
        (max, item) => Math.max(max, item.sequence ?? item.createdAt * 1000),
        0,
      ) + 1,
    );
    const item: QueuedGalleryMutation = {
      ...input,
      id: input.id ?? createUuid(),
      createdAt,
      sequence,
      attempts: 0,
      lastError: null,
    };
    if (typeof indexedDB === "undefined") {
      volatileFallback.set(item.id, item);
      return item;
    }
    return transaction<QueuedGalleryMutation>(
      "readwrite",
      (store, resolve, reject) => {
        const request = store.put(item);
        request.onsuccess = () => resolve(item);
        request.onerror = () => reject(request.error);
      },
    );
  });
  enqueueTail = pending.then(
    () => undefined,
    () => undefined,
  );
  return pending;
}

function mutationOrder(
  a: QueuedGalleryMutation,
  b: QueuedGalleryMutation,
): number {
  const aSequence = a.sequence ?? a.createdAt * 1000;
  const bSequence = b.sequence ?? b.createdAt * 1000;
  return aSequence - bSequence || a.id.localeCompare(b.id);
}

export function listGalleryMutations(): Promise<QueuedGalleryMutation[]> {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve([...volatileFallback.values()].sort(mutationOrder));
  }
  return transaction<QueuedGalleryMutation[]>(
    "readonly",
    (store, resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () =>
        resolve(
          (request.result as QueuedGalleryMutation[]).sort(mutationOrder),
        );
      request.onerror = () => reject(request.error);
    },
  );
}

export function removeGalleryMutation(id: string): Promise<void> {
  if (typeof indexedDB === "undefined") {
    volatileFallback.delete(id);
    return Promise.resolve();
  }
  return transaction<void>("readwrite", (store, resolve, reject) => {
    const request = store.delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export function updateGalleryMutationFailure(
  id: string,
  error: string,
): Promise<void> {
  if (typeof indexedDB === "undefined") {
    const item = volatileFallback.get(id);
    if (item)
      volatileFallback.set(id, {
        ...item,
        attempts: item.attempts + 1,
        lastError: error,
      });
    return Promise.resolve();
  }
  return transaction<void>("readwrite", (store, resolve, reject) => {
    const get = store.get(id);
    get.onsuccess = () => {
      const item = get.result as QueuedGalleryMutation | undefined;
      if (!item) return resolve();
      item.attempts += 1;
      item.lastError = error;
      const put = store.put(item);
      put.onsuccess = () => resolve();
      put.onerror = () => reject(put.error);
    };
    get.onerror = () => reject(get.error);
  });
}
