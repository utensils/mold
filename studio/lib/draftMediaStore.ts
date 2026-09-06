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
const durableDraftIds = new Set<string>();

interface InFlightWrite {
  records: readonly DraftMediaRecord[];
  promise: Promise<boolean>;
}

// IndexedDB serializes overlapping readwrite transactions eventually, but
// callers can open separate connections in an order that does not match the
// form snapshots that scheduled them. Keep one explicit commit order here and
// share an exact in-flight batch so a metadata-only edit cannot queue another
// structured clone of the same potentially enormous base64 payload.
let writeTail: Promise<void> = Promise.resolve();
const inFlightWrites = new Set<InFlightWrite>();
const latestWriteByDraftId = new Map<string, InFlightWrite>();

/** Copy record metadata without JSON-stringifying the potentially enormous
 * base64 payload. IndexedDB performs its own structured clone at the durable
 * boundary; this copy only protects the in-process cache from caller mutation. */
function cloneRecord<T extends DraftMediaRecord>(value: T): T {
  const { base64, ...metadata } = value;
  return {
    ...(structuredClone(metadata) as Omit<T, "base64">),
    ...(base64 === undefined ? {} : { base64 }),
  } as T;
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

function batchesMatch(
  left: readonly DraftMediaRecord[],
  right: readonly DraftMediaRecord[],
): boolean {
  return (
    left.length === right.length &&
    left.every((record, index) => recordsMatch(record, right[index]!))
  );
}

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (db: IDBDatabase | null) => {
      if (settled) {
        db?.close();
        return;
      }
      settled = true;
      resolve(db);
    };
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      finish(null);
      return;
    }
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE_NAME)) {
        req.result.createObjectStore(STORE_NAME, { keyPath: "draftId" });
      }
    };
    req.onsuccess = () => finish(req.result);
    req.onerror = () => finish(null);
    req.onblocked = () => finish(null);
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result: T | null) => {
      if (settled) return;
      settled = true;
      db.close();
      resolve(result);
    };
    try {
      const tx = db.transaction(STORE_NAME, mode);
      const req = fn(tx.objectStore(STORE_NAME));
      let result: T | null = null;
      req.onsuccess = () => {
        result = req.result ?? null;
      };
      req.onerror = () => finish(null);
      tx.oncomplete = () => finish(result);
      tx.onerror = () => finish(null);
      tx.onabort = () => finish(null);
    } catch {
      finish(null);
    }
  });
}

async function writeRecords(
  records: readonly DraftMediaRecord[],
): Promise<boolean> {
  const db = await openDb();
  if (!db) return false;
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result: boolean) => {
      if (settled) return;
      settled = true;
      db.close();
      resolve(result);
    };
    let tx: IDBTransaction | null = null;
    try {
      tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      for (const record of records) store.put(record);
      tx.oncomplete = () => finish(true);
      tx.onerror = () => finish(false);
      tx.onabort = () => finish(false);
    } catch {
      try {
        tx?.abort();
      } catch {
        // The transaction may already have auto-aborted after a clone/quota
        // failure. Either way the batch did not commit.
      }
      finish(false);
    }
  });
}

async function commitRecords(
  records: readonly DraftMediaRecord[],
): Promise<boolean> {
  // Re-evaluate after earlier queued writes finish. A record that was not yet
  // durable when this call was scheduled may now be durable and unchanged.
  const changed = records.filter((record) => {
    const draftId = record.draftId!;
    const existing = memory.get(draftId);
    return (
      !durableDraftIds.has(draftId) ||
      !existing ||
      !recordsMatch(existing, record)
    );
  });
  if (changed.length === 0) return true;
  const persisted = await writeRecords(changed);
  if (persisted) {
    for (const record of changed) {
      memory.set(record.draftId!, record);
      durableDraftIds.add(record.draftId!);
    }
  }
  return persisted;
}

function enqueueRecords(
  records: readonly DraftMediaRecord[],
): Promise<boolean> {
  if (records.length === 0) return Promise.resolve(true);
  const matching = latestWriteByDraftId.get(records[0]!.draftId!);
  // Do not share an older A write across an intervening B replacement for the
  // same draft ID (A -> B -> A). Every record must still name this exact write
  // as its latest queued authority before the batch can safely coalesce.
  const isLatestForEveryRecord = records.every(
    (record) => latestWriteByDraftId.get(record.draftId!) === matching,
  );
  if (
    matching &&
    isLatestForEveryRecord &&
    batchesMatch(matching.records, records)
  ) {
    return matching.promise;
  }

  const promise = writeTail.then(() => commitRecords(records));
  writeTail = promise.then(
    () => undefined,
    () => undefined,
  );
  const write = { records, promise };
  inFlightWrites.add(write);
  for (const record of records)
    latestWriteByDraftId.set(record.draftId!, write);
  const removeWrite = () => {
    inFlightWrites.delete(write);
    for (const record of records) {
      if (latestWriteByDraftId.get(record.draftId!) === write) {
        latestWriteByDraftId.delete(record.draftId!);
      }
    }
  };
  void promise.then(removeWrite, removeWrite);
  return promise;
}

export async function putDraftMedia<T extends DraftMediaRecord>(
  media: T,
): Promise<void> {
  if (!media.draftId || !media.base64) return;
  const existing = memory.get(media.draftId);
  if (
    durableDraftIds.has(media.draftId) &&
    existing &&
    recordsMatch(existing, media)
  ) {
    return;
  }
  let saved: T;
  try {
    saved = cloneRecord(media);
  } catch {
    return;
  }
  durableDraftIds.delete(media.draftId);
  memory.set(media.draftId, saved);
  await enqueueRecords([saved]);
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
  const existing = memory.get(media.draftId);
  if (
    durableDraftIds.has(media.draftId) &&
    existing &&
    recordsMatch(existing, media)
  ) {
    return true;
  }
  let saved: T;
  try {
    saved = cloneRecord(media);
  } catch {
    return false;
  }
  return enqueueRecords([saved]);
}

/** Commit a descriptor's complete media set atomically. Callers may write the
 * descriptor only after this returns true; one failed clone, quota check, or
 * transaction abort leaves the prior IndexedDB records intact. */
export async function putDurableMediaBatch<T extends DraftMediaRecord>(
  media: readonly T[],
): Promise<boolean> {
  if (media.some((record) => !record.draftId || !record.base64)) return false;
  if (media.length === 0) return true;
  let saved: T[];
  try {
    saved = media.map(cloneRecord);
  } catch {
    return false;
  }
  return enqueueRecords(saved);
}

export async function getDraftMedia<T extends DraftMediaRecord>(
  draftId: string,
): Promise<T | null> {
  const fromDb = await withStore<T>("readonly", (store) => store.get(draftId));
  if (fromDb) {
    memory.set(draftId, cloneRecord(fromDb));
    durableDraftIds.add(draftId);
    return fromDb;
  }
  const fromMemory = memory.get(draftId);
  return fromMemory ? cloneRecord(fromMemory as T) : null;
}

export async function deleteDraftMedia(draftId: string): Promise<void> {
  memory.delete(draftId);
  durableDraftIds.delete(draftId);
  await withStore("readwrite", (store) => store.delete(draftId));
}

/**
 * Delete every stored record whose draft id starts with `prefix`, and answer
 * with the ids removed.
 *
 * `deleteDraftMedia` can only reach a key the caller already knows. A retired
 * feature's blobs are exactly the ones nobody knows the ids of any more — the
 * draft that named them is itself being deleted, and may be unreadable — so
 * retirement needs to sweep by prefix or leave megabytes of base64 behind in
 * the user's quota forever.
 */
export async function deleteDraftMediaByPrefix(
  prefix: string,
): Promise<string[]> {
  const keys = await withStore<IDBValidKey[]>("readonly", (store) =>
    store.getAllKeys(),
  );
  const matched = (keys ?? [])
    .filter((key): key is string => typeof key === "string")
    .filter((key) => key.startsWith(prefix));
  for (const key of matched) await deleteDraftMedia(key);
  for (const key of [...memory.keys()]) {
    if (key.startsWith(prefix)) {
      memory.delete(key);
      durableDraftIds.delete(key);
      if (!matched.includes(key)) matched.push(key);
    }
  }
  return matched;
}

export function clearMemoryDraftsForTest() {
  memory.clear();
  durableDraftIds.clear();
}
