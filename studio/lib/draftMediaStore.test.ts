import { beforeEach, describe, expect, it } from "vitest";
import { IDBFactory } from "fake-indexeddb";
import {
  clearMemoryDraftsForTest,
  getDraftMedia,
  putDraftMedia,
  putDurableMediaBatch,
  type DraftMediaRecord,
} from "./draftMediaStore";

function abortFirstWriteFactory(): IDBFactory {
  const factory = new IDBFactory();
  const open = factory.open.bind(factory);
  let abortNextWrite = true;
  Object.defineProperty(factory, "open", {
    value: (name: string, version?: number) => {
      const request = version === undefined ? open(name) : open(name, version);
      request.addEventListener(
        "success",
        () => {
          const db = request.result;
          const transaction = db.transaction.bind(db);
          Object.defineProperty(db, "transaction", {
            value: (...args: Parameters<IDBDatabase["transaction"]>) => {
              const tx = transaction(...args);
              if (args[1] === "readwrite" && abortNextWrite) {
                abortNextWrite = false;
                queueMicrotask(() => tx.abort());
              }
              return tx;
            },
            configurable: true,
          });
        },
        { once: true },
      );
      return request;
    },
    configurable: true,
  });
  return factory;
}

function countingFactory(): {
  factory: IDBFactory;
  readwriteTransactions: () => number;
} {
  const factory = new IDBFactory();
  const open = factory.open.bind(factory);
  let readwriteTransactions = 0;
  Object.defineProperty(factory, "open", {
    value: (name: string, version?: number) => {
      const request = version === undefined ? open(name) : open(name, version);
      request.addEventListener(
        "success",
        () => {
          const db = request.result;
          const transaction = db.transaction.bind(db);
          Object.defineProperty(db, "transaction", {
            value: (...args: Parameters<IDBDatabase["transaction"]>) => {
              if (args[1] === "readwrite") readwriteTransactions += 1;
              return transaction(...args);
            },
            configurable: true,
          });
        },
        { once: true },
      );
      return request;
    },
    configurable: true,
  });
  return { factory, readwriteTransactions: () => readwriteTransactions };
}

describe("draft media store", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: new IDBFactory(),
      writable: true,
      configurable: true,
    });
    clearMemoryDraftsForTest();
  });

  it("commits a media set durably in one transaction", async () => {
    await expect(
      putDurableMediaBatch([
        { draftId: "first", base64: "FIRST_BYTES" },
        { draftId: "second", base64: "SECOND_BYTES" },
      ]),
    ).resolves.toBe(true);

    clearMemoryDraftsForTest();
    await expect(getDraftMedia("first")).resolves.toEqual({
      draftId: "first",
      base64: "FIRST_BYTES",
    });
    await expect(getDraftMedia("second")).resolves.toEqual({
      draftId: "second",
      base64: "SECOND_BYTES",
    });
  });

  it("aborts the whole batch when IndexedDB cannot clone one record", async () => {
    const uncloneable = {
      draftId: "uncloneable",
      base64: "BAD_BYTES",
      callback: () => undefined,
    } as DraftMediaRecord;

    await expect(
      putDurableMediaBatch([
        { draftId: "would-have-been-first", base64: "FIRST_BYTES" },
        uncloneable,
      ]),
    ).resolves.toBe(false);

    clearMemoryDraftsForTest();
    await expect(getDraftMedia("would-have-been-first")).resolves.toBeNull();
    await expect(getDraftMedia("uncloneable")).resolves.toBeNull();
  });

  it("rolls back every record when the readwrite transaction aborts", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: abortFirstWriteFactory(),
      writable: true,
      configurable: true,
    });

    await expect(
      putDurableMediaBatch([
        { draftId: "one", base64: "ONE_BYTES" },
        { draftId: "two", base64: "TWO_BYTES" },
      ]),
    ).resolves.toBe(false);

    clearMemoryDraftsForTest();
    await expect(getDraftMedia("one")).resolves.toBeNull();
    await expect(getDraftMedia("two")).resolves.toBeNull();
  });

  it("does not reopen IndexedDB for an unchanged known-durable record", async () => {
    const record = { draftId: "large-media", base64: "LARGE_BYTES" };
    await expect(putDurableMediaBatch([record])).resolves.toBe(true);

    const durableFactory = globalThis.indexedDB;
    Object.defineProperty(globalThis, "indexedDB", {
      value: undefined,
      writable: true,
      configurable: true,
    });
    await expect(putDurableMediaBatch([record])).resolves.toBe(true);
    await expect(
      putDurableMediaBatch([{ ...record, base64: "CHANGED_BYTES" }]),
    ).resolves.toBe(false);
    Object.defineProperty(globalThis, "indexedDB", {
      value: durableFactory,
      writable: true,
      configurable: true,
    });
  });

  it("shares one transaction between simultaneous identical batches", async () => {
    const observed = countingFactory();
    Object.defineProperty(globalThis, "indexedDB", {
      value: observed.factory,
      writable: true,
      configurable: true,
    });
    const record = { draftId: "large-concurrent", base64: "LARGE_BYTES" };

    await expect(
      Promise.all([
        putDurableMediaBatch([record]),
        putDurableMediaBatch([record]),
      ]),
    ).resolves.toEqual([true, true]);
    expect(observed.readwriteTransactions()).toBe(1);
  });

  it("serializes changed bytes after an in-flight write", async () => {
    const observed = countingFactory();
    Object.defineProperty(globalThis, "indexedDB", {
      value: observed.factory,
      writable: true,
      configurable: true,
    });
    const original = { draftId: "changed-concurrent", base64: "OLD_BYTES" };
    const replacement = { ...original, base64: "NEW_BYTES" };

    await expect(
      Promise.all([
        putDurableMediaBatch([original]),
        putDurableMediaBatch([replacement]),
      ]),
    ).resolves.toEqual([true, true]);
    expect(observed.readwriteTransactions()).toBe(2);
    clearMemoryDraftsForTest();
    await expect(getDraftMedia(original.draftId)).resolves.toEqual(replacement);
  });

  it("does not coalesce A through an intervening B replacement", async () => {
    const observed = countingFactory();
    Object.defineProperty(globalThis, "indexedDB", {
      value: observed.factory,
      writable: true,
      configurable: true,
    });
    const first = { draftId: "a-b-a", base64: "A_BYTES" };
    const replacement = { ...first, base64: "B_BYTES" };

    await expect(
      Promise.all([
        putDurableMediaBatch([first]),
        putDurableMediaBatch([replacement]),
        putDurableMediaBatch([first]),
      ]),
    ).resolves.toEqual([true, true, true]);
    expect(observed.readwriteTransactions()).toBe(3);
    clearMemoryDraftsForTest();
    await expect(getDraftMedia(first.draftId)).resolves.toEqual(first);
  });

  it("retries an identical batch after its in-flight write fails", async () => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: abortFirstWriteFactory(),
      writable: true,
      configurable: true,
    });
    const record = { draftId: "retry-after-abort", base64: "RETRY_BYTES" };

    await expect(putDurableMediaBatch([record])).resolves.toBe(false);
    await expect(putDurableMediaBatch([record])).resolves.toBe(true);
    clearMemoryDraftsForTest();
    await expect(getDraftMedia(record.draftId)).resolves.toEqual(record);
  });

  it("does not treat a failed best-effort replacement as durable", async () => {
    const original = { draftId: "shared-id", base64: "ORIGINAL_BYTES" };
    await expect(putDurableMediaBatch([original])).resolves.toBe(true);

    const durableFactory = globalThis.indexedDB;
    Object.defineProperty(globalThis, "indexedDB", {
      value: undefined,
      writable: true,
      configurable: true,
    });
    const replacement = { ...original, base64: "REPLACEMENT_BYTES" };
    await putDraftMedia(replacement);
    await expect(putDurableMediaBatch([replacement])).resolves.toBe(false);
    Object.defineProperty(globalThis, "indexedDB", {
      value: durableFactory,
      writable: true,
      configurable: true,
    });
  });
});
