import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { CopyReceipt } from "../api/galleryCopy";
import { IDBFactory, IDBDatabase } from "fake-indexeddb";

const originalLocks = Object.getOwnPropertyDescriptor(navigator, "locks");
const receipt = (source: string, recipe = "recipe"): CopyReceipt => ({
  version: 1,
  source,
  destination: "destination",
  sourceFilename: "print.png",
  destinationFilename: "copy.png",
  digest: "digest",
  size: 320,
  origin: JSON.stringify([source, "print.png"]),
  recipe,
  stage: "complete",
});
beforeEach(() => {
  localStorage.clear();
  vi.resetModules();
  vi.stubGlobal("indexedDB", new IDBFactory());
});
afterEach(() => {
  vi.unstubAllGlobals();
  if (originalLocks) Object.defineProperty(navigator, "locks", originalLocks);
  else Reflect.deleteProperty(navigator, "locks");
});
function installLocks() {
  const tails = new Map<string, Promise<unknown>>();
  const request = vi.fn(
    (_name: string, _options: unknown, work: () => Promise<unknown>) => {
      const tail = tails.get(_name) ?? Promise.resolve();
      const run = tail.catch(() => undefined).then(work);
      tails.set(_name, run);
      return run;
    },
  );
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: { request },
  });
  return request;
}
it("preserves receipts written by two tabs with independent module queues", async () => {
  const request = installLocks();
  const first = await import("./galleryCopyJournal");
  vi.resetModules();
  const second = await import("./galleryCopyJournal");
  await Promise.all([
    first
      .copyJournal(first.browserSyncStorage)
      .write("first", receipt("first")),
    second
      .copyJournal(second.browserSyncStorage)
      .write("second", receipt("second")),
  ]);
  expect(
    Object.keys((await first.browserSyncStorage.read())!.receipts).sort(),
  ).toEqual(["first", "second"]);
  expect(request).toHaveBeenCalledTimes(2);
});
it("serializes whole browser copies across tabs while allowing their journal checkpoints", async () => {
  const request = installLocks();
  const first = await import("./galleryCopyJournal");
  vi.resetModules();
  const second = await import("./galleryCopyJournal");
  const events: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const firstJournal = first.copyJournal(first.browserSyncStorage);
  const firstRun = firstJournal.runExclusive!(async () => {
    events.push("first started");
    await gate;
    await firstJournal.write("first", receipt("first"));
    events.push("first finished");
  });
  await Promise.resolve();
  const secondRun = second.copyJournal(second.browserSyncStorage).runExclusive!(
    async () => {
      events.push("second started");
    },
  );
  await Promise.resolve();
  expect(events).toEqual(["first started"]);
  release();
  await Promise.all([firstRun, secondRun]);
  expect(events).toEqual(["first started", "first finished", "second started"]);
  expect(request.mock.calls.map(([name]) => name)).toContain(
    "mold.library-sync.v1",
  );
});
it("refuses browser writes before publication when cross-tab locking is unavailable", async () => {
  vi.stubGlobal("indexedDB", undefined);
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: undefined,
  });
  const { copyJournal, browserSyncStorage } =
    await import("./galleryCopyJournal");
  await expect(
    copyJournal(browserSyncStorage).write("first", receipt("first")),
  ).rejects.toThrow("browser");
  expect(localStorage.getItem("mold.library-sync.v1")).toBeNull();
});
it("serializes insecure HTTP copies across tabs without blocking nested checkpoints", async () => {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: undefined,
  });
  const first = await import("./galleryCopyJournal");
  vi.resetModules();
  const second = await import("./galleryCopyJournal");
  const firstJournal = first.copyJournal(first.browserSyncStorage);
  const secondJournal = second.copyJournal(second.browserSyncStorage);
  const events: string[] = [];
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const a = firstJournal.runExclusive!(async () => {
    events.push("first");
    await gate;
    await firstJournal.write("first", receipt("first"));
  });
  await vi.waitFor(() => expect(events).toEqual(["first"]));
  const b = secondJournal.runExclusive!(async () => {
    events.push("second");
    await secondJournal.write("second", receipt("second"));
  });
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(events).toEqual(["first"]);
  release();
  await Promise.all([a, b]);
  expect(events).toEqual(["first", "second"]);
  expect(
    Object.keys((await first.browserSyncStorage.read())!.receipts).sort(),
  ).toEqual(["first", "second"]);
});
it("aborts a queued insecure HTTP copy without starting it and releases a failed copy", async () => {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: undefined,
  });
  const { copyJournal, browserSyncStorage } =
    await import("./galleryCopyJournal");
  const journal = copyJournal(browserSyncStorage);
  let release!: () => void;
  let started = false;
  const a = journal.runExclusive!(async () => {
    started = true;
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    throw new Error("copy failed");
  });
  const failed = expect(a).rejects.toThrow("copy failed");
  await vi.waitFor(() => expect(started).toBe(true));
  const controller = new AbortController();
  const callback = vi.fn(async () => undefined);
  const b = journal.runExclusive!(callback, controller.signal);
  const aborted = expect(b).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await aborted;
  release();
  await failed;
  await journal.runExclusive!(callback);
  expect(callback).toHaveBeenCalledTimes(1);
});
it("does not adopt lineage when identical bytes now carry a different recipe", async () => {
  const { copyJournal } = await import("./galleryCopyJournal");
  const saved = receipt("origin", "old-recipe");
  const journal = copyJournal({
    read: async () => ({ version: 1, keep: {}, receipts: { saved } }),
    write: async () => undefined,
  });
  expect(
    await journal.lineage!("destination", "copy.png", "digest", "new-recipe"),
  ).toBeNull();
  expect(
    await journal.lineage!("destination", "copy.png", "digest", "old-recipe"),
  ).toEqual(saved);
});
it("cancels active HTTP copy work if its database lock is interrupted", async () => {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: undefined,
  });
  let transaction: IDBTransaction | undefined;
  const original = IDBDatabase.prototype.transaction;
  vi.spyOn(IDBDatabase.prototype, "transaction").mockImplementation(function (
    this: InstanceType<typeof IDBDatabase>,
    ...args
  ) {
    const result = original.apply(this, args);
    transaction = result;
    return result;
  });
  const { copyJournal, browserSyncStorage } =
    await import("./galleryCopyJournal");
  let active: AbortSignal | undefined;
  let release!: () => void;
  let rejectedEarly = false;
  const run = copyJournal(browserSyncStorage).runExclusive!(async (signal) => {
    active = signal;
    await new Promise<void>((resolve) => {
      release = resolve;
    });
  });
  void run.catch(() => {
    rejectedEarly = true;
  });
  const rejected = expect(run).rejects.toBeDefined();
  await vi.waitFor(() => expect(active).toBeDefined());
  transaction!.abort();
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(active!.aborted).toBe(true);
  expect(rejectedEarly).toBe(false);
  release();
  await rejected;
  expect(active!.aborted).toBe(true);
});
it("uses the verified destination recipe when import bookkeeping differs", async () => {
  const { copyJournal } = await import("./galleryCopyJournal");
  const saved = {
    ...receipt("origin", "source-recipe"),
    destinationRecipe: "destination-recipe",
  };
  const journal = copyJournal({
    read: async () => ({ version: 1, keep: {}, receipts: { saved } }),
    write: async () => undefined,
  });
  expect(
    await journal.lineage!(
      "destination",
      "copy.png",
      "digest",
      "destination-recipe",
    ),
  ).toEqual(saved);
  expect(
    await journal.lineage!("destination", "copy.png", "digest", "other-recipe"),
  ).toBeNull();
});
