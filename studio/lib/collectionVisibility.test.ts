import { afterEach, beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({ updateCollectionHidden: vi.fn() }));
vi.mock("../api/galleryOrganization", () => api);
import {
  collectionAvailability,
  scopedCollectionInventory,
  rememberCollectionVisibility,
  reconcileCollectionVisibility,
  resetCollectionVisibilityForTests,
} from "./collectionVisibility";
const collection = {
  id: "a",
  slug: "private",
  name: "Private",
  hidden: true,
  count: 3,
  description: null,
  cover_filename: null,
  created_at: 1,
  updated_at: 1,
};
const host = {
  hostId: "one",
  target: { baseUrl: "http://one", apiKey: null },
  instanceId: "instance",
  collections: [collection],
  listingOk: true,
};
afterEach(() => {
  vi.unstubAllGlobals();
  resetCollectionVisibilityForTests();
});
beforeEach(() => {
  vi.stubGlobal("localStorage", { getItem: () => null, setItem: vi.fn() });
  resetCollectionVisibilityForTests();
  collection.hidden = true;
  api.updateCollectionHidden.mockReset();
});
it("distinguishes confirmed absence from unavailable inventory", () => {
  expect(collectionAvailability([], "two", true)).toBe("absent");
  expect(collectionAvailability([], "two", false)).toBe("unavailable");
  expect(collectionAvailability([{ hostId: "two" }], "two", true)).toBe(
    "present",
  );
});
it("scopes counts and covers without losing global visibility", () => {
  const [scoped] = scopedCollectionInventory(
    [
      {
        slug: "private",
        name: "Private",
        count: 7,
        hidden: true,
        hosts: [
          { hostId: "one", id: "a", count: 3 },
          { hostId: "two", id: "b", count: 4 },
        ],
        cover: { hostId: "one", filename: "a.png" },
      },
    ],
    "two",
  );
  expect(scoped!.count).toBe(4);
  expect(scoped!.cover).toBeNull();
  expect(scoped!.hidden).toBe(true);
});
it("retains a failed explicit show and retries rather than healing stale hidden state", async () => {
  rememberCollectionVisibility("private", false, [host]);
  api.updateCollectionHidden
    .mockRejectedValueOnce(new Error("offline"))
    .mockResolvedValue({ ...collection, hidden: false });
  await reconcileCollectionVisibility([host]);
  await reconcileCollectionVisibility([host]);
  expect(api.updateCollectionHidden.mock.calls.map((c) => c[2])).toEqual([
    false,
    false,
  ]);
});
it("refuses replay to a replaced installation or changed route", async () => {
  rememberCollectionVisibility("private", false, [host]);
  await reconcileCollectionVisibility([{ ...host, instanceId: "replacement" }]);
  expect(api.updateCollectionHidden).not.toHaveBeenCalled();
});
it("heals mixed hidden replicas", async () => {
  api.updateCollectionHidden.mockImplementation(async (_t, id, hidden) => ({
    ...collection,
    id,
    hidden,
  }));
  await reconcileCollectionVisibility([
    host,
    {
      ...host,
      hostId: "two",
      target: { baseUrl: "http://two", apiKey: null },
      collections: [{ ...collection, id: "b", hidden: false }],
    },
  ]);
  expect(api.updateCollectionHidden).toHaveBeenCalledWith(
    { baseUrl: "http://two", apiKey: null },
    "b",
    true,
  );
});
it("serializes Hide then Show while the earlier write is in flight", async () => {
  collection.hidden = false;
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  api.updateCollectionHidden
    .mockImplementationOnce(async () => {
      await held;
      return { ...collection, hidden: true };
    })
    .mockResolvedValue({ ...collection, hidden: false });
  rememberCollectionVisibility("private", true, [host]);
  const earlier = reconcileCollectionVisibility([host]);
  await Promise.resolve();
  rememberCollectionVisibility("private", false, [host]);
  const later = reconcileCollectionVisibility([host]);
  release();
  await Promise.all([earlier, later]);
  expect(api.updateCollectionHidden.mock.calls.map((c) => c[2])).toEqual([
    true,
    false,
  ]);
  expect(collection.hidden).toBe(false);
});
it("keeps Show pending until a fresh post-edit GET confirms it and protects older GETs", async () => {
  const {
    collectionVisibilityRevision,
    protectCollectionVisibilityListing,
    desiredCollectionHidden,
  } = await import("./collectionVisibility");
  const started = collectionVisibilityRevision();
  rememberCollectionVisibility("private", false, [host]);
  api.updateCollectionHidden.mockResolvedValue({
    ...collection,
    hidden: false,
  });
  await reconcileCollectionVisibility([{ ...host, readRevision: started }]);
  expect(desiredCollectionHidden("private", true)).toBe(false);
  expect(
    protectCollectionVisibilityListing(
      [{ ...collection, hidden: true }],
      started,
    )[0]!.hidden,
  ).toBe(false);
  await reconcileCollectionVisibility([
    { ...host, readRevision: collectionVisibilityRevision() },
  ]);
  expect(desiredCollectionHidden("private", true)).toBe(true);
  expect(
    protectCollectionVisibilityListing(
      [{ ...collection, hidden: true }],
      started,
    )[0]!.hidden,
  ).toBe(false);
});
it("checks the live registry before reaching the second host after a suspended write", async () => {
  const second = {
    ...host,
    hostId: "two",
    target: { baseUrl: "http://two", apiKey: null },
    collections: [{ ...collection, id: "two" }],
  };
  let live = [host, second];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  api.updateCollectionHidden.mockImplementationOnce(async () => {
    await held;
    return { ...collection, hidden: false };
  });
  rememberCollectionVisibility("private", false, live);
  const task = reconcileCollectionVisibility(live, () => live);
  await Promise.resolve();
  live = [host, { ...second, instanceId: "replacement" }];
  release();
  const errors = await task;
  expect(api.updateCollectionHidden).toHaveBeenCalledTimes(1);
  expect(errors.join(" ")).toContain("installation changed");
});
