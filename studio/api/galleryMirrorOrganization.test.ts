import { afterEach, beforeEach, expect, it, vi } from "vitest";
const api = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  listCollections: vi.fn(),
  createCollection: vi.fn(),
  updateCollection: vi.fn(),
  updateCollectionHidden: vi.fn(),
  patchGalleryImage: vi.fn(),
  setCollectionItems: vi.fn(),
}));
vi.mock("./client", () => ({ apiJsonTo: api.apiJsonTo }));
vi.mock("./galleryOrganization", () => api);
import {
  captureMirrorOrganization,
  applyMirrorOrganization,
} from "./galleryMirrorOrganization";
const target = { baseUrl: "http://host", apiKey: null };
const collection = {
  id: "source",
  name: "Private",
  slug: "private",
  description: "Shared",
  hidden: true,
  count: 1,
  cover_filename: null,
  created_at: 1,
  updated_at: 1,
};
afterEach(() => {
  localStorage.clear();
});
beforeEach(() => {
  localStorage.clear();
  for (const fn of Object.values(api)) fn.mockReset();
});
it("repairs existing media without overwriting destination-only organization", async () => {
  api.apiJsonTo.mockResolvedValue([
    { filename: "copy.png", title: "Keep", tags: ["Local"], favorite: true },
  ]);
  api.listCollections.mockResolvedValue([
    { ...collection, id: "dest", hidden: false, description: null },
  ]);
  api.updateCollectionHidden.mockResolvedValue({ ...collection, id: "dest" });
  await applyMirrorOrganization(target, "copy.png", {
    item: { title: "Source", tags: ["Shared"], collections: ["source"] },
    collections: [collection],
  });
  expect(api.patchGalleryImage).toHaveBeenCalledWith(target, "copy.png", {
    title: "Keep",
    tags: ["Local", "Shared"],
    favorite: true,
  });
  expect(api.updateCollectionHidden).toHaveBeenCalledWith(target, "dest", true);
  expect(api.setCollectionItems).toHaveBeenCalledWith(target, "dest", {
    add: ["copy.png"],
    remove: [],
  });
});
it("fails capture before copying when source inventory fails", async () => {
  api.apiJsonTo.mockRejectedValue(new Error("offline"));
  api.listCollections.mockResolvedValue([]);
  await expect(captureMirrorOrganization(target, "print.png")).rejects.toThrow(
    "offline",
  );
  expect(api.patchGalleryImage).not.toHaveBeenCalled();
});
it("does not attach organization to a missing destination", async () => {
  api.apiJsonTo.mockResolvedValue([]);
  api.listCollections.mockResolvedValue([]);
  await expect(
    applyMirrorOrganization(target, "missing.png", {
      item: {},
      collections: [],
    }),
  ).rejects.toThrow("unavailable");
  expect(api.patchGalleryImage).not.toHaveBeenCalled();
});
it("does not import stale hidden attributes after Show during transfer", async () => {
  const visibility = await import("../lib/collectionVisibility");
  visibility.resetCollectionVisibilityForTests();
  const revision = visibility.collectionVisibilityRevision();
  visibility.rememberCollectionVisibility("private", false, [
    {
      hostId: "local",
      target,
      instanceId: null,
      collections: [],
      listingOk: false,
    },
  ]);
  api.apiJsonTo.mockResolvedValue([{ filename: "copy.png" }]);
  api.listCollections.mockResolvedValue([
    { ...collection, id: "dest", hidden: false },
  ]);
  await applyMirrorOrganization(target, "copy.png", {
    item: {},
    collections: [collection],
    visibilityRevision: revision,
  });
  expect(api.updateCollectionHidden).not.toHaveBeenCalled();
  visibility.resetCollectionVisibilityForTests();
});
it("corrects an earlier mirror Hide when Show arrives during its write", async () => {
  const visibility = await import("../lib/collectionVisibility");
  visibility.resetCollectionVisibilityForTests();
  const revision = visibility.collectionVisibilityRevision();
  api.apiJsonTo.mockResolvedValue([{ filename: "copy.png" }]);
  api.listCollections.mockResolvedValue([
    { ...collection, id: "dest", hidden: false },
  ]);
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  api.updateCollectionHidden
    .mockImplementationOnce(async () => {
      await held;
      return { ...collection, id: "dest", hidden: true };
    })
    .mockResolvedValue({ ...collection, id: "dest", hidden: false });
  const task = applyMirrorOrganization(target, "copy.png", {
    item: {},
    collections: [collection],
    visibilityRevision: revision,
  });
  await vi.waitFor(() =>
    expect(api.updateCollectionHidden).toHaveBeenCalledTimes(1),
  );
  visibility.rememberCollectionVisibility("private", false, [
    {
      hostId: "local",
      target,
      instanceId: null,
      collections: [],
      listingOk: false,
    },
  ]);
  release();
  await task;
  expect(api.updateCollectionHidden.mock.calls.map((c) => c[2])).toEqual([
    true,
    false,
  ]);
  visibility.resetCollectionVisibilityForTests();
});
