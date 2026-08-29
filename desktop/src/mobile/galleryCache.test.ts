import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it } from "vitest";
import type { GalleryImage } from "../lib/api/types";
import type { MobileGalleryImage } from "./libraryOrganization";
import {
  captureCachedHostFence,
  createThumbnailRouteGenerationRegistry,
  type CachedGalleryMediaRef,
  loadCachedGallery,
  loadCachedGalleryMedia,
  loadCachedHostPresentation,
  clearCachedGalleryHosts,
  patchCachedGalleryPrints,
  probeCachedGalleryMedia,
  removeCachedGalleryPrints,
  removeCachedGalleryRows,
  storeCachedGallery,
  storeCachedGalleryMedia,
  storeCachedHostPresentation,
} from "./galleryCache";
import type { ModelEntry } from "../lib/api/types";

function print(filename: string, timestamp: number): GalleryImage {
  return {
    filename,
    timestamp,
    format: "png",
    metadata: {
      prompt: "",
      model: "flux-dev:q8",
      seed: timestamp,
      steps: 4,
      guidance: 3.5,
      width: 512,
      height: 512,
    },
  };
}

function mediaRef(
  hostId: string,
  filename: string,
  mediaVersion = "1:8",
  tier: 256 | 512 = 256,
): CachedGalleryMediaRef {
  return { hostId, filename, mediaVersion, tier };
}

function thumbnail(label = "thumbnail"): Blob {
  return new Blob([Uint8Array.of(0x89, 0x50, 0x4e, 0x47), label], { type: "image/png" });
}

beforeEach(() => {
  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: new IDBFactory(),
  });
});

describe("mobile gallery cache", () => {
  it("keeps route generations stable when an old route reports after its replacement", () => {
    const generationFor = createThumbnailRouteGenerationRegistry();
    const oldRoute = { baseUrl: "http://old-host:7680", apiKey: "old-key" };
    const newRoute = { baseUrl: "http://new-host:7680", apiKey: "new-key" };
    const oldGeneration = generationFor("studio-instance", oldRoute);
    const newGeneration = generationFor("studio-instance", newRoute);

    // A late completion/retry from the superseded route must not mutate the
    // replacement route's identity and strand its in-flight tile.
    expect(generationFor("studio-instance", oldRoute)).toBe(oldGeneration);
    expect(generationFor("studio-instance", newRoute)).toBe(newGeneration);
    expect(newGeneration).not.toBe(oldGeneration);
  });

  it("persists a bounded newest-first gallery without connection secrets", async () => {
    const prints = Array.from({ length: 4_005 }, (_, index) => print(`${index}.png`, index));
    await storeCachedGallery("studio", prints);

    const restored = await loadCachedGallery("studio");
    expect(restored).toHaveLength(4_000);
    expect(restored[0]?.filename).toBe("4004.png");
    expect(restored.at(-1)?.filename).toBe("5.png");
    expect(JSON.stringify(restored)).not.toContain("apiKey");
  });

  it("round-trips thumbnail blobs for offline browsing", async () => {
    const blob = thumbnail();
    const ref = mediaRef("studio", "one.png");

    await storeCachedGalleryMedia(ref, blob);

    expect((await loadCachedGalleryMedia(ref))?.size).toBe(blob.size);
    expect(await probeCachedGalleryMedia([ref])).toEqual([true]);
  });

  it("separates content versions and rendition tiers", async () => {
    const v1 = mediaRef("studio", "one.png", "1:8", 256);
    const v2 = mediaRef("studio", "one.png", "2:8", 256);
    const retina = mediaRef("studio", "one.png", "1:8", 512);
    await storeCachedGalleryMedia(v1, thumbnail("v1"));
    await storeCachedGalleryMedia(retina, thumbnail("retina"));

    expect(await probeCachedGalleryMedia([v1, v2, retina])).toEqual([true, false, true]);
  });

  it("persists non-secret model presentation data under the server identity", async () => {
    const model = {
      name: "flux-dev:q8",
      family: "flux",
      downloaded: true,
    } as ModelEntry;
    await storeCachedHostPresentation({
      hostId: "instance-1",
      updatedAt: 42,
      instanceId: "instance-1",
      serverVersion: "1.2.3",
      models: [model],
      capabilities: null,
    });

    const restored = await loadCachedHostPresentation("instance-1");
    expect(restored).toMatchObject({
      instanceId: "instance-1",
      serverVersion: "1.2.3",
      models: [model],
    });
    expect(JSON.stringify(restored)).not.toContain("apiKey");
  });

  it("does not resurrect a presentation when a host is cleared during refresh", async () => {
    const fence = captureCachedHostFence("removed-instance");
    await clearCachedGalleryHosts(["removed-instance"]);
    await storeCachedHostPresentation(
      {
        hostId: "removed-instance",
        updatedAt: 42,
        instanceId: "removed-instance",
        serverVersion: "1.2.3",
        models: [],
        capabilities: null,
      },
      fence,
    );

    expect(await loadCachedHostPresentation("removed-instance")).toBeNull();
  });

  it("removes metadata and thumbnail bytes after a successful delete", async () => {
    const ref = mediaRef("studio", "delete.png");
    await storeCachedGallery("studio", [print("keep.png", 2), print("delete.png", 1)]);
    await storeCachedGalleryMedia(ref, thumbnail("delete"));

    await removeCachedGalleryPrints([{ hostId: "studio", filename: "delete.png" }]);

    expect((await loadCachedGallery("studio")).map((entry) => entry.filename)).toEqual([
      "keep.png",
    ]);
    expect(await loadCachedGalleryMedia(ref)).toBeNull();
  });

  it("removes live metadata but preserves every thumbnail when moving to Trash", async () => {
    const ref = mediaRef("studio", "trash.png");
    await storeCachedGallery("studio", [print("trash.png", 1)]);
    await storeCachedGalleryMedia(ref, thumbnail("trash"));

    await removeCachedGalleryRows([{ hostId: "studio", filename: "trash.png" }]);

    expect(await loadCachedGallery("studio")).toEqual([]);
    expect(await loadCachedGalleryMedia(ref)).not.toBeNull();
  });

  it("does not resurrect thumbnail bytes when delete wins a pending cache write", async () => {
    let finishBytes!: (bytes: ArrayBuffer) => void;
    const pendingBlob = thumbnail("pending");
    Object.defineProperty(pendingBlob, "arrayBuffer", {
      value: () =>
        new Promise<ArrayBuffer>((resolve) => {
          finishBytes = resolve;
        }),
    });
    const ref = mediaRef("studio", "delete.png");
    const write = storeCachedGalleryMedia(ref, pendingBlob);
    await Promise.resolve();

    await removeCachedGalleryPrints([{ hostId: "studio", filename: "delete.png" }]);
    finishBytes(new Uint8Array(await thumbnail("late").arrayBuffer()).buffer);
    await write;

    expect(await loadCachedGalleryMedia(ref)).toBeNull();
  });

  it("purges instance-scoped metadata and media without allowing pending writes back", async () => {
    await storeCachedGallery("old-instance", [print("old.png", 1)]);
    let finishBytes!: (bytes: ArrayBuffer) => void;
    const pendingBlob = thumbnail("pending");
    Object.defineProperty(pendingBlob, "arrayBuffer", {
      value: () =>
        new Promise<ArrayBuffer>((resolve) => {
          finishBytes = resolve;
        }),
    });
    const ref = mediaRef("old-instance", "old.png");
    const write = storeCachedGalleryMedia(ref, pendingBlob);
    await storeCachedHostPresentation({
      hostId: "old-instance",
      updatedAt: 1,
      instanceId: "old-instance",
      serverVersion: "1.0.0",
      models: [],
      capabilities: null,
    });
    await Promise.resolve();

    await clearCachedGalleryHosts(["old-instance"]);
    finishBytes(new Uint8Array(await thumbnail("late").arrayBuffer()).buffer);
    await write;

    expect(await loadCachedGallery("old-instance")).toEqual([]);
    expect(await loadCachedGalleryMedia(ref)).toBeNull();
    expect(await loadCachedHostPresentation("old-instance")).toBeNull();
  });
});

describe("mobile gallery cache organization fields", () => {
  it("persists additive title / favorite / tags / collections through a round trip", async () => {
    const organized = {
      ...print("org.png", 5),
      title: "Grain test",
      favorite: true,
      tags: ["blue", "smurf"],
      collections: ["c1"],
    };
    await storeCachedGallery("studio", [organized]);

    const [restored] = (await loadCachedGallery("studio")) as Array<typeof organized>;
    expect(restored?.title).toBe("Grain test");
    expect(restored?.favorite).toBe(true);
    expect(restored?.tags).toEqual(["blue", "smurf"]);
    expect(restored?.collections).toEqual(["c1"]);
  });

  it("patches cached rows in place and fences out an older listing still in flight", async () => {
    await storeCachedGallery("studio", [print("a.png", 2), print("b.png", 1)]);

    // An older refresh captured the rows before the edit…
    const staleRow: MobileGalleryImage = { ...print("a.png", 2), favorite: false };
    const stale = storeCachedGallery("studio", [staleRow]);
    await patchCachedGalleryPrints("studio", [
      { filename: "a.png", patch: { favorite: true, title: "Named" } },
    ]);
    await stale;

    const rows = (await loadCachedGallery("studio")) as Array<
      ReturnType<typeof print> & { favorite?: boolean; title?: string | null }
    >;
    expect(rows.find((row) => row.filename === "a.png")).toMatchObject({
      favorite: true,
      title: "Named",
    });
    expect(rows.find((row) => row.filename === "b.png")).toBeDefined();
  });
});
