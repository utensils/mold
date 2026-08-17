import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it } from "vitest";
import type { GalleryImage } from "../lib/api/types";
import {
  loadCachedGallery,
  loadCachedGalleryMedia,
  clearCachedGalleryHosts,
  removeCachedGalleryPrints,
  storeCachedGallery,
  storeCachedGalleryMedia,
} from "./galleryCache";

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

beforeEach(() => {
  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: new IDBFactory(),
  });
});

describe("mobile gallery cache", () => {
  it("persists a bounded newest-first gallery without connection secrets", async () => {
    const prints = Array.from({ length: 505 }, (_, index) => print(`${index}.png`, index));
    await storeCachedGallery("studio", prints);

    const restored = await loadCachedGallery("studio");
    expect(restored).toHaveLength(500);
    expect(restored[0]?.filename).toBe("504.png");
    expect(restored.at(-1)?.filename).toBe("5.png");
    expect(JSON.stringify(restored)).not.toContain("apiKey");
  });

  it("round-trips thumbnail blobs for offline browsing", async () => {
    const thumbnail = new Blob(["thumbnail"], { type: "image/webp" });

    await storeCachedGalleryMedia("studio", "one.png", "thumbnail", thumbnail);

    expect(await (await loadCachedGalleryMedia("studio", "one.png", "thumbnail"))?.text()).toBe(
      "thumbnail",
    );
  });

  it("removes metadata and thumbnail bytes after a successful delete", async () => {
    await storeCachedGallery("studio", [print("keep.png", 2), print("delete.png", 1)]);
    await storeCachedGalleryMedia("studio", "delete.png", "thumbnail", new Blob(["thumb"]));

    await removeCachedGalleryPrints([{ hostId: "studio", filename: "delete.png" }]);

    expect((await loadCachedGallery("studio")).map((entry) => entry.filename)).toEqual([
      "keep.png",
    ]);
    expect(await loadCachedGalleryMedia("studio", "delete.png", "thumbnail")).toBeNull();
  });

  it("does not resurrect thumbnail bytes when delete wins a pending cache write", async () => {
    let finishBytes!: (bytes: ArrayBuffer) => void;
    const pendingBlob = {
      size: 5,
      type: "image/webp",
      arrayBuffer: () =>
        new Promise<ArrayBuffer>((resolve) => {
          finishBytes = resolve;
        }),
    } as Blob;
    const write = storeCachedGalleryMedia("studio", "delete.png", "thumbnail", pendingBlob);
    await Promise.resolve();

    await removeCachedGalleryPrints([{ hostId: "studio", filename: "delete.png" }]);
    finishBytes(new TextEncoder().encode("thumb").buffer);
    await write;

    expect(await loadCachedGalleryMedia("studio", "delete.png", "thumbnail")).toBeNull();
  });

  it("purges instance-scoped metadata and media without allowing pending writes back", async () => {
    await storeCachedGallery("old-instance", [print("old.png", 1)]);
    let finishBytes!: (bytes: ArrayBuffer) => void;
    const pendingBlob = {
      size: 5,
      type: "image/webp",
      arrayBuffer: () =>
        new Promise<ArrayBuffer>((resolve) => {
          finishBytes = resolve;
        }),
    } as Blob;
    const write = storeCachedGalleryMedia("old-instance", "old.png", "thumbnail", pendingBlob);
    await Promise.resolve();

    await clearCachedGalleryHosts(["old-instance"]);
    finishBytes(new TextEncoder().encode("thumb").buffer);
    await write;

    expect(await loadCachedGallery("old-instance")).toEqual([]);
    expect(await loadCachedGalleryMedia("old-instance", "old.png", "thumbnail")).toBeNull();
  });
});
