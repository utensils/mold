import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IDBFactory } from "fake-indexeddb";

import {
  enqueueGalleryMutation,
  galleryBulkRequest,
  listGalleryMutations,
  removeGalleryMutation,
  updateGalleryMutationFailure,
} from "./galleryMutationOutbox";

beforeEach(() => {
  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: new IDBFactory(),
  });
});

afterEach(() => vi.restoreAllMocks());

describe("gallery mutation outbox", () => {
  it("retains secret-free host-bound work and removes it after replay", async () => {
    await enqueueGalleryMutation({
      id: "op-1",
      hostId: "hal9000",
      hostInstanceId: "instance-1",
      hostName: "hal9000",
      op: {
        kind: "addTags",
        hostId: "hal9000",
        filenames: ["a.png", "b.png"],
        tags: ["blue"],
      },
    });
    await updateGalleryMutationFailure("op-1", "offline");

    expect(await listGalleryMutations()).toEqual([
      expect.objectContaining({
        id: "op-1",
        hostInstanceId: "instance-1",
        attempts: 1,
        lastError: "offline",
      }),
    ]);
    expect(JSON.stringify(await listGalleryMutations())).not.toContain(
      "apiKey",
    );

    await removeGalleryMutation("op-1");
    expect(await listGalleryMutations()).toEqual([]);
  });

  it("turns a large title selection into one bulk request", () => {
    const filenames = Array.from({ length: 30 }, (_, index) => `${index}.png`);
    expect(
      galleryBulkRequest("op-30", {
        kind: "setTitle",
        hostId: "hal9000",
        filenames,
        title: "Moon studies",
      }),
    ).toEqual({
      operation_id: "op-30",
      filenames: [],
      titles: filenames.map((filename) => ({
        filename,
        title: "Moon studies",
      })),
    });
  });

  it("persists strict enqueue order when inverse edits share a millisecond", async () => {
    vi.spyOn(Date, "now").mockReturnValue(1000);
    await Promise.all([
      enqueueGalleryMutation({
        id: "z-add",
        hostId: "hal9000",
        hostInstanceId: "instance-1",
        hostName: "hal9000",
        op: {
          kind: "addTags",
          hostId: "hal9000",
          filenames: ["a.png"],
          tags: ["blue"],
        },
      }),
      enqueueGalleryMutation({
        id: "a-remove",
        hostId: "hal9000",
        hostInstanceId: "instance-1",
        hostName: "hal9000",
        op: {
          kind: "removeTags",
          hostId: "hal9000",
          filenames: ["a.png"],
          tags: ["blue"],
        },
      }),
    ]);
    expect((await listGalleryMutations()).map((item) => item.id)).toEqual([
      "z-add",
      "a-remove",
    ]);
  });
});
