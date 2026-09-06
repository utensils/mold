import { beforeEach, describe, expect, it, vi } from "vitest";

const deleteDraftMediaByPrefix = vi.fn(
  async (_prefix: string) => [] as string[],
);
vi.mock("./draftMediaStore", () => ({
  deleteDraftMediaByPrefix: (prefix: string) =>
    deleteDraftMediaByPrefix(prefix),
}));

import { retireSequenceStorage } from "./retireSequenceStorage";

describe("retireSequenceStorage", () => {
  beforeEach(() => {
    localStorage.clear();
    deleteDraftMediaByPrefix.mockClear();
    deleteDraftMediaByPrefix.mockResolvedValue([]);
  });

  it("frees every key the retired sequence store owned", async () => {
    localStorage.setItem("mold.sequence.draft.v1", '{"clips":[]}');
    localStorage.setItem("mold.chain.draft.v2", "{}");
    localStorage.setItem("mold.composer.mode", "sequence");
    localStorage.setItem("mold.mobile.create-mode.v1", "sequence");
    localStorage.setItem("mold.create.tracked-sequences.v1", "[]");
    localStorage.setItem("mold.create.chain-job-host", "plato");

    const removed = await retireSequenceStorage();

    expect(removed.keys).toHaveLength(6);
    for (const key of removed.keys)
      expect(localStorage.getItem(key)).toBeNull();
  });

  // The clip and opening-image blobs share the ONE-SHOT composer's database,
  // so they cannot be dropped with the store: they have to be swept by the
  // prefix only sequence media ever used.
  it("sweeps the orphaned sequence media out of the shared draft database", async () => {
    deleteDraftMediaByPrefix.mockResolvedValue([
      "sequence-opening-image",
      "sequence-clip-3",
    ]);

    const removed = await retireSequenceStorage();

    expect(deleteDraftMediaByPrefix).toHaveBeenCalledWith("sequence-");
    expect(removed.media).toEqual([
      "sequence-opening-image",
      "sequence-clip-3",
    ]);
  });

  it("is a no-op on a browser that never authored a sequence", async () => {
    localStorage.setItem("mold.generate.jobs", '{"version":1,"jobs":[]}');

    const removed = await retireSequenceStorage();

    expect(removed.keys).toEqual([]);
    expect(localStorage.getItem("mold.generate.jobs")).toBe(
      '{"version":1,"jobs":[]}',
    );
  });

  // Boot must never depend on it: a blocked database frees no blobs, and the
  // localStorage keys still have to go.
  it("still frees the keys when the media database refuses", async () => {
    localStorage.setItem("mold.sequence.draft.v1", "{ broken");
    deleteDraftMediaByPrefix.mockRejectedValue(new Error("blocked"));

    const removed = await retireSequenceStorage();

    expect(removed.keys).toEqual(["mold.sequence.draft.v1"]);
    expect(removed.media).toEqual([]);
    expect(localStorage.getItem("mold.sequence.draft.v1")).toBeNull();
  });
});
