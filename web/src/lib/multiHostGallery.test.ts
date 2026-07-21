import { describe, expect, it, vi } from "vitest";
import { fetchMergedGallery, makePrintKey, printKey } from "./multiHostGallery";
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";
import type { GalleryImage } from "../types";

function host(id: string, name: string): HostEntry {
  return { id, name, url: id === "origin" ? "" : `https://${id}.local` };
}

function img(filename: string, timestamp: number): GalleryImage {
  return {
    filename,
    timestamp,
    format: "png",
    metadata: { prompt: filename, model: "flux", seed: 1 } as never,
  };
}

describe("fetchMergedGallery", () => {
  it("fetches every host and merges newest-first with host tags", async () => {
    const origin = host("origin", "this server");
    const remote = host("bender", "bender");
    const fetcher = vi.fn(async (h: HostEntry) =>
      h.id === "origin"
        ? [img("o1.png", 100), img("o2.png", 300)]
        : [img("r1.png", 200)],
    );

    const res = await fetchMergedGallery([origin, remote], fetcher);

    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(res.entries.map((e) => e.filename)).toEqual([
      "o2.png",
      "r1.png",
      "o1.png",
    ]);
    // Each entry carries its origin host.
    const r1 = res.entries.find((e) => e.filename === "r1.png")!;
    expect(r1.hostId).toBe("bender");
    expect(r1.hostLabel).toBe("bender");
    expect(res.reachableHostIds).toEqual(["origin", "bender"]);
    expect(res.unreachableHostIds).toEqual([]);
  });

  it("degrades gracefully: a failing host is recorded, the rest still render", async () => {
    const origin = host("origin", "this server");
    const down = host("bender", "bender");
    const fetcher = vi.fn(async (h: HostEntry) => {
      if (h.id === "bender") throw new Error("ECONNREFUSED");
      return [img("o1.png", 100)];
    });

    const res = await fetchMergedGallery([origin, down], fetcher);

    expect(res.entries.map((e) => e.filename)).toEqual(["o1.png"]);
    expect(res.reachableHostIds).toEqual(["origin"]);
    expect(res.unreachableHostIds).toEqual(["bender"]);
  });

  it("never returns a blank grid just because one box is down", async () => {
    const origin = host("origin", "this server");
    const down = host("bender", "bender");
    const fetcher = vi.fn(async (h: HostEntry) => {
      if (h.id === "origin") return [img("o1.png", 100), img("o2.png", 200)];
      throw new Error("down");
    });
    const res = await fetchMergedGallery([origin, down], fetcher);
    expect(res.entries.length).toBe(2);
  });

  it("counts only non-origin hosts as remotes for the honest header", async () => {
    const origin = host("origin", "this server");
    const a = host("a", "a");
    const b = host("b", "b");
    const fetcher = vi.fn(async () => [] as GalleryImage[]);
    const solo = await fetchMergedGallery([origin], fetcher);
    expect(solo.remoteHostCount).toBe(0);
    const multi = await fetchMergedGallery([origin, a, b], fetcher);
    expect(multi.remoteHostCount).toBe(2);
  });
});

describe("printKey", () => {
  it("keeps same-named prints on different hosts apart", () => {
    // mold names prints model+seed+timestamp, so two boxes routinely hold the
    // same filename. Filename alone is not an identity.
    const mine = printKey({ hostId: ORIGIN_HOST_ID, filename: "flux_42.png" });
    const theirs = printKey({ hostId: "bender", filename: "flux_42.png" });
    expect(mine).not.toBe(theirs);
    expect(mine).toBe(makePrintKey(ORIGIN_HOST_ID, "flux_42.png"));
    expect(theirs).toBe(makePrintKey("bender", "flux_42.png"));
  });

  it("treats an untagged entry as the origin's", () => {
    expect(printKey({ filename: "cat.png" })).toBe(
      makePrintKey(ORIGIN_HOST_ID, "cat.png"),
    );
  });

  it("stays stable for the same entry", () => {
    const entry = { hostId: "bender", filename: "cat.png" };
    expect(printKey(entry)).toBe(printKey({ ...entry }));
  });
});
