import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiFetch, apiFetchTo } from "../api/client";
import { authedMediaUrl, evictHostMedia, evictMedia, galleryMediaPath } from "./media";

vi.mock("../api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));

const blobResponse = () =>
  ({ blob: () => Promise.resolve(new Blob(["png bytes"])) }) as unknown as Response;

let objectUrlSeq = 0;
const revoked: string[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiFetch).mockImplementation(() => Promise.resolve(blobResponse()));
  vi.mocked(apiFetchTo).mockImplementation(() => Promise.resolve(blobResponse()));
  revoked.length = 0;
  URL.createObjectURL = vi.fn(() => `blob:mock-${++objectUrlSeq}`);
  URL.revokeObjectURL = vi.fn((url: string) => void revoked.push(url));
});

describe("galleryMediaPath", () => {
  it("keeps host gallery media on the authenticated API", () => {
    expect(galleryMediaPath("print one.png", "host")).toBe("/api/gallery/image/print%20one.png");
    expect(galleryMediaPath("print one.png", "host", true)).toBe(
      "/api/gallery/thumbnail/print%20one.png",
    );
  });

  it("routes local gallery media through the restricted native protocol", () => {
    expect(galleryMediaPath("print one.png", "local", true)).toBe(
      "mold-local://localhost/print%20one.png",
    );
  });
});

describe("authedMediaUrl host-keyed cache", () => {
  it("caches the same path separately per cacheKey", async () => {
    const path = "/api/gallery/thumbnail/same.png";
    const target = { baseUrl: "http://hal9000:7680", apiKey: "hk" };

    const a = await authedMediaUrl(path, { target, cacheKey: "hal9000-7680" });
    const b = await authedMediaUrl(path, { target, cacheKey: "okra-7680" });

    expect(a).not.toBe(b);
    expect(apiFetchTo).toHaveBeenCalledTimes(2);

    // A repeat on either key hits the cache — no third fetch.
    expect(await authedMediaUrl(path, { target, cacheKey: "hal9000-7680" })).toBe(a);
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
  });

  it("uses the primary connection when no target is given", async () => {
    await authedMediaUrl("/api/gallery/thumbnail/primary.png");
    expect(apiFetch).toHaveBeenCalledWith("/api/gallery/thumbnail/primary.png");
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("passes mold-local URLs through untouched", async () => {
    const url = await authedMediaUrl("mold-local://localhost/x.png", { cacheKey: "local" });
    expect(url).toBe("mold-local://localhost/x.png");
    expect(apiFetch).not.toHaveBeenCalled();
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("evictMedia drops one keyed entry only", async () => {
    const path = "/api/gallery/image/evict-one.png";
    const target = { baseUrl: "http://hal9000:7680", apiKey: null };
    const a = await authedMediaUrl(path, { target, cacheKey: "hostA" });
    const b = await authedMediaUrl(path, { target, cacheKey: "hostB" });

    evictMedia(path, "hostA");
    await Promise.resolve();

    expect(revoked).toEqual([a]);
    // hostB stays cached; hostA refetches.
    expect(await authedMediaUrl(path, { target, cacheKey: "hostB" })).toBe(b);
    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    await authedMediaUrl(path, { target, cacheKey: "hostA" });
    expect(apiFetchTo).toHaveBeenCalledTimes(3);
  });

  it("evictHostMedia clears one host's whole bucket and nothing else", async () => {
    const target = { baseUrl: "http://hal9000:7680", apiKey: null };
    const a1 = await authedMediaUrl("/api/gallery/image/a1.png", { target, cacheKey: "gone" });
    const a2 = await authedMediaUrl("/api/gallery/thumbnail/a1.png", { target, cacheKey: "gone" });
    const kept = await authedMediaUrl("/api/gallery/image/a1.png", { target, cacheKey: "kept" });

    evictHostMedia("gone");
    await Promise.resolve();

    expect(revoked.sort()).toEqual([a1, a2].sort());
    expect(await authedMediaUrl("/api/gallery/image/a1.png", { target, cacheKey: "kept" })).toBe(
      kept,
    );
    expect(apiFetchTo).toHaveBeenCalledTimes(3);
  });
});
