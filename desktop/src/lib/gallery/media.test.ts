import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch, apiFetchTo, currentTarget } from "../api/client";
import {
  authedMediaUrl,
  evictHostMedia,
  evictMedia,
  galleryMediaPath,
  streamableMediaUrl,
} from "./media";

vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api/client")>()),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(),
}));

const blobResponse = () =>
  ({ blob: () => Promise.resolve(new Blob(["png bytes"])) }) as unknown as Response;

let objectUrlSeq = 0;
const revoked: string[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiFetch).mockImplementation(() => Promise.resolve(blobResponse()));
  vi.mocked(apiFetchTo).mockImplementation(() => Promise.resolve(blobResponse()));
  vi.mocked(currentTarget).mockReturnValue({ baseUrl: "http://primary:7680", apiKey: null });
  revoked.length = 0;
  URL.createObjectURL = vi.fn(() => `blob:mock-${++objectUrlSeq}`);
  URL.revokeObjectURL = vi.fn((url: string) => void revoked.push(url));
  vi.stubGlobal("fetch", vi.fn());
});

describe("streamableMediaUrl", () => {
  it("uses the direct host URL when no API key is required", async () => {
    const target = { baseUrl: "http://studio.tailnet.ts.net:7680", apiKey: null };

    await expect(
      streamableMediaUrl("/api/gallery/image/clip.mp4", { target, cacheKey: "studio" }),
    ).resolves.toBe("http://studio.tailnet.ts.net:7680/api/gallery/image/clip.mp4");
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("exchanges an API key for a short-lived streamable gallery URL", async () => {
    const target = { baseUrl: "https://studio.example", apiKey: "secret" };
    vi.mocked(apiFetchTo).mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          token: "short_lived-ticket",
          expires_at: 1_800_000_000,
          auth_required: true,
        }),
    } as Response);

    const url = await streamableMediaUrl("/api/gallery/image/clip%20one.mp4", {
      target,
      cacheKey: "studio",
    });

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/media-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "/api/gallery/image/clip%20one.mp4" }),
    });
    expect(url).toBe(
      "https://studio.example/api/gallery/image/clip%20one.mp4?media_token=short_lived-ticket&expires=1800000000",
    );
  });

  it("uses a ticket for the desktop loopback server without leaking its durable key", async () => {
    const target = { baseUrl: "http://127.0.0.1:49152", apiKey: "desktop-key" };
    vi.mocked(apiFetchTo).mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          token: "loopback-ticket",
          expires_at: 1_800_000_000,
          auth_required: true,
        }),
    } as Response);

    const url = await streamableMediaUrl("/api/gallery/image/clip.mp4", {
      target,
      cacheKey: "local",
    });

    expect(url).toBe(
      "http://127.0.0.1:49152/api/gallery/image/clip.mp4?media_token=loopback-ticket&expires=1800000000",
    );
    expect(url).not.toContain("desktop-key");
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/media-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "/api/gallery/image/clip.mp4" }),
    });
  });

  it("uses the direct URL when a current auth-disabled host sees a stale saved key", async () => {
    const target = { baseUrl: "http://studio:7680", apiKey: "stale-key" };
    vi.mocked(apiFetchTo).mockResolvedValueOnce({
      json: () => Promise.resolve({ token: null, expires_at: null, auth_required: false }),
    } as Response);

    await expect(
      streamableMediaUrl("/api/gallery/image/clip.mp4", { target, cacheKey: "studio" }),
    ).resolves.toBe("http://studio:7680/api/gallery/image/clip.mp4");
  });

  it("probes and directly streams video from an older auth-disabled host", async () => {
    const target = { baseUrl: "http://old-public:7680", apiKey: "stale-key" };
    vi.mocked(apiFetchTo).mockRejectedValueOnce(new ApiError("missing", 404));
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true } as Response);

    await expect(
      streamableMediaUrl("/api/gallery/image/clip.mp4", { target, cacheKey: "old" }),
    ).resolves.toBe("http://old-public:7680/api/gallery/image/clip.mp4");
    expect(fetch).toHaveBeenCalledWith("http://old-public:7680/api/gallery/image/clip.mp4", {
      method: "HEAD",
    });
  });

  it("allows still images on older hosts to use the bounded blob path", async () => {
    const target = { baseUrl: "http://old-host:7680", apiKey: "secret" };
    vi.mocked(apiFetchTo)
      .mockRejectedValueOnce(new ApiError("missing", 404))
      .mockResolvedValueOnce(blobResponse());
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false } as Response);

    await expect(
      streamableMediaUrl("/api/gallery/image/print.png", {
        target,
        cacheKey: "old",
        allowLegacyBlob: true,
      }),
    ).resolves.toMatch(/^blob:mock-/);
  });

  it("refuses to buffer videos from hosts without streaming tickets", async () => {
    const target = { baseUrl: "http://old-host:7680", apiKey: "secret" };
    vi.mocked(apiFetchTo).mockRejectedValueOnce(new ApiError("missing", 404));
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false } as Response);

    await expect(
      streamableMediaUrl("/api/gallery/image/clip.mp4", { target, cacheKey: "old" }),
    ).rejects.toThrow("Update this Mold host");
  });
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
