import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch, apiFetchTo, currentTarget } from "../api/client";
import { inTauri, ipc } from "../ipc";
import {
  authedMediaUrl,
  evictHostMedia,
  evictMedia,
  fetchGalleryMediaBytes,
  fullSizeMediaUrl,
  galleryFilenameOfPath,
  galleryMediaPath,
  mediaMimeType,
  streamableMediaUrl,
} from "./media";

vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api/client")>()),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(),
}));
vi.mock("../ipc", () => ({
  inTauri: vi.fn(),
  ipc: { fetchGalleryThumbnail: vi.fn(), fetchGalleryMedia: vi.fn() },
}));

const blobResponse = () =>
  ({ blob: () => Promise.resolve(new Blob(["png bytes"])) }) as unknown as Response;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

let objectUrlSeq = 0;
const revoked: string[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiFetch).mockImplementation(() => Promise.resolve(blobResponse()));
  vi.mocked(apiFetchTo).mockImplementation(() => Promise.resolve(blobResponse()));
  vi.mocked(currentTarget).mockReturnValue({ baseUrl: "http://primary:7680", apiKey: null });
  vi.mocked(inTauri).mockReturnValue(false);
  vi.mocked(ipc.fetchGalleryThumbnail).mockReset();
  vi.mocked(ipc.fetchGalleryMedia).mockReset();
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

describe("fullSizeMediaUrl", () => {
  const target = { baseUrl: "http://hal9000:7680", apiKey: null };

  it("fetches desktop full-size media through native HTTP so held streams cannot starve it", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1, 2, 3]).buffer);

    const url = await fullSizeMediaUrl("/api/gallery/image/mold%20print.png", {
      target,
      cacheKey: "hal9000-7680",
    });

    expect(url).toMatch(/^blob:mock-/);
    expect(ipc.fetchGalleryMedia).toHaveBeenCalledWith(target, "mold print.png");
    // Never pointed the media element straight at the host.
    expect(apiFetchTo).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
    const blob = vi.mocked(URL.createObjectURL).mock.calls[0]?.[0] as Blob;
    expect(blob.type).toBe("image/png");
  });

  it("caches the native object URL per host bucket and path", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1]).buffer);

    const first = await fullSizeMediaUrl("/api/gallery/image/a.png", { target, cacheKey: "h" });
    const second = await fullSizeMediaUrl("/api/gallery/image/a.png", { target, cacheKey: "h" });
    const other = await fullSizeMediaUrl("/api/gallery/image/a.png", { target, cacheKey: "g" });

    expect(first).toBe(second);
    expect(other).not.toBe(first);
    expect(ipc.fetchGalleryMedia).toHaveBeenCalledTimes(2);
  });

  it("falls back to the streaming URL when the native route refuses", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockRejectedValue(
      new Error("The gallery file is unexpectedly large."),
    );

    await expect(
      fullSizeMediaUrl("/api/gallery/image/huge.png", { target, cacheKey: "h" }),
    ).resolves.toBe("http://hal9000:7680/api/gallery/image/huge.png");
    // A refused fetch must not poison the cache for the next attempt.
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1]).buffer);
    await expect(
      fullSizeMediaUrl("/api/gallery/image/huge.png", { target, cacheKey: "h" }),
    ).resolves.toMatch(/^blob:mock-/);
  });

  it("keeps video on the Range-friendly streaming URL instead of buffering it natively", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    await expect(
      fullSizeMediaUrl("/api/gallery/image/clip.mp4", { target, cacheKey: "h" }),
    ).resolves.toBe("http://hal9000:7680/api/gallery/image/clip.mp4");
    expect(ipc.fetchGalleryMedia).not.toHaveBeenCalled();
  });

  it("uses the streaming route outside Tauri and for mold-local paths", async () => {
    vi.mocked(inTauri).mockReturnValue(false);
    await expect(
      fullSizeMediaUrl("/api/gallery/image/a.png", { target, cacheKey: "h" }),
    ).resolves.toBe("http://hal9000:7680/api/gallery/image/a.png");
    expect(ipc.fetchGalleryMedia).not.toHaveBeenCalled();

    vi.mocked(inTauri).mockReturnValue(true);
    await expect(fullSizeMediaUrl("mold-local://localhost/a.png", { target })).resolves.toBe(
      "mold-local://localhost/a.png",
    );
    expect(ipc.fetchGalleryMedia).not.toHaveBeenCalled();
  });

  it("reads raw bytes natively and falls back to authenticated HTTP when refused", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([7, 8]).buffer);
    await expect(fetchGalleryMediaBytes("/api/gallery/image/a.png", target)).resolves.toEqual(
      new Uint8Array([7, 8]),
    );
    expect(apiFetchTo).not.toHaveBeenCalled();

    vi.mocked(ipc.fetchGalleryMedia).mockRejectedValue(new Error("refused"));
    vi.mocked(apiFetchTo).mockResolvedValue(
      new Response(new Uint8Array([9]), { status: 200 }) as unknown as Response,
    );
    await expect(fetchGalleryMediaBytes("/api/gallery/image/a.png", target)).resolves.toEqual(
      new Uint8Array([9]),
    );
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/a.png");
  });

  it("honours the caller's video flag over the filename extension", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1]).buffer);
    await expect(
      fullSizeMediaUrl("/api/gallery/image/anim.webp", { target, cacheKey: "h", video: true }),
    ).resolves.toBe("http://hal9000:7680/api/gallery/image/anim.webp");
    expect(ipc.fetchGalleryMedia).not.toHaveBeenCalled();
  });

  it("accepts postMessage-fallback number arrays as native bytes", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue([1, 2, 3]);
    await expect(
      fullSizeMediaUrl("/api/gallery/image/arr.png", { target, cacheKey: "h" }),
    ).resolves.toMatch(/^blob:mock-/);
    const blob = vi.mocked(URL.createObjectURL).mock.calls.at(-1)?.[0] as Blob;
    expect(blob.size).toBe(3);
    await expect(fetchGalleryMediaBytes("/api/gallery/image/arr.png", target)).resolves.toEqual(
      new Uint8Array([1, 2, 3]),
    );
  });

  it("keeps only a bounded LRU of full-size blobs and revokes the evicted ones", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1]).buffer);
    const urls: string[] = [];
    for (let i = 0; i < 10; i++) {
      urls.push(
        await fullSizeMediaUrl(`/api/gallery/image/lru-${i}.png`, { target, cacheKey: "lru" }),
      );
    }
    // The two oldest were evicted and revoked; the newest eight are retained.
    expect(revoked).toEqual(expect.arrayContaining([urls[0]!, urls[1]!]));
    expect(revoked).not.toContain(urls[9]!);
    const calls = vi.mocked(ipc.fetchGalleryMedia).mock.calls.length;
    await fullSizeMediaUrl("/api/gallery/image/lru-9.png", { target, cacheKey: "lru" });
    expect(vi.mocked(ipc.fetchGalleryMedia).mock.calls.length).toBe(calls);
    await fullSizeMediaUrl("/api/gallery/image/lru-0.png", { target, cacheKey: "lru" });
    expect(vi.mocked(ipc.fetchGalleryMedia).mock.calls.length).toBe(calls + 1);
  });

  it("evictMedia and evictHostMedia drop full-size blobs too", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryMedia).mockResolvedValue(new Uint8Array([1]).buffer);
    const a = await fullSizeMediaUrl("/api/gallery/image/ev-a.png", { target, cacheKey: "ev" });
    const b = await fullSizeMediaUrl("/api/gallery/image/ev-b.png", { target, cacheKey: "ev" });
    evictMedia("/api/gallery/image/ev-a.png", "ev");
    await Promise.resolve();
    expect(revoked).toContain(a);
    expect(revoked).not.toContain(b);
    evictHostMedia("ev");
    await Promise.resolve();
    expect(revoked).toContain(b);
  });

  it("derives the gallery filename and MIME type from the media path", () => {
    expect(galleryFilenameOfPath("/api/gallery/image/a%20b.PNG")).toBe("a b.PNG");
    expect(galleryFilenameOfPath("/api/gallery/thumbnail/a.png")).toBeNull();
    expect(galleryFilenameOfPath("/api/gallery/image/")).toBeNull();
    expect(galleryFilenameOfPath("/api/gallery/image/a/b.png")).toBeNull();
    expect(mediaMimeType("a b.PNG")).toBe("image/png");
    expect(mediaMimeType("clip.mp4")).toBe("video/mp4");
    expect(mediaMimeType("tone.wav")).toBe("audio/wav");
    expect(mediaMimeType("mystery.bin")).toBe("application/octet-stream");
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

  it("marks Trash-view local media so the native protocol reads `.trash/`", () => {
    // A trashed row must never be shadowed by a newer live file under the
    // same name — the query flips the protocol's live-first resolution.
    expect(galleryMediaPath("print one.png", "local", true, true)).toBe(
      "mold-local://localhost/print%20one.png?view=trash",
    );
    expect(galleryMediaPath("print one.png", "local", false, true)).toBe(
      "mold-local://localhost/print%20one.png?view=trash",
    );
    // Host media stays on the plain API path: the origin server resolves
    // its own trashed rows into `.trash/`.
    expect(galleryMediaPath("print one.png", "host", false, true)).toBe(
      "/api/gallery/image/print%20one.png",
    );
  });
});

describe("authedMediaUrl host-keyed cache", () => {
  it("loads desktop thumbnails through native HTTP so held generation streams cannot starve them", async () => {
    vi.mocked(inTauri).mockReturnValue(true);
    vi.mocked(ipc.fetchGalleryThumbnail).mockResolvedValue({
      base64: btoa("thumbnail bytes"),
      contentType: "image/png",
    });
    const target = { baseUrl: "http://hal9000:7680", apiKey: "hk" };

    await expect(
      authedMediaUrl("/api/gallery/thumbnail/new%20print.png", {
        target,
        cacheKey: "hal9000-7680",
      }),
    ).resolves.toMatch(/^blob:mock-/);

    expect(ipc.fetchGalleryThumbnail).toHaveBeenCalledWith(target, "new print.png");
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

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

  it("does not reuse an in-flight thumbnail after the host route changes", async () => {
    const oldResponse = deferred<Response>();
    const oldTarget = { baseUrl: "http://hal9000:7680", apiKey: "old-key" };
    const newTarget = { baseUrl: "http://hal9000:7680", apiKey: "new-key" };
    vi.mocked(apiFetchTo).mockImplementation((target) =>
      target.apiKey === "old-key" ? oldResponse.promise : Promise.resolve(blobResponse()),
    );
    const path = "/api/gallery/thumbnail/reconnected.png";

    const oldLoad = authedMediaUrl(path, { target: oldTarget, cacheKey: "hal9000-7680" });
    await Promise.resolve();
    const newLoad = authedMediaUrl(path, { target: newTarget, cacheKey: "hal9000-7680" });
    await expect(newLoad).resolves.toMatch(/^blob:mock-/);

    expect(apiFetchTo).toHaveBeenCalledTimes(2);
    expect(apiFetchTo).toHaveBeenLastCalledWith(newTarget, path);
    oldResponse.resolve(blobResponse());
    await expect(oldLoad).resolves.toMatch(/^blob:mock-/);
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
