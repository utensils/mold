import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  MediaUpgradeRequiredError,
  __resetGalleryMediaForTests,
  evictHostMedia,
  fetchGalleryBlob,
  fetchGalleryThumbnailBlob,
  needsAuthedMedia,
  resolveStreamableSrc,
  resolveThumbnailSrc,
} from "./galleryMedia";
import { ORIGIN_HOST_ID, type HostEntry } from "./hostRegistry";

const origin: HostEntry = { id: ORIGIN_HOST_ID, name: "this server", url: "" };
const keylessRemote: HostEntry = {
  id: "hal9000-7680",
  name: "hal9000",
  url: "http://hal9000:7680",
};
const authedRemote: HostEntry = {
  id: "halcyon-7680",
  name: "halcyon",
  url: "http://halcyon:7680",
  apiKey: "secret-key",
};

const originalFetch = globalThis.fetch;
let objectUrlCounter = 0;
const revoked: string[] = [];

beforeEach(() => {
  __resetGalleryMediaForTests();
  objectUrlCounter = 0;
  revoked.length = 0;
  globalThis.URL.createObjectURL = vi.fn(
    () => `blob:mock-${++objectUrlCounter}`,
  );
  globalThis.URL.revokeObjectURL = vi.fn((u: string) => {
    revoked.push(u);
  });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

function mockFetch(handler: (url: string, init?: RequestInit) => unknown) {
  globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) =>
    handler(url, init),
  ) as never;
}

describe("needsAuthedMedia", () => {
  it("is false for the keyless origin and keyless remotes, true for keyed hosts", () => {
    expect(needsAuthedMedia(origin)).toBe(false);
    expect(needsAuthedMedia(keylessRemote)).toBe(false);
    expect(needsAuthedMedia(authedRemote)).toBe(true);
  });
});

describe("resolveThumbnailSrc", () => {
  it("returns a plain relative URL for the origin (no fetch)", async () => {
    mockFetch(() => {
      throw new Error("should not fetch");
    });
    const src = await resolveThumbnailSrc(origin, "cat.png");
    // The rendition query asks a current server for the display's tier as
    // JPEG; an older server ignores it and answers its 256 px PNG.
    expect(src).toBe("/api/gallery/thumbnail/cat.png?size=256&fmt=jpeg");
  });

  it("returns a plain absolute URL for a keyless remote", async () => {
    const src = await resolveThumbnailSrc(keylessRemote, "cat.png");
    expect(src).toBe(
      "http://hal9000:7680/api/gallery/thumbnail/cat.png?size=256&fmt=jpeg",
    );
  });

  it("blob-fetches an authed remote thumbnail with the key and caches it", async () => {
    const calls: Array<{ url: string; key?: string }> = [];
    mockFetch((url, init) => {
      calls.push({
        url,
        key: (init?.headers as Record<string, string>)?.["x-api-key"],
      });
      return { ok: true, blob: async () => new Blob(["x"]) };
    });
    const first = await resolveThumbnailSrc(authedRemote, "cat.png");
    const second = await resolveThumbnailSrc(authedRemote, "cat.png");
    expect(first).toBe("blob:mock-1");
    expect(second).toBe("blob:mock-1"); // cached — no second fetch
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe(
      "http://halcyon:7680/api/gallery/thumbnail/cat.png?size=256&fmt=jpeg",
    );
    expect(calls[0]!.key).toBe("secret-key");
  });

  it("serves an authed tile from the persistent store with zero fetches, and fills it on a miss", async () => {
    // A minimal Cache API: insertion-ordered, keyed by request URL.
    const stored = new Map<string, Blob>();
    const fakeCache = {
      match: async (url: string) => {
        const blob = stored.get(url);
        return blob ? new Response(blob) : undefined;
      },
      put: async (url: string, response: Response) => {
        stored.set(url, await response.blob());
      },
      keys: async () => [...stored.keys()].map((u) => new Request(u)),
      delete: async (url: string | Request) => {
        return stored.delete(typeof url === "string" ? url : url.url);
      },
    };
    (globalThis as { caches?: unknown }).caches = {
      open: async () => fakeCache,
    };
    try {
      let fetches = 0;
      mockFetch(() => {
        fetches += 1;
        return { ok: true, blob: async () => new Blob(["tile"]) };
      });
      // Miss: one fetch, then the bytes land in the store.
      await resolveThumbnailSrc(authedRemote, "cat.png", { mediaVersion: "1:10" });
      await new Promise((r) => setTimeout(r, 0));
      expect(fetches).toBe(1);
      expect(stored.size).toBe(1);

      // A fresh page (memory cache cleared) hits the store: zero fetches.
      __resetGalleryMediaForTests();
      (globalThis as { caches?: unknown }).caches = { open: async () => fakeCache };
      const src = await resolveThumbnailSrc(authedRemote, "cat.png", { mediaVersion: "1:10" });
      expect(src.startsWith("blob:")).toBe(true);
      expect(fetches).toBe(1);

      // A new content version is a different key: fetched again.
      await resolveThumbnailSrc(authedRemote, "cat.png", { mediaVersion: "2:10" });
      expect(fetches).toBe(2);
      // Without a version nothing persists (a "legacy" key could go stale).
      __resetGalleryMediaForTests();
      (globalThis as { caches?: unknown }).caches = { open: async () => fakeCache };
      await resolveThumbnailSrc(authedRemote, "dog.png");
      await new Promise((r) => setTimeout(r, 0));
      expect(stored.size).toBe(2);
    } finally {
      delete (globalThis as { caches?: unknown }).caches;
    }
  });
});

describe("resolveStreamableSrc", () => {
  it("returns the plain direct URL for a keyless host", async () => {
    const src = await resolveStreamableSrc(keylessRemote, "clip.mp4");
    expect(src).toBe("http://hal9000:7680/api/gallery/image/clip.mp4");
  });

  it("appends a media_token ticket for an authed host", async () => {
    mockFetch((url) => {
      expect(url).toBe("http://halcyon:7680/api/gallery/media-token");
      return {
        ok: true,
        status: 200,
        json: async () => ({
          token: "tkt",
          expires_at: 1900,
          auth_required: true,
        }),
      };
    });
    const src = await resolveStreamableSrc(authedRemote, "clip.mp4");
    const url = new URL(src);
    expect(url.origin + url.pathname).toBe(
      "http://halcyon:7680/api/gallery/image/clip.mp4",
    );
    expect(url.searchParams.get("media_token")).toBe("tkt");
    expect(url.searchParams.get("expires")).toBe("1900");
  });

  it("uses the plain URL when the host reports auth is disabled", async () => {
    mockFetch(() => ({
      ok: true,
      status: 200,
      json: async () => ({
        token: null,
        expires_at: null,
        auth_required: false,
      }),
    }));
    const src = await resolveStreamableSrc(authedRemote, "clip.mp4");
    expect(src).toBe("http://halcyon:7680/api/gallery/image/clip.mp4");
  });

  it("rejects images on hosts without the current ticket endpoint", async () => {
    mockFetch((url) => {
      if (url.endsWith("/media-token")) return { ok: false, status: 404 };
      return { ok: true, blob: async () => new Blob(["img"]) };
    });
    await expect(
      resolveStreamableSrc(authedRemote, "cat.png"),
    ).rejects.toBeInstanceOf(MediaUpgradeRequiredError);
  });

  it("raises MediaUpgradeRequiredError for videos on ticket-less hosts", async () => {
    mockFetch((url) => {
      if (url.endsWith("/media-token")) return { ok: false, status: 404 };
      throw new Error("must not blob-fetch a video");
    });
    await expect(
      resolveStreamableSrc(authedRemote, "clip.mp4"),
    ).rejects.toBeInstanceOf(MediaUpgradeRequiredError);
  });
});

describe("fetchGalleryBlob", () => {
  it("fetches the full-size media with the host key", async () => {
    let sentKey: string | undefined;
    mockFetch((url, init) => {
      expect(url).toBe("http://halcyon:7680/api/gallery/image/cat.png");
      sentKey = (init?.headers as Record<string, string>)?.["x-api-key"];
      return { ok: true, blob: async () => new Blob(["bytes"]) };
    });
    const blob = await fetchGalleryBlob(authedRemote, "cat.png");
    expect(blob).toBeInstanceOf(Blob);
    expect(sentKey).toBe("secret-key");
  });
});

describe("fetchGalleryThumbnailBlob", () => {
  it("fetches the exact encoded poster path with the host key", async () => {
    let sentKey: string | undefined;
    mockFetch((url, init) => {
      expect(url).toBe(
        "http://halcyon:7680/api/gallery/thumbnail/clip%20one.mp4",
      );
      sentKey = (init?.headers as Record<string, string>)?.["x-api-key"];
      return { ok: true, blob: async () => new Blob(["poster"]) };
    });

    const blob = await fetchGalleryThumbnailBlob(authedRemote, "clip one.mp4");
    expect(blob).toBeInstanceOf(Blob);
    expect(sentKey).toBe("secret-key");
  });
});

describe("evictHostMedia", () => {
  it("revokes cached object URLs for the host", async () => {
    mockFetch(() => ({ ok: true, blob: async () => new Blob(["x"]) }));
    const src = await resolveThumbnailSrc(authedRemote, "cat.png");
    evictHostMedia(authedRemote.id);
    // eviction revokes asynchronously off the cached promise
    await Promise.resolve();
    await Promise.resolve();
    expect(revoked).toContain(src);
  });
});
