import { describe, expect, it } from "vitest";
import {
  persistentThumbnailKey,
  persistentThumbnailStore,
  THUMBNAIL_STORE_MAX_ENTRIES,
  thumbnailRenditionQuery,
  thumbnailTier,
} from "./thumbnailPersistentCache";

/** A minimal in-memory `CacheStorage` with insertion-ordered keys. */
function fakeCacheStorage(
  opts: { failOpen?: boolean; failPut?: boolean } = {},
) {
  const entries = new Map<string, Response>();
  const cache = {
    async match(request: RequestInfo | URL) {
      const url =
        typeof request === "string" ? request : (request as Request).url;
      const hit = entries.get(url);
      return hit ? hit.clone() : undefined;
    },
    async put(request: RequestInfo | URL, response: Response) {
      if (opts.failPut) throw new DOMException("quota", "QuotaExceededError");
      const url =
        typeof request === "string" ? request : (request as Request).url;
      entries.set(url, response);
    },
    async keys() {
      return [...entries.keys()].map((url) => new Request(url));
    },
    async delete(request: RequestInfo | URL) {
      const url =
        typeof request === "string" ? request : (request as Request).url;
      return entries.delete(url);
    },
  };
  const storage = {
    async open() {
      if (opts.failOpen) throw new Error("no storage");
      return cache;
    },
  } as unknown as CacheStorage;
  return { storage, entries };
}

describe("thumbnail rendition policy", () => {
  it("asks for 512 px on retina and 256 otherwise, always JPEG", () => {
    expect(thumbnailTier(1)).toBe(256);
    expect(thumbnailTier(2)).toBe(512);
    expect(thumbnailRenditionQuery(512)).toBe("size=512&fmt=jpeg");
    expect(thumbnailRenditionQuery(256)).toBe("size=256&fmt=jpeg");
  });

  it("keys on host, filename, version and tier", () => {
    const a = persistentThumbnailKey("plato", "a.png", "1:10", 512);
    expect(a).not.toBe(persistentThumbnailKey("plato", "a.png", "2:10", 512));
    expect(a).not.toBe(persistentThumbnailKey("plato", "a.png", "1:10", 256));
    expect(a).not.toBe(persistentThumbnailKey("hal", "a.png", "1:10", 512));
  });
});

describe("persistentThumbnailStore", () => {
  it("is absent without the Cache API (plain-http origins)", () => {
    expect(persistentThumbnailStore(undefined)).toBeNull();
  });

  it("round-trips a tile and evicts by host prefix", async () => {
    const { storage, entries } = fakeCacheStorage();
    const store = persistentThumbnailStore(storage)!;
    const key = persistentThumbnailKey("plato", "a b.png", "1:10", 512);
    expect(await store.get(key)).toBeNull();
    await store.put(key, new Blob(["jpeg bytes"], { type: "image/jpeg" }));
    const hit = await store.get(key);
    expect(hit?.type).toBe("image/jpeg");
    expect(await hit!.text()).toBe("jpeg bytes");
    expect(entries.size).toBe(1);

    await store.put(
      persistentThumbnailKey("hal", "z.png", "1:1", 256),
      new Blob(["x"]),
    );
    await store.evictPrefix("plato|");
    expect(await store.get(key)).toBeNull();
    expect(entries.size).toBe(1);
  });

  it("never throws when storage refuses", async () => {
    const refused = persistentThumbnailStore(
      fakeCacheStorage({ failOpen: true }).storage,
    )!;
    expect(await refused.get("k")).toBeNull();
    await expect(refused.put("k", new Blob(["x"]))).resolves.toBeUndefined();
    const quota = persistentThumbnailStore(
      fakeCacheStorage({ failPut: true }).storage,
    )!;
    await expect(quota.put("k", new Blob(["x"]))).resolves.toBeUndefined();
  });

  it("prunes oldest entries past the bound", async () => {
    const { storage, entries } = fakeCacheStorage();
    const store = persistentThumbnailStore(storage)!;
    for (let i = 0; i < THUMBNAIL_STORE_MAX_ENTRIES + 64; i++) {
      await store.put(
        persistentThumbnailKey("h", `${i}.png`, "1:1", 256),
        new Blob(["x"]),
      );
    }
    expect(entries.size).toBeLessThanOrEqual(THUMBNAIL_STORE_MAX_ENTRIES);
    expect(
      await store.get(persistentThumbnailKey("h", "0.png", "1:1", 256)),
    ).toBeNull();
    expect(
      await store.get(
        persistentThumbnailKey(
          "h",
          `${THUMBNAIL_STORE_MAX_ENTRIES + 63}.png`,
          "1:1",
          256,
        ),
      ),
    ).not.toBeNull();
  });
});
