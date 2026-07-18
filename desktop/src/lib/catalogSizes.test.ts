import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetCatalogSizeCache, resolveEntrySize } from "./catalogSizes";
import { fetchCatalogDetail } from "./api/catalog";
import type { CatalogEntry } from "./api/types";

// Size resolution reads THROUGH the shared detail fetch — one code path for
// `GET /api/catalog/:id` (drawer detail + lazy card sizes).
vi.mock("./api/catalog", () => ({
  fetchCatalogDetail: vi.fn(),
}));

const apiJsonMock = vi.mocked(fetchCatalogDetail);

function entry(part: Partial<CatalogEntry>): CatalogEntry {
  return {
    id: "hf:author/model",
    source: "hf",
    name: "model",
    family: "flux",
    kind: "checkpoint",
    nsfw: false,
    installed: false,
    size_bytes: null,
    ...part,
  };
}

/** Let queued microtasks (promise chains) settle. */
async function flush(): Promise<void> {
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
}

beforeEach(() => {
  resetCatalogSizeCache();
  apiJsonMock.mockReset();
});

describe("resolveEntrySize", () => {
  it("returns size_bytes immediately without fetching", async () => {
    const size = await resolveEntrySize(entry({ size_bytes: 6_400_000_000 }));
    expect(size).toBe(6_400_000_000);
    expect(apiJsonMock).not.toHaveBeenCalled();
  });

  it("resolves the size through the shared catalog detail fetch", async () => {
    apiJsonMock.mockResolvedValueOnce(entry({ size_bytes: 123 }));
    const size = await resolveEntrySize(entry({ id: "hf:author/model" }));
    expect(size).toBe(123);
    expect(apiJsonMock).toHaveBeenCalledWith("hf:author/model");
  });

  it("memoizes per id — one fetch even for concurrent callers", async () => {
    apiJsonMock.mockResolvedValue(entry({ size_bytes: 42 }));
    const e = entry({ id: "cv:8001" });
    const [a, b] = await Promise.all([resolveEntrySize(e), resolveEntrySize(e)]);
    const c = await resolveEntrySize(e);
    expect([a, b, c]).toEqual([42, 42, 42]);
    expect(apiJsonMock).toHaveBeenCalledTimes(1);
  });

  it("resolves null on failure without caching it — a later render retries", async () => {
    apiJsonMock.mockRejectedValueOnce(new Error("upstream 502"));
    apiJsonMock.mockResolvedValueOnce(entry({ size_bytes: 55 }));
    const e = entry({ id: "cv:9999" });
    expect(await resolveEntrySize(e)).toBeNull();
    expect(await resolveEntrySize(e)).toBe(55);
    expect(apiJsonMock).toHaveBeenCalledTimes(2);
  });

  it("shares one in-flight fetch between concurrent callers even when it fails", async () => {
    apiJsonMock.mockRejectedValue(new Error("upstream 502"));
    const e = entry({ id: "cv:9998" });
    const [a, b] = await Promise.all([resolveEntrySize(e), resolveEntrySize(e)]);
    expect([a, b]).toEqual([null, null]);
    expect(apiJsonMock).toHaveBeenCalledTimes(1);
  });

  it("treats a missing size_bytes in the detail as null", async () => {
    apiJsonMock.mockResolvedValueOnce(entry({ size_bytes: null }));
    expect(await resolveEntrySize(entry({ id: "cv:777" }))).toBeNull();
  });

  it("caps in-flight lookups at 4 and drains the queue as slots free up", async () => {
    const resolvers: Array<(value: CatalogEntry) => void> = [];
    apiJsonMock.mockImplementation(
      () =>
        new Promise<CatalogEntry>((resolve) => {
          resolvers.push(resolve);
        }) as Promise<never>,
    );

    const pending = Array.from({ length: 6 }, (_, i) => resolveEntrySize(entry({ id: `cv:${i}` })));
    await flush();
    expect(apiJsonMock).toHaveBeenCalledTimes(4);

    resolvers[0]?.(entry({ size_bytes: 1 }));
    await flush();
    expect(apiJsonMock).toHaveBeenCalledTimes(5);

    resolvers[1]?.(entry({ size_bytes: 2 }));
    await flush();
    expect(apiJsonMock).toHaveBeenCalledTimes(6);

    for (const resolve of resolvers.slice(2)) resolve(entry({ size_bytes: 9 }));
    const sizes = await Promise.all(pending);
    expect(sizes).toEqual([1, 2, 9, 9, 9, 9]);
  });

  it("resetCatalogSizeCache clears memoized results", async () => {
    apiJsonMock.mockResolvedValue(entry({ size_bytes: 7 }));
    await resolveEntrySize(entry({ id: "cv:1" }));
    resetCatalogSizeCache();
    await resolveEntrySize(entry({ id: "cv:1" }));
    expect(apiJsonMock).toHaveBeenCalledTimes(2);
  });
});
