import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __resetCatalogSingletonForTests, useCatalog } from "./useCatalog";

const originalFetch = globalThis.fetch;

// Make a 48-row entry array for a given page so we can verify append vs replace.
function makePageEntries(page: number, size: number) {
  return Array.from({ length: size }, (_, i) => {
    const idx = (page - 1) * size + i;
    return {
      id: `hf:row-${idx}`,
      name: `Row ${idx}`,
      family: "flux",
      supported: true,
      installed: false,
      source: "hf",
      source_id: `r${idx}`,
      author: null,
      family_role: "foundation",
      sub_family: null,
      modality: "image",
      kind: "checkpoint",
      file_format: "safetensors",
      bundling: "separated",
      size_bytes: 1,
      download_count: 100,
      rating: null,
      likes: 0,
      nsfw: false,
      thumbnail_url: null,
      description: null,
      license: null,
      license_flags: null,
      tags: [],
      companions: [],
      download_recipe: { files: [], needs_token: null },
      primary_path: null,
      created_at: null,
      updated_at: null,
      added_at: 0,
    };
  });
}

function installFetchMock(opts: { total: number; pageSize: number }) {
  globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
    if (url.startsWith("/api/catalog/families")) {
      return {
        ok: true,
        json: async () => ({
          families: [{ family: "flux" }],
        }),
      };
    }
    if (url.startsWith("/api/catalog/search")) {
      const params = new URL(url, "http://localhost").searchParams;
      const page = Number(params.get("page") ?? "1");
      const pageSize = Number(params.get("page_size") ?? String(opts.pageSize));
      const offset = (page - 1) * pageSize;
      const remaining = Math.max(0, opts.total - offset);
      const count = Math.min(pageSize, remaining);
      return {
        ok: true,
        json: async () => ({
          entries: makePageEntries(page, count),
          page,
          page_size: pageSize,
          total: opts.total,
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
}

/**
 * Serves a fixed list of pages whose rows carry the given modalities, so a
 * test can describe "page 1 is all image, page 2 has two video rows" and
 * exercise the client-side modality filter across page boundaries.
 */
function installModalityFetchMock(opts: { pages: string[][] }) {
  const total = opts.pages.reduce((n, p) => n + p.length, 0);
  globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
    if (url.startsWith("/api/catalog/families")) {
      return { ok: true, json: async () => ({ families: [] }) };
    }
    if (url.startsWith("/api/catalog/search")) {
      const params = new URL(url, "http://localhost").searchParams;
      const page = Number(params.get("page") ?? "1");
      const rows = opts.pages[page - 1] ?? [];
      const offset = opts.pages
        .slice(0, page - 1)
        .reduce((n, p) => n + p.length, 0);
      const entries = rows.map((modality, i) => ({
        ...makePageEntries(1, 1)[0],
        id: `hf:row-${offset + i}`,
        name: `Row ${offset + i}`,
        modality,
      }));
      return {
        ok: true,
        json: async () => ({ entries, page, page_size: 48, total }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
}

beforeEach(() => {
  __resetCatalogSingletonForTests();
  installFetchMock({ total: 1, pageSize: 48 });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("useCatalog", () => {
  it("loads families and entries on init", async () => {
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.entries.value.length).toBe(1);
    expect(cat.families.value[0].family).toBe("flux");
  });

  it("states include_nsfw=false by default so the unchecked box matches the server", async () => {
    // `GET /api/catalog/search` treats a missing `include_nsfw` as *true*.
    // Leaving it out of the default filter meant the topbar rendered an
    // unchecked box over an NSFW-inclusive result set.
    const cat = useCatalog();
    await cat.refresh();
    const searchUrl = (globalThis.fetch as any).mock.calls
      .map((c: any[]) => c[0] as string)
      .find((u: string) => u.startsWith("/api/catalog/search"));
    expect(searchUrl).toContain("include_nsfw=false");
  });

  it("setFilter triggers a re-fetch", async () => {
    const cat = useCatalog();
    await cat.refresh();
    (globalThis.fetch as any).mockClear();
    cat.setFilter({ family: "flux" });
    await new Promise((r) => setTimeout(r, 300)); // past the 250ms debounce
    expect(
      (globalThis.fetch as any).mock.calls.some((c: any[]) =>
        (c[0] as string).includes("family=flux"),
      ),
    ).toBe(true);
  });

  it("openDetail uses an entry already in the list without hitting /api/catalog/:id", async () => {
    // Live-search rows aren't in the DB-backed entry endpoint, so the
    // re-fetch would 404 and the unhandled rejection used to leave the
    // drawer unmounted. Regression test: clicking an entry must populate
    // `detail` from the in-memory list.
    const cat = useCatalog();
    await cat.refresh();
    const id = cat.entries.value[0].id;
    (globalThis.fetch as any).mockClear();
    await cat.openDetail(id);
    expect(cat.detail.value?.kind).toBe("catalog");
    expect(
      cat.detail.value?.kind === "catalog" ? cat.detail.value.entry.id : null,
    ).toBe(id);
    const detailCalls = (globalThis.fetch as any).mock.calls.filter(
      (c: any[]) =>
        (c[0] as string).startsWith("/api/catalog/") &&
        !(c[0] as string).startsWith("/api/catalog/search") &&
        !(c[0] as string).startsWith("/api/catalog/families"),
    );
    expect(detailCalls.length).toBe(0);
  });

  it("openDetail falls back to /api/catalog/:id when the entry isn't cached", async () => {
    const cat = useCatalog();
    await cat.refresh();
    const fallbackId = "hf:not-in-list";
    (globalThis.fetch as any).mockImplementationOnce(async (url: string) => {
      if (url === `/api/catalog/${encodeURIComponent(fallbackId)}`) {
        return {
          ok: true,
          json: async () => ({
            id: fallbackId,
            name: "Fallback",
            family: "flux",
            supported: true,
            installed: false,
            source: "hf",
            source_id: "x",
            author: null,
            family_role: "foundation",
            sub_family: null,
            modality: "image",
            kind: "checkpoint",
            file_format: "safetensors",
            bundling: "separated",
            size_bytes: 1,
            download_count: 0,
            rating: null,
            likes: 0,
            nsfw: false,
            thumbnail_url: null,
            description: null,
            license: null,
            license_flags: null,
            tags: [],
            companions: [],
            download_recipe: { files: [], needs_token: null },
            primary_path: null,
            created_at: null,
            updated_at: null,
            added_at: 0,
          }),
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    await cat.openDetail(fallbackId);
    expect(
      cat.detail.value?.kind === "catalog" ? cat.detail.value.entry.id : null,
    ).toBe(fallbackId);
  });

  it("openDetail leaves detail null when the fallback fetch 404s", async () => {
    // The drawer's v-if guards on detail; a 404 must not crash openDetail
    // either, since the SPA's `void cat.openDetail(id)` swallows rejections.
    const cat = useCatalog();
    await cat.refresh();
    const missingId = "cv:99999999";
    (globalThis.fetch as any).mockImplementationOnce(async () => ({
      ok: false,
      status: 404,
    }));
    await cat.openDetail(missingId);
    expect(cat.detail.value).toBeNull();
  });

  it("openDetail on a malformed cached entry sets a retryable error, not a dead drawer", async () => {
    // A row that survived the list render but can't back a detail view (no
    // name/family) must open the error state — never a `detail` the drawer
    // can only paint blank, and never an exception out of the async call.
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.startsWith("/api/catalog/families")) {
        return { ok: true, json: async () => ({ families: [] }) };
      }
      if (url.startsWith("/api/catalog/search")) {
        return {
          ok: true,
          json: async () => ({
            entries: [{ id: "hf:bad" }],
            page: 1,
            page_size: 48,
            total: 1,
          }),
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    }) as typeof fetch;
    const cat = useCatalog();
    await cat.refresh();
    await cat.openDetail("hf:bad");
    expect(cat.detail.value).toBeNull();
    expect(cat.detailLoadingId.value).toBeNull();
    expect(cat.detailError.value?.id).toBe("hf:bad");
  });

  it("openDetail survives a null row sitting in the cached list", async () => {
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url.startsWith("/api/catalog/families")) {
        return { ok: true, json: async () => ({ families: [] }) };
      }
      if (url.startsWith("/api/catalog/search")) {
        return {
          ok: true,
          json: async () => ({
            entries: [null, ...makePageEntries(1, 1)],
            page: 1,
            page_size: 48,
            total: 2,
          }),
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    }) as typeof fetch;
    const cat = useCatalog();
    await cat.refresh();
    await cat.openDetail("hf:row-0");
    expect(cat.detail.value?.kind).toBe("catalog");
    expect(cat.detailError.value).toBeNull();
  });

  it("uses the direct catalog support capability", async () => {
    const cat = useCatalog();
    expect(cat.canDownload({ supported: true } as any)).toBe(true);
    expect(cat.canDownload({ supported: false } as any)).toBe(false);
  });
});

describe("useCatalog modality filter", () => {
  // `GET /api/catalog/search` has no `modality` parameter — Axum drops it
  // silently, so the Image/Video chips filtered nothing. Every entry does
  // carry a `modality`, so the filter is applied to the fetched rows.
  it("narrows visibleEntries to the selected modality", async () => {
    installModalityFetchMock({ pages: [["image", "video", "image", "video"]] });
    const cat = useCatalog();
    cat.setFilter({ modality: "video" });
    await cat.refresh();
    expect(cat.entries.value.length).toBe(4);
    expect(cat.visibleEntries.value.map((e) => e.id)).toEqual([
      "hf:row-1",
      "hf:row-3",
    ]);
  });

  it("leaves visibleEntries untouched when no modality is selected", async () => {
    installModalityFetchMock({ pages: [["image", "video"]] });
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.visibleEntries.value.length).toBe(2);
  });

  it("layers on top of the server-side filters instead of replacing them", async () => {
    // The regression the user hit was combinational: picking Video alongside
    // a kind/source must keep sending those to the server and only then
    // narrow by modality locally.
    installModalityFetchMock({ pages: [["image", "video"]] });
    const cat = useCatalog();
    cat.setFilter({ modality: "video", kind: "lora", source: "civitai" });
    await cat.refresh();
    const url = (globalThis.fetch as any).mock.calls
      .map((c: any[]) => c[0] as string)
      .find((u: string) => u.startsWith("/api/catalog/search"));
    expect(url).toContain("kind=lora");
    expect(url).toContain("source=civitai");
    expect(url).toContain("include_nsfw=false");
    expect(url).not.toContain("modality=");
    expect(cat.visibleEntries.value.map((e) => e.id)).toEqual(["hf:row-1"]);
  });

  it("resultCount reports the surviving rows while a modality filter is on", async () => {
    installModalityFetchMock({ pages: [["image", "video", "image"]] });
    const cat = useCatalog();
    cat.setFilter({ modality: "video" });
    await cat.refresh();
    // The server total (3) describes the unfiltered feed and would overstate
    // what the grid actually shows.
    expect(cat.total.value).toBe(3);
    expect(cat.resultCount.value).toBe(1);
  });

  it("resultCount reports the server total when no modality filter is on", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.resultCount.value).toBe(130);
  });
});

describe("useCatalog infinite scroll", () => {
  it("exposes total and hasMore from the list response", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.entries.value.length).toBe(48);
    expect(cat.total.value).toBe(130);
    expect(cat.hasMore.value).toBe(true);
  });

  it("loadMore appends the next page without replacing existing entries", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.entries.value.length).toBe(48);
    expect(cat.entries.value[0].id).toBe("hf:row-0");

    await cat.loadMore();
    expect(cat.entries.value.length).toBe(96);
    expect(cat.entries.value[0].id).toBe("hf:row-0");
    expect(cat.entries.value[48].id).toBe("hf:row-48");

    await cat.loadMore();
    expect(cat.entries.value.length).toBe(130);
    expect(cat.hasMore.value).toBe(false);
  });

  it("loadMore is a no-op when hasMore is false", async () => {
    installFetchMock({ total: 10, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.hasMore.value).toBe(false);
    (globalThis.fetch as any).mockClear();
    await cat.loadMore();
    expect((globalThis.fetch as any).mock.calls.length).toBe(0);
  });

  it("setFilter resets entries to the first page (does not append)", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    await cat.loadMore();
    expect(cat.entries.value.length).toBe(96);

    cat.setFilter({ family: "flux" });
    await new Promise((r) => setTimeout(r, 300)); // past debounce
    expect(cat.entries.value.length).toBe(48);
    expect(cat.entries.value[0].id).toBe("hf:row-0");
  });

  it("keeps fetching while a modality filter hides every row of a page", async () => {
    // Client-side filtering after a paged fetch can hide an entire page. The
    // grid must not settle on "No models found." while the server still has
    // pages left — keep pulling until something survives or the feed ends.
    installModalityFetchMock({
      pages: [
        Array(48).fill("image"),
        Array(48).fill("image"),
        ["video", "image", "video"],
      ],
    });
    const cat = useCatalog();
    cat.setFilter({ modality: "video" });
    await cat.refresh();
    expect(cat.visibleEntries.value.length).toBe(2);
    expect(cat.visibleEntries.value.every((e) => e.modality === "video")).toBe(
      true,
    );
  });

  it("stops auto-filling once the feed is exhausted instead of looping", async () => {
    installModalityFetchMock({
      pages: [Array(48).fill("image"), Array(20).fill("image")],
    });
    const cat = useCatalog();
    cat.setFilter({ modality: "video" });
    await cat.refresh();
    expect(cat.visibleEntries.value.length).toBe(0);
    expect(cat.hasMore.value).toBe(false);
    const calls = (globalThis.fetch as any).mock.calls.filter((c: any[]) =>
      (c[0] as string).startsWith("/api/catalog/search"),
    );
    expect(calls.length).toBe(2);
  });

  it("concurrent loadMore calls do not double-fetch the same page", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    (globalThis.fetch as any).mockClear();
    await Promise.all([cat.loadMore(), cat.loadMore(), cat.loadMore()]);
    const listCalls = (globalThis.fetch as any).mock.calls.filter((c: any[]) =>
      (c[0] as string).startsWith("/api/catalog/search"),
    );
    expect(listCalls.length).toBe(1);
    expect(cat.entries.value.length).toBe(96);
  });
});
