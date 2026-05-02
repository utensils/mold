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
      engine_phase: 1,
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
          families: [{ family: "flux", foundation: 1, finetune: 4 }],
        }),
      };
    }
    if (url.startsWith("/api/catalog?")) {
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

  it("enables download for engine_phase 1 + 2, disables for engine_phase >= 3", async () => {
    const cat = useCatalog();
    expect(cat.canDownload({ engine_phase: 1 } as any)).toBe(true);
    expect(cat.canDownload({ engine_phase: 2 } as any)).toBe(true);
    expect(cat.canDownload({ engine_phase: 3 } as any)).toBe(false);
    expect(cat.canDownload({ engine_phase: 99 } as any)).toBe(false);
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

  it("concurrent loadMore calls do not double-fetch the same page", async () => {
    installFetchMock({ total: 130, pageSize: 48 });
    const cat = useCatalog();
    await cat.refresh();
    (globalThis.fetch as any).mockClear();
    await Promise.all([cat.loadMore(), cat.loadMore(), cat.loadMore()]);
    const listCalls = (globalThis.fetch as any).mock.calls.filter((c: any[]) =>
      (c[0] as string).startsWith("/api/catalog?"),
    );
    expect(listCalls.length).toBe(1);
    expect(cat.entries.value.length).toBe(96);
  });
});
