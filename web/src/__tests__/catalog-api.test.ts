import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCatalog,
  fetchCatalogEntry,
  fetchCatalogFamilies,
  postCatalogRefresh,
  postCatalogDownload,
} from "../api";

const originalFetch = globalThis.fetch;

describe("catalog api", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn() as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("fetchCatalog passes filters as query params", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalog({ family: "flux", q: "juggernaut", page: 2 });
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as string;
    expect(call).toContain("/api/catalog");
    expect(call).toContain("family=flux");
    expect(call).toContain("q=juggernaut");
    expect(call).toContain("page=2");
  });

  it("fetchCatalogEntry url-encodes the id", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    await fetchCatalogEntry("hf:foo/bar baz");
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as string;
    expect(call).toContain("/api/catalog/hf%3Afoo%2Fbar%20baz");
  });

  it("postCatalogRefresh sends an empty JSON body when no filters", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ id: "abc" }),
    });
    const out = await postCatalogRefresh({});
    expect(out.id).toBe("abc");
    const init = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][1];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({}));
  });

  it("fetchCatalogFamilies returns the families array", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({
        families: [{ family: "flux", foundation: 2, finetune: 3 }],
      }),
    });
    const out = await fetchCatalogFamilies();
    expect(out.families[0].family).toBe("flux");
  });

  it("postCatalogDownload returns the job_ids", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ job_ids: ["x"] }),
    });
    const out = await postCatalogDownload("hf:bfl/FLUX.1-dev");
    expect(out.job_ids).toEqual(["x"]);
  });
});
