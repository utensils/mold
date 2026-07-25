import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCatalogEntry,
  fetchCatalogFamilies,
  fetchCatalogSearch,
  getCatalogCredentialStatus,
  looksLikeCatalogId,
  postCatalogDownload,
  putCatalogCredential,
  deleteCatalogCredential,
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

  it("fetchCatalogSearch passes filters as query params", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalogSearch({ family: "flux", q: "juggernaut", page: 2 });
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as string;
    expect(call).toContain("/api/catalog/search");
    expect(call).toContain("family=flux");
    expect(call).toContain("q=juggernaut");
    expect(call).toContain("page=2");
  });

  it("does not forward legacy browser tokens now that credentials belong to the server", async () => {
    localStorage.setItem(
      "mold.web.accounts.v1",
      JSON.stringify({ hfToken: "hf_x1234", civitaiToken: "cv_y5678" }),
    );
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalogSearch({ q: "flux" });
    const opts = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][1];
    expect(opts?.headers).toBeUndefined();
    localStorage.clear();
  });

  it("sends no auth headers when no token is stored", async () => {
    localStorage.clear();
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalogSearch({ q: "flux" });
    const opts = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][1];
    expect(opts?.headers).toBeUndefined();
  });

  it("fetchCatalogSearch passes component kind filters as query params", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({ entries: [], page: 1, page_size: 48 }),
    });
    await fetchCatalogSearch({ kind: "tokenizer", q: "qwen tokenizer" });
    const call = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as string;
    expect(call).toContain("kind=tokenizer");
    expect(call).toContain("q=qwen+tokenizer");
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

  it("fetchCatalogFamilies returns the families array", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({
        families: [{ family: "flux" }],
      }),
    });
    const out = await fetchCatalogFamilies();
    expect(out.families[0].family).toBe("flux");
  });

  it("looksLikeCatalogId matches cv:/hf: but not manifest tags", () => {
    expect(looksLikeCatalogId("cv:232703")).toBe(true);
    expect(looksLikeCatalogId("hf:RunDiffusion/Juggernaut-XL-v9")).toBe(true);
    expect(looksLikeCatalogId("flux-dev:q4")).toBe(false);
    expect(looksLikeCatalogId("flux-dev")).toBe(false);
    expect(looksLikeCatalogId("")).toBe(false);
  });

  it("postCatalogDownload returns primary_job_id + companion_jobs", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => ({
        primary_job_id: "primary-uuid",
        companion_jobs: [
          { name: "clip-l", job_id: "c1" },
          { name: "sdxl-vae", job_id: "c2" },
        ],
      }),
    });
    const out = await postCatalogDownload("cv:sdxl-pony");
    expect(out.primary_job_id).toBe("primary-uuid");
    expect(out.companion_jobs).toHaveLength(2);
    expect(out.companion_jobs[0].name).toBe("clip-l");
    expect(out.companion_jobs[0].job_id).toBe("c1");
  });

  it("reads, saves, and clears server-owned catalog credentials", async () => {
    const status = {
      hf: { configured: false, source: null, masked: null },
      civitai: {
        configured: true,
        source: "server",
        masked: "cv_••••5678",
      },
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => status,
    });

    await expect(getCatalogCredentialStatus()).resolves.toEqual(status);
    await putCatalogCredential("civitai", "cv_secret5678");
    await deleteCatalogCredential("civitai");

    const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls[0][0]).toBe("/api/catalog/credentials");
    expect(calls[1]).toEqual([
      "/api/catalog/credentials/civitai",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ token: "cv_secret5678" }),
      }),
    ]);
    expect(calls[2]).toEqual([
      "/api/catalog/credentials/civitai",
      expect.objectContaining({ method: "DELETE" }),
    ]);
  });
});
