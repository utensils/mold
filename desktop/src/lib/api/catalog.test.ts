import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiJsonTo, apiFetchTo, currentTarget } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(),
}));
vi.mock("./client", () => ({
  apiJsonTo,
  apiFetchTo,
  currentTarget,
}));
vi.mock("../catalogCredentials", () => ({
  catalogCredentialHeaders: vi.fn((forward: boolean) => {
    const headers = new Headers();
    if (forward) headers.set("X-Mold-HF-Token", "hf-secret");
    return Promise.resolve(headers);
  }),
}));

import {
  fetchCatalogDetail,
  fetchCatalogInstalled,
  searchCatalog,
  startCatalogDownload,
} from "./catalog";
import type { CatalogEntry } from "./types";

const PRIMARY = { baseUrl: "http://127.0.0.1:49152", apiKey: null };

beforeEach(() => {
  vi.clearAllMocks();
  currentTarget.mockReturnValue(PRIMARY);
});

describe("searchCatalog", () => {
  it("puts kind and sort on the wire", async () => {
    apiJsonTo.mockResolvedValueOnce({ entries: [], page: 1, page_size: 24, total: 0 });
    await searchCatalog({ kind: "lora", sort: "rating" });
    expect(apiJsonTo).toHaveBeenCalledWith(
      PRIMARY,
      "/api/catalog/search?kind=lora&sort=rating",
      expect.anything(),
    );
  });

  it("omits kind and sort when unset", async () => {
    apiJsonTo.mockResolvedValueOnce({ entries: [], page: 1, page_size: 24, total: 0 });
    await searchCatalog({ q: "flux" });
    const [, path] = apiJsonTo.mock.calls[0] as [unknown, string];
    expect(path).toBe("/api/catalog/search?q=flux");
    expect(path).not.toContain("kind=");
    expect(path).not.toContain("sort=");
  });
});

describe("fetchCatalogDetail", () => {
  it("hits GET /api/catalog/:id with the RAW id against the current target", async () => {
    apiJsonTo.mockResolvedValueOnce({ id: "hf:author/model" });
    await fetchCatalogDetail("hf:author/model");
    // Raw id — colons and slashes are part of the wildcard route match.
    expect(apiJsonTo).toHaveBeenCalledWith(
      PRIMARY,
      "/api/catalog/hf:author/model",
      expect.anything(),
    );
  });

  it("targets an explicit host when given one (multi-host detail fetch)", async () => {
    apiJsonTo.mockResolvedValueOnce({ id: "cv:8001" });
    const remote = { baseUrl: "http://hal9000:7680", apiKey: "hk" };
    await fetchCatalogDetail("cv:8001", true, remote);
    expect(currentTarget).not.toHaveBeenCalled();
    const [target, path, init] = apiJsonTo.mock.calls[0] as [
      typeof remote,
      string,
      { headers: Headers },
    ];
    expect(target).toEqual(remote);
    expect(path).toBe("/api/catalog/cv:8001");
    // Forwarded catalog credentials ride along for remote hosts.
    expect(init.headers.get("X-Mold-HF-Token")).toBe("hf-secret");
  });

  it("passes older-server responses through untouched — descriptive fields stay optional", async () => {
    // An old server that predates description/license/tags on the wire.
    const sparse = {
      id: "hf:a/b",
      source: "hf",
      name: "b",
      family: "flux",
      kind: "checkpoint",
      nsfw: false,
      installed: false,
      size_bytes: 123,
    };
    apiJsonTo.mockResolvedValueOnce(sparse);
    const detail: CatalogEntry = await fetchCatalogDetail("hf:a/b");
    expect(detail).toEqual(sparse);
    expect(detail.description).toBeUndefined();
    expect(detail.license).toBeUndefined();
    expect(detail.tags).toBeUndefined();
    expect(detail.download_recipe).toBeUndefined();
  });
});

describe("fetchCatalogInstalled", () => {
  it("hits GET /api/catalog/installed with kind/family filters against the current target", async () => {
    apiJsonTo.mockResolvedValueOnce({ entries: [], page: 1, page_size: 0, total: 0 });
    await fetchCatalogInstalled({ family: "sd15", kind: "control-net" });
    expect(apiJsonTo).toHaveBeenCalledWith(
      PRIMARY,
      "/api/catalog/installed?family=sd15&kind=control-net",
    );
  });

  it("omits absent params from the query string", async () => {
    apiJsonTo.mockResolvedValueOnce({ entries: [], page: 1, page_size: 0, total: 0 });
    await fetchCatalogInstalled({ kind: "control-net" });
    expect(apiJsonTo).toHaveBeenCalledWith(PRIMARY, "/api/catalog/installed?kind=control-net");
  });

  it("targets an explicit host when given one (multi-host installed listing)", async () => {
    apiJsonTo.mockResolvedValueOnce({ entries: [], page: 1, page_size: 0, total: 0 });
    const remote = { baseUrl: "http://hal9000:7680", apiKey: "hk" };
    await fetchCatalogInstalled({ kind: "lora" }, remote);
    expect(currentTarget).not.toHaveBeenCalled();
    expect(apiJsonTo).toHaveBeenCalledWith(remote, "/api/catalog/installed?kind=lora");
  });

  it("returns the entries envelope untouched", async () => {
    const envelope = {
      entries: [{ id: "cv:1", name: "cn", installed: true, primary_path: "/m/cn.safetensors" }],
      page: 1,
      page_size: 1,
      total: 1,
    };
    apiJsonTo.mockResolvedValueOnce(envelope);
    expect(await fetchCatalogInstalled({})).toEqual(envelope);
  });
});

describe("startCatalogDownload", () => {
  it("surfaces an accepted catalog pull that enqueued no work", async () => {
    apiFetchTo.mockResolvedValueOnce(
      new Response(JSON.stringify({ primary_job_id: null, companion_jobs: [] }), { status: 202 }),
    );

    await expect(startCatalogDownload("hf:lightx2v/Qwen-Image-Lightning")).rejects.toThrow(
      "did not queue any files",
    );
  });

  it("returns a companion job when the catalog entry has no primary", async () => {
    apiFetchTo.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          primary_job_id: null,
          companion_jobs: [{ name: "clip-l", job_id: "companion-1" }],
        }),
        { status: 202 },
      ),
    );

    await expect(startCatalogDownload("hf:companion/test")).resolves.toBe("companion-1");
  });
});
