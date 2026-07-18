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

import { fetchCatalogDetail, fetchCatalogInstalled } from "./catalog";
import type { CatalogEntry } from "./types";

const PRIMARY = { baseUrl: "http://127.0.0.1:49152", apiKey: null };

beforeEach(() => {
  vi.clearAllMocks();
  currentTarget.mockReturnValue(PRIMARY);
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
