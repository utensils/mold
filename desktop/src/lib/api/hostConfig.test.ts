import { beforeEach, describe, expect, it, vi } from "vitest";

const apiFetchTo = vi.fn();
const apiJsonTo = vi.fn();
vi.mock("./client", () => ({
  apiFetchTo: (...args: unknown[]) => apiFetchTo(...args),
  apiJsonTo: (...args: unknown[]) => apiJsonTo(...args),
}));

import { fetchHostConfigKey, resetHostConfigKey, setHostConfigKey } from "./hostConfig";

const target = { baseUrl: "http://hal9000:7680", apiKey: "k" };

beforeEach(() => {
  apiFetchTo.mockReset();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 200 }));
  apiJsonTo.mockReset();
});

describe("fetchHostConfigKey", () => {
  it("GETs the encoded key on the named host and returns the row", async () => {
    apiJsonTo.mockResolvedValue({
      key: "gallery.trash_retention_days",
      value: 30,
      source: "default",
    });
    const row = await fetchHostConfigKey(target, "gallery.trash_retention_days");
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/config/gallery.trash_retention_days",
      expect.objectContaining({ signal: null }),
    );
    expect(row).toEqual({ key: "gallery.trash_retention_days", value: 30, source: "default" });
  });

  it("encodes keys that need it and forwards an abort signal", async () => {
    apiJsonTo.mockResolvedValue({ key: "a/b", value: null, source: "default" });
    const controller = new AbortController();
    await fetchHostConfigKey(target, "a/b", controller.signal);
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/config/a%2Fb",
      expect.objectContaining({ signal: controller.signal }),
    );
  });
});

describe("setHostConfigKey", () => {
  it("PUTs {value} as JSON on the named host", async () => {
    await setHostConfigKey(target, "gallery.trash_retention_days", 7);
    expect(apiFetchTo).toHaveBeenCalledTimes(1);
    const [calledTarget, path, init] = apiFetchTo.mock.calls[0] as [
      typeof target,
      string,
      RequestInit,
    ];
    expect(calledTarget).toBe(target);
    expect(path).toBe("/api/config/gallery.trash_retention_days");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ value: 7 });
    expect(new Headers(init.headers).get("content-type")).toBe("application/json");
  });

  it("sends 0 (keep forever) as a real number, never as absent", async () => {
    await setHostConfigKey(target, "gallery.trash_retention_days", 0);
    const init = apiFetchTo.mock.calls[0]![2] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ value: 0 });
  });
});

describe("resetHostConfigKey", () => {
  it("DELETEs the key on the named host", async () => {
    await resetHostConfigKey(target, "gallery.trash_retention_days");
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/config/gallery.trash_retention_days", {
      method: "DELETE",
    });
  });
});
