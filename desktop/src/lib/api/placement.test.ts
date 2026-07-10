import { beforeEach, describe, expect, it, vi } from "vitest";

const apiFetch = vi.fn();
const apiJson = vi.fn();
// Hoisted so the (hoisted) vi.mock factory can reference the class without a TDZ.
const { ApiError } = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(
      message: string,
      public status: number,
      public body: unknown = null,
    ) {
      super(message);
      this.name = "ApiError";
    }
  }
  return { ApiError };
});
vi.mock("./client", () => ({
  apiFetch: (...args: unknown[]) => apiFetch(...args),
  apiJson: (...args: unknown[]) => apiJson(...args),
  ApiError,
}));

import { deleteModelPlacement, getModelPlacement, putModelPlacement } from "./placement";
import type { DevicePlacement } from "../placement";

beforeEach(() => {
  apiFetch.mockReset();
  apiFetch.mockResolvedValue(new Response(null, { status: 200 }));
  apiJson.mockReset();
});

describe("putModelPlacement", () => {
  it("PUTs to the encoded model path with a JSON body", async () => {
    const placement: DevicePlacement = {
      text_encoders: { kind: "cpu" },
      advanced: null,
    };
    await putModelPlacement("flux-dev:q4", placement);

    expect(apiFetch).toHaveBeenCalledTimes(1);
    const [path, init] = apiFetch.mock.calls[0]!;
    expect(path).toBe("/api/config/model/flux-dev%3Aq4/placement");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual(placement);
    expect(new Headers(init.headers).get("content-type")).toBe("application/json");
  });

  it("percent-encodes slashes and colons in the model name", async () => {
    await putModelPlacement("org/model:tag", { text_encoders: { kind: "auto" }, advanced: null });
    const [path] = apiFetch.mock.calls[0]!;
    expect(path).toBe("/api/config/model/org%2Fmodel%3Atag/placement");
  });
});

describe("deleteModelPlacement", () => {
  it("DELETEs the encoded model path", async () => {
    await deleteModelPlacement("qwen-image:fp16");
    expect(apiFetch).toHaveBeenCalledTimes(1);
    const [path, init] = apiFetch.mock.calls[0]!;
    expect(path).toBe("/api/config/model/qwen-image%3Afp16/placement");
    expect(init.method).toBe("DELETE");
  });
});

describe("getModelPlacement", () => {
  it("GETs the encoded model path and returns the saved placement", async () => {
    const saved: DevicePlacement = { text_encoders: { kind: "cpu" }, advanced: null };
    apiJson.mockResolvedValue(saved);
    const result = await getModelPlacement("flux-dev:q4");
    expect(apiJson).toHaveBeenCalledWith("/api/config/model/flux-dev%3Aq4/placement");
    expect(result).toEqual(saved);
  });

  it("returns null on a 404 (nothing saved)", async () => {
    apiJson.mockRejectedValue(new ApiError("no placement", 404));
    expect(await getModelPlacement("flux-dev:q4")).toBeNull();
  });

  it("rethrows non-404 errors", async () => {
    apiJson.mockRejectedValue(new ApiError("boom", 500));
    await expect(getModelPlacement("flux-dev:q4")).rejects.toThrow("boom");
  });
});
