import { beforeEach, describe, expect, it, vi } from "vitest";
import { catalogCredentialHeaders } from "./catalogCredentials";

const secretGet = vi.fn();

vi.mock("./ipc", () => ({
  ipc: {
    secretGet: (name: string) => secretGet(name),
  },
}));

describe("catalogCredentialHeaders", () => {
  beforeEach(() => secretGet.mockReset());

  it("forwards locally stored catalog credentials to a remote engine", async () => {
    secretGet.mockImplementation(async (name: string) =>
      name === "hf-token" ? "hf_local" : name === "civitai-token" ? "cv_local" : null,
    );

    const headers = await catalogCredentialHeaders(true);

    expect(headers.get("X-Mold-HF-Token")).toBe("hf_local");
    expect(headers.get("X-Mold-Civitai-Token")).toBe("cv_local");
  });

  it("does not expose catalog credentials to a local engine", async () => {
    const headers = await catalogCredentialHeaders(false);

    expect([...headers]).toEqual([]);
    expect(secretGet).not.toHaveBeenCalled();
  });
});
