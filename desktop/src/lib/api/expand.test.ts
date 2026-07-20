import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiJson, apiJsonTo } from "./client";
import { expandPrompt } from "./expand";

vi.mock("./client", () => ({
  apiJson: vi.fn(),
  apiJsonTo: vi.fn(),
}));

const apiJsonMock = vi.mocked(apiJson);
const apiJsonToMock = vi.mocked(apiJsonTo);

function sentBody(): Record<string, unknown> {
  const [, init] = apiJsonMock.mock.calls.at(-1)!;
  return JSON.parse((init as RequestInit).body as string) as Record<string, unknown>;
}

describe("expandPrompt", () => {
  beforeEach(() => {
    apiJsonMock.mockReset();
    apiJsonMock.mockResolvedValue({ original: "a cat", expanded: ["a detailed cat"] });
    apiJsonToMock.mockReset();
    apiJsonToMock.mockResolvedValue({ original: "a cat", expanded: ["a detailed cat"] });
  });

  it("defaults to a single variation and omits model_family", async () => {
    await expandPrompt("a cat");
    expect(apiJsonMock).toHaveBeenCalledWith("/api/expand", expect.anything());
    const body = sentBody();
    expect(body).toEqual({ prompt: "a cat", variations: 1 });
    expect("model_family" in body).toBe(false);
  });

  it("sends the requested variations and family override", async () => {
    await expandPrompt("a cat", { modelFamily: "sdxl", variations: 5 });
    expect(sentBody()).toEqual({ prompt: "a cat", model_family: "sdxl", variations: 5 });
  });

  it("sends variations alone when no family is given", async () => {
    await expandPrompt("a cat", { variations: 3 });
    const body = sentBody();
    expect(body).toEqual({ prompt: "a cat", variations: 3 });
    expect("model_family" in body).toBe(false);
  });

  it("omits a whitespace-only family instead of sending it", async () => {
    await expandPrompt("a cat", { modelFamily: "   ", variations: 2 });
    const body = sentBody();
    expect(body).toEqual({ prompt: "a cat", variations: 2 });
    expect("model_family" in body).toBe(false);
  });

  it("trims surrounding whitespace from the family override", async () => {
    await expandPrompt("a cat", { modelFamily: "  sdxl  " });
    expect(sentBody()).toEqual({ prompt: "a cat", model_family: "sdxl", variations: 1 });
  });

  it("directs expansion to the selected remote host", async () => {
    const target = { baseUrl: "https://studio.tailnet.ts.net", apiKey: "secret" };

    await expandPrompt("a cat", { modelFamily: "flux" }, target);

    expect(apiJsonToMock).toHaveBeenCalledWith(target, "/api/expand", expect.anything());
    expect(apiJsonMock).not.toHaveBeenCalled();
    const [, , init] = apiJsonToMock.mock.calls.at(-1)!;
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      prompt: "a cat",
      model_family: "flux",
      variations: 1,
    });
  });
});
