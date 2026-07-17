import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiJson } from "./client";
import { expandPrompt } from "./expand";

vi.mock("./client", () => ({
  apiJson: vi.fn(),
}));

const apiJsonMock = vi.mocked(apiJson);

function sentBody(): Record<string, unknown> {
  const [, init] = apiJsonMock.mock.calls.at(-1)!;
  return JSON.parse((init as RequestInit).body as string) as Record<string, unknown>;
}

describe("expandPrompt", () => {
  beforeEach(() => {
    apiJsonMock.mockReset();
    apiJsonMock.mockResolvedValue({ original: "a cat", expanded: ["a detailed cat"] });
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
});
