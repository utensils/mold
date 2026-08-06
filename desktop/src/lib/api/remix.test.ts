import { beforeEach, describe, expect, it, vi } from "vitest";
import { remixPrompt } from "./remix";
import { apiJsonTo } from "./client";

vi.mock("./client", () => ({ apiJson: vi.fn(), apiJsonTo: vi.fn() }));

describe("remixPrompt", () => {
  beforeEach(() => vi.clearAllMocks());

  it("posts the complete request to the exact selected host", async () => {
    vi.mocked(apiJsonTo).mockResolvedValue({
      source_prompt: "a lighthouse",
      source_kind: "direct",
      variants: [],
    });
    const request = {
      source_prompt: "a lighthouse",
      source_kind: "direct" as const,
      model_family: "flux",
      variations: 3,
      task: "text-to-image" as const,
      dimensions: ["camera" as const],
    };
    const target = { baseUrl: "http://studio:7680", apiKey: "secret" };
    await remixPrompt(request, target);
    expect(apiJsonTo).toHaveBeenCalledWith(
      target,
      "/api/remix",
      expect.objectContaining({ method: "POST", body: JSON.stringify(request) }),
    );
  });
});
