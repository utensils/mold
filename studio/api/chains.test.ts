import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChainRequestWire } from "../lib/api/chainTypes";
import { validateChain } from "./chains";

const request: ChainRequestWire = {
  model: "ltx-2-19b-distilled:fp8",
  stages: [{ prompt: "opening", frames: 97 }],
  width: 1216,
  height: 704,
  fps: 24,
  steps: 8,
  guidance: 3,
};

afterEach(() => vi.unstubAllGlobals());

describe("validateChain", () => {
  it("posts the current request to the exact authenticated host", async () => {
    const response = {
      model: request.model,
      width: 1216,
      height: 704,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 1,
      estimated_total_frames: 97,
      estimated_duration_ms: 4042,
      stages: [
        {
          prompt: "opening",
          frames: 97,
          output_frames: 97,
          transition: "smooth",
          fade_frames: null,
          has_source_image: false,
          has_negative_prompt: false,
        },
      ],
      warnings: [],
      vram_estimate: null,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify(response), { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      validateChain(request, {
        baseUrl: "http://render-box:7680",
        apiKey: "secret",
      }),
    ).resolves.toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://render-box:7680/api/generate/chain/validate",
      expect.objectContaining({
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(request),
      }),
    );
    const headers = fetchMock.mock.calls[0]![1].headers as Headers;
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("x-api-key")).toBe("secret");
  });
});
