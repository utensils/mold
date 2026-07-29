import { describe, expect, it } from "vitest";
import type { ChainRequestWire as StudioChainRequestWire } from "@studio/lib/api/chainTypes";
import type { ChainRequestWire as WebChainRequestWire } from "./types";

function serializeWebChainRequest(request: WebChainRequestWire): string {
  return JSON.stringify(request);
}

describe("durable chain request wire contract", () => {
  it("is assignable to the web chain request and serializes seed offsets as numbers", () => {
    const request: StudioChainRequestWire = {
      model: "ltx-2.3:bf16",
      stages: [
        { prompt: "opening", frames: 97 },
        { prompt: "continuation", frames: 97, seed_offset: 7 },
      ],
      motion_tail_frames: 17,
      width: 768,
      height: 512,
      fps: 24,
      seed: 42,
      steps: 20,
      guidance: 3.5,
      output_format: "mp4",
    };

    const encoded = serializeWebChainRequest(request);

    expect(JSON.parse(encoded).stages[1].seed_offset).toBe(7);
  });
});
