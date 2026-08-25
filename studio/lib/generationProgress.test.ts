import { describe, expect, it } from "vitest";
import {
  generationProgressCopy,
  phaseForStageStart,
} from "./generationProgress";

describe("generation progress phases", () => {
  it("keeps a nested transformer-block stage inside denoising", () => {
    expect(phaseForStageStart("denoising", 4, 20)).toBe("denoising");
    expect(
      generationProgressCopy({
        phase: "denoising",
        step: 4,
        total: 20,
        stage: "Streaming MiniMax H3 transformer blocks",
      }),
    ).toBe("Developing 4/20 — Streaming MiniMax H3 transformer blocks");
  });

  it("reserves finalizing for a stage after the last denoise evaluation", () => {
    expect(phaseForStageStart("denoising", 20, 20)).toBe("finalizing");
    expect(
      generationProgressCopy({
        phase: "finalizing",
        step: 20,
        total: 20,
        stage: "Decoding MiniMax H3 video",
      }),
    ).toBe("Finalizing — Decoding MiniMax H3 video");
  });
});
