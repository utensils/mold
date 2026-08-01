import { describe, expect, it } from "vitest";
import {
  baseGenerationCapabilities,
  isAdvancedVideoFamily,
} from "./generationCapabilities";

describe("baseGenerationCapabilities", () => {
  it("keeps supported family aliases on the same capability policy", () => {
    for (const family of ["sd15", "sd1.5", "stable-diffusion-1.5"]) {
      expect(baseGenerationCapabilities(family)).toMatchObject({
        supportsScheduler: true,
        supportsControlNet: true,
      });
    }
    for (const family of ["flux2", "flux.2", "flux-2"]) {
      expect(baseGenerationCapabilities(family)).toMatchObject({
        supportsNegativePrompt: false,
        supportsLora: family === "flux2",
      });
    }
    for (const family of ["ltx2", "ltx-2"]) {
      expect(baseGenerationCapabilities(family)).toMatchObject({
        supportsVideo: true,
        supportsAudio: true,
      });
      expect(isAdvancedVideoFamily(family)).toBe(true);
    }
  });

  it("returns independent scheduler option lists", () => {
    const first = baseGenerationCapabilities("sdxl").schedulerOptions;
    first.pop();
    expect(baseGenerationCapabilities("sdxl").schedulerOptions).toEqual([
      "default",
      "ddim",
      "euler-ancestral",
      "unipc",
    ]);
  });
});
