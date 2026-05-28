import { describe, expect, it } from "vitest";
import {
  generationCapabilitiesForFamily,
  schedulerOptionsForFamily,
} from "./generateCapabilities";

describe("generationCapabilities", () => {
  it("centralizes family gates used by Generate controls", () => {
    expect(generationCapabilitiesForFamily("sdxl")).toMatchObject({
      supportsNegativePrompt: true,
      supportsScheduler: true,
      supportsCfgPlus: false,
      supportsVideo: false,
      supportsAudio: false,
      supportsLora: true,
      sourceImageMode: "single",
      supportsMask: true,
    });

    expect(generationCapabilitiesForFamily("flux")).toMatchObject({
      supportsNegativePrompt: false,
      supportsScheduler: false,
      supportsCfgPlus: false,
      supportsVideo: false,
      supportsAudio: false,
      supportsLora: true,
      sourceImageMode: "single",
      supportsMask: true,
    });

    expect(generationCapabilitiesForFamily("sd3.5")).toMatchObject({
      supportsNegativePrompt: true,
      supportsScheduler: false,
      supportsCfgPlus: true,
    });

    expect(generationCapabilitiesForFamily("qwen-image-edit")).toMatchObject({
      supportsNegativePrompt: false,
      supportsScheduler: false,
      supportsCfgPlus: false,
      supportsVideo: false,
      supportsLora: true,
      sourceImageMode: "qwen-edit",
      supportsMask: false,
      forcesBatchSizeOne: true,
    });

    expect(generationCapabilitiesForFamily("ltx2")).toMatchObject({
      supportsVideo: true,
      supportsAudio: true,
      supportsLora: true,
    });
  });

  it("returns scheduler options only for families that honor them", () => {
    expect(schedulerOptionsForFamily("sdxl")).toEqual([
      "default",
      "ddim",
      "euler-ancestral",
      "unipc",
    ]);
    expect(schedulerOptionsForFamily("sd15")).toEqual([
      "default",
      "ddim",
      "euler-ancestral",
      "unipc",
    ]);
    expect(schedulerOptionsForFamily("flux")).toEqual([]);
    expect(schedulerOptionsForFamily("qwen-image")).toEqual([]);
  });
});
