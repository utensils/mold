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
    for (const family of ["sdxl", "sd15"]) {
      expect(schedulerOptionsForFamily(family)).toEqual([
        "default",
        "ddim",
        "euler-ancestral",
        "uni-pc",
      ]);
    }
    // Wan's flow solvers are a disjoint set from the UNet schedulers; the
    // server rejects each one for the other's families.
    expect(schedulerOptionsForFamily("wan")).toEqual([
      "default",
      "uni-pc",
      "euler",
      "dpm-pp",
    ]);
    expect(schedulerOptionsForFamily("flux")).toEqual([]);
    expect(schedulerOptionsForFamily("qwen-image")).toEqual([]);
  });

  it("distinguishes FLUX.2 Dev reference editing from Klein img2img", () => {
    expect(
      generationCapabilitiesForFamily("flux2", "flux2-dev:bf16"),
    ).toMatchObject({
      sourceImageMode: "references",
      supportsMask: false,
      supportsLora: false,
      forcesBatchSizeOne: false,
    });
    expect(
      generationCapabilitiesForFamily("flux2", "flux2-klein:bf16"),
    ).toMatchObject({
      sourceImageMode: "single",
      supportsMask: true,
      supportsLora: true,
    });
  });
});
