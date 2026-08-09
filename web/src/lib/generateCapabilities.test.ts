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

  // #772: `/api/models[].source_image` is additive, so absence has to keep
  // today's family answer — an older server enforces nothing at admission and
  // would reject nothing a client newly withheld.
  it("falls back to the family heuristic when the server advertises nothing", () => {
    expect(
      generationCapabilitiesForFamily("wan", "wan22-t2v-a14b:q5"),
    ).toMatchObject({
      sourceImageCapability: "optional",
      supportsSourceImage: true,
      requiresSourceImage: false,
      supportsEndFrame: false,
    });
    expect(generationCapabilitiesForFamily("sdxl")).toMatchObject({
      sourceImageCapability: "optional",
      supportsSourceImage: true,
      requiresSourceImage: false,
    });
    expect(generationCapabilitiesForFamily("ltx-video")).toMatchObject({
      sourceImageCapability: "unsupported",
      supportsSourceImage: false,
    });
  });

  it("threads the advertised source-image contract through the fifth argument", () => {
    expect(
      generationCapabilitiesForFamily(
        "wan",
        "wan22-t2v-a14b:q5",
        null,
        null,
        "unsupported",
      ),
    ).toMatchObject({
      sourceImageCapability: "unsupported",
      supportsSourceImage: false,
      requiresSourceImage: false,
      supportsEndFrame: false,
    });
    expect(
      generationCapabilitiesForFamily(
        "wan",
        "wan22-i2v-a14b:q8",
        null,
        null,
        "required",
      ),
    ).toMatchObject({
      sourceImageCapability: "required",
      supportsSourceImage: true,
      requiresSourceImage: true,
      supportsEndFrame: true,
    });
    expect(
      generationCapabilitiesForFamily(
        "wan",
        "wan22-ti2v-5b:fp16",
        null,
        null,
        "optional",
      ).supportsEndFrame,
    ).toBe(true);
    // First/last frames are wan's layout alone; LTX-2 keyframes are their own
    // control and must not be duplicated by an End frame well.
    expect(
      generationCapabilitiesForFamily(
        "ltx2",
        "ltx-2-19b:fp8",
        null,
        null,
        "optional",
      ).supportsEndFrame,
    ).toBe(false);
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
