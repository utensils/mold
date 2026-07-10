import { describe, expect, it } from "vitest";
import {
  generationCapabilitiesForFamily,
  outputFormatsForFamily,
  pruneRequestForFamily,
  schedulerOptionsForFamily,
  supportsAdvancedVideo,
} from "./capabilities";
import type { GenerateRequest } from "./api/types";

describe("generationCapabilitiesForFamily", () => {
  it("gates negative prompt to CFG families only", () => {
    for (const cfg of ["sd15", "sdxl", "sd3", "sd3.5", "wuerstchen"]) {
      // wuerstchen isn't in the no-negative set, so it keeps negative prompts.
      expect(generationCapabilitiesForFamily(cfg).supportsNegativePrompt).toBe(true);
    }
    for (const noneg of ["flux", "flux2", "z-image", "qwen-image", "qwen-image-edit"]) {
      expect(generationCapabilitiesForFamily(noneg).supportsNegativePrompt).toBe(false);
    }
  });

  it("offers a scheduler only for sd15 and sdxl", () => {
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
    expect(schedulerOptionsForFamily("sd3")).toEqual([]);
    expect(schedulerOptionsForFamily("qwen-image")).toEqual([]);
  });

  it("gates CFG++ to sd3 / sd3.5 per the matrix", () => {
    expect(generationCapabilitiesForFamily("sd3").supportsCfgPlus).toBe(true);
    expect(generationCapabilitiesForFamily("sd3.5").supportsCfgPlus).toBe(true);
    expect(generationCapabilitiesForFamily("sdxl").supportsCfgPlus).toBe(false);
    expect(generationCapabilitiesForFamily("flux").supportsCfgPlus).toBe(false);
  });

  it("supports img2img on image families but not video families", () => {
    expect(generationCapabilitiesForFamily("flux").supportsImg2img).toBe(true);
    expect(generationCapabilitiesForFamily("sdxl").supportsImg2img).toBe(true);
    expect(generationCapabilitiesForFamily("ltx2").supportsImg2img).toBe(false);
    expect(generationCapabilitiesForFamily("ltx-video").supportsImg2img).toBe(false);
  });

  it("puts qwen-image-edit in edit mode with batch locked to 1 and no mask", () => {
    const caps = generationCapabilitiesForFamily("qwen-image-edit");
    expect(caps.sourceImageMode).toBe("qwen-edit");
    expect(caps.forcesBatchSizeOne).toBe(true);
    expect(caps.supportsMask).toBe(false);
  });

  it("exposes video + audio only on the ltx families", () => {
    expect(generationCapabilitiesForFamily("ltx2")).toMatchObject({
      supportsVideo: true,
      supportsAudio: true,
    });
    expect(generationCapabilitiesForFamily("ltx-video")).toMatchObject({
      supportsVideo: true,
      supportsAudio: false,
    });
    expect(generationCapabilitiesForFamily("flux").supportsVideo).toBe(false);
  });

  it("restricts ControlNet to sd15", () => {
    expect(generationCapabilitiesForFamily("sd15").supportsControlNet).toBe(true);
    expect(generationCapabilitiesForFamily("sdxl").supportsControlNet).toBe(false);
  });

  it("orders output formats — images png-first, video mp4-first", () => {
    expect(outputFormatsForFamily("flux")).toEqual(["png", "jpeg", "webp"]);
    expect(outputFormatsForFamily("ltx2")).toEqual(["mp4", "gif", "apng", "webp"]);
  });
});

describe("pruneRequestForFamily", () => {
  const full: GenerateRequest = {
    prompt: "a cat",
    model: "x",
    width: 1024,
    height: 1024,
    steps: 20,
    negative_prompt: "blurry",
    scheduler: "ddim",
    cfg_plus: true,
    source_image: "AAAA",
    strength: 0.6,
    mask_image: "BBBB",
    control_image: "CCCC",
    control_model: "controlnet-canny-sd15",
    control_scale: 0.9,
    loras: [{ path: "/l.safetensors", scale: 1 }],
    batch_size: 4,
    output_format: "png",
  };

  it("strips CFG-only + scheduler + control + cfg_plus for FLUX", () => {
    const pruned = pruneRequestForFamily(full, "flux");
    expect(pruned.negative_prompt).toBeUndefined();
    expect(pruned.scheduler).toBeUndefined();
    expect(pruned.cfg_plus).toBeUndefined();
    expect(pruned.control_image).toBeUndefined();
    expect(pruned.control_model).toBeUndefined();
    // FLUX keeps img2img + loras.
    expect(pruned.source_image).toBe("AAAA");
    expect(pruned.loras).toBeDefined();
  });

  it("keeps scheduler + control for sd15 but drops cfg_plus (not sd3)", () => {
    const pruned = pruneRequestForFamily(full, "sd15");
    expect(pruned.scheduler).toBe("ddim");
    expect(pruned.control_image).toBe("CCCC");
    expect(pruned.negative_prompt).toBe("blurry");
    expect(pruned.cfg_plus).toBeUndefined();
  });

  it("keeps cfg_plus for sd3 but strips scheduler", () => {
    const pruned = pruneRequestForFamily(full, "sd3");
    expect(pruned.cfg_plus).toBe(true);
    expect(pruned.scheduler).toBeUndefined();
  });

  it("locks batch to 1 and drops source image for qwen-image-edit", () => {
    const pruned = pruneRequestForFamily(full, "qwen-image-edit");
    expect(pruned.batch_size).toBe(1);
    expect(pruned.source_image).toBeUndefined();
    expect(pruned.strength).toBeUndefined();
    expect(pruned.mask_image).toBeUndefined();
  });

  it("drops img2img + loras for a video family and fixes the format", () => {
    const pruned = pruneRequestForFamily(full, "ltx2");
    expect(pruned.source_image).toBeUndefined();
    expect(pruned.loras).toBe(full.loras); // ltx2 keeps loras
    expect(pruned.output_format).toBe("mp4"); // png isn't valid for video
  });

  it("does not mutate the input", () => {
    const snapshot = JSON.stringify(full);
    pruneRequestForFamily(full, "flux");
    expect(JSON.stringify(full)).toBe(snapshot);
  });

  it("strips LTX-2 advanced video fields for non-ltx2 families", () => {
    const advanced: GenerateRequest = {
      ...full,
      source_video: "VVVV",
      keyframes: [{ frame: 0, image: "KKKK" }],
      pipeline: "two-stage",
      retake_range: { start_seconds: 0, end_seconds: 1 },
      spatial_upscale: "x2",
      temporal_upscale: "x2",
    };
    // ltx-video is a video family but NOT ltx2 → advanced fields are pruned.
    const pruned = pruneRequestForFamily(advanced, "ltx-video");
    expect(pruned.source_video).toBeUndefined();
    expect(pruned.keyframes).toBeUndefined();
    expect(pruned.pipeline).toBeUndefined();
    expect(pruned.retake_range).toBeUndefined();
    expect(pruned.spatial_upscale).toBeUndefined();
    expect(pruned.temporal_upscale).toBeUndefined();
    // flux (still image) also prunes them.
    expect(pruneRequestForFamily(advanced, "flux").pipeline).toBeUndefined();
  });

  it("keeps LTX-2 advanced video fields for ltx2", () => {
    const advanced: GenerateRequest = {
      ...full,
      source_video: "VVVV",
      keyframes: [{ frame: 8, image: "KKKK" }],
      pipeline: "keyframe",
      spatial_upscale: "x1-5",
      temporal_upscale: "x2",
    };
    const pruned = pruneRequestForFamily(advanced, "ltx2");
    expect(pruned.source_video).toBe("VVVV");
    expect(pruned.keyframes).toEqual([{ frame: 8, image: "KKKK" }]);
    expect(pruned.pipeline).toBe("keyframe");
    expect(pruned.spatial_upscale).toBe("x1-5");
  });
});

describe("supportsAdvancedVideo", () => {
  it("is true only for ltx2 (not plain ltx-video)", () => {
    for (const f of ["ltx2", "ltx-2", "LTX2"]) {
      expect(supportsAdvancedVideo(f)).toBe(true);
      expect(generationCapabilitiesForFamily(f).supportsAdvancedVideo).toBe(true);
    }
    for (const f of ["ltx-video", "flux", "sdxl", "z-image"]) {
      expect(supportsAdvancedVideo(f)).toBe(false);
      expect(generationCapabilitiesForFamily(f).supportsAdvancedVideo).toBe(false);
    }
  });
});
