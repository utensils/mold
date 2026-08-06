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

  it("treats wan as video without audio or a bespoke advanced panel", () => {
    // The three flags are independent and wan sits differently on each. Video:
    // yes. Audio: no — wan checkpoints ship no audio VAE and no vocoder, so an
    // audio toggle would offer a request the server rejects before denoising.
    // Advanced panel: no — wan needs only the generic steps/guidance/negative
    // controls, unlike LTX-2's guidance overrides and spatial rungs.
    expect(baseGenerationCapabilities("wan")).toMatchObject({
      supportsVideo: true,
      supportsAudio: false,
    });
    expect(isAdvancedVideoFamily("wan")).toBe(false);
  });

  it("offers wan the LoRA control its engine supports", () => {
    // Mirrors `workflows.lora: true` for the wan family in
    // `crates/mold-inference/src/batch.rs`. Without wan in
    // LORA_CAPABLE_FAMILIES the control is hidden on every surface even
    // though the server would accept the request — which is how the A14B
    // fast tier's four-step distill would become unreachable from the UI.
    expect(baseGenerationCapabilities("wan").supportsLora).toBe(true);
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

  it("resolves LTX guidance from both checkpoint and explicit pipeline", () => {
    expect(
      baseGenerationCapabilities("ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toMatchObject({
      guidanceAdjustable: false,
      fixedGuidance: 1,
      supportsNegativePrompt: false,
    });
    expect(
      baseGenerationCapabilities("ltx2", "ltx-2.3-22b-dev:fp8"),
    ).toMatchObject({
      guidanceAdjustable: true,
      fixedGuidance: null,
      supportsNegativePrompt: true,
    });
    expect(
      baseGenerationCapabilities("ltx2", "ltx-2.3-22b-dev:fp8", "distilled"),
    ).toMatchObject({
      guidanceAdjustable: false,
      supportsNegativePrompt: false,
    });
    expect(
      baseGenerationCapabilities(
        "ltx2",
        "ltx-2.3-22b-distilled:fp8",
        "two-stage",
      ),
    ).toMatchObject({ guidanceAdjustable: true, supportsNegativePrompt: true });
    expect(
      baseGenerationCapabilities(
        "ltx-video",
        "ltx-video-0.9.8-13b-distilled:bf16",
      ),
    ).toMatchObject({
      guidanceAdjustable: false,
      supportsNegativePrompt: false,
    });
    expect(
      baseGenerationCapabilities("ltx2", "hf:opaque/checkpoint", null, {
        adjustable: false,
        supports_negative_prompt: false,
        fixed_scale: 1,
      }),
    ).toMatchObject({
      guidanceAdjustable: false,
      fixedGuidance: 1,
      supportsNegativePrompt: false,
    });
  });
});
