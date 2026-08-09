import { describe, expect, it } from "vitest";
import {
  baseGenerationCapabilities,
  isAdvancedVideoFamily,
  isImageConditionedVideoFamily,
  isWanFamily,
  schedulerLabel,
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

  it("gives H3 task partitions synchronized AV without generic controls", () => {
    expect(
      baseGenerationCapabilities(
        "minimax-h3",
        "minimax-h3-fl2va:comfy-pruned-int8",
      ),
    ).toMatchObject({
      supportsVideo: true,
      supportsAudio: true,
      supportsNegativePrompt: false,
      guidanceAdjustable: false,
      fixedGuidance: 0,
      supportsScheduler: false,
      supportsCfgPlus: false,
      supportsLora: false,
      supportsControlNet: false,
      sourceImageMode: "h3-boundaries",
      supportsMask: false,
      forcesBatchSizeOne: true,
    });
    expect(
      baseGenerationCapabilities(
        "minimax_h3",
        "minimax_h3_ref2va:official-bf16",
      ).sourceImageMode,
    ).toBe("ordered-references");
    expect(
      baseGenerationCapabilities("", "minimax-h3-ref2va:comfy-pruned-int8"),
    ).toMatchObject({
      supportsVideo: true,
      supportsAudio: true,
      sourceImageMode: "ordered-references",
    });
    expect(isAdvancedVideoFamily("minimax-h3")).toBe(false);
  });

  it("offers wan the LoRA control its engine supports", () => {
    // Mirrors `workflows.lora: true` for the wan family in
    // `crates/mold-inference/src/batch.rs`. Without wan in
    // LORA_CAPABLE_FAMILIES the control is hidden on every surface even
    // though the server would accept the request — which is how the A14B
    // fast tier's four-step distill would become unreachable from the UI.
    expect(baseGenerationCapabilities("wan").supportsLora).toBe(true);
  });

  it("hides strength and mask for wan — pinned frames take neither", () => {
    // Wan pins conditioning frames exactly: the engine never reads
    // `strength` and rejects `mask_image`, so showing either control
    // advertises a knob the render ignores or refuses.
    const wan = baseGenerationCapabilities("wan", "wan22-i2v-a14b:q8");
    expect(wan.supportsStrength).toBe(false);
    expect(wan.supportsMask).toBe(false);
    // LTX-2 image-to-video genuinely consumes strength.
    expect(baseGenerationCapabilities("ltx2").supportsStrength).toBe(true);
  });

  it("withholds the LoRA control from the wan fp8-scaled tier", () => {
    // `WanTransformer::from_safetensors_with_loras` refuses every adapter
    // stack on fp8-scaled weights (merging would re-round the delta to three
    // mantissa bits), so offering the control advertises a load that always
    // fails. GGUF and bf16 tiers keep it.
    expect(
      baseGenerationCapabilities("wan", "wan22-t2v-a14b:fp8").supportsLora,
    ).toBe(false);
    expect(
      baseGenerationCapabilities("wan", "wan22-i2v-a14b:fp8").supportsLora,
    ).toBe(false);
    expect(
      baseGenerationCapabilities("wan", "wan22-t2v-a14b:q8").supportsLora,
    ).toBe(true);
  });

  it("separates image conditioning from the advanced-video panel", () => {
    // Two independent questions that happened to have the same answer while
    // LTX-2 was the only image-conditioned video family. Wan is the case that
    // separates them, and a consumer that keeps deriving one from the other
    // hides the source-image control for a family that requires it.
    expect(isImageConditionedVideoFamily("wan")).toBe(true);
    expect(isAdvancedVideoFamily("wan")).toBe(false);

    for (const family of ["ltx2", "ltx-2"]) {
      expect(isImageConditionedVideoFamily(family)).toBe(true);
      expect(isAdvancedVideoFamily(family)).toBe(true);
    }

    // Plain ltx-video has no image-to-video path and would ignore an image.
    expect(isImageConditionedVideoFamily("ltx-video")).toBe(false);
    // Image families are not video-image-conditioned; they have their own
    // source-image handling and must not be routed through this predicate.
    expect(isImageConditionedVideoFamily("flux")).toBe(false);
    expect(isImageConditionedVideoFamily("")).toBe(false);
  });

  it("resolves the source-image contract per model, family as fallback", () => {
    // Advertised wins: the three wan checkpoints split three ways and the
    // family cannot tell them apart.
    expect(
      baseGenerationCapabilities(
        "wan",
        "wan22-t2v-a14b",
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
      baseGenerationCapabilities(
        "wan",
        "wan22-i2v-a14b",
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
      baseGenerationCapabilities(
        "wan",
        "wan22-ti2v-5b",
        null,
        null,
        "optional",
      ),
    ).toMatchObject({
      sourceImageCapability: "optional",
      supportsSourceImage: true,
      requiresSourceImage: false,
      supportsEndFrame: true,
    });

    // An older server advertises nothing, so every wan checkpoint keeps
    // today's optional well and nothing is ever gated on a guess.
    expect(baseGenerationCapabilities("wan", "wan22-i2v-a14b")).toMatchObject({
      sourceImageCapability: "optional",
      supportsSourceImage: true,
      requiresSourceImage: false,
      supportsEndFrame: false,
    });

    // Image families read a source image; a video family with no
    // image-to-video path does not.
    expect(baseGenerationCapabilities("flux").supportsSourceImage).toBe(true);
    expect(baseGenerationCapabilities("ltx2").supportsSourceImage).toBe(true);
    expect(baseGenerationCapabilities("ltx-video").supportsSourceImage).toBe(
      false,
    );

    // The end frame is wan's alone even where a source image is advertised.
    expect(
      baseGenerationCapabilities("ltx2", "", null, null, "optional")
        .supportsEndFrame,
    ).toBe(false);
  });

  it("returns independent scheduler option lists", () => {
    const first = baseGenerationCapabilities("sdxl").schedulerOptions;
    first.pop();
    expect(baseGenerationCapabilities("sdxl").schedulerOptions).toEqual([
      "default",
      "ddim",
      "euler-ancestral",
      "uni-pc",
    ]);
  });

  it("offers wan its sample solvers and no UNet scheduler", () => {
    // The two sets are disjoint on the server: validation rejects `ddim` /
    // `euler-ancestral` for wan and rejects `euler` / `dpm-pp` everywhere
    // else. `uni-pc` is the one solver both sides accept.
    expect(baseGenerationCapabilities("wan").schedulerOptions).toEqual([
      "default",
      "uni-pc",
      "euler",
      "dpm-pp",
    ]);
    expect(baseGenerationCapabilities("wan").supportsScheduler).toBe(true);
    for (const family of ["sd15", "sdxl"]) {
      const options = baseGenerationCapabilities(family).schedulerOptions;
      expect(options).not.toContain("euler");
      expect(options).not.toContain("dpm-pp");
    }
  });

  it("gates the wan recipe controls on family and distill tier", () => {
    // Flow shift and the solver apply to every wan checkpoint; the per-expert
    // distill strengths only to the A14B tiers that actually ship a Lightning
    // adapter, which is what the engine checks against the resolved paths.
    for (const model of ["wan22-t2v-a14b:q5", "wan22-i2v-a14b:q4"]) {
      expect(baseGenerationCapabilities("wan", model).wanRecipe).toEqual({
        supported: true,
        supportsDistillStrength: true,
      });
    }
    for (const model of [
      "wan22-t2v-a14b:q8",
      "wan22-ti2v-5b:fp16",
      "wan21-t2v-1.3b:bf16",
      "hf:opaque/wan-checkpoint",
    ]) {
      expect(baseGenerationCapabilities("wan", model).wanRecipe).toEqual({
        supported: true,
        supportsDistillStrength: false,
      });
    }
    // Off-family the whole group is unavailable — the server rejects every one
    // of these fields for a non-wan model rather than ignoring it.
    for (const family of ["ltx2", "flux", "sdxl", "minimax-h3"]) {
      expect(baseGenerationCapabilities(family).wanRecipe).toEqual({
        supported: false,
        supportsDistillStrength: false,
      });
    }
  });

  it("names every solver the same way on every surface", () => {
    expect(schedulerLabel("uni-pc")).toBe("UniPC");
    expect(schedulerLabel("dpm-pp")).toBe("DPM++");
    expect(schedulerLabel("euler-ancestral")).toBe("Euler ancestral");
    // An unknown value from a newer server renders verbatim, never blank.
    expect(schedulerLabel("res-multistep")).toBe("res-multistep");
  });

  it("recognizes the wan family regardless of casing and padding", () => {
    expect(isWanFamily(" WAN ")).toBe(true);
    expect(isWanFamily("wan21")).toBe(false);
    expect(isWanFamily("")).toBe(false);
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
