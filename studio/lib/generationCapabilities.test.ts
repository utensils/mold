import { describe, expect, it } from "vitest";
import {
  baseGenerationCapabilities,
  isAdvancedVideoFamily,
  isImageConditionedVideoFamily,
  isWanFamily,
  schedulerLabel,
} from "./generationCapabilities";
import type { GenerationRecipeProfile } from "./generationProfile";
import { hunyuan3dRecipe, sdxlRecipe } from "./generationProfile.testFixtures";

describe("baseGenerationCapabilities", () => {
  it("takes advanced, scheduler, and output policy from the resolved recipe", () => {
    const recipe = {
      capabilities: {
        guidance: { adjustable: false, supports_negative_prompt: false },
        negative_prompt: { mode: "hidden", required: false },
        supports_audio: false,
        source_video: { mode: "hidden", required: false },
        mask: { mode: "hidden", required: false },
        keyframes: { mode: "hidden", required: false },
        audio: { mode: "hidden", required: false },
        lora: { mode: "hidden", max_count: 0 },
        controlnet: { mode: "adjustable", max_count: 1 },
        output: {
          default_format: "jpeg",
          formats: ["jpeg"],
          audio_requires_mp4: false,
        },
        wan_recipe: {
          mode: "hidden",
          supports_distill_strength: false,
          supports_first_last_frame: false,
        },
        schedulers: ["ddim"],
      },
    } as unknown as GenerationRecipeProfile;
    expect(
      baseGenerationCapabilities("flux", "", null, null, null, recipe),
    ).toMatchObject({
      supportsNegativePrompt: false,
      supportsControlNet: true,
      supportsLora: false,
      supportsMask: false,
      supportsScheduler: true,
      schedulerOptions: ["default", "ddim"],
      outputFormats: ["jpeg"],
      defaultOutputFormat: "jpeg",
    });
  });

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
        offersAudioControl: true,
      });
      expect(isAdvancedVideoFamily(family)).toBe(true);
    }
  });

  it("keeps the LTX audio control visible when a checkpoint lacks audio assets", () => {
    const videoOnlyRecipe = {
      capabilities: {
        guidance: { adjustable: false, supports_negative_prompt: false },
        negative_prompt: { mode: "hidden", required: false },
        supports_audio: false,
        source_video: { mode: "adjustable", required: false },
        mask: { mode: "hidden", required: false },
        keyframes: { mode: "adjustable", required: false },
        audio: { mode: "hidden", required: false },
        lora: { mode: "adjustable", max_count: 4 },
        controlnet: { mode: "hidden", max_count: 0 },
        output: {
          default_format: "mp4",
          formats: ["mp4"],
          audio_requires_mp4: false,
        },
        wan_recipe: {
          mode: "hidden",
          supports_distill_strength: false,
          supports_first_last_frame: false,
        },
        schedulers: [],
      },
    } as unknown as GenerationRecipeProfile;

    for (const family of ["ltx2", "ltx-2"]) {
      expect(
        baseGenerationCapabilities(
          family,
          "ltx-video-only",
          null,
          null,
          null,
          videoOnlyRecipe,
        ),
      ).toMatchObject({
        supportsAudio: false,
        offersAudioControl: true,
      });
    }

    expect(baseGenerationCapabilities("ltx-video")).toMatchObject({
      supportsAudio: false,
      offersAudioControl: false,
    });
    expect(baseGenerationCapabilities("wan")).toMatchObject({
      supportsAudio: false,
      offersAudioControl: false,
    });
    expect(
      baseGenerationCapabilities(
        "minimax-h3",
        "minimax-h3-fl2va:official-bf16",
      ),
    ).toMatchObject({
      supportsAudio: true,
      offersAudioControl: false,
    });
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

  it("offers no scheduler when a wan recipe advertises an empty list", () => {
    // The DMD tiers pin their own sampler, so the recipe carries
    // `schedulers: []` — and the server drops the key on the wire
    // (`skip_serializing_if = "Vec::is_empty"`). A recipe IS in hand, so the
    // missing key means "none", never the legacy-host solver fallback: every
    // option the fallback would offer is a 422 on this tier.
    const dmdRecipe = JSON.parse(
      JSON.stringify({
        capabilities: {
          guidance: { adjustable: false, supports_negative_prompt: false },
          negative_prompt: { mode: "hidden", required: false },
          supports_audio: false,
          source_video: { mode: "hidden", required: false },
          mask: { mode: "hidden", required: false },
          keyframes: { mode: "hidden", required: false },
          audio: { mode: "hidden", required: false },
          lora: { mode: "adjustable", max_count: 4 },
          controlnet: { mode: "hidden", max_count: 0 },
          output: {
            default_format: "mp4",
            formats: ["mp4"],
            audio_requires_mp4: false,
          },
          wan_recipe: {
            mode: "hidden",
            supports_distill_strength: false,
            supports_first_last_frame: false,
          },
          // `skip_serializing_if = "Vec::is_empty"` — the round-trip below
          // removes the key exactly as the server's JSON does.
          schedulers: undefined,
        },
      }),
    ) as GenerationRecipeProfile;
    expect(
      "schedulers" in
        (dmdRecipe.capabilities as unknown as Record<string, unknown>),
    ).toBe(false);

    const caps = baseGenerationCapabilities(
      "wan",
      "wan21-t2v-1.3b:turbo",
      null,
      null,
      null,
      dmdRecipe,
    );
    expect(caps.schedulerOptions).toEqual([]);
    expect(caps.supportsScheduler).toBe(false);
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

describe("Flux.2 negative prompt", () => {
  // The undistilled [klein] base checkpoints are the one Flux.2 tier that
  // samples with a real unconditional branch, so they are the one tier whose
  // negative prompt reaches the render. Mirrors
  // `mold_core::validation::is_flux2_base_model`. A host that advertises its
  // guidance capabilities still wins; this is the no-advertisement fallback.
  it("is offered for undistilled base tiers only", () => {
    for (const base of [
      "flux2-klein-base:bf16",
      "flux2-klein-base:q4",
      "flux2-klein-base-9b:q8",
    ]) {
      expect(
        baseGenerationCapabilities("flux2", base).supportsNegativePrompt,
      ).toBe(true);
    }
    for (const distilled of [
      "flux2-klein:bf16",
      "flux2-klein-9b:q8",
      "flux2-dev:q4",
    ]) {
      expect(
        baseGenerationCapabilities("flux2", distilled).supportsNegativePrompt,
      ).toBe(false);
    }
  });

  it("still defers to what the host advertises", () => {
    expect(
      baseGenerationCapabilities("flux2", "flux2-klein-base:q4", null, {
        adjustable: true,
        supports_negative_prompt: false,
      }).supportsNegativePrompt,
    ).toBe(false);
  });
});

describe("prompt mode, strength, and mesh from the advertised recipe", () => {
  it("reads the hunyuan3d contract: no canvas, no strength, prompt ignored, mesh controls", () => {
    const caps = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      hunyuan3dRecipe(),
    );
    expect(caps.supportsStrength).toBe(false);
    expect(caps.promptMode).toBe("ignored");
    expect(caps.canvasless).toBe(true);
    expect(caps.mesh).toBeDefined();
    expect(caps.mesh?.octree_default).toBe(256);
    expect(caps.mesh?.octree_resolutions).toEqual([128, 192, 256, 320, 384]);
    expect(caps.mesh?.texture.mode).toBe("hidden");
    expect(caps.requiresSourceImage).toBe(true);
    expect(caps.outputFormats).toEqual(["glb"]);
  });

  it("reads the sdxl contract: canvas, strength, prompt required, no mesh", () => {
    const caps = baseGenerationCapabilities(
      "sdxl",
      "cyberrealistic-pony:fp16",
      null,
      null,
      null,
      sdxlRecipe(),
    );
    expect(caps.supportsStrength).toBe(true);
    expect(caps.promptMode).toBe("required");
    expect(caps.canvasless).toBe(false);
    expect(caps.mesh).toBeUndefined();
  });

  /**
   * A recipe is not always in hand: Create aimed at a machine that must
   * download the checkpoint first resolves no model row, and an older host
   * advertises no profile at all. The pre-profile family rule answers then —
   * a 3-D print has no pixel canvas either way, and reading absence as "has
   * one" put a Shape and Resolution pair on a mesh model's 0 × 0.
   */
  it("keeps a mesh family canvasless when no recipe is advertised", () => {
    const caps = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      null,
    );
    expect(caps.canvasless).toBe(true);
    // Nothing can invent the advertised block; only the canvas rule survives.
    expect(caps.mesh).toBeUndefined();
  });

  /**
   * The canvas was only the first thing the family rule had to answer. A 3-D
   * engine denoises nothing, repaints nothing, has no unconditional branch to
   * steer, and stores exactly one container — so with no recipe in hand the
   * legacy rules must say all four, or the panel offers a Denoise slider, an
   * Edit-mask control, a Negative-prompt field and a png/jpeg/webp format
   * picker for a print that is none of those things.
   */
  it("refuses strength, mask and a negative prompt for a mesh family with no recipe", () => {
    const caps = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      null,
    );
    expect(caps.supportsStrength).toBe(false);
    expect(caps.supportsMask).toBe(false);
    expect(caps.supportsNegativePrompt).toBe(false);
  });

  it("stores only the glTF container for a mesh family with no recipe", () => {
    const caps = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      null,
    );
    expect(caps.outputFormats).toEqual(["glb"]);
    expect(caps.defaultOutputFormat).toBe("glb");
  });

  /**
   * The whole point of the fallback is that a client aimed at a machine
   * without the checkpoint shows the SAME contract as one aimed at a machine
   * that has it. Only the advertised mesh block, which nothing can invent,
   * may differ.
   */
  it("agrees with the advertised recipe on every contract the panel renders", () => {
    const advertised = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      hunyuan3dRecipe(),
    );
    const fallback = baseGenerationCapabilities(
      "hunyuan3d",
      "hunyuan3d-mini-turbo:fp16",
      null,
      null,
      "required",
      null,
    );
    for (const key of [
      "canvasless",
      "supportsStrength",
      "supportsMask",
      "supportsNegativePrompt",
      "defaultOutputFormat",
    ] as const) {
      expect(fallback[key], key).toEqual(advertised[key]);
    }
    expect(fallback.outputFormats).toEqual(advertised.outputFormats);
  });

  it("leaves a raster family's strength, mask, negative and formats alone", () => {
    const caps = baseGenerationCapabilities(
      "sdxl",
      "cyberrealistic-pony:fp16",
      null,
      null,
      null,
      null,
    );
    expect(caps.supportsStrength).toBe(true);
    expect(caps.supportsMask).toBe(true);
    expect(caps.supportsNegativePrompt).toBe(true);
    expect(caps.outputFormats).toEqual(["png", "jpeg", "webp"]);
  });

  it("keeps a raster family's canvas when no recipe is advertised", () => {
    const caps = baseGenerationCapabilities(
      "sdxl",
      "cyberrealistic-pony:fp16",
      null,
      null,
      null,
      null,
    );
    expect(caps.canvasless).toBe(false);
  });

  it("trusts an advertised supports_strength over the family heuristic", () => {
    // The host says a flux checkpoint does not read strength; the client
    // must not overrule it with the old "every image family does" rule.
    const recipe = sdxlRecipe();
    recipe.capabilities.supports_strength = false;
    expect(
      baseGenerationCapabilities("flux", "", null, null, null, recipe)
        .supportsStrength,
    ).toBe(false);
  });

  it("falls back to the legacy strength and prompt rules when the host is silent", () => {
    // An older host's recipe carries neither field; its serde default would
    // be `false`, which is not an assertion that strength is unsupported.
    const recipe = sdxlRecipe();
    delete (recipe.capabilities as { supports_strength?: unknown })
      .supports_strength;
    delete (recipe.capabilities as { prompt?: unknown }).prompt;
    const flux = baseGenerationCapabilities(
      "flux",
      "",
      null,
      null,
      null,
      recipe,
    );
    expect(flux.supportsStrength).toBe(true);
    expect(flux.promptMode).toBe("required");
    const wan = baseGenerationCapabilities(
      "wan",
      "wan22-i2v-a14b:q8",
      null,
      null,
      null,
      recipe,
    );
    expect(wan.supportsStrength).toBe(false);
    expect(
      baseGenerationCapabilities("ltx2", "", null, null, null, recipe)
        .promptMode,
    ).toBe("optional");
    // No recipe at all: the same legacy answers.
    expect(baseGenerationCapabilities("ltx2")).toMatchObject({
      promptMode: "optional",
      supportsStrength: true,
      canvasless: false,
      mesh: undefined,
    });
  });
});
