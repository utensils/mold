import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe, sdxlRecipe } from "@studio/lib/generationProfile.testFixtures";
import type { GenerationProfileSet, GenerationRecipeProfile } from "@studio/lib/generationProfile";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import { buildRequest, newGenerateForm } from "../lib/generateForm";
import { applyMobileGalleryMetadata } from "./reuse";

const model: ModelEntry = {
  name: "ltx2:q8",
  family: "ltx2",
  size_gb: 20,
  is_loaded: false,
  hf_repo: "example/ltx2",
  default_steps: 30,
  default_guidance: 3,
  default_width: 768,
  default_height: 512,
  description: "Video model",
  downloaded: true,
};

const metadata: OutputMetadata = {
  prompt: "a ship crossing violet lightning",
  original_prompt: "a ship",
  negative_prompt: "calm water",
  model: model.name,
  seed: 77,
  steps: 28,
  guidance: 4.25,
  width: 1536,
  height: 1024,
  generation_width: 768,
  generation_height: 512,
  output_format: "mp4",
  scheduler: "ddim",
  cfg_plus: true,
  loras: [{ path: "hidden.safetensors", scale: 0.8 }],
  upscale_model: "hidden-upscaler",
  enable_audio: true,
  frames: 121,
  fps: 30,
  pipeline: "two-stage-hq",
  pipeline_requested: true,
  spatial_upscale: "x2",
  temporal_upscale: "x2",
  guidance_overrides: {
    stg_scale: 1.25,
    stg_blocks: [28, 29],
  },
};

describe("applyMobileGalleryMetadata", () => {
  it("preserves canonical multiline prompts through mobile Library reuse", () => {
    const form = newGenerateForm();
    applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        prompt: "first line\n\nsecond line",
        negative_prompt: "blur\nwatermark",
        original_prompt: "source\nidea",
      },
      [model],
    );

    expect(form.prompt).toBe("first line\n\nsecond line");
    expect(form.negativePrompt).toBe("blur\nwatermark");
    expect(form.originalPrompt).toBe("source\nidea");
    expect(buildRequest(form)).toMatchObject({
      prompt: "first line\n\nsecond line",
      negative_prompt: "blur\nwatermark",
      original_prompt: "source\nidea",
    });
  });

  it("keeps an auto-chained One shot's own frame count", () => {
    const form = newGenerateForm();
    applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        pipeline_requested: false,
        output_mode: "one-shot",
        chain_job_id: "internal-chain",
        chain: {
          stage_count: 3,
          motion_tail_frames: 17,
          stages: [
            { prompt: metadata.prompt, frames: 97, transition: "smooth" },
            { prompt: metadata.prompt, frames: 97, transition: "smooth" },
            { prompt: metadata.prompt, frames: 97, transition: "smooth" },
          ],
        },
      },
      [model],
    );

    expect(form.frames).toBe(metadata.frames);
    expect(form.pipeline).toBeNull();
  });

  it("does not promote an ordinary nightly output's resolved pipeline to an override", () => {
    const form = newGenerateForm();
    applyMobileGalleryMetadata(
      form,
      {
        prompt: "the live nightly metadata shape",
        model: "ltx-2.3-22b-distilled:fp8",
        seed: 44,
        steps: 8,
        guidance: 1,
        width: 768,
        height: 768,
        frames: 217,
        fps: 24,
        pipeline: "distilled",
        output_mode: "one-shot",
        version: "0.23.3 (b3e803c 2026-08-21)",
      },
      [{ ...model, name: "ltx-2.3-22b-distilled:fp8" }],
    );

    expect(form.pipeline).toBeNull();
    expect(buildRequest(form).pipeline).toBeUndefined();
  });

  it("restores the desktop's full-fidelity metadata for mobile generation", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, metadata, [model]);

    expect(result).toEqual({
      modelName: model.name,
      substitutedModel: false,
      title: "",
    });

    expect(form).toMatchObject({
      prompt: metadata.prompt,
      originalPrompt: metadata.original_prompt,
      negativePrompt: metadata.negative_prompt,
      model: model.name,
      width: 768,
      height: 512,
      steps: 28,
      guidance: 4.25,
      seed: "77",
      outputFormat: "mp4",
      frames: 121,
      fps: 30,
      scheduler: "ddim",
      cfgPlus: true,
      loras: [
        {
          path: "hidden.safetensors",
          name: "hidden",
          scale: 0.8,
          trainedWords: [],
        },
      ],
      upscaleModel: "hidden-upscaler",
      enableAudio: true,
      pipeline: "two-stage-hq",
      spatialUpscale: "x2",
      temporalUpscale: "x2",
      guidanceOverrides: {
        stgScale: 1.25,
        stgBlocks: "28, 29",
      },
    });

    const request = buildRequest(form);
    expect(request).toMatchObject({
      original_prompt: "a ship",
      negative_prompt: "calm water",
      loras: [{ path: "hidden.safetensors", scale: 0.8 }],
      enable_audio: true,
      pipeline: "two-stage-hq",
      spatial_upscale: "x2",
      temporal_upscale: "x2",
      guidance_overrides: {
        stg_scale: 1.25,
        stg_blocks: [28, 29],
      },
    });
    expect(request.scheduler).toBeUndefined();
    expect(request.cfg_plus).toBeUndefined();
    expect(request.upscale_model).toBe("hidden-upscaler");
  });

  it("hands back the print's saved title, and an empty one for untitled prints", () => {
    const form = newGenerateForm();
    const titled = { ...metadata, title: "  Grain test 01 " } as OutputMetadata;
    expect(applyMobileGalleryMetadata(form, titled, [model]).title).toBe("Grain test 01");
    expect(applyMobileGalleryMetadata(form, metadata, [model]).title).toBe("");
  });

  it("clears stale provenance when reusing a promptless print", () => {
    const form = newGenerateForm();
    form.prompt = "previous expanded prompt";
    form.originalPrompt = "previous source prompt";

    applyMobileGalleryMetadata(form, { ...metadata, prompt: "", original_prompt: "stale" }, [
      model,
    ]);

    expect(form.prompt).toBe("");
    expect(form.originalPrompt).toBeNull();
    form.prompt = "a newly typed prompt";
    expect(buildRequest(form).original_prompt).toBeUndefined();
  });

  it("keeps generation valid when the print's original model is no longer installed", () => {
    const replacement = {
      ...model,
      name: "flux:replacement",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    };
    const form = newGenerateForm();

    const result = applyMobileGalleryMetadata(form, metadata, [replacement]);

    expect(result).toEqual({
      modelName: replacement.name,
      substitutedModel: true,
      title: "",
    });
    expect(form.model).toBe(replacement.name);
    expect(form.family).toBe(replacement.family);
    expect(form.outputFormat).toBe("png");
    expect(form.loras).toEqual([]);
    expect(form.upscaleModel).toBe("");
    expect(form.controlModel).toBe("");
    expect(form.cameraControl).toBeNull();
    expect(form.guidanceOverrides).toEqual({
      stgScale: null,
      stgBlocks: "",
      rescaleScale: null,
      modalityScale: null,
      skipStep: null,
    });
    expect(buildRequest(form)).toMatchObject({ model: replacement.name });
    expect(buildRequest(form).guidance_overrides).toBeUndefined();
    expect(buildRequest(form).loras).toBeUndefined();
    expect(buildRequest(form).upscale_model).toBeUndefined();
  });

  it("preserves a recorded Ref2VA partition instead of substituting another model", () => {
    const replacement = {
      ...model,
      name: "flux:replacement",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    } as ModelEntry;
    const form = newGenerateForm();

    const result = applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        model: "MiniMax_H3_Ref2VA",
        output_format: "mp4",
        references: [
          {
            kind: "image",
            index: 1,
            name: "subject.png",
            sha256: "a".repeat(64),
            mime_type: "image/png",
            width: 1024,
            height: 768,
          },
        ],
      },
      [replacement],
    );

    expect(result).toMatchObject({
      modelName: "minimax-h3-ref2va:comfy-pruned-int8",
      substitutedModel: false,
    });
    expect(form).toMatchObject({
      model: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      outputFormat: "mp4",
    });
    expect(form.h3Authoring?.references[0]?.reference).toMatchObject({
      kind: "image",
      media: { authority: "descriptor" },
      provenance: { name: "subject.png" },
    });
  });

  it("matches a legacy Ref2VA alias to its installed canonical checkpoint", () => {
    const ref2va = {
      ...model,
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      default_width: 1344,
      default_height: 768,
      default_steps: 50,
      default_guidance: 0,
    } as ModelEntry;
    const form = newGenerateForm();

    const result = applyMobileGalleryMetadata(
      form,
      { ...metadata, model: "minimax_h3_ref2va", output_format: "mp4" },
      [ref2va],
    );

    expect(result.substitutedModel).toBe(false);
    expect(form.model).toBe(ref2va.name);
    expect(form.family).toBe("minimax-h3");
  });

  it("preserves a missing official FL2VA checkpoint for exact-model recovery", () => {
    const replacement = {
      ...model,
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
    } as ModelEntry;
    const form = newGenerateForm();

    const result = applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        model: "minimax-h3-fl2va:official-bf16",
        output_format: "mp4",
      },
      [replacement],
    );

    expect(result).toMatchObject({
      modelName: "minimax-h3-fl2va:official-bf16",
      substitutedModel: false,
    });
    expect(form.model).toBe("minimax-h3-fl2va:official-bf16");
    expect(form.family).toBe("minimax-h3");
  });

  it.each([
    ["unavailable", []],
    [
      "installed",
      [
        {
          ...model,
          name: "minimax-h3-ref2va:future-layout",
          family: "minimax-h3",
        } as ModelEntry,
      ],
    ],
  ])("keeps an %s unknown H3 one-shot inside its fail-closed family", (_state, installed) => {
    const replacement = {
      ...model,
      name: "flux:replacement",
      family: "flux",
    } as ModelEntry;
    const form = newGenerateForm();
    const unknown = "minimax-h3-ref2va:future-layout";

    const result = applyMobileGalleryMetadata(
      form,
      { ...metadata, model: unknown, output_format: "mp4" },
      [...installed, replacement],
    );

    expect(result).toMatchObject({
      modelName: unknown,
      substitutedModel: false,
    });
    expect(form.model).toBe(unknown);
    expect(form.family).toBe("minimax-h3");
    expect(form.model).not.toBe(replacement.name);
  });

  it("restores a wan print's solver and recipe", () => {
    const wan = {
      ...model,
      name: "wan22-t2v-a14b:q5",
      family: "wan",
      default_width: 832,
      default_height: 480,
    } as ModelEntry;
    const form = newGenerateForm();

    applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        model: wan.name,
        scheduler: "euler",
        sample_shift: 12,
        distill_strength_high: 1.8,
        distill_strength_low: 0.9,
      },
      [wan],
    );

    expect(form.scheduler).toBe("euler");
    expect(buildRequest(form)).toMatchObject({
      scheduler: "euler",
      sample_shift: 12,
      distill_strength_high: 1.8,
      distill_strength_low: 0.9,
    });
  });

  it("drops a wan solver and recipe when the substituted model is another family", () => {
    const replacement = {
      ...model,
      name: "flux:replacement",
      family: "flux",
      default_width: 1024,
      default_height: 1024,
    } as ModelEntry;
    const form = newGenerateForm();

    applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        model: "wan22-t2v-a14b:q5",
        scheduler: "dpm-pp",
        sample_shift: 12,
        distill_strength_high: 1.8,
      },
      [replacement],
    );

    expect(form.scheduler).toBe("default");
    expect(form.wanRecipe).toEqual({
      sampleShift: null,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
    const request = buildRequest(form);
    expect("sample_shift" in request).toBe(false);
    expect("distill_strength_high" in request).toBe(false);
  });
});

/**
 * A print stitched from several clips restores as an ORDINARY ONE SHOT. There
 * is no clip rail on any surface any more, and the print is never refused —
 * whatever the host recorded comes back as a single render.
 *
 * The prompt is the whole subtlety: `metadata.prompt` on a stitched print is
 * every clip's prompt joined by newlines, which nobody wrote and nobody can
 * re-render. Clip 1's own prompt from `metadata.chain.stages[0]` is the only
 * honest answer, and it is the SHARED mapper that resolves it — asserted here
 * because the phone is one of its callers, not because it holds a second copy.
 */
describe("applyMobileGalleryMetadata — stitched prints", () => {
  const sequenceModel = {
    ...model,
    name: "ltx-video-0.9.8-2b-distilled:bf16",
    family: "ltx-video",
    supports_sequence: true,
  } as ModelEntry;

  function chainPrint(frames: number[], extra: Partial<OutputMetadata> = {}): OutputMetadata {
    return {
      ...metadata,
      model: sequenceModel.name,
      prompt: frames.map((_, i) => `clip ${i + 1}`).join("\n"),
      chain_job_id: "job-9",
      chain: {
        stage_count: frames.length,
        motion_tail_frames: 0,
        stages: frames.map((f, i) => ({
          prompt: `clip ${i + 1}`,
          frames: f,
          transition: "smooth" as const,
        })),
      },
      ...extra,
    } as OutputMetadata;
  }

  it("restores the FIRST clip's prompt, never the newline-joined blob", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 33]), [sequenceModel]);

    expect(form.prompt).toBe("clip 1");
    expect(form.prompt).not.toContain("clip 2");
    expect(result).toEqual({
      modelName: sequenceModel.name,
      substitutedModel: false,
      title: "",
    });
  });

  it("restores the recorded model, canvas, fps and seed as a one shot", () => {
    const form = newGenerateForm();
    applyMobileGalleryMetadata(
      form,
      chainPrint([25, 33], {
        width: 1024,
        height: 576,
        generation_width: 1024,
        generation_height: 576,
        fps: 30,
        seed: 4242,
      }),
      [sequenceModel],
    );

    expect(form.model).toBe(sequenceModel.name);
    expect(form.width).toBe(1024);
    expect(form.height).toBe(576);
    expect(form.fps).toBe(30);
    expect(form.seed).toBe("4242");
  });

  it("never refuses a stitched print, including an H3 one", () => {
    const h3 = {
      ...sequenceModel,
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
    } as ModelEntry;
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 25], { model: h3.name }), [h3]);

    expect(result.modelName).toBe(h3.name);
    expect(result.substitutedModel).toBe(false);
    expect(form.prompt).toBe("clip 1");
  });

  it("substitutes an installed model when the recorded one is gone", () => {
    const other = {
      ...sequenceModel,
      name: "wan22-ti2v-5b:dmd",
      family: "wan",
    } as ModelEntry;
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 25]), [other]);

    expect(result.substitutedModel).toBe(true);
    expect(form.model).toBe(other.name);
    expect(form.prompt).toBe("clip 1");
  });

  it("keeps an ephemeral auto-chain's own prompt untouched", () => {
    // Every stage of an auto-chained long video carries the SAME prompt, so
    // the recorded prompt and clip 1's are the same string — nothing to undo.
    const form = newGenerateForm();
    applyMobileGalleryMetadata(
      form,
      {
        ...metadata,
        model: sequenceModel.name,
        output_mode: "one-shot",
        chain: {
          stage_count: 2,
          motion_tail_frames: 17,
          stages: [
            { prompt: metadata.prompt, frames: 97, transition: "smooth" as const },
            { prompt: metadata.prompt, frames: 97, transition: "smooth" as const },
          ],
        },
      } as OutputMetadata,
      [sequenceModel],
    );

    expect(form.prompt).toBe(metadata.prompt);
    expect(form.originalPrompt).toBe(metadata.original_prompt);
  });
});

/**
 * A 3-D print restores through the same shared mapper as every other print,
 * so the phone must not undo its answers afterwards: the format stays the
 * `glb` the recipe advertises, the canvas stays the zero canvas a canvasless
 * recipe renders with, and the mesh controls the print recorded come back.
 */
describe("applyMobileGalleryMetadata on a mesh print", () => {
  function profile(recipe: GenerationRecipeProfile): GenerationProfileSet {
    return {
      schema_version: 1,
      profile_id: "mesh-reuse",
      profile_hash: "mesh-reuse-hash",
      default_recipe_id: recipe.id,
      recipes: [recipe],
    };
  }

  const meshModel: ModelEntry = {
    name: "hunyuan3d-mini-turbo:fp16",
    family: "hunyuan3d",
    size_gb: 2,
    is_loaded: false,
    hf_repo: "tencent/Hunyuan3D-2mini",
    default_steps: 5,
    default_guidance: 5,
    default_width: 1024,
    default_height: 1024,
    description: "Mesh model",
    downloaded: true,
    source_image: "required",
    generation_profile: profile(hunyuan3dRecipe()),
  };

  const meshPrint: OutputMetadata = {
    prompt: "an armchair",
    model: meshModel.name,
    seed: 4,
    steps: 5,
    guidance: 5,
    // A mesh print's width/height describe its POSTER, never a canvas.
    width: 512,
    height: 512,
    output_format: "glb",
    strength: 0.42,
    source_fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    mesh: { octree_resolution: 384, threshold: 0.42, target_faces: 25_000 },
  };

  it("keeps glb, the zero canvas, and the recorded mesh controls", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, meshPrint, [meshModel]);

    expect(result.modelName).toBe(meshModel.name);
    expect(result.substitutedModel).toBe(false);
    expect(form.outputFormat).toBe("glb");
    expect(form.width).toBe(0);
    expect(form.height).toBe(0);
    expect(form.mesh).toEqual({
      octreeResolution: 384,
      threshold: 0.42,
      targetFaces: 25_000,
    });

    form.sourceImage = "c291cmNl";
    const request = buildRequest(form);
    expect(request.output_format).toBe("glb");
    expect(request.width).toBe(0);
    expect(request.height).toBe(0);
    expect(request.source_fit).toBeUndefined();
    expect(request.strength).toBeUndefined();
    expect(request.mesh).toEqual({
      octree_resolution: 384,
      threshold: 0.42,
      target_faces: 25_000,
    });
  });

  /**
   * The phone corrected the restored format against the pre-profile FAMILY
   * list, which recognizes exactly one mesh family name. A host that ships a
   * second one still advertises `glb` in its recipe, and the family list
   * would answer `png` — turning a reuse into a request the mesh engine
   * refuses. The advertised recipe is the authority.
   */
  it("keeps an advertised mesh format the legacy family list does not know", () => {
    const futureMesh: ModelEntry = {
      ...meshModel,
      name: "hunyuan3d-2.1-standard:fp16",
      family: "hunyuan3d-2.1",
    };

    const form = newGenerateForm();
    applyMobileGalleryMetadata(form, { ...meshPrint, model: futureMesh.name }, [futureMesh]);

    expect(form.outputFormat).toBe("glb");
    expect(buildRequest(form).output_format).toBe("glb");
  });

  /**
   * The same rule in the other direction: the substituted model's own
   * advertised contract, not the wider family list, corrects the format.
   */
  it("corrects the format against the substituted model's recipe, not its family", () => {
    const narrowRecipe = sdxlRecipe();
    narrowRecipe.capabilities.output = {
      ...narrowRecipe.capabilities.output,
      default_format: "png",
      formats: ["png"],
    };
    const fallback: ModelEntry = {
      name: "sdxl-base:fp16",
      family: "sdxl",
      size_gb: 6,
      is_loaded: false,
      hf_repo: "sdxl",
      default_steps: 30,
      default_guidance: 7,
      default_width: 1024,
      default_height: 1024,
      description: "Image model",
      downloaded: true,
      generation_profile: profile(narrowRecipe),
    };

    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, { ...meshPrint, output_format: "webp" }, [
      fallback,
    ]);

    expect(result.substitutedModel).toBe(true);
    // `webp` is in the legacy sdxl family list but NOT in this recipe's.
    expect(form.outputFormat).toBe("png");
  });
});
