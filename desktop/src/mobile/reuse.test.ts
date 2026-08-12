import { describe, expect, it } from "vitest";
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
  spatial_upscale: "x2",
  temporal_upscale: "x2",
  guidance_overrides: {
    stg_scale: 1.25,
    stg_blocks: [28, 29],
  },
};

describe("applyMobileGalleryMetadata", () => {
  it("restores the desktop's full-fidelity metadata for mobile generation", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, metadata, [model]);

    expect(result).toEqual({
      modelName: model.name,
      substitutedModel: false,
      sequence: null,
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
    expect(request.upscale_model).toBeUndefined();
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
      sequence: null,
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
      sequence: null,
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
      sequence: null,
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

// iPhone gets Reuse only in this pass (Edit sequence needs a chain-detail
// fetch on the recovery route), so the clip rail is the whole contract here.
describe("applyMobileGalleryMetadata — sequence prints", () => {
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

  it("returns the clip rail without overwriting the one-shot prompt", () => {
    const form = newGenerateForm();
    form.prompt = "parked one shot";
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 33]), [sequenceModel]);

    expect(result.sequence?.clips.map((c) => c.prompt)).toEqual(["clip 1", "clip 2"]);
    expect(result.sequence?.clips.map((c) => c.frames)).toEqual([25, 33]);
    expect(result.sequence?.raised).toBe(0);
    expect(form.prompt).toBe("parked one shot");
  });

  it("raises clips that no longer clear the model's motion tail", () => {
    const ltx2 = {
      ...sequenceModel,
      name: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
    } as ModelEntry;
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([9, 65], { model: ltx2.name }), [
      ltx2,
    ]);

    expect(result.sequence?.raised).toBe(1);
    expect(result.sequence?.clips.every((c) => c.frames > 17)).toBe(true);
  });

  it("substitutes a SEQUENCE-capable model when the recorded one is gone", () => {
    // Falling back to the first installed model would hand the clip rail an
    // image model that cannot render a sequence at all.
    const still = {
      ...model,
      name: "flux-dev:q8",
      family: "flux",
      supports_sequence: false,
    } as ModelEntry;
    const other = {
      ...sequenceModel,
      name: "ltx-video-0.9.8-2b-distilled:q8",
    } as ModelEntry;
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 25]), [still, other]);

    expect(result.substitutedModel).toBe(true);
    expect(result.modelName).toBe(other.name);
    expect(result.sequence?.clips).toHaveLength(2);
  });

  it("fails an H3 sequence explicitly without substituting another partition", () => {
    const other = {
      ...sequenceModel,
      name: "ltx-video-0.9.8-2b-distilled:q8",
    } as ModelEntry;
    const form = newGenerateForm();
    const initialModel = form.model;
    const result = applyMobileGalleryMetadata(
      form,
      chainPrint([25, 25], { model: "minimax-h3-fl2va:official-bf16" }),
      [other],
    );

    expect(result).toMatchObject({
      modelName: "minimax-h3-fl2va:official-bf16",
      substitutedModel: false,
      sequence: null,
    });
    expect(result.sequenceUnsupportedReason).toContain("cannot render a clip sequence");
    expect(form.model).toBe(initialModel);
    expect(form.model).not.toBe(other.name);
  });

  it.each(["unavailable", "installed"])(
    "refuses an %s unknown H3 sequence without mutating the form",
    (availability) => {
      const unknown = "minimax-h3-ref2va:future-layout";
      const other = {
        ...sequenceModel,
        name: "ltx-video-0.9.8-2b-distilled:q8",
      } as ModelEntry;
      const installedUnknown = {
        ...sequenceModel,
        name: unknown,
        family: "minimax-h3",
      } as ModelEntry;
      const form = newGenerateForm();
      form.prompt = "parked one shot";
      const before = structuredClone(form);

      const result = applyMobileGalleryMetadata(
        form,
        chainPrint([25, 25], { model: unknown }),
        availability === "installed" ? [installedUnknown, other] : [other],
      );

      expect(result).toMatchObject({
        modelName: unknown,
        substitutedModel: false,
        sequence: null,
      });
      expect(result.sequenceUnsupportedReason).toContain("cannot render a clip sequence");
      expect(form).toEqual(before);
    },
  );

  it("reports what the print could not give back", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(
      form,
      chainPrint([25, 25], { negative_prompt: "calm water" }),
      [sequenceModel],
    );
    expect(result.sequence?.lossy.negatives).toBe(true);
    expect(result.sequence?.clips[0]!.negativePrompt).toBe("calm water");
    expect(result.sequence?.clips[1]!.negativePrompt).toBe("");
  });

  /**
   * Wan's seam carries context only for an image-conditioned checkpoint
   * (#783), so the live tail the reused clips are clamped against belongs to
   * the resolved model's advertised `source_image`, not to the family. Reuse
   * passed bare name/family strings, which made every wan sequence look
   * tail-free. A single-frame clip is the boundary the one-frame handoff
   * moves.
   */
  it("clamps against an image-conditioned wan checkpoint's one-frame handoff", () => {
    const wan = (name: string, sourceImage: string) =>
      ({
        ...sequenceModel,
        name,
        family: "wan",
        source_image: sourceImage,
      }) as ModelEntry;

    const conditioned = wan("wan22-i2v-a14b:q5", "required");
    expect(
      applyMobileGalleryMetadata(
        newGenerateForm(),
        chainPrint([1, 53], { model: conditioned.name }),
        [conditioned],
      ).sequence?.raised,
    ).toBe(1);

    // A text-to-video checkpoint genuinely carries nothing across the seam.
    const unconditioned = wan("wan22-t2v-a14b:q5", "unsupported");
    expect(
      applyMobileGalleryMetadata(
        newGenerateForm(),
        chainPrint([1, 53], { model: unconditioned.name }),
        [unconditioned],
      ).sequence?.raised,
    ).toBe(0);
  });
});
