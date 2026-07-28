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
    expect(buildRequest(form)).toMatchObject({ model: replacement.name });
    expect(buildRequest(form).loras).toBeUndefined();
    expect(buildRequest(form).upscale_model).toBeUndefined();
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

  it("returns the clip rail with clip 1's prompt on the form", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, chainPrint([25, 33]), [sequenceModel]);

    expect(result.sequence?.clips.map((c) => c.prompt)).toEqual(["clip 1", "clip 2"]);
    expect(result.sequence?.clips.map((c) => c.frames)).toEqual([25, 33]);
    expect(result.sequence?.raised).toBe(0);
    // Never the newline join — the whole point of a sequence-aware reuse.
    expect(form.prompt).toBe("clip 1");
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
});
