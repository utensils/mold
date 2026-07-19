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
  it("restores visible prompt controls without retaining invisible advanced settings", () => {
    const form = newGenerateForm();
    const result = applyMobileGalleryMetadata(form, metadata, [model]);

    expect(result).toEqual({ modelName: model.name, substitutedModel: false });

    expect(form).toMatchObject({
      prompt: metadata.prompt,
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
      scheduler: "default",
      cfgPlus: false,
      loras: [],
      upscaleModel: "",
      enableAudio: false,
      pipeline: null,
      spatialUpscale: null,
      temporalUpscale: null,
    });

    expect(buildRequest(form)).not.toMatchObject({
      loras: expect.anything(),
      enable_audio: expect.anything(),
      pipeline: expect.anything(),
      spatial_upscale: expect.anything(),
      temporal_upscale: expect.anything(),
    });
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

    expect(result).toEqual({ modelName: replacement.name, substitutedModel: true });
    expect(form.model).toBe(replacement.name);
    expect(form.family).toBe(replacement.family);
    expect(form.outputFormat).toBe("png");
    expect(buildRequest(form).model).toBe(replacement.name);
  });
});
