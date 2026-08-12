import { describe, expect, it } from "vitest";
import {
  advertisedGenerationProfile,
  effectiveGenerationRecipe,
  generationRecipeSelectionError,
  profileAspectOptions,
  resolutionProfileError,
  type GenerationProfileModel,
  type GenerationProfileSet,
  type GenerationRecipeProfile,
} from "./generationProfile";

function profileModel(): GenerationProfileModel {
  const recipe = (
    id: string,
    pipeline: Exclude<
      GenerationRecipeProfile["request_selector"]["pipeline"],
      undefined
    >,
    alignment: number,
  ): GenerationRecipeProfile => ({
    id,
    label: id,
    request_selector: { pipeline },
    defaults: { width: 1024, height: 576, steps: 9, guidance: 0 },
    resolution: {
      domain: "buckets" as const,
      alignment,
      min_width: 64,
      min_height: 64,
      max_pixels: 1_800_000,
      max_axis_pixels: 2048,
      aspect_groups: [
        {
          id: "16:9",
          label: "16:9",
          presets: [
            { id: "1024x576", width: 1024, height: 576, tier: "recommended" },
          ],
        },
      ],
    },
    steps: {
      default: 9,
      min: 1,
      max: 100,
      step: 1,
      mode: "adjustable" as const,
    },
    guidance: {
      default: 0,
      min: 0,
      max: 100,
      step: 0.1,
      mode: "fixed" as const,
    },
    capabilities: {
      guidance: {
        adjustable: false,
        supports_negative_prompt: false,
        fixed_scale: 0,
      },
      negative_prompt: { mode: "hidden" as const, required: false },
      supports_lora: false,
      supports_controlnet: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: { mode: "hidden" as const, required: false },
      mask: { mode: "hidden" as const, required: false },
      keyframes: { mode: "hidden" as const, required: false },
      audio: { mode: "hidden" as const, required: false },
      lora: { mode: "hidden" as const, max_count: 0 },
      controlnet: { mode: "hidden" as const, max_count: 0 },
      output: {
        default_format: "mp4" as const,
        formats: ["mp4" as const],
        audio_requires_mp4: false,
      },
      wan_recipe: {
        mode: "hidden" as const,
        supports_distill_strength: false,
        supports_first_last_frame: false,
      },
      schedulers: [],
    },
    provenance: [],
  });
  return {
    name: "fixture",
    family: "fixture",
    default_width: 1024,
    default_height: 576,
    generation_profile: {
      schema_version: 1,
      profile_id: "fixture.v1",
      profile_hash: "hash",
      default_recipe_id: "one-stage",
      recipes: [
        recipe("one-stage", "one-stage", 32),
        recipe("two-stage", "two-stage", 64),
      ],
    },
  };
}

describe("generation profile contract", () => {
  it("resolves the complete recipe selected by pipeline", () => {
    const model = profileModel();
    expect(
      effectiveGenerationRecipe(model, "one-stage")?.resolution.alignment,
    ).toBe(32);
    expect(
      effectiveGenerationRecipe(model, "two-stage")?.resolution.alignment,
    ).toBe(64);
  });

  it("rejects unknown schemas instead of guessing their meaning", () => {
    const model = profileModel();
    model.generation_profile = {
      ...(model.generation_profile as GenerationProfileSet),
      schema_version: 2,
    } as unknown as GenerationProfileSet;
    expect(advertisedGenerationProfile(model)).toBeNull();
  });

  it("rejects malformed nested controls instead of trusting the outer version", () => {
    const model = profileModel();
    const recipe = model.generation_profile?.recipes[0];
    if (!recipe) throw new Error("missing fixture recipe");
    recipe.resolution.alignment = 0;
    expect(advertisedGenerationProfile(model)).toBeNull();
  });

  it("fails closed for an explicit pipeline absent from a valid v1 profile", () => {
    const model = profileModel();
    expect(effectiveGenerationRecipe(model, "future-pipeline")).toBeNull();
    expect(generationRecipeSelectionError(model, "future-pipeline")).toContain(
      "not supported",
    );
  });

  it("enforces bucket membership after grid and budget validation", () => {
    const resolution = effectiveGenerationRecipe(
      profileModel(),
      "one-stage",
    )?.resolution;
    expect(resolutionProfileError(1024, 576, resolution)).toBeNull();
    expect(resolutionProfileError(1008, 576, resolution)).toContain(
      "multiples of 32",
    );
    expect(resolutionProfileError(1024, 1024, resolution)).toContain(
      "supported resolution buckets",
    );
  });

  it("renders exactly the authored aspect groups", () => {
    expect(profileAspectOptions(profileModel(), "one-stage")).toEqual([
      { id: "16:9", label: "16:9", ratio: 16 / 9 },
    ]);
  });
});
