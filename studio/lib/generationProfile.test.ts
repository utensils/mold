import { describe, expect, it } from "vitest";
import {
  advertisedGenerationProfile,
  closestProfileAspect,
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
  generationRecipeSelectionError,
  profileAspectOptions,
  resolutionProfileError,
  resolutionProfileFinding,
  resolutionProfileWarning,
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
      supports_identity: false,
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

  it("softens bucket membership to an advisory for a warn-policy recipe", () => {
    const strict = effectiveGenerationRecipe(
      profileModel(),
      "one-stage",
    )?.resolution;
    // Absent policy (older servers) keeps the fail-closed blocker and no
    // advisory — the server would refuse the request.
    expect(resolutionProfileWarning(1024, 1024, strict)).toBeNull();

    const advisory = { ...strict!, off_bucket: "warn" as const };
    expect(resolutionProfileError(1024, 1024, advisory)).toBeNull();
    expect(resolutionProfileWarning(1024, 1024, advisory)).toContain(
      "results may vary",
    );
    // An exact bucket stays silent, and a size the recipe refuses outright
    // (off-grid) never earns the softer message.
    expect(resolutionProfileWarning(1024, 576, advisory)).toBeNull();
    expect(resolutionProfileWarning(1008, 576, advisory)).toBeNull();
  });

  it("renders exactly the authored aspect groups", () => {
    expect(profileAspectOptions(profileModel(), "one-stage")).toEqual([
      { id: "16:9", label: "16:9", ratio: 16 / 9 },
    ]);
  });

  it("snaps form values onto a recipe's fixed controls", () => {
    // The fixture recipe fixes guidance at 0 and keeps steps adjustable —
    // a stale form value on a fixed control is never user authority (the
    // control is disabled), so it snaps instead of stranding Generate
    // behind an error the user cannot correct. Desktop and web share this.
    // Its single reject-policy bucket is fixed authority too, exactly like
    // H3's reviewed compact envelope.
    const recipe = effectiveGenerationRecipe(profileModel(), "one-stage");
    expect(fixedRecipeControlOverrides(recipe)).toEqual({
      guidance: 0,
      width: 1024,
      height: 576,
    });
    expect(fixedRecipeControlOverrides(null)).toEqual({});
  });

  it("treats a single reject-policy bucket as fixed resolution authority", () => {
    const base = effectiveGenerationRecipe(profileModel(), "one-stage")!;
    const single = (
      overrides: Partial<GenerationRecipeProfile["resolution"]>,
    ): GenerationRecipeProfile => ({
      ...base,
      resolution: { ...base.resolution, ...overrides },
    });

    // An absent policy means Reject — the fail-closed reading every other
    // client path already takes — so it snaps just like an explicit one.
    expect(
      fixedRecipeControlOverrides(single({ off_bucket: "reject" })),
    ).toMatchObject({
      width: 1024,
      height: 576,
    });
    expect(fixedRecipeControlOverrides(single({}))).toMatchObject({
      width: 1024,
      height: 576,
    });

    // Wan admits an off-bucket size with an advisory, so its buckets are
    // recommendations rather than the only runnable canvas.
    const warn = fixedRecipeControlOverrides(single({ off_bucket: "warn" }));
    expect(warn.width).toBeUndefined();
    expect(warn.height).toBeUndefined();

    // More than one advertised bucket is a choice, not an envelope.
    const multiple = fixedRecipeControlOverrides(
      single({
        aspect_groups: [
          ...base.resolution.aspect_groups,
          {
            id: "1:1",
            label: "1:1",
            presets: [
              { id: "768x768", width: 768, height: 768, tier: "recommended" },
            ],
          },
        ],
      }),
    );
    expect(multiple.width).toBeUndefined();

    // Wuerstchen advertises one preset on a dynamic canvas — a preset is a
    // suggestion there, and any aligned size is admitted.
    const dynamic = fixedRecipeControlOverrides(single({ domain: "dynamic" }));
    expect(dynamic.width).toBeUndefined();
  });

  it("snaps a fixed temporal frame count", () => {
    const base = effectiveGenerationRecipe(profileModel(), "one-stage")!;
    const adjustable: GenerationRecipeProfile = {
      ...base,
      temporal: {
        frames: {
          default: 124,
          min: 124,
          max: 345,
          step: 17,
          mode: "adjustable",
        },
        frame_offset: 5,
        fps: { mode: "fixed", value: 24 },
      },
    };
    expect(fixedRecipeControlOverrides(adjustable).frames).toBeUndefined();

    const fixed: GenerationRecipeProfile = {
      ...adjustable,
      temporal: {
        ...adjustable.temporal!,
        frames: { ...adjustable.temporal!.frames, max: 124, mode: "fixed" },
      },
    };
    expect(fixedRecipeControlOverrides(fixed).frames).toBe(124);
  });
});

describe("resolutionProfileFinding — the client never blocks a size", () => {
  const resolution = () =>
    effectiveGenerationRecipe(profileModel(), "one-stage")!.resolution;

  it("downgrades an undersized custom size to an advisory", () => {
    const strict = { ...resolution(), min_width: 1344, min_height: 768 };
    const finding = resolutionProfileFinding(1024, 576, strict);
    expect(finding?.level).toBe("warn");
    expect(finding?.message).toContain("1344 × 768");
    expect(finding?.message).toContain("server may reject");
  });

  it("downgrades off-grid, over-budget, and off-bucket sizes to advisories", () => {
    expect(resolutionProfileFinding(1008, 576, resolution())).toMatchObject({
      level: "warn",
    });
    expect(resolutionProfileFinding(4096, 2048, resolution())).toMatchObject({
      level: "warn",
    });
    expect(resolutionProfileFinding(1024, 1024, resolution())).toMatchObject({
      level: "warn",
    });
  });

  it("still blocks malformed input that cannot form a request", () => {
    expect(resolutionProfileFinding(1024.5, 576, resolution())).toMatchObject({
      level: "block",
    });
    expect(
      resolutionProfileFinding(Number.NaN, 576, resolution()),
    ).toMatchObject({ level: "block" });
  });

  it("keeps the warn-policy bucket advisory and stays silent on a preset", () => {
    expect(resolutionProfileFinding(1024, 576, resolution())).toBeNull();
    const advisory = { ...resolution(), off_bucket: "warn" as const };
    expect(resolutionProfileFinding(1024, 1024, advisory)?.message).toContain(
      "results may vary",
    );
  });

  it("returns null with no recipe — legacy hosts keep their own checks", () => {
    expect(resolutionProfileFinding(1024, 576, null)).toBeNull();
  });
});

describe("closestProfileAspect — custom sizes highlight the nearest shape", () => {
  it("matches an exact preset", () => {
    expect(
      closestProfileAspect(profileModel(), "one-stage", 1024, 576),
    ).toEqual({ id: "16:9", label: "16:9", ratio: 16 / 9, exact: true });
  });

  it("returns the nearest group for a custom size in a buckets domain", () => {
    const closest = closestProfileAspect(
      profileModel(),
      "one-stage",
      1000,
      600,
    );
    expect(closest).toMatchObject({ id: "16:9", exact: false });
  });

  it("has no tolerance cutoff — a square custom size still maps somewhere", () => {
    expect(
      closestProfileAspect(profileModel(), "one-stage", 1024, 1024),
    ).toMatchObject({ id: "16:9", exact: false });
  });

  it("returns null without a recipe or with invalid dimensions", () => {
    expect(closestProfileAspect(null, null, 1024, 576)).toBeNull();
    expect(
      closestProfileAspect(profileModel(), "one-stage", 0, 576),
    ).toBeNull();
  });
});
