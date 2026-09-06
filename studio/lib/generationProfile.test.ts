import { describe, expect, it } from "vitest";
import {
  advertisedGenerationProfile,
  closestProfileAspect,
  controlNote,
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
  generationRecipeSelectionError,
  profileAspectOptions,
  recipeIsCanvasless,
  resolutionProfileError,
  resolutionProfileFinding,
  resolutionProfileWarning,
  type GenerationProfileModel,
  type GenerationProfileSet,
  type GenerationRecipeProfile,
} from "./generationProfile";
import { hunyuan3dRecipe, sdxlRecipe } from "./generationProfile.testFixtures";

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
      prompt: { mode: "required" },
      supports_strength: false,
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

describe("output format gate", () => {
  it("accepts the canvasless GLB contract a 3-D family advertises", () => {
    // `OUTPUT_FORMATS` is a runtime GATE, not a type: a missing format makes
    // `isOutputCapabilities` reject the whole profile, and the model then
    // renders with the legacy raster fallback — canvas controls and PNG
    // output — instead of the contract the server actually sent. Nothing
    // fails loudly, which is why this is asserted rather than reviewed.
    const model = profileModel();
    const set = model.generation_profile as GenerationProfileSet;
    for (const recipe of set.recipes) {
      recipe.capabilities.output = {
        default_format: "glb",
        formats: ["glb"],
        audio_requires_mp4: false,
      };
    }
    expect(advertisedGenerationProfile(model)).not.toBeNull();
  });

  it("still rejects a format the server could not have produced", () => {
    const model = profileModel();
    const set = model.generation_profile as GenerationProfileSet;
    for (const recipe of set.recipes) {
      recipe.capabilities.output = {
        default_format: "tiff",
        formats: ["tiff"],
        audio_requires_mp4: false,
      } as unknown as GenerationRecipeProfile["capabilities"]["output"];
    }
    expect(advertisedGenerationProfile(model)).toBeNull();
  });
});

describe("controlNote", () => {
  it("returns the server's own sentence for a fixed control", () => {
    expect(
      controlNote({
        mode: "fixed",
        note: "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.",
      }),
    ).toBe(
      "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.",
    );
  });

  it("invents nothing when the control is fixed but silent", () => {
    // Exactly what an older server's response deserializes to.
    expect(controlNote({ mode: "fixed" })).toBeNull();
    expect(controlNote({ mode: "fixed", note: null })).toBeNull();
    expect(controlNote({ mode: "fixed", note: "   " })).toBeNull();
  });

  it("stays silent for adjustable, hidden, and absent controls", () => {
    expect(controlNote({ mode: "adjustable", note: "unreachable" })).toBeNull();
    expect(controlNote({ mode: "hidden", note: "unreachable" })).toBeNull();
    expect(controlNote(null)).toBeNull();
    expect(controlNote(undefined)).toBeNull();
  });
});

describe("prompt, strength, and mesh contract", () => {
  function modelWith(recipe: GenerationRecipeProfile): GenerationProfileModel {
    return {
      name: "fixture",
      family: "fixture",
      ...(recipe.defaults.width
        ? { default_width: recipe.defaults.width }
        : {}),
      ...(recipe.defaults.height
        ? { default_height: recipe.defaults.height }
        : {}),
      generation_profile: {
        schema_version: 1,
        profile_id: "fixture.v1",
        profile_hash: "hash",
        default_recipe_id: recipe.id,
        recipes: [recipe],
      },
    };
  }
  type LooseCaps = Record<string, unknown>;
  function caps(recipe: GenerationRecipeProfile): LooseCaps {
    return recipe.capabilities as unknown as LooseCaps;
  }

  it("accepts the real hunyuan3d recipe the server emits", () => {
    const model = modelWith(hunyuan3dRecipe());
    const recipe = effectiveGenerationRecipe(model);
    expect(recipe?.legacy_adapter).toBeUndefined();
    expect(recipe?.capabilities.prompt.mode).toBe("ignored");
    expect(recipe?.capabilities.supports_strength).toBe(false);
    expect(recipe?.capabilities.mesh?.octree_default).toBe(256);
  });

  it("accepts an older host's recipe that carries none of the new fields", () => {
    // Serde defaults `prompt` to Required, `supports_strength` to false, and
    // `mesh` to absent; the client must accept the wire shape that produced
    // those defaults instead of rejecting the whole profile.
    const recipe = sdxlRecipe();
    delete caps(recipe).prompt;
    delete caps(recipe).supports_strength;
    delete caps(recipe).mesh;
    const resolved = effectiveGenerationRecipe(modelWith(recipe));
    expect(resolved).not.toBeNull();
    expect(resolved?.legacy_adapter).toBeUndefined();
  });

  it("accepts a null mesh block and an absent prompt reason", () => {
    const recipe = sdxlRecipe();
    caps(recipe).mesh = null;
    caps(recipe).prompt = { mode: "optional", reason: null };
    expect(advertisedGenerationProfile(modelWith(recipe))).not.toBeNull();
  });

  it("rejects a prompt mode this client does not understand", () => {
    const recipe = sdxlRecipe();
    caps(recipe).prompt = { mode: "maybe" };
    expect(advertisedGenerationProfile(modelWith(recipe))).toBeNull();
    caps(recipe).prompt = "required";
    expect(advertisedGenerationProfile(modelWith(recipe))).toBeNull();
  });

  it("rejects a supports_strength that is not a boolean", () => {
    const recipe = sdxlRecipe();
    caps(recipe).supports_strength = "yes";
    expect(advertisedGenerationProfile(modelWith(recipe))).toBeNull();
  });

  it("rejects malformed mesh controls instead of trusting the outer version", () => {
    const octree = (value: unknown) => {
      const recipe = hunyuan3dRecipe();
      (caps(recipe).mesh as LooseCaps).octree_resolutions = value;
      return advertisedGenerationProfile(modelWith(recipe));
    };
    expect(octree("256")).toBeNull();
    expect(octree([])).toBeNull();
    expect(octree([128, 0])).toBeNull();
    expect(octree([128, 1.5])).toBeNull();
    expect(octree([128, -256])).toBeNull();

    const defaultOff = hunyuan3dRecipe();
    (caps(defaultOff).mesh as LooseCaps).octree_default = 300;
    expect(advertisedGenerationProfile(modelWith(defaultOff))).toBeNull();

    const badThreshold = hunyuan3dRecipe();
    (caps(badThreshold).mesh as LooseCaps).threshold = { default: 0.6 };
    expect(advertisedGenerationProfile(modelWith(badThreshold))).toBeNull();
    (caps(badThreshold).mesh as LooseCaps).threshold = {
      default: 2,
      min: 0,
      max: 1,
      step: 0.01,
      mode: "adjustable",
    };
    expect(advertisedGenerationProfile(modelWith(badThreshold))).toBeNull();

    const faces = hunyuan3dRecipe();
    (caps(faces).mesh as LooseCaps).target_faces_min = 5_000_000;
    expect(advertisedGenerationProfile(modelWith(faces))).toBeNull();

    const texture = hunyuan3dRecipe();
    (caps(texture).mesh as LooseCaps).texture = { mode: "sometimes" };
    expect(advertisedGenerationProfile(modelWith(texture))).toBeNull();

    const notRecord = hunyuan3dRecipe();
    caps(notRecord).mesh = [];
    expect(advertisedGenerationProfile(modelWith(notRecord))).toBeNull();
  });

  it("accepts the complete mesh workflow contract and rejects malformed additions", () => {
    const complete = hunyuan3dRecipe();
    Object.assign(caps(complete).mesh as LooseCaps, {
      named_views: {
        mode: "hidden",
        roles: ["front", "left", "back", "right"],
        min_count: 0,
        max_count: 0,
        reason: "Named camera views require a 2mv checkpoint",
      },
      mesh_input: {
        mode: "hidden",
        formats: ["glb", "obj"],
        max_count: 0,
        max_bytes: 268_435_456,
        up_axes: ["y", "z"],
        meters_per_unit_min: 0.000_001,
        meters_per_unit_max: 1_000_000,
      },
      texture_resolutions: [1024, 2048, 4096],
      texture_default_resolution: 2048,
      texture_view_count: {
        default: 6,
        min: 6,
        max: 6,
        step: 1,
        mode: "fixed",
      },
      matting: { mode: "hidden", required: false },
      delight: { mode: "hidden", required: false },
      workflow_modes: ["image_to_mesh"],
    });
    expect(advertisedGenerationProfile(modelWith(complete))).not.toBeNull();

    const duplicateRole = structuredClone(complete);
    (caps(duplicateRole).mesh as LooseCaps).named_views = {
      mode: "adjustable",
      roles: ["front", "front"],
      min_count: 1,
      max_count: 2,
    };
    expect(advertisedGenerationProfile(modelWith(duplicateRole))).toBeNull();

    const badTextureDefault = structuredClone(complete);
    (caps(badTextureDefault).mesh as LooseCaps).texture_default_resolution =
      3072;
    expect(
      advertisedGenerationProfile(modelWith(badTextureDefault)),
    ).toBeNull();

    const unknownWorkflow = structuredClone(complete);
    (caps(unknownWorkflow).mesh as LooseCaps).workflow_modes = ["magic_mesh"];
    expect(advertisedGenerationProfile(modelWith(unknownWorkflow))).toBeNull();
  });

  it("fills the legacy adapter from the pre-profile family rules", () => {
    // A host that predates the profile still has the old client rules
    // applied to it, so behaviour there is unchanged: LTX-2 with visual
    // conditioning was optional, wan never read strength, flux did.
    const legacy = (family: string, name: string) =>
      effectiveGenerationRecipe({
        name,
        family,
        default_width: 1024,
        default_height: 1024,
      });
    expect(legacy("ltx2", "ltx2-19b:q8")?.capabilities.prompt.mode).toBe(
      "optional",
    );
    expect(legacy("flux", "flux-dev:q8")?.capabilities.prompt.mode).toBe(
      "required",
    );
    expect(legacy("flux", "flux-dev:q8")?.capabilities.supports_strength).toBe(
      true,
    );
    expect(
      legacy("wan", "wan22-i2v-a14b:q8")?.capabilities.supports_strength,
    ).toBe(false);
    expect(
      legacy("qwen-image-edit", "qwen-image-edit:q8")?.capabilities
        .supports_strength,
    ).toBe(false);
    expect(legacy("flux", "flux-dev:q8")?.capabilities.mesh).toBeUndefined();
    expect(legacy("flux", "flux-dev:q8")?.legacy_adapter).toBe(true);
  });
});

describe("recipeIsCanvasless", () => {
  it("is true only for a recipe with no resolution domain", () => {
    expect(recipeIsCanvasless(hunyuan3dRecipe())).toBe(true);
    expect(recipeIsCanvasless(sdxlRecipe())).toBe(false);
    expect(recipeIsCanvasless(null)).toBe(false);
    expect(recipeIsCanvasless(undefined)).toBe(false);
  });
});
