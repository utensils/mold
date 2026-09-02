import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import { newGenerateForm } from "./generateForm";
import {
  identityConditioningValidationError,
  inlineGenerationMediaBytes,
  meshTargetFacesError,
  meshTargetFacesValidationError,
  sourceConditioningValidationError,
  resolutionValidationWarning,
  resolutionValidationError,
} from "./generateValidation";
import type { GenerateForm } from "./generateForm";

function formWithEveryH3MediaField() {
  const form = newGenerateForm();
  form.h3Authoring = {
    firstFrame: {
      filename: "first.png",
      mimeType: "image/png",
      width: 32,
      height: 32,
      data: "QUJD",
    },
    lastFrame: {
      filename: "last.png",
      mimeType: "image/png",
      width: 32,
      height: 32,
      data: "REVG",
    },
    references: [
      {
        reference: {
          kind: "image",
          media: { authority: "inline", data: "R0hJ" },
          provenance: { name: "reference.png" },
          mime_type: "image/png",
          width: 32,
          height: 32,
        },
      },
    ],
  };
  return form;
}

describe("inlineGenerationMediaBytes — MiniMax H3 active partition", () => {
  it("counts only FL2VA boundaries and preserves replacement exclusion", () => {
    const form = formWithEveryH3MediaField();
    form.model = "minimax-h3-fl2va:official-bf16";

    expect(inlineGenerationMediaBytes(form)).toBe(6);
    expect(inlineGenerationMediaBytes(form, "h3FirstFrame")).toBe(3);
    expect(inlineGenerationMediaBytes(form, "h3References")).toBe(6);
  });

  it("counts only ordered Ref2VA media and preserves replacement exclusion", () => {
    const form = formWithEveryH3MediaField();
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";

    expect(inlineGenerationMediaBytes(form)).toBe(3);
    expect(inlineGenerationMediaBytes(form, "h3References")).toBe(0);
    expect(inlineGenerationMediaBytes(form, "h3FirstFrame")).toBe(3);
  });

  it.each(["flux:replacement", "minimax-h3-ref2va:future-layout"])(
    "ignores parked H3 media when %s has no released H3 wire task",
    (model) => {
      const form = formWithEveryH3MediaField();
      form.model = model;

      expect(inlineGenerationMediaBytes(form)).toBe(0);
    },
  );
});

/**
 * #783: a continuation supplies its own first frames from the tail of the clip
 * it continues. The server counts that as carrying source
 * (`mold_core::validation::request_carries_source_frames`, called from
 * `enforce_source_image_capability`), so the client's own contract gate has to
 * agree — otherwise the Continue-a-video control the same branch just made
 * visible for a Wan I2V checkpoint offers work that submit refuses.
 *
 * This drives the real entry point desktop's `GenerateView`, `SourceImageWell`,
 * and both iPhone surfaces (`MobileApp`, `MobileSourceControls`) call.
 */
describe("sourceConditioningValidationError — a continuation carries source frames", () => {
  function wanContinuation(model = "wan22-i2v-a14b:q8") {
    const form = newGenerateForm();
    form.family = "wan";
    form.model = model;
    form.sourceImageCapability = "required";
    form.frames = 49;
    form.fps = 16;
    form.extendVideo = { filename: "clip.mp4", base64: "Q0xJUA==" };
    return form;
  }

  it("admits a Wan I2V continuation that has no attached image", () => {
    expect(sourceConditioningValidationError(wanContinuation())).toBeNull();
  });

  it("still blocks a plain Wan I2V render with no image", () => {
    const form = wanContinuation();
    form.extendVideo = null;
    expect(sourceConditioningValidationError(form)).toMatch(/image-to-video only/);
  });

  it("refuses a continuation aimed at a text-to-video-only checkpoint", () => {
    const form = wanContinuation("wan22-t2v-a14b:q8");
    form.sourceImageCapability = "unsupported";
    expect(sourceConditioningValidationError(form)).toMatch(
      /text-to-video only and cannot continue/,
    );
  });

  it("ignores a parked source image for a text-to-video-only checkpoint", () => {
    const form = wanContinuation("wan22-t2v-a14b:q8");
    form.sourceImageCapability = "unsupported";
    form.extendVideo = null;
    form.sourceImage = "UEFSS0VE";

    expect(
      sourceConditioningValidationError(form, {
        ignoreUnsupportedStagedSource: true,
      }),
    ).toBeNull();
    expect(sourceConditioningValidationError(form)).toMatch(/text-to-video only/);
  });

  it("ignores a staged clip on a family with no continuation path", () => {
    // Both request builders drop `extend_video` outside an extend-capable
    // family, so a stale staged clip must not smuggle the requirement away.
    const form = wanContinuation();
    form.family = "ltx-video";
    expect(sourceConditioningValidationError(form)).toMatch(/image-to-video only/);
  });
});

/**
 * A canvasless recipe (a 3-D mesh) reconstructs its source image; it is not
 * an image-to-VIDEO checkpoint, and a fresh Hunyuan3D form's first blocker
 * must not say it is. The snapshot is the authority; the family name is the
 * fallback for a form restored before the profile landed.
 */
describe("sourceConditioningValidationError — a canvasless mesh recipe", () => {
  function meshForm() {
    const form = newGenerateForm();
    form.family = "hunyuan3d";
    form.model = "hunyuan3d-mini-turbo:fp16";
    form.sourceImageCapability = "required";
    form.recipeCapabilities = {
      outputFormats: ["glb"],
      defaultOutputFormat: "glb",
      promptMode: "ignored",
      supportsStrength: false,
      canvasless: true,
      mesh: null,
    };
    return form;
  }

  it("names the source image the model reconstructs, never a first frame", () => {
    const error = sourceConditioningValidationError(meshForm());
    expect(error).toBe("This model reconstructs a source image; attach one to generate.");
  });

  it("falls back to the family when no snapshot has landed yet", () => {
    const form = meshForm();
    form.recipeCapabilities = null;
    expect(sourceConditioningValidationError(form)).toMatch(/reconstructs a source image/);
  });

  it("clears once a source image is attached", () => {
    const form = meshForm();
    form.sourceImage = "c291cmNl";
    expect(sourceConditioningValidationError(form)).toBeNull();
  });

  it("leaves the image-to-video wording to video checkpoints", () => {
    const form = newGenerateForm();
    form.family = "wan";
    form.model = "wan22-i2v-a14b:q8";
    form.sourceImageCapability = "required";
    expect(sourceConditioningValidationError(form)).toMatch(/image-to-video only/);
  });
});

/**
 * Target faces is optional, but a typed budget outside the recipe's advertised
 * bounds is a 422 at admission. Name it inline, in the server's terms, the way
 * `integerControlError` names Steps — never snap it silently.
 */
describe("meshTargetFacesError", () => {
  // The advertised block exactly as a host sends it: faces 100–2,000,000.
  const mesh = hunyuan3dRecipe().capabilities.mesh!;

  it("accepts blank (the raw surface) and any whole number within the bounds", () => {
    expect(meshTargetFacesError(null, mesh)).toBeNull();
    expect(meshTargetFacesError(100, mesh)).toBeNull();
    expect(meshTargetFacesError(50_000, mesh)).toBeNull();
    expect(meshTargetFacesError(2_000_000, mesh)).toBeNull();
  });

  it("names the advertised bounds for a budget outside them", () => {
    expect(meshTargetFacesError(10, mesh)).toBe(
      "Target faces must be a whole number from 100 to 2000000.",
    );
    expect(meshTargetFacesError(9_000_000, mesh)).toMatch(/from 100 to 2000000/);
    expect(meshTargetFacesError(150.5, mesh)).toMatch(/whole number/);
  });

  it("says nothing on a recipe with no mesh block", () => {
    expect(meshTargetFacesError(10, null)).toBeNull();
  });

  it("reads the live form's snapshot and controls", () => {
    const form = newGenerateForm();
    form.mesh.targetFaces = 10;
    expect(meshTargetFacesValidationError(form)).toBeNull();
    form.recipeCapabilities = {
      outputFormats: ["glb"],
      defaultOutputFormat: "glb",
      promptMode: "ignored",
      supportsStrength: false,
      canvasless: true,
      mesh,
    };
    expect(meshTargetFacesValidationError(form)).toMatch(/from 100 to 2000000/);
    form.mesh.targetFaces = null;
    expect(meshTargetFacesValidationError(form)).toBeNull();
  });
});

describe("resolutionValidationWarning", () => {
  it("surfaces the warn-policy off-bucket advisory without ever blocking", () => {
    const recipe = {
      id: "default",
      label: "Default",
      request_selector: {},
      defaults: { width: 1280, height: 704, steps: 20, guidance: 5 },
      resolution: {
        domain: "buckets",
        alignment: 16,
        min_width: 64,
        min_height: 64,
        max_pixels: 2_000_000,
        off_bucket: "warn",
        aspect_groups: [
          {
            id: "16:9",
            label: "16:9",
            presets: [{ id: "1280x704", width: 1280, height: 704, tier: "recommended" }],
          },
        ],
      },
      steps: { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
      guidance: { default: 5, min: 0, max: 20, step: 0.1, mode: "adjustable" },
      capabilities: {
        guidance: {
          adjustable: true,
          supports_negative_prompt: true,
          fixed_scale: null,
        },
        negative_prompt: { mode: "adjustable", required: false },
        supports_lora: false,
        supports_controlnet: false,
        supports_identity: false,
        supports_sequence: false,
        supports_extend: false,
        supports_audio: false,
        source_video: { mode: "hidden", required: false },
        mask: { mode: "hidden", required: false },
        keyframes: { mode: "hidden", required: false },
        audio: { mode: "hidden", required: false },
        lora: { mode: "hidden", max_count: 0 },
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
      provenance: [],
    };
    const model = {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      generation_profile: {
        schema_version: 1,
        profile_id: "wan.v1",
        profile_hash: "hash",
        default_recipe_id: "default",
        recipes: [recipe],
      },
    } as unknown as Parameters<typeof resolutionValidationWarning>[2];

    // The bucket size is silent; an aligned off-bucket size is admitted
    // (no blocking error) and earns the advisory instead.
    expect(resolutionValidationWarning(1280, 704, model)).toBeNull();
    expect(resolutionValidationError(1024, 1024, model)).toBeNull();
    expect(resolutionValidationWarning(1024, 1024, model)).toContain("results may vary");
  });
});

describe("resolution checks never block a custom size (server is the authority)", () => {
  const rejectModel = () => {
    const recipe = {
      id: "default",
      label: "Default",
      request_selector: {},
      defaults: { width: 1344, height: 768, steps: 20, guidance: 5 },
      resolution: {
        domain: "buckets",
        alignment: 32,
        min_width: 1344,
        min_height: 768,
        max_pixels: 1_032_192,
        off_bucket: "reject",
        aspect_groups: [
          {
            id: "7:4",
            label: "7:4",
            presets: [{ id: "1344x768", width: 1344, height: 768, tier: "recommended" }],
          },
        ],
      },
      steps: { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
      guidance: { default: 5, min: 0, max: 20, step: 0.1, mode: "adjustable" },
      capabilities: {
        guidance: { adjustable: true, supports_negative_prompt: true, fixed_scale: null },
        negative_prompt: { mode: "adjustable", required: false },
        supports_lora: false,
        supports_controlnet: false,
        supports_identity: false,
        supports_sequence: false,
        supports_extend: false,
        supports_audio: false,
        source_video: { mode: "hidden", required: false },
        mask: { mode: "hidden", required: false },
        keyframes: { mode: "hidden", required: false },
        audio: { mode: "hidden", required: false },
        lora: { mode: "hidden", max_count: 0 },
        controlnet: { mode: "hidden", max_count: 0 },
        output: { default_format: "mp4", formats: ["mp4"], audio_requires_mp4: false },
        wan_recipe: {
          mode: "hidden",
          supports_distill_strength: false,
          supports_first_last_frame: false,
        },
        schedulers: [],
      },
      provenance: [],
    };
    return {
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      generation_profile: {
        schema_version: 1,
        profile_id: "h3.v1",
        profile_hash: "hash",
        default_recipe_id: "default",
        recipes: [recipe],
      },
    } as unknown as Parameters<typeof resolutionValidationError>[2];
  };

  it("downgrades an undersized custom size to an advisory", () => {
    expect(resolutionValidationError(1024, 576, rejectModel())).toBeNull();
    const warning = resolutionValidationWarning(1024, 576, rejectModel());
    expect(warning).toContain("1344 × 768");
    expect(warning).toContain("server may reject");
  });

  it("still blocks input that cannot form a request", () => {
    expect(resolutionValidationError(1024.5, 576, rejectModel())).toContain("whole numbers");
  });

  it("downgrades the legacy no-recipe limits to advisories too", () => {
    // No generation_profile at all: the family constants used to hard-block.
    const legacy = { name: "flux-dev:q8", family: "flux" } as Parameters<
      typeof resolutionValidationError
    >[2];
    expect(resolutionValidationError(48, 48, legacy)).toBeNull();
    expect(resolutionValidationWarning(48, 48, legacy)).toContain("server may reject");
    expect(resolutionValidationError(1000, 576, legacy)).toBeNull();
    expect(resolutionValidationWarning(1000, 576, legacy)).toContain("multiples of");
    expect(resolutionValidationError(4096, 4096, legacy)).toBeNull();
    expect(resolutionValidationWarning(4096, 4096, legacy)).toBeTruthy();
    // A well-formed on-grid size stays silent on both channels.
    expect(resolutionValidationError(1024, 576, legacy)).toBeNull();
    expect(resolutionValidationWarning(1024, 576, legacy)).toBeNull();
  });
});

// A PNG whose header declares `width × height`. Only the 24-byte header is
// read by every identity pre-check, so the payload never has to be real.
function pngBase64(width: number, height: number): string {
  const bytes = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const u32 = (value: number) => [
    (value >>> 24) & 0xff,
    (value >>> 16) & 0xff,
    (value >>> 8) & 0xff,
    value & 0xff,
  ];
  bytes.push(...u32(13), 0x49, 0x48, 0x44, 0x52, ...u32(width), ...u32(height), 8, 6, 0, 0, 0);
  return btoa(String.fromCharCode(...bytes));
}

describe("identityConditioningValidationError", () => {
  function identityForm(): GenerateForm {
    const form = newGenerateForm();
    form.model = "flux-dev:q8";
    form.family = "flux";
    form.steps = 20;
    form.identitySupported = true;
    form.identityImage = { filename: "face.png", base64: pngBase64(768, 768) };
    return form;
  }

  it("is silent for a form that does not mention identity at all", () => {
    expect(identityConditioningValidationError(newGenerateForm())).toBeNull();
  });

  it("accepts a qualified checkpoint carrying only a photo", () => {
    expect(identityConditioningValidationError(identityForm())).toBeNull();
  });

  it("refuses a knob with no photo", () => {
    const form = identityForm();
    form.identityImage = null;
    form.identityWeight = 1.5;
    expect(identityConditioningValidationError(form)).toContain("Attach an identity photo");
  });

  it("parks — never refuses — on a checkpoint that is not identity-qualified", () => {
    // The partition never reaches the wire on such a checkpoint, so there is
    // nothing for Generate to block on: the photo stays staged and the well
    // is simply not rendered, exactly as staged LTX-2 media behaves.
    const form = identityForm();
    form.identitySupported = false;
    expect(identityConditioningValidationError(form)).toBeNull();
    // An unread capability is the same parked state, never a refusal.
    form.identitySupported = null;
    expect(identityConditioningValidationError(form)).toBeNull();
    // Not even a combination the qualified path refuses outright.
    form.loras = [{ path: "style.safetensors", name: "style", scale: 1, trainedWords: [] }];
    form.identityWeight = 99;
    expect(identityConditioningValidationError(form)).toBeNull();
  });

  it("refuses the combination with a LoRA", () => {
    const form = identityForm();
    form.loras = [{ path: "style.safetensors", name: "style", scale: 1, trainedWords: [] }];
    expect(identityConditioningValidationError(form)).toContain("LoRA");
  });

  it("refuses the combination with an img2img source image", () => {
    const form = identityForm();
    form.family = "sd15";
    form.model = "sd15:fp16";
    form.sourceImage = "c291cmNl";
    expect(identityConditioningValidationError(form)).toContain("source image");
  });

  it("ignores a parked source image the selected checkpoint would drop", () => {
    // Source media is parked across model switches so switching back restores
    // the draft. A checkpoint that advertises no source image drops it in
    // `buildRequest`, so it must not refuse the identity partition either.
    const form = identityForm();
    form.sourceImageCapability = "unsupported";
    form.sourceImage = "c291cmNl";
    expect(identityConditioningValidationError(form)).toBeNull();
  });

  it("refuses an out-of-range strength", () => {
    const form = identityForm();
    form.identityWeight = 3.5;
    expect(identityConditioningValidationError(form)).toContain("0 to 3");
    form.identityWeight = -0.1;
    expect(identityConditioningValidationError(form)).toContain("0 to 3");
    form.identityWeight = 3;
    expect(identityConditioningValidationError(form)).toBeNull();
  });

  it("refuses a start step at or beyond the steps this print renders", () => {
    const form = identityForm();
    form.steps = 8;
    form.identityStartStep = 8;
    expect(identityConditioningValidationError(form)).toContain("0 to 7");
    form.identityStartStep = 7;
    expect(identityConditioningValidationError(form)).toBeNull();
  });

  it("refuses bytes that are not a PNG or JPEG, and an oversized photo", () => {
    const form = identityForm();
    form.identityImage = { filename: "face.gif", base64: "R0lGODlhAQABAAAAACw=" };
    expect(identityConditioningValidationError(form)).toContain("PNG or JPEG");
    form.identityImage = { filename: "huge.png", base64: pngBase64(9000, 4000) };
    expect(identityConditioningValidationError(form)).toContain("8192");
  });

  it("counts the identity photo against the inline media budget", () => {
    const form = identityForm();
    const bytes = inlineGenerationMediaBytes(form);
    expect(bytes).toBeGreaterThan(0);
    expect(inlineGenerationMediaBytes(form, "identityImage")).toBe(0);
  });
});
