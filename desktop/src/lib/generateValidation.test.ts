import { describe, expect, it } from "vitest";
import { newGenerateForm } from "./generateForm";
import {
  inlineGenerationMediaBytes,
  sourceConditioningValidationError,
  resolutionValidationWarning,
  resolutionValidationError,
} from "./generateValidation";

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

  it("ignores a staged clip on a family with no continuation path", () => {
    // Both request builders drop `extend_video` outside an extend-capable
    // family, so a stale staged clip must not smuggle the requirement away.
    const form = wanContinuation();
    form.family = "ltx-video";
    expect(sourceConditioningValidationError(form)).toMatch(/image-to-video only/);
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
            presets: [
              { id: "1280x704", width: 1280, height: 704, tier: "recommended" },
            ],
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
    expect(resolutionValidationWarning(1024, 1024, model)).toContain(
      "results may vary",
    );
  });
});
