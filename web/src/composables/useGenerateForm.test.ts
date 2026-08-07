import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import {
  applyMetadataToForm,
  cloneTemplateForm,
  promptWithStyle,
  sanitizePersistedForm,
  useGenerateForm,
  __testing__,
} from "./useGenerateForm";
import type {
  GenerateFormState,
  ModelInfoExtended,
  OutputMetadata,
} from "../types";

const STORAGE_KEY = "mold.generate.form";

function makeModel(
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name: "flux2-klein:q4",
    family: "flux2",
    size_gb: 6,
    is_loaded: false,
    last_used: null,
    hf_repo: "black-forest-labs/FLUX.2-Klein",
    downloaded: true,
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...overrides,
  };
}

describe("useGenerateForm", () => {
  beforeEach(() => {
    localStorage.clear();
    __testing__.resetForTest();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("hydrates defaults when localStorage is empty", () => {
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.height).toBe(1024);
    expect(form.state.value.steps).toBe(20);
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.imageAttachments).toEqual([]);
    expect(form.state.value.icLoraControl).toBeNull();
    expect(form.state.value.sourceFitPolicy).toEqual({ mode: "pad-repaint" });
  });

  it("serializes a host-provided IC-LoRA control without replacing custom LoRAs", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2.3-22b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.icLoraControl = "motion-track";
    form.state.value.sourceVideoPath = "/guides/trajectory.mp4";
    form.state.value.loras = [{ path: "/loras/style.safetensors", scale: 0.8 }];
    expect(form.toRequest()).toMatchObject({
      pipeline: "ic-lora",
      ic_lora_control: "motion-track",
      source_video_path: "/guides/trajectory.mp4",
      loras: [{ path: "/loras/style.safetensors", scale: 0.8 }],
    });
  });

  it("shares a singleton state across route remounts", () => {
    const first = useGenerateForm();
    first.state.value.prompt = "persistent cat";
    first.state.value.width = 768;
    first.state.value.upscaleModel = "real-esrgan-x4plus:fp16";
    first.state.value.imageAttachments = [
      {
        kind: "upload",
        filename: "source.png",
        base64: "SOURCE_BYTES",
        draftId: "draft-source",
        width: 640,
        height: 480,
        mime: "image/png",
      },
    ];
    first.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK_BYTES",
      draftId: "draft-mask",
      width: 640,
      height: 480,
      mime: "image/png",
    };

    const second = useGenerateForm();

    expect(second.state).toBe(first.state);
    expect(second.state.value.prompt).toBe("persistent cat");
    expect(second.state.value.width).toBe(768);
    expect(second.state.value.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    expect(second.state.value.imageAttachments[0]?.base64).toBe("SOURCE_BYTES");
    expect(second.state.value.maskImage?.base64).toBe("MASK_BYTES");
  });

  it("discards persisted snapshots from pre-current schemas", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        prompt: "a cat",
        model: "flux-dev:q4",
        width: 512,
        height: 768,
        // sourceImage should never be read back from storage even if someone
        // injects it — base64 lives in memory only.
        sourceImage: { kind: "upload", filename: "x.png", base64: "AAAA" },
        imageAttachments: [
          { kind: "upload", filename: "y.png", base64: "BBBB" },
        ],
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value).toMatchObject({ version: 3, prompt: "" });
  });

  it("loads a version 3 snapshot preserving a saved stylePreset", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 3, prompt: "a cat", stylePreset: "cinematic" }),
    );
    const form = useGenerateForm();
    expect(form.state.value.stylePreset).toBe("cinematic");
  });

  it("upgrades a version 3 camera picker value into the visible LoRA stack", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 3,
        model: "ltx-2-19b-distilled:fp8",
        modelFamily: "ltx2",
        cameraControl: "dolly-in",
        loras: [],
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value.cameraControl).toBe("dolly-in");
    expect(form.state.value.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        scale: 1,
        trainedWords: [],
      },
    ]);
  });

  it("clears a legacy picker-only camera value when all LoRA slots are occupied", () => {
    const loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
      trainedWords: [],
    }));
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 3,
        model: "ltx-2-19b-distilled:fp8",
        modelFamily: "ltx2",
        cameraControl: "dolly-in",
        loras,
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value.cameraControl).toBeNull();
    expect(form.state.value.loras).toEqual(loras);
  });

  it("toRequest bakes the shared kit's style template without mutating the prompt", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.stylePreset = "cinematic";
    expect(form.toRequest().prompt).toBe(
      "cinematic film still of a lighthouse in a storm, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    // The textarea content itself is never rewritten by the style row.
    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
  });

  it("toRequest merges the preset's curated negative for families that take one", () => {
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    // User fragments first, preset fragments appended.
    expect(form.toRequest().negative_prompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
    // The visible negative field is untouched — composition happens on the way out.
    expect(form.state.value.negativePrompt).toBe("text");
  });

  it("toRequest never ships a preset negative to a family that rejects one", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.modelFamily = "flux2";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    expect(form.toRequest().negative_prompt).toBeNull();
  });

  it("toRequest sends the bare prompt when no style preset is active", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.stylePreset = null;
    expect(form.toRequest().prompt).toBe("a lighthouse in a storm");
  });

  it("promptWithStyle leaves an empty prompt empty (a template has nothing to wrap)", () => {
    const form = useGenerateForm();
    form.state.value.prompt = "";
    form.state.value.stylePreset = "anime";
    expect(promptWithStyle(form.state.value)).toBe("");
  });

  it("discards a snapshot with a mismatched version to avoid stale schemas", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 99, prompt: "stale" }),
    );
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("swallows malformed JSON without throwing", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("debounces persistence and strips attachment bytes from the written snapshot", async () => {
    const form = useGenerateForm();
    form.state.value.prompt = "a fox";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "x.png", base64: "SECRETBYTES" },
    ];
    await nextTick();

    // Watch fires but persist is debounced by 300ms.
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    vi.advanceTimersByTime(300);

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.prompt).toBe("a fox");
    expect(parsed.imageAttachments[0]).toMatchObject({
      filename: "x.png",
      draftId: expect.any(String),
    });
    expect(parsed.imageAttachments[0].base64).toBeUndefined();
    expect(raw).not.toContain("SECRETBYTES");
  });

  it("hydrates media drafts from the draft store after refresh", async () => {
    const first = useGenerateForm();
    first.state.value.imageAttachments = [
      { kind: "upload", filename: "source.png", base64: "SOURCE_BYTES" },
    ];
    first.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK_BYTES",
    };
    first.state.value.controlImage = {
      kind: "upload",
      filename: "control.png",
      base64: "CONTROL_BYTES",
    };
    await nextTick();
    vi.advanceTimersByTime(300);
    await __testing__.flushDraftWrites();

    __testing__.resetForTest();
    const second = useGenerateForm();
    await __testing__.flushHydration();

    expect(second.state.value.imageAttachments[0]?.base64).toBe("SOURCE_BYTES");
    expect(second.state.value.maskImage?.base64).toBe("MASK_BYTES");
    expect(second.state.value.controlImage?.base64).toBe("CONTROL_BYTES");
    expect(localStorage.getItem(STORAGE_KEY)).not.toContain("_BYTES");
  });

  it("applyModelDefaults copies model defaults and clears video fields for non-video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 25;
    form.state.value.fps = 24;

    form.applyModelDefaults(
      makeModel({
        name: "sdxl:fp16",
        family: "sdxl",
        default_width: 1024,
        default_height: 1024,
        default_steps: 30,
        default_guidance: 7.5,
      }),
    );

    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.frames).toBeNull();
    expect(form.state.value.fps).toBeNull();
  });

  it("applyModelDefaults seeds frame/fps for video families when absent", () => {
    const form = useGenerateForm();
    form.state.value.frames = null;
    form.state.value.fps = null;

    form.applyModelDefaults(
      makeModel({ name: "ltx-video:fp16", family: "ltx-video" }),
    );

    expect(form.state.value.frames).toBe(25);
    expect(form.state.value.fps).toBe(24);
  });

  it("applyModelDefaults takes the model's advertised fps, like steps and guidance", () => {
    const form = useGenerateForm();
    form.state.value.fps = 24;

    form.applyModelDefaults(
      makeModel({
        name: "ltx-video-0.9.6-distilled:bf16",
        family: "ltx-video",
        default_fps: 30,
      }),
    );

    expect(form.state.value.fps).toBe(30);
  });

  it("applyModelDefaults preserves user-chosen frame/fps for video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 49;
    form.state.value.fps = 30;

    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));

    expect(form.state.value.frames).toBe(49);
    expect(form.state.value.fps).toBe(30);
  });

  it("resetSettings restores the selected model's defaults", () => {
    const form = useGenerateForm();
    const model = makeModel({
      name: "sdxl:fp16",
      family: "sdxl",
      default_width: 1024,
      default_height: 1024,
      default_steps: 30,
      default_guidance: 7.5,
    });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      width: 512,
      height: 512,
      steps: 4,
      guidance: 11,
      seedMode: "static",
      seed: 42,
      strength: 0.3,
      negativePrompt: "blurry",
      loras: [{ path: "a.safetensors", scale: 0.8 }],
      upscaleModel: "real-esrgan-x4plus:fp16",
      scheduler: "ddim",
      cfgPlus: true,
      outputFormat: "webp",
      sourceFitPolicy: { mode: "crop-fill" },
      imageAttachments: [
        { kind: "upload", filename: "src.png", base64: "AAAA" },
      ],
      gifPreview: true,
    });

    form.resetSettings(model);

    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.height).toBe(1024);
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.seedMode).toBe("random");
    expect(form.state.value.seed).toBeNull();
    expect(form.state.value.strength).toBe(0.75);
    expect(form.state.value.negativePrompt).toBe("");
    expect(form.state.value.loras).toEqual([]);
    expect(form.state.value.upscaleModel).toBe("");
    expect(form.state.value.scheduler).toBeNull();
    expect(form.state.value.cfgPlus).toBe(false);
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.sourceFitPolicy).toEqual({ mode: "pad-repaint" });
    expect(form.state.value.imageAttachments).toEqual([]);
    expect(form.state.value.gifPreview).toBe(false);
  });

  it("resetSettings preserves the prompt, style, model and batch size", () => {
    const form = useGenerateForm();
    const model = makeModel({ name: "sdxl:fp16", family: "sdxl" });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      prompt: "a lighthouse in a storm",
      stylePreset: "cinematic",
      // Prepared batch work is never silently resized (CLAUDE.md), so the
      // batch stepper survives a settings reset.
      batchSize: 4,
      steps: 3,
    });

    form.resetSettings(model);

    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
    expect(form.state.value.stylePreset).toBe("cinematic");
    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.modelFamily).toBe("sdxl");
    expect(form.state.value.batchSize).toBe(4);
    expect(form.state.value.steps).toBe(20);
  });

  it("resetSettings falls back to plain defaults with no resolved model row", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "mystery:fp16",
      modelFamily: "flux",
      prompt: "a cat",
      steps: 3,
      guidance: 11,
    });

    form.resetSettings(null);

    expect(form.state.value.model).toBe("mystery:fp16");
    expect(form.state.value.modelFamily).toBe("flux");
    expect(form.state.value.prompt).toBe("a cat");
    expect(form.state.value.steps).toBe(20);
    expect(form.state.value.guidance).toBe(3.5);
  });

  it("resetSettings restores video defaults for video families", () => {
    const form = useGenerateForm();
    const model = makeModel({ name: "ltx2:fp8", family: "ltx2" });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      frames: 97,
      fps: 30,
      pipeline: "two-stage",
      spatialUpscale: "x2",
      keyframes: [
        {
          frame: 8,
          image: { kind: "upload", filename: "k.png", base64: "AA" },
        },
      ],
    });

    form.resetSettings(model);

    expect(form.state.value.frames).toBe(25);
    expect(form.state.value.fps).toBe(24);
    expect(form.state.value.pipeline).toBeNull();
    expect(form.state.value.spatialUpscale).toBeNull();
    expect(form.state.value.keyframes).toEqual([]);
    expect(form.state.value.outputFormat).toBe("mp4");
  });

  it("toRequest maps camelCase state to snake_case wire payload", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      prompt: "a cat",
      negativePrompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seedMode: "static",
      seed: 42,
      batchSize: 2,
      outputFormat: "png",
      scheduler: "ddim",
      strength: 0.8,
      frames: null,
      fps: null,
      expand: { enabled: true, variations: 3, familyOverride: null },
      imageAttachments: [
        { kind: "upload", filename: "src.png", base64: "AAAA" },
      ],
    });

    const wire = form.toRequest();
    expect(wire).toMatchObject({
      prompt: "a cat",
      negative_prompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seed: 42,
      batch_size: 2,
      output_format: "png",
      scheduler: "ddim",
      strength: 0.8,
      source_image: "AAAA",
      expand: true,
    });
    expect(wire.edit_images).toBeUndefined();
  });

  it("serializes ordered Qwen edit attachments as edit_images and omits source_image", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "qwen-image-edit:q4",
      modelFamily: "qwen-image-edit",
      batchSize: 4,
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
        { kind: "upload", filename: "ref-a.png", base64: "REF_A" },
        { kind: "gallery", filename: "ref-b.png", base64: "REF_B" },
      ],
    });

    const wire = form.toRequest();
    expect(wire.edit_images).toEqual(["TARGET", "REF_A", "REF_B"]);
    expect(wire.source_image).toBeUndefined();
    expect(wire.batch_size).toBe(1);
    expect(wire.strength).toBeUndefined();
  });

  it("omits stale mask state for Qwen edit requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "qwen-image-edit:q4",
      modelFamily: "qwen-image-edit",
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();

    expect(wire.edit_images).toEqual(["TARGET"]);
    expect(wire.mask_image).toBeUndefined();
    expect(wire.source_image).toBeUndefined();
  });

  it("serializes FLUX.2 Dev attachments as ordered references", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      batchSize: 3,
      imageAttachments: [
        { kind: "upload", filename: "one.png", base64: "REF_ONE" },
        { kind: "upload", filename: "two.png", base64: "REF_TWO" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();
    expect(wire.edit_images).toEqual(["REF_ONE", "REF_TWO"]);
    expect(wire.source_image).toBeUndefined();
    expect(wire.mask_image).toBeUndefined();
    expect(wire.strength).toBeUndefined();
    expect(wire.batch_size).toBe(1);
  });

  it("omits stale LoRA state from FLUX.2 Dev requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      loras: [{ path: "/loras/stale.safetensors", scale: 0.8 }],
    });

    expect(form.toRequest().loras).toBeUndefined();
  });

  it("preserves text-only FLUX.2 Dev batches", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      batchSize: 3,
      imageAttachments: [],
    });

    expect(form.toRequest().batch_size).toBe(3);
  });

  it("serializes an uploaded mask image for non-edit img2img requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [
        { kind: "upload", filename: "source.png", base64: "SOURCE" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();

    expect(wire.source_image).toBe("SOURCE");
    expect(wire.source_image_name).toBe("source.png");
    expect(wire.mask_image).toBe("MASK");
    expect(wire.edit_images).toBeUndefined();
  });

  it("serializes LoRAs in the visible stack order", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.loras = [
      { path: "/loras/second.safetensors", scale: 0.9 },
      { path: "/loras/first.safetensors", scale: 1.2 },
      { path: "/loras/third.safetensors", scale: 0.5 },
    ];

    expect(form.toRequest().loras).toEqual([
      { path: "/loras/second.safetensors", scale: 0.9 },
      { path: "/loras/first.safetensors", scale: 1.2 },
      { path: "/loras/third.safetensors", scale: 0.5 },
    ]);
  });

  it("serializes an LTX-2 camera preset as the camera-control LoRA alias", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    expect(form.toRequest().loras).toContainEqual({
      path: "camera-control:dolly-in",
      scale: 1,
    });
  });

  it("uses the camera strength edited in the visible LoRA stack", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    form.state.value.loras = [{ path: "camera-control:dolly-in", scale: 0.45 }];
    expect(form.toRequest().loras).toEqual([
      { path: "camera-control:dolly-in", scale: 0.45 },
    ]);
  });

  it("never evicts a user LoRA from a full stack to serialize camera control", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    form.state.value.loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
    }));

    expect(form.toRequest().loras).toEqual([
      { path: "one", scale: 1 },
      { path: "two", scale: 1 },
      { path: "three", scale: 1 },
      { path: "four", scale: 1 },
    ]);
  });

  it("serializes only the first attachment as source_image for non-edit families", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
        { kind: "upload", filename: "ignored.png", base64: "IGNORED" },
      ],
    });

    const wire = form.toRequest();
    expect(wire.source_image).toBe("TARGET");
    expect(wire.edit_images).toBeUndefined();
  });

  it("prunes source and video fields that are irrelevant to the selected family", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [],
      strength: 0.42,
      frames: 97,
      fps: 30,
      gifPreview: true,
    });

    const wire = form.toRequest();
    expect(wire.strength).toBeUndefined();
    expect(wire.frames).toBeUndefined();
    expect(wire.fps).toBeUndefined();
    expect(wire.gif_preview).toBeUndefined();
  });

  it("keeps video fields for a wan model whose stored family is empty", () => {
    // `selectedFamily` falls back to the model name only when `modelFamily` is
    // absent — a restored draft that predates the field. Without the wan arm
    // the family resolves to "", `supportsVideo` is false, and `toRequest`
    // prunes exactly the fields that make it a video request: the same request
    // is then submitted as a still.
    const form = useGenerateForm();
    for (const model of [
      "wan22-t2v-a14b:q5",
      "wan22-i2v-a14b:q8",
      "wan21-t2v-1.3b:bf16",
      "wan22-ti2v-5b:fp16",
    ]) {
      Object.assign(form.state.value, {
        model,
        modelFamily: "",
        frames: 81,
        fps: 16,
        gifPreview: true,
      });
      const wire = form.toRequest();
      expect(wire.frames, model).toBe(81);
      expect(wire.fps, model).toBe(16);
    }

    // A server-supplied family still wins over the name heuristic.
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "sdxl",
      frames: 81,
      fps: 16,
    });
    expect(form.toRequest().frames).toBeUndefined();
  });

  it("serializes backend-supported advanced generation knobs", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2.3-22b-distilled:fp8",
      modelFamily: "ltx2",
      seedMode: "static",
      seed: 123,
      cfgPlus: true,
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL",
      },
      controlModel: "controlnet-canny-sd15",
      controlScale: 0.8,
      upscaleModel: "real-esrgan-x4plus:fp16",
      gifPreview: true,
      audioFile: { kind: "upload", filename: "voice.wav", base64: "VOICE" },
      audioFilePath: "",
      sourceVideo: { kind: "upload", filename: "clip.mp4", base64: "VIDEO" },
      sourceVideoPath: "",
      keyframes: [
        {
          frame: 0,
          image: { kind: "upload", filename: "first.png", base64: "FIRST" },
        },
        {
          frame: 24,
          image: { kind: "upload", filename: "last.png", base64: "LAST" },
        },
      ],
      pipeline: "keyframe",
      retakeRange: { start_seconds: 1.25, end_seconds: 3.5 },
      spatialUpscale: "x1-5",
      temporalUpscale: "x2",
    });

    const wire = form.toRequest();

    expect(wire.seed).toBe(123);
    expect(wire.cfg_plus).toBeUndefined();
    expect(wire.mask_image).toBe("MASK");
    expect(wire.control_image).toBeUndefined();
    expect(wire.control_model).toBeUndefined();
    expect(wire.control_scale).toBeUndefined();
    expect(wire.upscale_model).toBe("real-esrgan-x4plus:fp16");
    expect(wire.gif_preview).toBe(true);
    expect(wire.audio_file).toBe("VOICE");
    expect(wire.audio_file_path).toBeUndefined();
    expect(wire.source_video).toBe("VIDEO");
    expect(wire.source_video_path).toBeUndefined();
    expect(wire.keyframes).toEqual([
      { frame: 0, image: "FIRST" },
      { frame: 24, image: "LAST" },
    ]);
    expect(wire.pipeline).toBe("keyframe");
    expect(wire.retake_range).toEqual({
      start_seconds: 1.25,
      end_seconds: 3.5,
    });
    expect(wire.spatial_upscale).toBe("x1-5");
    expect(wire.temporal_upscale).toBe("x2");
  });

  it("omits guidance overrides until one is set", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2-19b-distilled:fp8",
      modelFamily: "ltx2",
    });

    expect(form.toRequest().guidance_overrides).toBeUndefined();

    form.state.value.guidanceOverrides = {
      stgScale: 1.5,
      stgBlocks: "28, 29",
      rescaleScale: null,
      modalityScale: null,
      skipStep: 2,
    };

    expect(form.toRequest().guidance_overrides).toEqual({
      stg_scale: 1.5,
      stg_blocks: [28, 29],
      skip_step: 2,
    });
  });

  it("serializes CFG++ only for SD3-family models", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sd3.5-large:fp16",
      modelFamily: "sd3.5",
      cfgPlus: true,
    });

    expect(form.toRequest().cfg_plus).toBe(true);
  });

  it("serializes ControlNet inputs for SD1.5 models", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sd15:fp16",
      modelFamily: "sd15",
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL",
      },
      controlModel: "controlnet-canny-sd15:fp16",
      controlScale: 0.8,
    });

    const wire = form.toRequest();

    expect(wire.control_image).toBe("CONTROL");
    expect(wire.control_model).toBe("controlnet-canny-sd15:fp16");
    expect(wire.control_scale).toBe(0.8);
  });

  it("omits seed when seed mode is random even if a numeric seed is present", () => {
    const form = useGenerateForm();
    form.state.value.seedMode = "random";
    form.state.value.seed = 123;

    expect(form.toRequest().seed).toBeNull();
  });

  it("model switching forces Qwen edit batch to 1 and trims multi-image attachments for non-edit families", () => {
    const form = useGenerateForm();
    form.state.value.batchSize = 3;
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "target.png", base64: "TARGET" },
      { kind: "upload", filename: "ref.png", base64: "REF" },
    ];

    form.applyModelDefaults(
      makeModel({ name: "qwen-image-edit:q4", family: "qwen-image-edit" }),
    );
    expect(form.state.value.modelFamily).toBe("qwen-image-edit");
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.imageAttachments).toHaveLength(2);

    form.applyModelDefaults(makeModel({ name: "sdxl:fp16", family: "sdxl" }));
    expect(form.state.value.modelFamily).toBe("sdxl");
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.imageAttachments).toHaveLength(1);
    expect(form.state.value.imageAttachments[0]?.base64).toBe("TARGET");
  });

  it("model switching preserves a text-only batch for FLUX.2 Dev", () => {
    const form = useGenerateForm();
    form.state.value.batchSize = 4;

    form.applyModelDefaults(
      makeModel({ name: "flux2-dev:bf16", family: "flux2" }),
    );

    expect(form.state.value.batchSize).toBe(4);
    expect(form.state.value.imageAttachments).toEqual([]);
  });

  it("toRequest omits expand entirely when disabled (server treats missing/false the same, but this keeps payload minimal)", () => {
    const form = useGenerateForm();
    form.state.value.expand.enabled = false;
    const wire = form.toRequest();
    expect(wire.expand).toBeUndefined();
  });

  it("toRequest maps empty negativePrompt to null so server skips CFG", () => {
    const form = useGenerateForm();
    form.state.value.negativePrompt = "";
    expect(form.toRequest().negative_prompt).toBeNull();
  });

  it("toRequest omits stale scheduler and negative prompt for families that ignore them", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux-dev:q4",
      modelFamily: "flux",
      negativePrompt: "blurry",
      scheduler: "ddim",
    });

    const wire = form.toRequest();
    expect(wire.negative_prompt).toBeNull();
    expect(wire.scheduler).toBeUndefined();
  });

  it("preserves a distilled LTX negative in form state but omits it from the request", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "hf:opaque/distilled-checkpoint",
      modelFamily: "ltx2",
      pipeline: null,
      negativePrompt: "flicker",
      guidance: 7,
      guidanceCapabilities: {
        adjustable: false,
        supports_negative_prompt: false,
        fixed_scale: 1,
      },
    });
    expect(form.toRequest().negative_prompt).toBeNull();
    expect(form.state.value.negativePrompt).toBe("flicker");
    expect(form.state.value.guidance).toBe(7);
  });

  it("family-capability helpers match the documented allow-lists", () => {
    const form = useGenerateForm();
    // Video families.
    expect(form.isVideoFamily("ltx-video")).toBe(true);
    expect(form.isVideoFamily("ltx2")).toBe(true);
    expect(form.isVideoFamily("flux")).toBe(false);

    // CFG (negative prompt) support — flow-matching families skip CFG.
    expect(form.supportsNegativePrompt("sdxl")).toBe(true);
    expect(form.supportsNegativePrompt("sd15")).toBe(true);
    expect(form.supportsNegativePrompt("flux")).toBe(false);
    expect(form.supportsNegativePrompt("z-image")).toBe(false);

    // Scheduler override — UNet families only.
    expect(form.supportsScheduler("sdxl")).toBe(true);
    expect(form.supportsScheduler("sd15")).toBe(true);
    expect(form.supportsScheduler("flux")).toBe(false);

    // LoRA picker visibility — mirrors the server-side validation gate.
    for (const family of [
      "flux",
      "flux2",
      "ltx2",
      "sd15",
      "sd3",
      "sdxl",
      "qwen-image",
      "qwen-image-edit",
      "z-image",
    ]) {
      expect(form.supportsLora(family)).toBe(true);
    }
    expect(form.supportsLora("wuerstchen")).toBe(false);
  });

  it("reset() restores defaults", () => {
    const form = useGenerateForm();
    form.state.value.prompt = "dirty";
    form.state.value.steps = 99;
    form.reset();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.steps).toBe(20);
  });
});

describe("useGenerateForm placement", () => {
  it("carries placement into the outgoing request wire", () => {
    const form = useGenerateForm();
    form.state.value.placement = {
      text_encoders: { kind: "cpu" },
      advanced: {
        transformer: { kind: "gpu", ordinal: 1 },
        vae: { kind: "auto" },
        t5: { kind: "cpu" },
      },
    };
    const wire = form.toRequest();
    expect(wire.placement).toEqual({
      text_encoders: { kind: "cpu" },
      advanced: {
        transformer: { kind: "gpu", ordinal: 1 },
        vae: { kind: "auto" },
        t5: { kind: "cpu" },
      },
    });
  });

  it("omits placement from the request when null", () => {
    const form = useGenerateForm();
    form.state.value.placement = null;
    const wire = form.toRequest();
    expect(wire.placement).toBeUndefined();
  });
});

describe("useGenerateForm — enableAudio (LTX-2 / LTX-2.3)", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("defaults enableAudio to null on hydrate so the wire stays clean for image families", () => {
    const form = useGenerateForm();
    expect(form.state.value.enableAudio).toBeNull();
    expect(form.toRequest().enable_audio).toBeUndefined();
  });

  it("auto-enables audio when an LTX-2 model is selected so the AV path is on by default", () => {
    // Mirrors the server-side `family_supports_audio("ltx2")` truth table.
    // Switching to LTX-2 should turn audio on so the user gets the AV
    // capability they expect from the model — they can still uncheck it.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    expect(form.state.value.enableAudio).toBe(true);
    expect(form.toRequest().enable_audio).toBe(true);
  });

  it("keeps audio off for an LTX-2 catalog checkpoint whose assets are video-only", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(
      makeModel({
        name: "cv:3143864",
        family: "ltx2",
        supports_audio: false,
      }),
    );
    expect(form.state.value.enableAudio).toBe(false);
    expect(form.toRequest().enable_audio).toBe(false);
  });

  it("forces audio off at request time when persisted form state is stale", () => {
    const form = useGenerateForm();
    form.state.value.model = "cv:3143864";
    form.state.value.modelFamily = "ltx2";
    form.state.value.enableAudio = true;

    const wire = form.toRequest(
      makeModel({
        name: "cv:3143864",
        family: "ltx2",
        supports_audio: false,
      }),
    );

    expect(wire.enable_audio).toBe(false);
  });

  it("clears enableAudio back to null when switching from an AV family to an image family", () => {
    // The server rejects `enable_audio: true` for non-AV families; the
    // form must drop the toggle on family change so the user doesn't get
    // a 400 they didn't ask for.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    expect(form.state.value.enableAudio).toBe(true);
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.enableAudio).toBeNull();
    expect(form.toRequest().enable_audio).toBeUndefined();
  });

  it("clears enableAudio when switching to LTX-Video (video but no audio path)", () => {
    // LTX-Video is a video family but has no audio decode path. The toggle
    // must NOT auto-on here even though the family is video — only LTX-2 /
    // LTX-2.3 (`family === "ltx2"`) advertises audio support.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    form.applyModelDefaults(
      makeModel({ name: "ltx-video:fp16", family: "ltx-video" }),
    );
    expect(form.state.value.enableAudio).toBeNull();
  });

  it("forwards a user-set enableAudio onto the wire as enable_audio", () => {
    const form = useGenerateForm();
    form.state.value.enableAudio = false; // user explicitly disabled audio
    expect(form.toRequest().enable_audio).toBe(false);
    form.state.value.enableAudio = true;
    expect(form.toRequest().enable_audio).toBe(true);
  });
});

describe("generate form serialization helpers", () => {
  function makeForm(
    overrides: Partial<GenerateFormState> = {},
  ): GenerateFormState {
    return {
      version: 3,
      stylePreset: null,
      prompt: "a cat",
      negativePrompt: "",
      model: "flux-dev:q4",
      modelFamily: "flux",
      width: 1024,
      height: 1024,
      steps: 28,
      guidance: 3.5,
      seedMode: "random",
      seed: null,
      batchSize: 1,
      strength: 0.75,
      frames: null,
      fps: null,
      scheduler: null,
      cfgPlus: false,
      outputFormat: "png",
      expand: { enabled: false, variations: 1, familyOverride: null },
      imageAttachments: [],
      maskImage: null,
      controlImage: null,
      controlModel: "",
      controlScale: 1,
      upscaleModel: "",
      gifPreview: false,
      audioFile: null,
      audioFilePath: "",
      sourceVideo: null,
      extendVideo: null,
      extendVideoPath: "",
      extendOverlapFrames: null,
      sourceVideoPath: "",
      keyframes: [],
      pipeline: null,
      retakeRange: null,
      spatialUpscale: null,
      temporalUpscale: null,
      cameraControl: null,
      placement: null,
      loras: [],
      enableAudio: null,
      sourceFitPolicy: { mode: "pad-repaint" },
      ...overrides,
    };
  }

  it("sanitizePersistedForm strips binary media while preserving safe path references", () => {
    const form = makeForm({
      imageAttachments: [
        { kind: "gallery", filename: "source.png", base64: "SOURCE_BYTES" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK_BYTES" },
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL_BYTES",
      },
      audioFile: {
        kind: "upload",
        filename: "voice.wav",
        base64: "VOICE_BYTES",
      },
      audioFilePath: "/srv/voice.wav",
      sourceVideo: {
        kind: "upload",
        filename: "clip.mp4",
        base64: "VIDEO_BYTES",
      },
      sourceVideoPath: "/srv/clip.mp4",
      keyframes: [
        {
          frame: 24,
          image: { kind: "upload", filename: "kf.png", base64: "KF_BYTES" },
        },
      ],
    });

    const sanitized = sanitizePersistedForm(form);

    expect(sanitized.imageAttachments).toEqual([
      { kind: "gallery", filename: "source.png" },
    ]);
    expect(sanitized.maskImage).toEqual({
      kind: "upload",
      filename: "mask.png",
    });
    expect(sanitized.controlImage).toEqual({
      kind: "upload",
      filename: "control.png",
    });
    expect(sanitized.audioFile).toEqual({
      kind: "upload",
      filename: "voice.wav",
    });
    expect(sanitized.sourceVideo).toEqual({
      kind: "upload",
      filename: "clip.mp4",
    });
    expect(sanitized.keyframes).toEqual([
      { frame: 24, image: { kind: "upload", filename: "kf.png" } },
    ]);
    expect(sanitized.audioFilePath).toBe("/srv/voice.wav");
    expect(sanitized.sourceVideoPath).toBe("/srv/clip.mp4");
    expect(JSON.stringify(sanitized)).not.toContain("_BYTES");
  });

  it("cloneTemplateForm preserves generation config including static seed and ordered LoRAs", () => {
    const form = makeForm({
      seedMode: "static",
      seed: 777,
      scheduler: "ddim",
      loras: [
        { path: "/loras/a.safetensors", scale: 0.5, trainedWords: ["a"] },
        { path: "/loras/b.safetensors", scale: 1.2, trainedWords: ["b"] },
      ],
      placement: { text_encoders: { kind: "cpu" } },
    });

    const cloned = cloneTemplateForm(form);
    cloned.loras[0].scale = 2;

    expect(cloned.seedMode).toBe("static");
    expect(cloned.seed).toBe(777);
    expect(cloned.scheduler).toBe("ddim");
    expect(cloned.loras).toEqual([
      { path: "/loras/a.safetensors", scale: 2, trainedWords: ["a"] },
      { path: "/loras/b.safetensors", scale: 1.2, trainedWords: ["b"] },
    ]);
    expect(form.loras[0].scale).toBe(0.5);
  });

  it("applyMetadataToForm restores recreate-safe metadata without carrying stale binary inputs", () => {
    const current = makeForm({
      prompt: "old prompt",
      imageAttachments: [
        { kind: "upload", filename: "old.png", base64: "OLD_BYTES" },
      ],
    });
    const metadata: OutputMetadata = {
      prompt: "new prompt",
      negative_prompt: "blurry",
      model: "sdxl:fp16",
      seed: 42,
      steps: 30,
      guidance: 7.5,
      width: 768,
      height: 512,
      generation_width: 192,
      generation_height: 128,
      strength: 0.6,
      scheduler: "ddim",
      output_format: "jpeg",
      loras: [
        { path: "/loras/one.safetensors", scale: 0.8 },
        { path: "/loras/two.safetensors", scale: 1.1 },
      ],
      control_model: "controlnet-canny-sd15",
      control_scale: 0.7,
      upscale_model: "real-esrgan-x4plus:fp16",
      gif_preview: true,
      version: "0.12.0",
    };

    const next = applyMetadataToForm(current, metadata, {
      format: "png",
      models: [makeModel({ name: "sdxl:fp16", family: "sdxl" })],
    });

    expect(next.prompt).toBe("new prompt");
    expect(next.negativePrompt).toBe("blurry");
    expect(next.model).toBe("sdxl:fp16");
    expect(next.modelFamily).toBe("sdxl");
    expect(next.seedMode).toBe("static");
    expect(next.seed).toBe(42);
    expect(next.outputFormat).toBe("jpeg");
    expect(next.width).toBe(192);
    expect(next.height).toBe(128);
    expect(next.imageAttachments).toEqual([]);
    expect(next.loras).toEqual([
      { path: "/loras/one.safetensors", scale: 0.8 },
      { path: "/loras/two.safetensors", scale: 1.1 },
    ]);
  });

  it("does not infer camera motion from a metadata LoRA beyond the visible cap", () => {
    const current = makeForm();
    const next = applyMetadataToForm(
      current,
      {
        prompt: "tracking shot",
        model: "ltx-2-19b-distilled:fp8",
        seed: 7,
        steps: 8,
        guidance: 3,
        width: 768,
        height: 512,
        version: "0.1.0",
        loras: [
          { path: "one", scale: 1 },
          { path: "two", scale: 1 },
          { path: "three", scale: 1 },
          { path: "four", scale: 1 },
          { path: "camera-control:dolly-in", scale: 0.45 },
        ],
      },
      {
        models: [
          makeModel({
            name: "ltx-2-19b-distilled:fp8",
            family: "ltx2",
          }),
        ],
      },
    );

    expect(next.loras.map((lora) => lora.path)).toEqual([
      "one",
      "two",
      "three",
      "four",
    ]);
    expect(next.cameraControl).toBeNull();
  });
});
