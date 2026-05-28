import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import {
  applyMetadataToForm,
  cloneTemplateForm,
  sanitizePersistedForm,
  useGenerateForm,
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
  });

  it("migrates a version 1 persisted snapshot but drops source image bytes", () => {
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
    expect(form.state.value.prompt).toBe("a cat");
    expect(form.state.value.model).toBe("flux-dev:q4");
    expect(form.state.value.width).toBe(512);
    expect(form.state.value.height).toBe(768);
    expect(form.state.value.imageAttachments).toEqual([]);
    // Untouched fields fall back to defaults.
    expect(form.state.value.steps).toBe(20);
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
    expect(parsed.sourceImage).toBeUndefined();
    expect(parsed.imageAttachments).toEqual([]);
    expect(raw).not.toContain("SECRETBYTES");
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

  it("applyModelDefaults preserves user-chosen frame/fps for video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 49;
    form.state.value.fps = 30;

    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));

    expect(form.state.value.frames).toBe(49);
    expect(form.state.value.fps).toBe(30);
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

  it("serializes LoRAs in the visible stack order", () => {
    const form = useGenerateForm();
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
    expect(wire.control_image).toBe("CONTROL");
    expect(wire.control_model).toBe("controlnet-canny-sd15");
    expect(wire.control_scale).toBe(0.8);
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

  it("serializes CFG++ only for SD3-family models", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sd3.5-large:fp16",
      modelFamily: "sd3.5",
      cfgPlus: true,
    });

    expect(form.toRequest().cfg_plus).toBe(true);
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
      version: 2,
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
      sourceVideoPath: "",
      keyframes: [],
      pipeline: null,
      retakeRange: null,
      spatialUpscale: null,
      temporalUpscale: null,
      placement: null,
      loras: [],
      enableAudio: null,
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

    expect(sanitized.imageAttachments).toEqual([]);
    expect(sanitized.maskImage).toBeNull();
    expect(sanitized.controlImage).toBeNull();
    expect(sanitized.audioFile).toBeNull();
    expect(sanitized.sourceVideo).toBeNull();
    expect(sanitized.keyframes).toEqual([]);
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
    expect(next.imageAttachments).toEqual([]);
    expect(next.loras).toEqual([
      { path: "/loras/one.safetensors", scale: 0.8 },
      { path: "/loras/two.safetensors", scale: 1.1 },
    ]);
  });
});
