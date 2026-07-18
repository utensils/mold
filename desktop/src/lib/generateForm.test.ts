import { describe, expect, it } from "vitest";
import {
  applyMetadataToForm,
  applyModelDefaults,
  applyPrefillToForm,
  buildRequest,
  newGenerateForm,
  seedMode,
} from "./generateForm";
import { MAX_LORA_STACK } from "./capabilities";
import type { ModelEntry, OutputMetadata } from "./api/types";

function ltx2Model(): ModelEntry {
  return {
    name: "ltx2:q8",
    family: "ltx2",
    size_gb: 1,
    is_loaded: false,
    hf_repo: "r",
    default_steps: 30,
    default_guidance: 3,
    default_width: 768,
    default_height: 512,
    description: "",
    downloaded: true,
  };
}

function ltx2Form() {
  const form = newGenerateForm();
  applyModelDefaults(form, ltx2Model());
  return form;
}

describe("seedMode", () => {
  it("derives random from an empty field and fixed from any number", () => {
    expect(seedMode("")).toBe("random");
    expect(seedMode("   ")).toBe("random");
    expect(seedMode("42")).toBe("fixed");
  });
});

describe("buildRequest — post-generate upscale", () => {
  it("ships upscale_model for image families and omits it when off", () => {
    const form = newGenerateForm();
    form.model = "flux2-klein";
    form.family = "flux2";
    form.prompt = "a cat";
    expect(buildRequest(form).upscale_model).toBeUndefined();
    form.upscaleModel = "real-esrgan-x4plus";
    expect(buildRequest(form).upscale_model).toBe("real-esrgan-x4plus");
  });

  it("never ships upscale_model for video families", () => {
    const form = ltx2Form();
    form.prompt = "a ship";
    form.upscaleModel = "real-esrgan-x4plus";
    expect(buildRequest(form).upscale_model).toBeUndefined();
  });
});

describe("newGenerateForm advanced-video defaults", () => {
  it("starts with the LTX-2 advanced fields empty (optional-safe)", () => {
    const form = newGenerateForm();
    expect(form.sourceVideo).toBeNull();
    expect(form.keyframes).toEqual([]);
    expect(form.pipeline).toBeNull();
    expect(form.retakeRange).toBeNull();
    expect(form.spatialUpscale).toBeNull();
    expect(form.temporalUpscale).toBeNull();
  });
});

describe("buildRequest — LTX-2 advanced video", () => {
  it("maps advanced fields to their kebab-case wire values", () => {
    const form = ltx2Form();
    form.prompt = "a river";
    form.pipeline = "two-stage-hq";
    form.spatialUpscale = "x1-5";
    form.temporalUpscale = "x2";
    form.sourceVideo = { filename: "clip.mp4", base64: "VIDEOB64" };
    form.keyframes = [
      { frame: 0, image: { filename: "k0.png", base64: "K0" } },
      { frame: 24, image: { filename: "k1.png", base64: "K1" } },
    ];

    const req = buildRequest(form);
    expect(req.pipeline).toBe("two-stage-hq");
    expect(req.spatial_upscale).toBe("x1-5");
    expect(req.temporal_upscale).toBe("x2");
    expect(req.source_video).toBe("VIDEOB64");
    // Keyframe wire shape is { frame, image: base64 } — no filename on the wire.
    expect(req.keyframes).toEqual([
      { frame: 0, image: "K0" },
      { frame: 24, image: "K1" },
    ]);
  });

  it("emits audio_file only for the a2vid pipeline", () => {
    const form = ltx2Form();
    form.audioFile = { filename: "voice.wav", base64: "AUDIOB64" };
    // Not a2vid → audio is not sent (server would reject it, or it's irrelevant).
    form.pipeline = "keyframe";
    expect(buildRequest(form).audio_file).toBeUndefined();
    // a2vid → the conditioning audio ships as base64.
    form.pipeline = "a2vid";
    expect(buildRequest(form).audio_file).toBe("AUDIOB64");
  });

  it("omits audio_file for a2vid when no audio was picked", () => {
    const form = ltx2Form();
    form.pipeline = "a2vid";
    expect("audio_file" in buildRequest(form)).toBe(false);
  });

  it("does not ship audio_file for a non-ltx2 family", () => {
    const form = ltx2Form();
    form.pipeline = "a2vid";
    form.audioFile = { filename: "voice.wav", base64: "AUDIOB64" };
    form.family = "flux";
    expect(buildRequest(form).audio_file).toBeUndefined();
  });

  it("includes retake_range only when set", () => {
    const form = ltx2Form();
    expect(buildRequest(form).retake_range).toBeUndefined();
    form.pipeline = "retake";
    form.retakeRange = { start_seconds: 0.5, end_seconds: 2.5 };
    expect(buildRequest(form).retake_range).toEqual({ start_seconds: 0.5, end_seconds: 2.5 });
  });

  it("omits null / empty advanced fields entirely", () => {
    const form = ltx2Form();
    const req = buildRequest(form);
    expect("pipeline" in req).toBe(false);
    expect("source_video" in req).toBe(false);
    expect("keyframes" in req).toBe(false);
    expect("spatial_upscale" in req).toBe(false);
  });

  it("does not ship advanced fields for a non-ltx2 family", () => {
    const form = ltx2Form();
    form.pipeline = "keyframe";
    form.sourceVideo = { filename: "c.mp4", base64: "V" };
    // Switch to a still-image family; prune drops the advanced surface.
    form.family = "flux";
    const req = buildRequest(form);
    expect(req.pipeline).toBeUndefined();
    expect(req.source_video).toBeUndefined();
  });
});

describe("buildRequest — camera control (LTX-2 motion LoRA presets)", () => {
  it("appends a camera-control:<preset> lora with scale 1.0 for ltx2", () => {
    const form = ltx2Form();
    form.prompt = "a cave";
    form.cameraControl = "dolly-in";
    expect(buildRequest(form).loras).toEqual([{ path: "camera-control:dolly-in", scale: 1.0 }]);
  });

  it("appends the camera-control entry after user loras, preserving order", () => {
    const form = ltx2Form();
    form.loras = [{ path: "my-style.safetensors", name: "my-style", scale: 0.8, trainedWords: [] }];
    form.cameraControl = "jib-up";
    expect(buildRequest(form).loras).toEqual([
      { path: "my-style.safetensors", scale: 0.8 },
      { path: "camera-control:jib-up", scale: 1.0 },
    ]);
  });

  it("passes a custom .safetensors path through raw — no prefix (mirrors the CLI)", () => {
    const form = ltx2Form();
    form.cameraControl = "/loras/pan-up.safetensors";
    expect(buildRequest(form).loras).toEqual([{ path: "/loras/pan-up.safetensors", scale: 1.0 }]);
  });

  it("never leaks into requests for non-ltx2 families", () => {
    const form = ltx2Form();
    form.cameraControl = "dolly-out";
    form.family = "flux";
    form.model = "flux-schnell:q4";
    expect(buildRequest(form).loras).toBeUndefined();
    form.family = "ltx-video";
    form.model = "ltx-video-0.9.6:bf16";
    expect(buildRequest(form).loras).toBeUndefined();
  });

  it("skips presets for LTX-2.3 models (no published camera LoRAs) but keeps custom paths", () => {
    const form = ltx2Form();
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.cameraControl = "dolly-in";
    expect(buildRequest(form).loras).toBeUndefined();
    form.cameraControl = "/loras/dolly.safetensors";
    expect(buildRequest(form).loras).toEqual([{ path: "/loras/dolly.safetensors", scale: 1.0 }]);
  });

  it("omits loras entirely when camera control is unset or blank", () => {
    const form = ltx2Form();
    expect(buildRequest(form).loras).toBeUndefined();
    form.cameraControl = "   ";
    expect(buildRequest(form).loras).toBeUndefined();
  });

  it("clears cameraControl when switching to a family without advanced video", () => {
    const form = ltx2Form();
    form.cameraControl = "static";
    applyModelDefaults(form, { ...ltx2Model(), name: "flux:q8", family: "flux" });
    expect(form.cameraControl).toBeNull();
  });

  it("clears a stale preset when switching from 19B to LTX-2.3 (buildRequest would drop it)", () => {
    // Presets are published for 19B only; on LTX-2.3 buildRequest silently
    // drops a preset value. Clear it on the model switch so the UI matches.
    const form = ltx2Form();
    form.model = "ltx-2-19b:fp8";
    form.cameraControl = "dolly-in";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-2.3-22b-distilled:fp8", family: "ltx2" });
    expect(form.cameraControl).toBeNull();
  });

  it("keeps a custom .safetensors path when switching into LTX-2.3 (still valid there)", () => {
    const form = ltx2Form();
    form.model = "ltx-2-19b:fp8";
    form.cameraControl = "/loras/pan-up.safetensors";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-2.3-22b-distilled:fp8", family: "ltx2" });
    expect(form.cameraControl).toBe("/loras/pan-up.safetensors");
  });
});

describe("applyModelDefaults resets advanced video on family change", () => {
  it("clears the LTX-2 fields when the new family has no advanced video", () => {
    const form = ltx2Form();
    form.pipeline = "two-stage";
    form.keyframes = [{ frame: 0, image: { filename: "k.png", base64: "K" } }];
    form.sourceVideo = { filename: "c.mp4", base64: "V" };
    form.audioFile = { filename: "voice.wav", base64: "A" };
    form.spatialUpscale = "x2";

    applyModelDefaults(form, {
      ...ltx2Model(),
      name: "flux:q8",
      family: "flux",
    });

    expect(form.pipeline).toBeNull();
    expect(form.keyframes).toEqual([]);
    expect(form.sourceVideo).toBeNull();
    expect(form.audioFile).toBeNull();
    expect(form.spatialUpscale).toBeNull();
  });
});

// ── applyMetadataToForm (gallery "Reuse settings" full-fidelity restore) ────

function sd15Model(): ModelEntry {
  return {
    ...ltx2Model(),
    name: "sd15:fp16",
    family: "sd15",
    default_steps: 20,
    default_guidance: 7.5,
    default_width: 512,
    default_height: 512,
  };
}

function richImageMetadata(): OutputMetadata {
  return {
    prompt: "a lighthouse at dusk",
    negative_prompt: "blurry, low quality",
    original_prompt: "lighthouse",
    model: "sd15:fp16",
    seed: 42,
    steps: 30,
    guidance: 7.0,
    width: 2048,
    height: 2048,
    generation_width: 512,
    generation_height: 768,
    strength: 0.6,
    scheduler: "ddim",
    output_format: "jpeg",
    cfg_plus: true,
    loras: [
      { path: "detail-tweaker.safetensors", scale: 0.8 },
      { path: "film-grain.safetensors", scale: 0.5 },
    ],
    control_model: "sd15-controlnet-canny",
    control_scale: 0.9,
    upscale_model: "real-esrgan-x4plus:fp16",
    version: "0.17.1",
  };
}

describe("applyMetadataToForm", () => {
  it("restores the full serialized parameter set for an installed model", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, richImageMetadata(), [sd15Model()]);

    expect(form.model).toBe("sd15:fp16");
    expect(form.family).toBe("sd15");
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.negativePrompt).toBe("blurry, low quality");
    expect(form.originalPrompt).toBe("lighthouse");
    expect(form.steps).toBe(30);
    expect(form.guidance).toBe(7.0);
    expect(form.scheduler).toBe("ddim");
    expect(form.cfgPlus).toBe(true);
    expect(form.strength).toBe(0.6);
    expect(form.loras).toEqual([
      expect.objectContaining({ path: "detail-tweaker.safetensors", scale: 0.8 }),
      expect.objectContaining({ path: "film-grain.safetensors", scale: 0.5 }),
    ]);
    expect(form.controlModel).toBe("sd15-controlnet-canny");
    expect(form.controlScale).toBe(0.9);
    expect(form.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    expect(form.outputFormat).toBe("jpeg");
  });

  it("prefers the pre-upscale generation canvas over the saved raster size", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, richImageMetadata(), [sd15Model()]);
    expect(form.width).toBe(512);
    expect(form.height).toBe(768);
  });

  it("uses static-seed semantics: the recorded seed lands as a fixed seed", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, richImageMetadata(), [sd15Model()]);
    expect(form.seed).toBe("42");
    expect(seedMode(form.seed)).toBe("fixed");
  });

  it("clears stale binary media — metadata never carries source bytes", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.maskImage = "MASK";
    form.controlImage = "CTRL";
    form.sourceVideo = { filename: "v.mp4", base64: "V" };
    form.keyframes = [{ frame: 0, image: { filename: "k.png", base64: "K" } }];
    form.audioFile = { filename: "a.wav", base64: "A" };

    applyMetadataToForm(form, richImageMetadata(), [sd15Model()]);

    expect(form.sourceImage).toBeNull();
    expect(form.maskImage).toBeNull();
    expect(form.controlImage).toBeNull();
    expect(form.sourceVideo).toBeNull();
    expect(form.keyframes).toEqual([]);
    expect(form.audioFile).toBeNull();
  });

  it("falls back gracefully when the metadata's model is not installed", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, richImageMetadata(), []);
    expect(form.model).toBe("sd15:fp16");
    expect(form.family).toBe("");
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.negativePrompt).toBe("blurry, low quality");
    expect(form.seed).toBe("42");
  });

  it("restores video params for an ltx2 print", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      {
        prompt: "a ship in a storm",
        model: "ltx2:q8",
        seed: 7,
        steps: 30,
        guidance: 3,
        width: 768,
        height: 512,
        frames: 121,
        fps: 30,
        enable_audio: true,
        pipeline: "two-stage",
        spatial_upscale: "x2",
        output_format: "mp4",
      },
      [ltx2Model()],
    );
    expect(form.frames).toBe(121);
    expect(form.fps).toBe(30);
    expect(form.enableAudio).toBe(true);
    expect(form.pipeline).toBe("two-stage");
    expect(form.spatialUpscale).toBe("x2");
    expect(form.outputFormat).toBe("mp4");
  });

  it("promotes a legacy single lora/lora_scale pair into the stack", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      { ...richImageMetadata(), loras: null, lora: "old.safetensors", lora_scale: 0.7 },
      [sd15Model()],
    );
    expect(form.loras).toEqual([expect.objectContaining({ path: "old.safetensors", scale: 0.7 })]);
  });

  it("caps the restored LoRA stack at MAX_LORA_STACK", () => {
    const form = newGenerateForm();
    const many = Array.from({ length: 6 }, (_, i) => ({ path: `l${i}.safetensors`, scale: 1 }));
    applyMetadataToForm(form, { ...richImageMetadata(), loras: many }, [sd15Model()]);
    expect(form.loras).toHaveLength(MAX_LORA_STACK);
  });

  it("normalizes unknown or tagged scheduler values instead of crashing", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      { ...richImageMetadata(), scheduler: { ddim: { steps: 4 } } as never },
      [sd15Model()],
    );
    expect(form.scheduler).toBe("ddim");

    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: "warp-drive" }, [sd15Model()]);
    expect(form.scheduler).toBe("default");
  });

  it("accepts the server's hyphenated/underscored scheduler spellings", () => {
    const form = newGenerateForm();

    // mold-core `Display for Scheduler` serializes UniPc as "uni-pc".
    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: "uni-pc" }, [sd15Model()]);
    expect(form.scheduler).toBe("unipc");

    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: "uni_pc" }, [sd15Model()]);
    expect(form.scheduler).toBe("unipc");

    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: { "uni-pc": {} } as never }, [
      sd15Model(),
    ]);
    expect(form.scheduler).toBe("unipc");

    // The canonical euler-ancestral spelling still round-trips.
    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: "euler-ancestral" }, [
      sd15Model(),
    ]);
    expect(form.scheduler).toBe("euler-ancestral");
  });
});

// ── applyPrefillToForm (composer routing: metadata vs legacy scalar) ────────

describe("applyPrefillToForm", () => {
  it("keeps the legacy scalar path byte-for-byte (palette / history / jobs)", () => {
    const form = newGenerateForm();
    applyPrefillToForm(
      form,
      {
        prompt: "a cat",
        model: "sd15:fp16",
        seed: 99,
        width: 640,
        height: 480,
        steps: 12,
        guidance: 5,
        upscaleModel: "real-esrgan-x4plus",
      },
      [sd15Model()],
    );
    expect(form.prompt).toBe("a cat");
    expect(form.model).toBe("sd15:fp16");
    expect(form.family).toBe("sd15");
    expect(form.seed).toBe("99");
    expect(form.width).toBe(640);
    expect(form.height).toBe(480);
    expect(form.steps).toBe(12);
    expect(form.guidance).toBe(5);
    expect(form.upscaleModel).toBe("real-esrgan-x4plus");
  });

  it("scalar path: null seed means random and missing upscaleModel clears it", () => {
    const form = newGenerateForm();
    form.upscaleModel = "stale";
    applyPrefillToForm(
      form,
      { prompt: "p", model: "nope", seed: null, width: 1, height: 2, steps: 3, guidance: 4 },
      [],
    );
    expect(form.seed).toBe("");
    expect(form.upscaleModel).toBe("");
    expect(form.family).toBe("");
  });

  it("routes a metadata prefill through the full-fidelity restore", () => {
    const form = newGenerateForm();
    applyPrefillToForm(form, { metadata: richImageMetadata() }, [sd15Model()]);
    expect(form.negativePrompt).toBe("blurry, low quality");
    expect(form.loras).toHaveLength(2);
    expect(form.width).toBe(512);
  });
});
