import { describe, expect, it } from "vitest";
import { applyModelDefaults, buildRequest, newGenerateForm, seedMode } from "./generateForm";
import type { ModelEntry } from "./api/types";

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
