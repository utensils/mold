import { describe, expect, it } from "vitest";
import {
  applyMetadataToForm,
  applyModelDefaults,
  applyPrefillToForm,
  buildRequest,
  cloneGenerateForm,
  newGenerateForm,
  resetFormToModelDefaults,
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

describe("model-specific audio capability", () => {
  it("turns audio off when an LTX-2 checkpoint has video-only assets", () => {
    const form = newGenerateForm();
    form.enableAudio = true;
    applyModelDefaults(form, { ...ltx2Model(), name: "cv:3143864", supports_audio: false });
    expect(form.enableAudio).toBe(false);
  });

  it("enters H3 on its advertised frame minimum", () => {
    const form = newGenerateForm();
    form.frames = 25;
    applyModelDefaults(form, {
      ...ltx2Model(),
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      default_frames: 124,
      default_fps: 24,
    });
    expect(form.frames).toBe(124);
    expect(form.fps).toBe(24);
    expect(form.outputFormat).toBe("mp4");
  });
});

describe("advertised default negative prompt (#787)", () => {
  const WAN_DEFAULT = "色调艳丽，过曝，静态，细节模糊不清";

  function wanModel(): ModelEntry {
    return {
      ...ltx2Model(),
      name: "wan22-t2v-a14b:q5",
      family: "wan",
      default_negative_prompt: WAN_DEFAULT,
      guidance_capabilities: {
        adjustable: true,
        supports_negative_prompt: true,
      },
    };
  }

  it("prefills the advertised default when a wan model is selected", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, wanModel());
    expect(form.negativePrompt).toBe(WAN_DEFAULT);
    expect(form.negativePromptDefault).toBe(WAN_DEFAULT);
  });

  it("keeps an untouched default absent on the wire", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, wanModel());
    form.prompt = "a cat";
    expect(buildRequest(form).negative_prompt).toBeUndefined();
  });

  it("ships the explicit empty opt-out when the user clears the field", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, wanModel());
    form.prompt = "a cat";
    form.negativePrompt = "";
    expect(buildRequest(form).negative_prompt).toBe("");
  });

  it("ships typed text verbatim", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, wanModel());
    form.prompt = "a cat";
    form.negativePrompt = "hands";
    expect(buildRequest(form).negative_prompt).toBe("hands");
  });

  it("withdraws an untouched default when leaving the family, but keeps typed text", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, wanModel());
    applyModelDefaults(form, ltx2Model());
    expect(form.negativePrompt).toBe("");
    expect(form.negativePromptDefault).toBe("");

    applyModelDefaults(form, wanModel());
    form.negativePrompt = "hands";
    applyModelDefaults(form, ltx2Model());
    expect(form.negativePrompt).toBe("hands");
  });

  it("keeps today's omit-empty behavior for models without a default", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, ltx2Model());
    form.prompt = "a cat";
    expect(buildRequest(form).negative_prompt).toBeUndefined();
  });

  it('restores absent metadata negatives as the model\'s default, and "" as empty', () => {
    const metadata = {
      prompt: "a cat",
      model: "wan22-t2v-a14b:q5",
      seed: 7,
      steps: 30,
      guidance: 6,
      width: 832,
      height: 480,
      version: "1",
    } as OutputMetadata;
    const form = newGenerateForm();
    applyMetadataToForm(form, metadata, [wanModel()]);
    expect(form.negativePrompt).toBe(WAN_DEFAULT);

    const optedOut = newGenerateForm();
    applyMetadataToForm(optedOut, { ...metadata, negative_prompt: "" }, [wanModel()]);
    expect(optedOut.negativePrompt).toBe("");
    expect(buildRequest(optedOut).negative_prompt).toBe("");
  });
});

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

describe("buildRequest — MiniMax H3 authoring", () => {
  it("serializes FL2VA endpoints and fixed AV parameters from the shared contract", () => {
    const form = newGenerateForm();
    form.model = "minimax-h3-fl2va:official-bf16";
    // Exact prepared/retry snapshots can outlive their inventory row. The
    // released model partition remains sufficient authority when family
    // metadata is temporarily absent.
    form.family = "";
    form.prompt = "a synchronized shot";
    form.frames = 360;
    form.fps = 30;
    form.guidance = 6;
    form.outputFormat = "gif";
    form.h3Authoring = {
      firstFrame: {
        filename: "first.png",
        mimeType: "image/png",
        width: 1280,
        height: 720,
        data: "FIRST",
      },
      lastFrame: {
        filename: "last.png",
        mimeType: "image/png",
        width: 1280,
        height: 720,
        data: "LAST",
      },
      references: [],
    };

    const request = buildRequest(form);
    expect(request).toMatchObject({
      frames: 362,
      fps: 24,
      guidance: 0,
      strength: 1,
      batch_size: 1,
      output_format: "mp4",
      source_image: "FIRST",
      source_image_name: "first.png",
      keyframes: [{ frame: 361, image: "LAST", name: "last.png" }],
    });
    expect(request.enable_audio).toBeUndefined();
    expect(request.negative_prompt).toBeUndefined();
  });
});

describe("buildRequest — wan first/last frames", () => {
  function wanFlfForm() {
    const form = newGenerateForm();
    applyModelDefaults(form, {
      ...ltx2Model(),
      name: "wan22-i2v-a14b:q8",
      family: "wan",
      default_frames: 81,
      default_fps: 16,
      source_image: "optional",
    });
    form.frames = 81;
    form.sourceImage = "FIRST";
    form.sourceImageName = "open.png";
    return form;
  }

  it("ships the pair only as keyframes — never beside source_image", () => {
    const form = wanFlfForm();
    form.endFrame = { filename: "close.png", base64: "LAST" };
    const req = buildRequest(form);
    // The engine refuses `source_image` + `keyframes` together ("not both"),
    // and admission counts keyframes as source presence.
    expect(req.source_image).toBeUndefined();
    expect(req.source_image_name).toBeUndefined();
    expect(req.strength).toBeUndefined();
    expect(req.keyframes).toEqual([
      { frame: 0, image: "FIRST", name: "open.png" },
      { frame: 80, image: "LAST", name: "close.png" },
    ]);
  });

  it("keeps a lone source image an ordinary image-to-video request", () => {
    const req = buildRequest(wanFlfForm());
    expect(req.source_image).toBe("FIRST");
    expect(req.keyframes).toBeUndefined();
  });

  it("clears a staged end frame on metadata reuse — metadata carries no bytes", () => {
    const form = wanFlfForm();
    form.endFrame = { filename: "stale.png", base64: "STALE" };
    applyMetadataToForm(form, {
      prompt: "a lantern drifting downriver",
      model: "wan22-i2v-a14b:q8",
      seed: 7,
      steps: 20,
      guidance: 3.5,
      width: 832,
      height: 480,
      version: "test",
    } as OutputMetadata);
    // A previous draft's closing image must never silently pair with this
    // print's restored or newly attached opening image.
    expect(form.endFrame).toBeNull();
  });

  it("restores a keyframes-only request back into the two wells", () => {
    const form = wanFlfForm();
    form.endFrame = { filename: "close.png", base64: "LAST" };
    const req = buildRequest(form);

    const restored = newGenerateForm();
    applyPrefillToForm(restored, { request: req }, [
      {
        ...ltx2Model(),
        name: "wan22-i2v-a14b:q8",
        family: "wan",
        default_frames: 81,
        default_fps: 16,
        source_image: "optional",
      },
    ]);
    expect(restored.sourceImage).toBe("FIRST");
    expect(restored.sourceImageName).toBe("open.png");
    expect(restored.endFrame).toEqual({ filename: "close.png", base64: "LAST" });
    // Wan has no mid-clip keyframe panel; the raw list must not linger.
    expect(restored.keyframes).toEqual([]);
  });
});

describe("newGenerateForm advanced-video defaults", () => {
  it("starts with the LTX-2 advanced fields empty (optional-safe)", () => {
    const form = newGenerateForm();
    expect(form.sourceVideo).toBeNull();
    expect(form.keyframes).toEqual([]);
    expect(form.pipeline).toBeNull();
    expect(form.icLoraControl).toBeNull();
    expect(form.retakeRange).toBeNull();
    expect(form.spatialUpscale).toBeNull();
    expect(form.temporalUpscale).toBeNull();
    expect(form.guidanceOverrides).toEqual({
      stgScale: null,
      stgBlocks: "",
      rescaleScale: null,
      modalityScale: null,
      skipStep: null,
    });
  });
});

describe("cloneGenerateForm", () => {
  it("creates an independent snapshot of every nested mutable field", () => {
    const form = ltx2Form();
    form.imageAttachments = ["EDIT"];
    form.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: "realesrgan",
      fit: { mode: "crop-fill", alignX: "left", alignY: "top" },
    };
    form.loras = [{ path: "film.safetensors", name: "Film", scale: 0.8, trainedWords: ["film"] }];
    form.sourceVideo = { filename: "source.mp4", base64: "VIDEO" };
    form.audioFile = { filename: "sound.wav", base64: "AUDIO" };
    form.keyframes = [{ frame: 0, image: { filename: "first.png", base64: "FRAME" } }];
    form.retakeRange = { start_seconds: 1, end_seconds: 2 };
    form.guidanceOverrides.stgBlocks = "28, 29";

    const snapshot = cloneGenerateForm(form);
    snapshot.imageAttachments.push("NEXT");
    if (snapshot.sourceFit.mode === "upscale-then-fit") snapshot.sourceFit.fit.mode = "pad-fit";
    snapshot.loras[0]!.trainedWords.push("grain");
    snapshot.keyframes[0]!.image.filename = "changed.png";
    snapshot.retakeRange!.start_seconds = 9;
    snapshot.guidanceOverrides.stgBlocks = "30";

    expect(form.imageAttachments).toEqual(["EDIT"]);
    expect(form.sourceFit).toMatchObject({ fit: { mode: "crop-fill" } });
    expect(form.loras[0]!.trainedWords).toEqual(["film"]);
    expect(form.keyframes[0]!.image.filename).toBe("first.png");
    expect(form.retakeRange!.start_seconds).toBe(1);
    expect(form.guidanceOverrides.stgBlocks).toBe("28, 29");
    expect(snapshot).not.toBe(form);
  });
});

describe("buildRequest — LTX-2 advanced video", () => {
  it("preserves distilled guidance state while omitting an ineffective negative", () => {
    const form = ltx2Form();
    form.model = "hf:opaque/distilled-checkpoint";
    form.guidance = 7;
    form.negativePrompt = "flicker";
    form.guidanceCapabilities = {
      adjustable: false,
      supports_negative_prompt: false,
      fixed_scale: 1,
    };
    expect(buildRequest(form).negative_prompt).toBeUndefined();
    expect(form.negativePrompt).toBe("flicker");
    expect(form.guidance).toBe(7);
  });
  it("round-trips a built-in reference control beside custom LoRAs", () => {
    const form = ltx2Form();
    form.prompt = "a guided dancer";
    form.sourceVideo = { filename: "pose.mp4", base64: "POSE" };
    form.icLoraControl = "pose";
    form.loras = [
      { path: "/loras/style.safetensors", name: "Style", scale: 0.7, trainedWords: [] },
    ];
    const req = buildRequest(form);
    expect(req).toMatchObject({
      pipeline: "ic-lora",
      ic_lora_control: "pose",
      source_video: "POSE",
      loras: [{ path: "/loras/style.safetensors", scale: 0.7 }],
    });
  });

  it("routes the lip-dub adapter to its own pipeline, not ic-lora", () => {
    // The lip-dub adapter keeps its LoRA loaded for both stages and
    // conditions on the reference clip's speech. Sending `ic-lora` with it
    // would load the right weights and run the wrong graph — the server 422s
    // that pairing rather than honouring it.
    const form = ltx2Form();
    form.prompt = "she says something else entirely";
    form.sourceVideo = { filename: "speaker.mp4", base64: "SPEAKER" };
    form.icLoraControl = "lipdub";

    const req = buildRequest(form);

    expect(req).toMatchObject({
      pipeline: "lip-dub",
      ic_lora_control: "lipdub",
      source_video: "SPEAKER",
    });
  });

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
    // Names are provenance only; the engine still consumes frame + bytes.
    expect(req.keyframes).toEqual([
      { frame: 0, image: "K0", name: "k0.png" },
      { frame: 24, image: "K1", name: "k1.png" },
    ]);
  });

  it("emits audio_file only for the a2vid pipeline", () => {
    const form = ltx2Form();
    form.audioFile = { filename: "voice.wav", base64: "AUDIOB64" };
    // Not a2vid → audio is not sent (server would reject it, or it's irrelevant).
    form.pipeline = "keyframe";
    expect(buildRequest(form).audio_file).toBeUndefined();
    // a2vid → the conditioning audio ships as base64.
    form.pipeline = "a2-vid";
    expect(buildRequest(form).audio_file).toBe("AUDIOB64");
  });

  it("never sends enable_audio=false for the audio-only t2a pipeline", () => {
    // A fresh form has `enableAudio = false`, and picking `t2a` only moves the
    // output format — so every first-time desktop text-to-audio request used
    // to arrive as `pipeline=t2a` + `enable_audio=false`, which the server
    // rejects outright ("pipeline=t2a cannot be combined with
    // enable_audio=false"). Audio is what t2a renders; the flag cannot
    // contradict it.
    const form = ltx2Form();
    expect(form.enableAudio).toBe(false);
    form.pipeline = "t2a";
    form.outputFormat = "wav";
    expect(buildRequest(form).enable_audio).not.toBe(false);

    // An explicit opt-in stays exactly what the user asked for.
    form.enableAudio = true;
    expect(buildRequest(form).enable_audio).not.toBe(false);
  });

  it("drops stale conditioning and upscalers when the form is switched to t2a", () => {
    // The t2a controls stay on screen with a hint rather than vanishing, but
    // nothing was clearing their values — so a form that had a source video
    // sent it anyway and the server refused the request.
    const form = ltx2Form();
    form.sourceVideo = { filename: "clip.mp4", base64: "VIDEOB64" };
    form.keyframes = [{ frame: 0, image: { filename: "k0.png", base64: "K0" } }];
    form.spatialUpscale = "x2";
    form.temporalUpscale = "x2";
    form.pipeline = "t2a";
    form.outputFormat = "wav";

    const req = buildRequest(form);
    expect("source_video" in req).toBe(false);
    expect("keyframes" in req).toBe(false);
    expect("spatial_upscale" in req).toBe(false);
    expect("temporal_upscale" in req).toBe(false);
    // Duration and the prompt are the whole input — those must survive.
    expect(req.frames).toBe(form.frames);
    expect(req.fps).toBe(form.fps);
    expect(req.pipeline).toBe("t2a");

    // And the form itself keeps the clip, so switching back restores it.
    expect(form.sourceVideo?.base64).toBe("VIDEOB64");
  });

  it("still ships an explicit enable_audio for ordinary video pipelines", () => {
    const form = ltx2Form();
    form.pipeline = "two-stage";
    expect(buildRequest(form).enable_audio).toBe(false);
    form.enableAudio = true;
    expect(buildRequest(form).enable_audio).toBe(true);
  });

  it("omits audio_file for a2vid when no audio was picked", () => {
    const form = ltx2Form();
    form.pipeline = "a2-vid";
    expect("audio_file" in buildRequest(form)).toBe(false);
  });

  it("does not ship audio_file for a non-ltx2 family", () => {
    const form = ltx2Form();
    form.pipeline = "a2-vid";
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

  it("serializes guidance overrides only when LTX-2 controls are active", () => {
    const form = ltx2Form();
    expect(buildRequest(form).guidance_overrides).toBeUndefined();

    form.guidanceOverrides = {
      stgScale: 1.5,
      stgBlocks: "28, 29",
      rescaleScale: 0.7,
      modalityScale: 3,
      skipStep: 2,
    };
    expect(buildRequest(form).guidance_overrides).toEqual({
      stg_scale: 1.5,
      stg_blocks: [28, 29],
      rescale_scale: 0.7,
      modality_scale: 3,
      skip_step: 2,
    });

    form.family = "flux";
    expect(buildRequest(form).guidance_overrides).toBeUndefined();
  });
});

describe("buildRequest — camera control (LTX-2 motion LoRA presets)", () => {
  it("appends a camera-control:<preset> lora with scale 1.0 for ltx2", () => {
    const form = ltx2Form();
    form.prompt = "a cave";
    form.cameraControl = "dolly-in";
    expect(buildRequest(form).loras).toEqual([{ path: "camera-control:dolly-in", scale: 1.0 }]);
  });

  it("uses the camera adapter strength edited in the visible LoRA stack", () => {
    const form = ltx2Form();
    form.cameraControl = "dolly-in";
    form.loras = [
      {
        path: "camera-control:dolly-in",
        name: "Dolly in camera control",
        scale: 0.5,
        trainedWords: [],
      },
    ];
    expect(buildRequest(form).loras).toEqual([{ path: "camera-control:dolly-in", scale: 0.5 }]);
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

  it("serializes host-vetted presets without guessing compatibility from the public model id", () => {
    const form = ltx2Form();
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.cameraControl = "dolly-in";
    expect(buildRequest(form).loras).toEqual([{ path: "camera-control:dolly-in", scale: 1.0 }]);
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

  it("does not guess camera compatibility while model defaults change", () => {
    const form = ltx2Form();
    form.model = "ltx-2-19b:fp8";
    form.cameraControl = "dolly-in";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-2.3-22b-distilled:fp8", family: "ltx2" });
    expect(form.cameraControl).toBe("dolly-in");
  });

  it("keeps a custom .safetensors path when switching into LTX-2.3 (still valid there)", () => {
    const form = ltx2Form();
    form.model = "ltx-2-19b:fp8";
    form.cameraControl = "/loras/pan-up.safetensors";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-2.3-22b-distilled:fp8", family: "ltx2" });
    expect(form.cameraControl).toBe("/loras/pan-up.safetensors");
  });
});

describe("applyModelDefaults — model-advertised fps", () => {
  it("takes the model's default_fps, exactly like steps and guidance", () => {
    const form = ltx2Form();
    expect(form.fps).toBe(24);

    applyModelDefaults(form, {
      ...ltx2Model(),
      name: "ltx-video-0.9.6-distilled:bf16",
      family: "ltx-video",
      default_fps: 30,
    });

    expect(form.fps).toBe(30);
  });

  it("keeps the current fps when the server advertises none", () => {
    const form = ltx2Form();
    form.fps = 16;
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx2:fp8" });
    expect(form.fps).toBe(16);
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

// ── qwen-edit Target + Reference attachments ────────────────────────────────

function qwenEditModel(): ModelEntry {
  return {
    ...ltx2Model(),
    name: "qwen-image-edit-2511:q4",
    family: "qwen-image-edit",
    default_steps: 20,
    default_guidance: 4,
    default_width: 1024,
    default_height: 1024,
  };
}

describe("buildRequest — qwen-edit edit_images", () => {
  function qwenEditForm() {
    const form = newGenerateForm();
    applyModelDefaults(form, qwenEditModel());
    form.prompt = "make the sky pink";
    return form;
  }

  it("ships ordered edit_images (first = target, rest = references) and never source_image/strength", () => {
    const form = qwenEditForm();
    form.imageAttachments = ["TARGET", "REF_A", "REF_B"];
    const req = buildRequest(form);
    expect(req.edit_images).toEqual(["TARGET", "REF_A", "REF_B"]);
    expect("source_image" in req).toBe(false);
    expect("strength" in req).toBe(false);
    expect("mask_image" in req).toBe(false);
  });

  it("forces batch_size to 1", () => {
    const form = qwenEditForm();
    form.imageAttachments = ["TARGET"];
    form.batchSize = 4; // stale value a template restore could leave behind
    expect(buildRequest(form).batch_size).toBe(1);
  });

  it("omits edit_images when no attachments are set (prompt-only edit request)", () => {
    const req = buildRequest(qwenEditForm());
    expect("edit_images" in req).toBe(false);
  });

  it("never ships edit_images for single-mode families", () => {
    const form = newGenerateForm();
    form.model = "flux-dev:q8";
    form.family = "flux";
    form.prompt = "a cat";
    form.imageAttachments = ["STALE"];
    form.sourceImage = "SRC";
    const req = buildRequest(form);
    expect("edit_images" in req).toBe(false);
    expect(req.source_image).toBe("SRC");
  });
});

describe("buildRequest — FLUX.2 Dev references", () => {
  it("uses edit_images while keeping Klein on classic source_image", () => {
    const dev = newGenerateForm();
    dev.model = "flux2-dev:bf16";
    dev.family = "flux2";
    dev.prompt = "preserve the subject";
    dev.imageAttachments = ["REF_ONE", "REF_TWO"];
    dev.batchSize = 2;
    const devRequest = buildRequest(dev);
    expect(devRequest.edit_images).toEqual(["REF_ONE", "REF_TWO"]);
    expect("source_image" in devRequest).toBe(false);
    expect(devRequest.batch_size).toBe(1);

    const klein = newGenerateForm();
    klein.model = "flux2-klein:q8";
    klein.family = "flux2";
    klein.prompt = "a cat";
    klein.sourceImage = "SOURCE";
    const kleinRequest = buildRequest(klein);
    expect(kleinRequest.source_image).toBe("SOURCE");
    expect("edit_images" in kleinRequest).toBe(false);
  });

  it("preserves text-only batches and locks only reference requests", () => {
    const dev = newGenerateForm();
    dev.model = "flux2-dev:bf16";
    dev.family = "flux2";
    dev.prompt = "three landscapes";
    dev.batchSize = 3;

    expect(buildRequest(dev).batch_size).toBe(3);
    dev.imageAttachments = ["REFERENCE"];
    expect(buildRequest(dev).batch_size).toBe(1);
  });
});

describe("buildRequest — independent ControlNet conditioning", () => {
  it("ships control_image, model, and scale without source_image", () => {
    const form = newGenerateForm();
    form.model = "sd15:fp16";
    form.family = "sd15";
    form.prompt = "a mountain observatory";
    form.controlImage = "CONTROL";
    form.controlModel = "/models/controlnet-canny.safetensors";
    form.controlScale = 0.85;

    const req = buildRequest(form);
    expect(req.control_image).toBe("CONTROL");
    expect(req.control_model).toBe("/models/controlnet-canny.safetensors");
    expect(req.control_scale).toBe(0.85);
    expect(req.source_image).toBeUndefined();
    expect(req.source_image_name).toBeUndefined();
    expect(req.strength).toBeUndefined();
  });

  it("still strips stale ControlNet fields from unsupported families", () => {
    const form = newGenerateForm();
    form.model = "sdxl:fp16";
    form.family = "sdxl";
    form.prompt = "a mountain observatory";
    form.controlImage = "STALE";
    form.controlModel = "stale-control";
    form.controlScale = 0.5;

    const req = buildRequest(form);
    expect(req.control_image).toBeUndefined();
    expect(req.control_model).toBeUndefined();
    expect(req.control_scale).toBeUndefined();
  });
});

describe("applyModelDefaults — qwen-edit attachment seeding", () => {
  it("seeds the strip with the single-mode source as Target when switching to qwen-edit", () => {
    const form = newGenerateForm();
    form.family = "flux";
    form.sourceImage = "SRC";
    applyModelDefaults(form, qwenEditModel());
    expect(form.imageAttachments).toEqual(["SRC"]);
    expect(form.sourceImage).toBeNull();
    expect(form.maskImage).toBeNull();
  });

  it("keeps existing attachments when re-applying a qwen-edit model", () => {
    const form = newGenerateForm();
    form.imageAttachments = ["T", "R"];
    applyModelDefaults(form, qwenEditModel());
    expect(form.imageAttachments).toEqual(["T", "R"]);
  });

  it("promotes the Target back to the single source when leaving qwen-edit", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, qwenEditModel());
    form.imageAttachments = ["T", "R1", "R2"];
    applyModelDefaults(form, { ...ltx2Model(), name: "flux:q8", family: "flux" });
    expect(form.sourceImage).toBe("T");
    expect(form.imageAttachments).toEqual([]);
  });

  it("promotes the Target to the img2video source when switching to ltx2", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, qwenEditModel());
    form.imageAttachments = ["T"];
    applyModelDefaults(form, ltx2Model());
    expect(form.imageAttachments).toEqual([]);
    expect(form.sourceImage).toBe("T");
  });

  it("clears attachments when switching to ltx-video (no img2img at all)", () => {
    const form = newGenerateForm();
    applyModelDefaults(form, qwenEditModel());
    form.imageAttachments = ["T"];
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-video:q8", family: "ltx-video" });
    expect(form.imageAttachments).toEqual([]);
    expect(form.sourceImage).toBeNull();
  });

  it("locks the batch size to 1 on switch to qwen-edit", () => {
    const form = newGenerateForm();
    form.batchSize = 4;
    applyModelDefaults(form, qwenEditModel());
    expect(form.batchSize).toBe(1);
  });
});

function wanModel(name = "wan22-t2v-a14b:q5"): ModelEntry {
  return {
    ...ltx2Model(),
    name,
    family: "wan",
    default_steps: 4,
    default_guidance: 1,
    default_width: 832,
    default_height: 480,
  };
}

function wanForm(name?: string) {
  const form = newGenerateForm();
  applyModelDefaults(form, wanModel(name));
  return form;
}

describe("buildRequest — wan sampler recipe", () => {
  it("omits every recipe field while the controls are untouched", () => {
    const req = buildRequest(wanForm());
    expect("sample_shift" in req).toBe(false);
    expect("distill_strength_high" in req).toBe(false);
    expect("distill_strength_low" in req).toBe(false);
    expect(req.scheduler).toBeUndefined();
  });

  it("sends only what the user touched", () => {
    const form = wanForm();
    form.scheduler = "euler";
    form.wanRecipe = { sampleShift: 12, distillStrengthHigh: 1.8, distillStrengthLow: null };
    const req = buildRequest(form);
    expect(req).toMatchObject({ scheduler: "euler", sample_shift: 12, distill_strength_high: 1.8 });
    expect("distill_strength_low" in req).toBe(false);
  });

  it("drops the strengths on a wan tier that ships no distill", () => {
    const form = wanForm("wan22-t2v-a14b:q8");
    form.wanRecipe = { sampleShift: 8, distillStrengthHigh: 1.8, distillStrengthLow: 1 };
    const req = buildRequest(form);
    expect(req.sample_shift).toBe(8);
    expect("distill_strength_high" in req).toBe(false);
    expect("distill_strength_low" in req).toBe(false);
  });

  it("never ships a recipe field for another family", () => {
    const form = ltx2Form();
    form.wanRecipe = { sampleShift: 12, distillStrengthHigh: 1.8, distillStrengthLow: 1 };
    const req = buildRequest(form);
    expect("sample_shift" in req).toBe(false);
    expect("distill_strength_high" in req).toBe(false);
  });

  it("omits a value the server would reject rather than sending it", () => {
    const form = wanForm();
    form.wanRecipe = { sampleShift: 0, distillStrengthHigh: 9, distillStrengthLow: null };
    const req = buildRequest(form);
    expect("sample_shift" in req).toBe(false);
    expect("distill_strength_high" in req).toBe(false);
  });
});

describe("applyModelDefaults — wan sampler recipe and solver", () => {
  it("clears the recipe when the new model leaves the family", () => {
    const form = wanForm();
    form.wanRecipe = { sampleShift: 12, distillStrengthHigh: 1.8, distillStrengthLow: 1 };
    applyModelDefaults(form, sd15Model());
    expect(form.wanRecipe).toEqual({
      sampleShift: null,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });

  it("keeps the shift but clears the strengths on a tier without a distill", () => {
    const form = wanForm();
    form.wanRecipe = { sampleShift: 12, distillStrengthHigh: 1.8, distillStrengthLow: 1 };
    applyModelDefaults(form, wanModel("wan22-t2v-a14b:q8"));
    expect(form.wanRecipe).toEqual({
      sampleShift: 12,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });

  it("resets a solver the new family's server would reject", () => {
    // The two option sets are disjoint on the server, so carrying `dpm-pp`
    // into SDXL (or `ddim` into wan) is a 422, not a no-op.
    const wan = wanForm();
    wan.scheduler = "dpm-pp";
    applyModelDefaults(wan, sd15Model());
    expect(wan.scheduler).toBe("default");

    const sd = newGenerateForm();
    applyModelDefaults(sd, sd15Model());
    sd.scheduler = "ddim";
    applyModelDefaults(sd, wanModel());
    expect(sd.scheduler).toBe("default");
  });
});

describe("newGenerateForm source-fit default", () => {
  it("starts on pad-repaint, matching the web SPA's default policy", () => {
    expect(newGenerateForm().sourceFit).toEqual({ mode: "pad-repaint" });
  });

  it("keeps the chosen policy across a model change (web parity)", () => {
    const form = newGenerateForm();
    form.sourceFit = { mode: "crop-fill", alignX: "center", alignY: "center" };
    applyModelDefaults(form, ltx2Model());
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });
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
    form.imageAttachments = ["T", "R"];
    form.sourceVideo = { filename: "v.mp4", base64: "V" };
    form.keyframes = [{ frame: 0, image: { filename: "k.png", base64: "K" } }];
    form.audioFile = { filename: "a.wav", base64: "A" };
    form.cameraControl = "dolly-in";

    applyMetadataToForm(form, richImageMetadata(), [sd15Model()]);

    expect(form.sourceImage).toBeNull();
    expect(form.maskImage).toBeNull();
    expect(form.controlImage).toBeNull();
    expect(form.imageAttachments).toEqual([]);
    expect(form.sourceVideo).toBeNull();
    expect(form.keyframes).toEqual([]);
    expect(form.audioFile).toBeNull();
    expect(form.cameraControl).toBeNull();
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

  it("preserves H3 boundary provenance while requiring reattachment", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      {
        ...richImageMetadata(),
        model: "minimax-h3-fl2va:official-bf16",
        guidance: 0,
        output_format: "mp4",
        source_image_name: "opening.png",
        source_image_sha256: "a".repeat(64),
        frames: 124,
        keyframes: [{ frame: 123, name: "closing.png", sha256: "b".repeat(64) }],
      },
      [],
    );
    expect(form.h3Authoring?.firstFrame).toMatchObject({
      filename: "opening.png",
      data: "",
      sha256: "a".repeat(64),
    });
    expect(form.h3Authoring?.lastFrame).toMatchObject({
      filename: "closing.png",
      data: "",
      sha256: "b".repeat(64),
    });
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
        guidance_overrides: {
          stg_scale: 1.25,
          stg_blocks: [28, 29],
          rescale_scale: 0.6,
          modality_scale: 2.5,
          skip_step: 1,
        },
        output_format: "mp4",
      },
      [ltx2Model()],
    );
    expect(form.frames).toBe(121);
    expect(form.fps).toBe(30);
    expect(form.enableAudio).toBe(true);
    expect(form.pipeline).toBe("two-stage");
    expect(form.spatialUpscale).toBe("x2");
    expect(form.guidanceOverrides).toEqual({
      stgScale: 1.25,
      stgBlocks: "28, 29",
      rescaleScale: 0.6,
      modalityScale: 2.5,
      skipStep: 1,
    });
    expect(form.outputFormat).toBe("mp4");
  });

  it("restores a camera preset and its strength from the saved LoRA stack", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      {
        prompt: "a tracking shot",
        model: "ltx2:q8",
        seed: 7,
        width: 768,
        height: 512,
        steps: 20,
        guidance: 3,
        loras: [{ path: "camera-control:dolly-in", scale: 0.45 }],
      },
      [ltx2Model()],
    );

    expect(form.cameraControl).toBe("dolly-in");
    expect(form.loras).toEqual([
      expect.objectContaining({
        path: "camera-control:dolly-in",
        name: "Dolly in camera control",
        scale: 0.45,
      }),
    ]);
    expect(buildRequest(form).loras).toEqual([{ path: "camera-control:dolly-in", scale: 0.45 }]);
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

    // mold-core `Display for Scheduler` serializes UniPc as "uni-pc"; the
    // separator-squashed legacy spellings still restore onto it.
    for (const spelling of ["uni-pc", "uni_pc", "unipc"]) {
      applyMetadataToForm(form, { ...richImageMetadata(), scheduler: spelling }, [sd15Model()]);
      expect(form.scheduler).toBe("uni-pc");
    }

    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: { "uni-pc": {} } as never }, [
      sd15Model(),
    ]);
    expect(form.scheduler).toBe("uni-pc");

    // The canonical euler-ancestral spelling still round-trips.
    applyMetadataToForm(form, { ...richImageMetadata(), scheduler: "euler-ancestral" }, [
      sd15Model(),
    ]);
    expect(form.scheduler).toBe("euler-ancestral");
  });

  it("round-trips the wan recipe a print was rendered with", () => {
    const form = newGenerateForm();
    applyMetadataToForm(
      form,
      {
        ...richImageMetadata(),
        model: "wan22-t2v-a14b:q5",
        scheduler: "dpm-pp",
        sample_shift: 12,
        distill_strength_high: 1.8,
        distill_strength_low: 0.9,
      },
      [wanModel()],
    );
    expect(form.scheduler).toBe("dpm-pp");
    expect(form.wanRecipe).toEqual({
      sampleShift: 12,
      distillStrengthHigh: 1.8,
      distillStrengthLow: 0.9,
    });
    expect(buildRequest(form)).toMatchObject({
      scheduler: "dpm-pp",
      sample_shift: 12,
      distill_strength_high: 1.8,
      distill_strength_low: 0.9,
    });
  });

  it("leaves the recipe untouched for a print that carried none", () => {
    const form = newGenerateForm();
    applyMetadataToForm(form, { ...richImageMetadata(), model: "wan22-t2v-a14b:q5" }, [wanModel()]);
    expect(form.wanRecipe).toEqual({
      sampleShift: null,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
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

  it("restores an exact queued request without retaining stale advanced inputs", () => {
    const form = newGenerateForm();
    form.controlModel = "stale-control";
    applyPrefillToForm(
      form,
      {
        request: {
          prompt: "a moving train",
          negative_prompt: "blurry",
          original_prompt: "a train",
          model: "ltx2:q8",
          width: 1280,
          height: 720,
          steps: 18,
          guidance: 2.5,
          seed: 77,
          scheduler: "euler-ancestral",
          output_format: "mp4",
          source_image: "SOURCE",
          source_image_name: "frame.png",
          loras: [{ path: "/models/motion.safetensors", scale: 0.7 }],
          frames: 97,
          fps: 25,
          enable_audio: true,
          source_video: "VIDEO",
          audio_file: "AUDIO",
          keyframes: [{ frame: 48, image: "KEYFRAME" }],
          pipeline: "retake",
          retake_range: { start_seconds: 1, end_seconds: 2 },
          spatial_upscale: "x2",
          temporal_upscale: "x2",
          guidance_overrides: {
            stg_scale: 1.5,
            stg_blocks: [28],
            rescale_scale: 0.5,
          },
        },
      },
      [ltx2Model()],
    );

    expect(form).toMatchObject({
      prompt: "a moving train",
      originalPrompt: "a train",
      negativePrompt: "blurry",
      model: "ltx2:q8",
      family: "ltx2",
      seed: "77",
      scheduler: "euler-ancestral",
      outputFormat: "mp4",
      sourceImage: "SOURCE",
      sourceImageName: "frame.png",
      sourceFit: { mode: "crop-fill", alignX: "center", alignY: "center" },
      controlModel: "",
      frames: 97,
      fps: 25,
      enableAudio: true,
      pipeline: "retake",
      spatialUpscale: "x2",
      temporalUpscale: "x2",
      guidanceOverrides: {
        stgScale: 1.5,
        stgBlocks: "28",
        rescaleScale: 0.5,
        modalityScale: null,
        skipStep: null,
      },
    });
    expect(form.loras).toEqual([
      expect.objectContaining({ path: "/models/motion.safetensors", scale: 0.7 }),
    ]);
    expect(form.sourceVideo?.base64).toBe("VIDEO");
    expect(form.audioFile?.base64).toBe("AUDIO");
    expect(form.keyframes[0]?.image.base64).toBe("KEYFRAME");
  });
});

describe("LTX-2 img2img (image-to-video)", () => {
  it("keeps the source image when switching from an image family into ltx2", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.maskImage = "MASK";
    applyModelDefaults(form, ltx2Model());
    expect(form.sourceImage).toBe("SRC"); // seeds frame-0 conditioning
    expect(form.maskImage).toBeNull(); // no inpainting on video
  });

  it("clears the source when switching to plain ltx-video (no img2vid path)", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-video:q8", family: "ltx-video" });
    expect(form.sourceImage).toBeNull();
  });

  it("coerces a mask-dependent source-fit policy to crop-fill on entry", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    expect(form.sourceFit).toEqual({ mode: "pad-repaint" });
    applyModelDefaults(form, ltx2Model());
    expect(form.sourceFit).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });
  });

  it("ships source_image + strength for ltx2 but never mask_image", () => {
    const form = ltx2Form();
    form.prompt = "a cat";
    form.sourceImage = "SRC";
    form.strength = 0.6;
    form.maskImage = "MASK"; // stale value must not leak
    const req = buildRequest(form);
    expect(req.source_image).toBe("SRC");
    expect(req.strength).toBe(0.6);
    expect(req.mask_image).toBeUndefined();
  });
});

describe("source image provenance (Reuse-settings restore)", () => {
  it("ships source_image_name only alongside a source image", () => {
    const form = newGenerateForm();
    form.model = "flux-dev:q8";
    form.family = "flux";
    form.prompt = "a cat";
    form.sourceImageName = "mold-flux-1-2.png";
    // Name without an image never ships.
    expect(buildRequest(form).source_image_name).toBeUndefined();

    form.sourceImage = "SRC";
    const req = buildRequest(form);
    expect(req.source_image).toBe("SRC");
    expect(req.source_image_name).toBe("mold-flux-1-2.png");
  });

  it("clears the label whenever the source is cleared on model switch", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    applyModelDefaults(form, { ...ltx2Model(), name: "ltx-video:q8", family: "ltx-video" });
    expect(form.sourceImage).toBeNull();
    expect(form.sourceImageName).toBeNull();
  });

  it("keeps the label when the source survives a switch into ltx2", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    applyModelDefaults(form, ltx2Model());
    expect(form.sourceImage).toBe("SRC");
    expect(form.sourceImageName).toBe("pic.png");
  });

  it("drops the label when entering qwen-edit (attachments are unlabeled)", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    applyModelDefaults(form, qwenEditModel());
    expect(form.imageAttachments).toEqual(["SRC"]);
    expect(form.sourceImageName).toBeNull();
  });
});

describe("resetFormToModelDefaults", () => {
  const sdxl: ModelEntry = {
    name: "sdxl:base",
    family: "sdxl",
    size_gb: 7,
    is_loaded: false,
    hf_repo: "r",
    default_steps: 30,
    default_guidance: 7,
    default_width: 1024,
    default_height: 768,
    description: "",
    downloaded: true,
  };

  function dirtyForm() {
    const form = newGenerateForm();
    form.prompt = "a lighthouse at dusk";
    form.originalPrompt = "a lighthouse";
    form.model = "sdxl:base";
    form.family = "sdxl";
    form.batchSize = 4;
    form.negativePrompt = "blurry";
    form.scheduler = "ddim";
    form.cfgPlus = true;
    form.steps = 12;
    form.guidance = 1.5;
    form.width = 512;
    form.height = 512;
    form.seed = "1234";
    form.strength = 0.2;
    form.upscaleModel = "esrgan";
    form.stylePreset = "cinematic";
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    form.loras = [{ path: "/l.safetensors", name: "l", scale: 0.8, trainedWords: [] }];
    return form;
  }

  it("preserves the prompt, its expand provenance, the model, and the prepared batch size", () => {
    const form = dirtyForm();
    resetFormToModelDefaults(form, sdxl);
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.originalPrompt).toBe("a lighthouse");
    expect(form.model).toBe("sdxl:base");
    expect(form.family).toBe("sdxl");
    expect(form.batchSize).toBe(4);
  });

  it("restores every other field to its default", () => {
    const form = dirtyForm();
    const defaults = newGenerateForm();
    resetFormToModelDefaults(form, sdxl);
    expect(form.negativePrompt).toBe(defaults.negativePrompt);
    expect(form.scheduler).toBe(defaults.scheduler);
    expect(form.cfgPlus).toBe(defaults.cfgPlus);
    expect(form.seed).toBe(defaults.seed);
    expect(form.strength).toBe(defaults.strength);
    expect(form.upscaleModel).toBe(defaults.upscaleModel);
    expect(form.stylePreset).toBe(defaults.stylePreset);
    expect(form.sourceImage).toBeNull();
    expect(form.sourceImageName).toBeNull();
    expect(form.loras).toEqual([]);
  });

  it("applies the selected model's dimension, step, and guidance defaults", () => {
    const form = dirtyForm();
    resetFormToModelDefaults(form, sdxl);
    expect(form.width).toBe(1024);
    expect(form.height).toBe(768);
    expect(form.steps).toBe(30);
    expect(form.guidance).toBe(7);
  });

  it("keeps the named model and family when no model entry is available", () => {
    const form = dirtyForm();
    const defaults = newGenerateForm();
    resetFormToModelDefaults(form, null);
    expect(form.model).toBe("sdxl:base");
    expect(form.family).toBe("sdxl");
    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.batchSize).toBe(4);
    expect(form.steps).toBe(defaults.steps);
    expect(form.negativePrompt).toBe("");
  });

  it("locks batch to one for a family that renders a single print at a time", () => {
    const form = dirtyForm();
    const editModel: ModelEntry = { ...sdxl, name: "qwen-image-edit", family: "qwen-image-edit" };
    resetFormToModelDefaults(form, editModel);
    expect(form.batchSize).toBe(1);
  });
});

describe("source label clearing invariants (review findings)", () => {
  it("applyMetadataToForm clears the label with the image", () => {
    const form = newGenerateForm();
    form.sourceImage = "SRC";
    form.sourceImageName = "pic.png";
    applyMetadataToForm(form, { ...richImageMetadata() });
    expect(form.sourceImage).toBeNull();
    expect(form.sourceImageName).toBeNull();
  });
});
