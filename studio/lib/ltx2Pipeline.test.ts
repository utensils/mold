import { describe, expect, it } from "vitest";
import {
  AUDIO_ONLY_PIPELINE,
  isAudioCompletion,
  isAudioOnlyPipeline,
  stripAudioOnlyIncompatibleFields,
} from "./ltx2Pipeline";

describe("isAudioOnlyPipeline", () => {
  it("recognises the audio-only pipeline", () => {
    expect(isAudioOnlyPipeline(AUDIO_ONLY_PIPELINE)).toBe(true);
    expect(isAudioOnlyPipeline("t2a")).toBe(true);
  });

  it("treats every video pipeline, and an unset one, as not audio-only", () => {
    for (const pipeline of [
      "one-stage",
      "two-stage",
      "two-stage-hq",
      "distilled",
      "ic-lora",
      "keyframe",
      // Easy to confuse with t2a: a2-vid consumes audio, it does not render it.
      "a2-vid",
      "retake",
      "lip-dub",
    ]) {
      expect(isAudioOnlyPipeline(pipeline)).toBe(false);
    }
    expect(isAudioOnlyPipeline(null)).toBe(false);
    expect(isAudioOnlyPipeline(undefined)).toBe(false);
    expect(isAudioOnlyPipeline("")).toBe(false);
  });
});

describe("isAudioCompletion", () => {
  // Must be probed before any video or image branch: an audio print carries
  // the WAV in `image` and has no frames, so a video-shaped probe falls
  // through to the still branch and renders the WAV as a picture.
  it("recognises an audio completion by its sample rate", () => {
    expect(isAudioCompletion({ audio_sample_rate: 24000 })).toBe(true);
    // 0 Hz is nonsense, but it is still an audio-shaped completion — the
    // discriminator is presence, never truthiness.
    expect(isAudioCompletion({ audio_sample_rate: 0 })).toBe(true);
  });

  it("leaves stills, clips, and absent results alone", () => {
    expect(isAudioCompletion({})).toBe(false);
    expect(isAudioCompletion({ audio_sample_rate: null })).toBe(false);
    expect(isAudioCompletion(null)).toBe(false);
    expect(isAudioCompletion(undefined)).toBe(false);
  });
});

describe("stripAudioOnlyIncompatibleFields", () => {
  const loaded = {
    prompt: "heavy rain on a tin roof",
    model: "ltx-2-19b-dev:fp8",
    frames: 121,
    fps: 24,
    source_image: "PNGB64",
    source_image_name: "still.png",
    strength: 0.75,
    mask_image: "MASKB64",
    source_video: "CLIPB64",
    source_video_path: "/srv/clip.mp4",
    audio_file: "WAVB64",
    audio_file_path: "/srv/voice.wav",
    extend_video: "CLIPB64",
    extend_video_path: "/srv/tail.mp4",
    extend_overlap_frames: 9,
    keyframes: [{ frame: 0, image: "K0" }],
    retake_range: { start_seconds: 0, end_seconds: 1 },
    spatial_upscale: "x2",
    temporal_upscale: "x2",
    upscale_model: "real-esrgan-x4plus:fp16",
    control_image: "CTRL",
    control_model: "controlnet-canny-sd15",
    control_scale: 0.8,
    enable_audio: false,
  };

  // A user who picks a source video, switches to t2a and hits Generate would
  // otherwise send hidden state the server refuses one field at a time.
  it("drops every field the audio-only pipeline cannot carry", () => {
    const request = stripAudioOnlyIncompatibleFields({
      ...loaded,
      pipeline: "t2a",
    });

    for (const field of [
      "source_image",
      "source_image_name",
      "strength",
      "mask_image",
      "source_video",
      "source_video_path",
      "audio_file",
      "audio_file_path",
      "extend_video",
      "extend_video_path",
      "extend_overlap_frames",
      "keyframes",
      "retake_range",
      "spatial_upscale",
      "temporal_upscale",
      "upscale_model",
      "control_image",
      "control_model",
      "control_scale",
      // Audio is what the pipeline renders; `false` can only contradict it.
      "enable_audio",
    ]) {
      expect(field in request).toBe(false);
    }

    // What the render actually needs is untouched — `frames` / `fps` are the
    // duration, and the prompt is the whole input.
    expect(request.prompt).toBe("heavy rain on a tin roof");
    expect(request.frames).toBe(121);
    expect(request.fps).toBe(24);
    expect(request.pipeline).toBe("t2a");
  });

  it("keeps an explicit enable_audio: true", () => {
    const request = stripAudioOnlyIncompatibleFields({
      pipeline: "t2a",
      enable_audio: true,
    });
    expect(request.enable_audio).toBe(true);
  });

  it("returns every other pipeline's request untouched", () => {
    const video = { ...loaded, pipeline: "two-stage" };
    expect(stripAudioOnlyIncompatibleFields(video)).toBe(video);
    const auto = { ...loaded, pipeline: null };
    expect(stripAudioOnlyIncompatibleFields(auto)).toBe(auto);
  });
});
