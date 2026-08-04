import { describe, expect, it } from "vitest";
import { AUDIO_ONLY_PIPELINE, isAudioOnlyPipeline } from "./ltx2Pipeline";

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
