import { describe, expect, it } from "vitest";
import { requestVideoOnly, videoOnlyBlockedReason } from "./videoOnly";

const clear = {
  audioEnabled: false,
  audioOnlyPipeline: false,
  hasConditioningAudio: false,
  isExtend: false,
};

describe("videoOnly policy", () => {
  it("admits the plain opt-in", () => {
    expect(videoOnlyBlockedReason(clear)).toBeNull();
    expect(requestVideoOnly(true, clear)).toBe(true);
  });

  it("stays absent when not opted in", () => {
    expect(requestVideoOnly(false, clear)).toBeUndefined();
  });

  it("names each conflict and keeps the field off the wire", () => {
    for (const inputs of [
      { ...clear, audioEnabled: true },
      { ...clear, audioOnlyPipeline: true },
      { ...clear, hasConditioningAudio: true },
      { ...clear, isExtend: true },
    ]) {
      expect(videoOnlyBlockedReason(inputs)).toBeTruthy();
      expect(requestVideoOnly(true, inputs)).toBeUndefined();
    }
  });

  it("prefers the pipeline explanation over the audio toggle", () => {
    expect(
      videoOnlyBlockedReason({
        ...clear,
        audioEnabled: true,
        audioOnlyPipeline: true,
      }),
    ).toContain("Text-to-audio");
  });
});
