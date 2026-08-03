import { describe, expect, it } from "vitest";
import {
  clampVideoFrames,
  formatVideoDuration,
  framesForVideoDuration,
  maxVideoFrames,
} from "./videoDuration";

const ltx2 = {
  default_frames: 97,
  default_fps: 24,
  max_frames: 481,
  max_runtime_seconds: 20,
  max_frames_absolute: 604,
  frame_step: 8,
};

describe("video duration controls", () => {
  it("maps seconds onto the selected model's valid frame grid", () => {
    expect(framesForVideoDuration(10, 24, ltx2)).toBe(241);
    expect(formatVideoDuration(241, 24)).toBe("10s");
  });

  it("recomputes duration-based model ceilings at the selected fps", () => {
    expect(maxVideoFrames(ltx2, 12)).toBe(241);
    expect(maxVideoFrames(ltx2, 24)).toBe(481);
    expect(maxVideoFrames(ltx2, 48)).toBe(601);
  });

  it("uses fixed per-model caps for frame-limited video models", () => {
    const ltxVideo = { max_frames: 257, frame_step: 8 };
    expect(maxVideoFrames(ltxVideo, 24)).toBe(257);
    expect(maxVideoFrames(ltxVideo, 60)).toBe(257);
  });

  it("retains the LTX-2 duration budget for older hosts", () => {
    expect(maxVideoFrames({ family: "ltx2" }, 24)).toBe(481);
  });

  it("clamps manual values to the requestable model grid", () => {
    expect(clampVideoFrames(999, 24, ltx2)).toBe(481);
    expect(clampVideoFrames(1, 24, ltx2)).toBe(1);
  });
});
