import { describe, expect, it } from "vitest";
import {
  clampVideoFrames,
  formatVideoDuration,
  framesForVideoDuration,
  maxVideoFrames,
  snapVideoFrames,
  videoFramesError,
  videoFrameStep,
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

  it("snaps wan onto its 4n+1 grid, not the legacy 8n+1 default", () => {
    // The load-bearing case: a wan row that names its family but carries no
    // `frame_step`. The legacy fallback is 8, and wan's grid is 4, so without
    // the family arm every count that is 4n+1 but not 8n+1 gets rewritten —
    // 49 becomes 41 — or submitted and rejected by the server's validator.
    const wan = { family: "wan" };
    expect(videoFrameStep(wan)).toBe(4);
    expect(snapVideoFrames(49, wan)).toBe(49);
    expect(snapVideoFrames(81, wan)).toBe(81);
    expect(clampVideoFrames(49, 16, wan)).toBe(49);

    // The same counts read on the legacy 8-step grid get moved: 45 is a valid
    // wan length and an invalid LTX-2 one, so the 8-step rounds it away.
    expect(snapVideoFrames(45, wan)).toBe(45);
    expect(snapVideoFrames(45, { family: "ltx2" })).toBe(49);
    expect(snapVideoFrames(53, wan)).toBe(53);
    expect(snapVideoFrames(53, { family: "ltx2" })).toBe(57);
  });

  it("validates frame counts against the selected model grid", () => {
    expect(videoFramesError(45, { family: "wan" })).toBeNull();
    expect(videoFramesError(45, { family: "ltx2" })).toBe(
      "Frames must be 8n+1 — try 41 or 49.",
    );
    expect(videoFramesError(46, { family: "wan" })).toBe(
      "Frames must be 4n+1 — try 45 or 49.",
    );
  });

  it("prefers a server-advertised frame_step over the family fallback", () => {
    // The server stays authoritative: a host that advertises the grid wins,
    // including if a future wan checkpoint changes it.
    expect(videoFrameStep({ family: "wan", frame_step: 8 })).toBe(8);
    expect(videoFrameStep({ family: "flux" })).toBe(8);
    expect(videoFrameStep(null)).toBe(8);
  });

  it("gives wan the flat 257 ceiling on a host that advertises no max", () => {
    const wan = { family: "wan", default_fps: 16 };
    expect(maxVideoFrames(wan, 16)).toBe(257);
    // Flat, not duration-derived: unchanged across fps, unlike LTX-2.
    expect(maxVideoFrames(wan, 30)).toBe(257);
    expect((maxVideoFrames(wan, 16) - 1) % 4).toBe(0);
  });

  it("clamps manual values to the requestable model grid", () => {
    expect(clampVideoFrames(999, 24, ltx2)).toBe(481);
    expect(clampVideoFrames(1, 24, ltx2)).toBe(1);
    expect(formatVideoDuration(1, 24)).toBe("0.04s");
  });
});
