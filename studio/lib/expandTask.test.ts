import { describe, expect, it } from "vitest";
import { expansionTaskForRequest } from "./expandTask";

describe("expansionTaskForRequest", () => {
  it("keeps image families on image expansion", () => {
    expect(expansionTaskForRequest("flux", { source_image: "bytes" })).toBe(
      "text-to-image",
    );
  });

  it("detects text and image conditioned video", () => {
    expect(expansionTaskForRequest("ltx-video", {})).toBe("text-to-video");
    expect(
      expansionTaskForRequest("ltx2", { source_image: "opening-frame" }),
    ).toBe("image-to-video");
  });

  it("prioritizes retake, keyframes, and audio conditioning", () => {
    expect(
      expansionTaskForRequest("ltx2", {
        source_video: "video",
        retake_range: { start: 1, end: 2 },
      }),
    ).toBe("retake");
    expect(
      expansionTaskForRequest("ltx2", {
        source_image: "image",
        keyframes: [{ frame: 0 }],
      }),
    ).toBe("image-to-video");
    expect(
      expansionTaskForRequest("ltx2", {
        source_image: "image",
        keyframes: [{ frame: 0 }, { frame: 8 }],
      }),
    ).toBe("keyframe-interpolation");
    expect(
      expansionTaskForRequest("ltx2", {
        source_video: "video",
        audio_file: "audio",
      }),
    ).toBe("audio-driven-video");
    expect(
      expansionTaskForRequest("ltx2", {
        source_video: "speaking video",
        pipeline: "lip-dub",
      }),
    ).toBe("audio-driven-video");
  });

  it("recognizes continuation and audio-only pipelines", () => {
    expect(
      expansionTaskForRequest("ltx2", { extend_video_path: "/clip.mp4" }),
    ).toBe("video-to-video");
    expect(expansionTaskForRequest("ltx2", { pipeline: "t2a" })).toBe(
      "text-to-audio",
    );
  });
});
