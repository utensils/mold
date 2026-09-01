import { describe, expect, it } from "vitest";
import {
  expansionContextForRequest,
  expansionTaskForRequest,
} from "./expandTask";

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

  it("classifies wan as a video family (twin of the Rust resolver)", () => {
    expect(expansionTaskForRequest("wan", {})).toBe("text-to-video");
    expect(
      expansionTaskForRequest("wan", { source_image: "opening-frame" }),
    ).toBe("image-to-video");
  });

  it("classifies a single-frame wan render as image work (#798)", () => {
    expect(expansionTaskForRequest("wan", { frames: 1 })).toBe("text-to-image");
    // A source-conditioned one-frame request keeps its source-preserving
    // contract: the source image stays the visual authority.
    expect(
      expansionTaskForRequest("wan", { frames: 1, source_image: "still" }),
    ).toBe("image-to-video");
    expect(expansionTaskForRequest("wan", { frames: 81 })).toBe(
      "text-to-video",
    );
    // The still contract is wan's: LTX keeps its video classification.
    expect(expansionTaskForRequest("ltx2", { frames: 1 })).toBe(
      "text-to-video",
    );
  });

  it("distinguishes H3 ordered-reference resynthesis from FL2VA", () => {
    expect(expansionTaskForRequest("minimax-h3", {})).toBe("text-to-video");
    expect(
      expansionTaskForRequest("minimax-h3", {
        source_image: "opening-frame",
      }),
    ).toBe("image-to-video");
    expect(
      expansionTaskForRequest("minimax-h3", {
        keyframes: [{ frame: 361, image: "closing-frame" }],
      }),
    ).toBe("image-to-video");
    expect(
      expansionTaskForRequest("minimax-h3", {
        source_image: "opening-frame",
        keyframes: [{ frame: 361, image: "closing-frame" }],
      }),
    ).toBe("keyframe-interpolation");
    expect(
      expansionTaskForRequest("minimax-h3", {
        references: [{ kind: "image" }, { kind: "audio" }],
      }),
    ).toBe("reference-to-audio-video");
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

describe("expansionContextForRequest", () => {
  it("mirrors mold_core::ExpandContext::for_generation", () => {
    expect(
      expansionContextForRequest("wan", {
        model: "wan22-i2v-a14b:q5",
        width: 832,
        height: 480,
        frames: 81,
        fps: 16,
        enable_audio: false,
        source_image: "png",
        loras: [{ path: "/adapters/paper-boat.safetensors" }],
      }),
    ).toEqual({
      model: "wan22-i2v-a14b:q5",
      width: 832,
      height: 480,
      frames: 81,
      fps: 16,
      audio: false,
      references: [{ kind: "image", role: "first-frame" }],
      loras: ["paper-boat"],
    });
  });

  it("keeps ordered H3 references and marks a video soundtrack", () => {
    expect(
      expansionContextForRequest("minimax-h3", {
        model: "minimax-h3-ref2va:comfy-pruned-int8",
        references: [
          { kind: "video", has_audio: true },
          { kind: "image" },
          { kind: "audio" },
        ],
      }).references,
    ).toEqual([
      { kind: "video", has_audio: true, role: "reference" },
      { kind: "image", role: "reference" },
      { kind: "audio", role: "reference" },
    ]);
  });

  it("omits absent facts and classifies image-family sources", () => {
    expect(expansionContextForRequest("flux", {})).toEqual({});
    expect(
      expansionContextForRequest("qwen-image-edit", {
        source_image: "png",
        edit_images: ["png"],
        id_image: "png",
      }).references,
    ).toEqual([
      { kind: "image", role: "edit" },
      { kind: "image", role: "edit" },
      { kind: "image", role: "identity" },
    ]);
    expect(
      expansionContextForRequest("sdxl", { source_image: "png" }).references,
    ).toEqual([{ kind: "image", role: "source" }]);
  });
});
