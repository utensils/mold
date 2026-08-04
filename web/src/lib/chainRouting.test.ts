import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOTION_TAIL,
  LTX2_DEFAULT_CLIP_FRAMES,
  MAX_CHAIN_STAGES,
  decideChainRouting,
  decideGenerateRequestRouting,
} from "./chainRouting";

describe("decideChainRouting", () => {
  it("returns single when frames is null/undefined/zero", () => {
    expect(
      decideChainRouting(null, "ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toEqual({ kind: "single" });
    expect(
      decideChainRouting(undefined, "ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(0, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
  });

  it("returns single for ltx2-distilled at-or-below the cap", () => {
    expect(
      decideChainRouting(
        LTX2_DEFAULT_CLIP_FRAMES,
        "ltx2",
        "ltx-2.3-22b-distilled:fp8",
      ),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(25, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual(
      { kind: "single" },
    );
  });

  it("chains ltx2-distilled requests above the cap", () => {
    // 241 frames, clip=97, DEFAULT_MOTION_TAIL=17 → effective=80,
    // remainder=144, stageCount = 1 + ceil(144/80) = 1 + 2 = 3.
    const d = decideChainRouting(241, "ltx2", "ltx-2.3-22b-distilled:fp8");
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
  });

  it("chains installed catalog LTX-2 checkpoints whose stable id has no pipeline label", () => {
    const d = decideChainRouting(241, "ltx2", "cv:3143864");
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
  });

  it("chain stage count matches Rust normalise() expectations", () => {
    // Mirrors crates/mold-core/src/chain.rs test cases:
    //   (400, 97, 4, 5)  — 97 + 4*93 = 469 ≥ 400
    //   (200, 97, 4, 3)  — 97 + 2*93 = 283 ≥ 200
    //   (97,  97, 4, 1)  — single clip hits 97 exactly (handled as "single")
    expect(
      decideChainRouting(400, "ltx2", "ltx-2.3-22b-distilled:fp8", 4),
    ).toMatchObject({ kind: "chain", stageCount: 5 });
    expect(
      decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 4),
    ).toMatchObject({ kind: "chain", stageCount: 3 });
  });

  it("chains every ltx2 checkpoint past the single-clip budget", () => {
    // Chain capability is a property of the family: every LTX-2 pipeline
    // renders sequence clips, so a dev checkpoint routes exactly like a
    // distilled one. It used to be rejected outright.
    for (const model of [
      "ltx-2-19b:fp8",
      "ltx-2-19b-dev:fp8",
      "ltx-2.3-22b-dev:fp8",
      "ltx-2.3-22b-distilled:fp8",
    ]) {
      expect(decideChainRouting(489, "ltx2", model).kind).toBe("chain");
    }
  });

  it("rejects entirely-unknown families when frames exceed the single-clip budget", () => {
    const d = decideChainRouting(241, "flux", "flux-schnell:q4");
    expect(d.kind).toBe("reject");
  });

  it("stays single when frames are within budget regardless of family", () => {
    expect(decideChainRouting(49, "ltx-video", "ltx-video-0.9.6:bf16")).toEqual(
      { kind: "single" },
    );
    expect(decideChainRouting(97, "ltx2", "ltx-2-19b:fp8")).toEqual({
      kind: "single",
    });
  });

  it("chains ltx-video models above the cap with motion_tail=0 (no context handoff)", () => {
    // ltx-video has no img2vid path on the server, so motion_tail is forced
    // to 0 — the SPA mirrors that. 241 frames @ clip=97, tail=0 →
    // effective=97, remainder=144, stageCount = 1 + ceil(144/97) = 1 + 2 = 3.
    const d = decideChainRouting(
      241,
      "ltx-video",
      "ltx-video-0.9.8-13b-dev:bf16",
    );
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: 0,
      stageCount: 3,
    });
  });

  it("ignores caller-supplied motionTail for ltx-video (zeroed server-side)", () => {
    const d = decideChainRouting(
      300,
      "ltx-video",
      "ltx-video-0.9.8-13b-distilled:bf16",
      17,
    );
    expect(d).toMatchObject({ kind: "chain", motionTail: 0 });
  });

  it("rejects when motion tail is >= clip frames (only relevant for ltx2 chains)", () => {
    const d = decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 97);
    expect(d.kind).toBe("reject");
  });

  it("enforces the server's sixteen-stage chain ceiling", () => {
    expect(
      decideChainRouting(1297, "ltx2", "ltx-2-19b-distilled:fp8"),
    ).toMatchObject({
      kind: "chain",
      stageCount: MAX_CHAIN_STAGES,
    });
    expect(
      decideChainRouting(1305, "ltx2", "ltx-2-19b-distilled:fp8"),
    ).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("at most 1297 frames"),
    });
  });

  it("keeps temporal x2 on ordinary generation up to the duration budget", () => {
    // Temporal x2 halves the stage-1 frames AND the stage-1 fps, so it renders
    // the same runtime — the ceiling is the plain fps-derived one, not double.
    const request = {
      model: "ltx-2-19b:fp8",
      frames: 481,
      fps: 24,
      temporal_upscale: "x2" as const,
    };
    expect(decideGenerateRequestRouting(request, "ltx2")).toEqual({
      kind: "single",
    });
    expect(
      decideGenerateRequestRouting({ ...request, frames: 489 }, "ltx2"),
    ).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("484 frames"),
    });
  });

  it("moves the temporal x2 ceiling with fps", () => {
    const at12 = {
      model: "ltx-2-19b:fp8",
      frames: 249,
      fps: 12,
      temporal_upscale: "x2" as const,
    };
    expect(decideGenerateRequestRouting(at12, "ltx2")).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("244 frames"),
    });
    expect(decideGenerateRequestRouting({ ...at12, fps: 24 }, "ltx2")).toEqual({
      kind: "single",
    });
  });

  it("preserves advanced LTX-2 inputs by staying single-shot inside the model budget", () => {
    expect(
      decideGenerateRequestRouting(
        {
          frames: 153,
          fps: 24,
          model: "ltx-2.3-22b-distilled:fp8",
          negative_prompt: "blurry",
          guidance_overrides: { stg_scale: 1.5 },
        },
        "ltx2",
      ),
    ).toEqual({
      kind: "single",
      preservedAutoChainFields: ["negative_prompt", "guidance_overrides"],
    });
  });

  it("treats video continuation as single-shot instead of dropping it during chaining", () => {
    expect(
      decideGenerateRequestRouting(
        {
          frames: 153,
          fps: 24,
          model: "ltx-2.3-22b-distilled:fp8",
          extend_video: "base64-video",
          extend_overlap_frames: 17,
        },
        "ltx2",
      ),
    ).toEqual({
      kind: "single",
      preservedAutoChainFields: ["extend_video"],
    });
  });

  it("returns single when family is missing", () => {
    expect(decideChainRouting(50, null, "anything")).toEqual({
      kind: "single",
    });
    expect(decideChainRouting(50, undefined, "anything")).toEqual({
      kind: "single",
    });
  });
});
