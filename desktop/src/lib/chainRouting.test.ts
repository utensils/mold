import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOTION_TAIL,
  LTX2_DEFAULT_CLIP_FRAMES,
  MAX_CHAIN_STAGES,
  buildAutoChainRequest,
  buildGenerationEstimateRequest,
  decideChainRouting,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
} from "./chainRouting";
import type { GenerateRequest } from "./api/types";

describe("decideChainRouting", () => {
  it("returns single when frames is null/undefined/zero", () => {
    expect(decideChainRouting(null, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
    expect(decideChainRouting(undefined, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
    expect(decideChainRouting(0, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
  });

  it("returns single for ltx2-distilled at-or-below the cap", () => {
    expect(
      decideChainRouting(LTX2_DEFAULT_CLIP_FRAMES, "ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(25, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
  });

  it("chains ltx2-distilled requests one past the cap boundary (98)", () => {
    // 98 frames, clip=97, tail=17 → remainder=1, stageCount = 1 + ceil(1/80) = 2.
    const d = decideChainRouting(LTX2_DEFAULT_CLIP_FRAMES + 1, "ltx2", "ltx-2.3-22b-distilled:fp8");
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 2,
    });
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
    expect(decideChainRouting(400, "ltx2", "ltx-2.3-22b-distilled:fp8", 4)).toMatchObject({
      kind: "chain",
      stageCount: 5,
    });
    expect(decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 4)).toMatchObject({
      kind: "chain",
      stageCount: 3,
    });
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
    expect(decideChainRouting(49, "ltx-video", "ltx-video-0.9.6:bf16")).toEqual({
      kind: "single",
    });
    expect(decideChainRouting(97, "ltx2", "ltx-2-19b:fp8")).toEqual({
      kind: "single",
    });
  });

  it("chains ltx-video models above the cap with motion_tail=0 (no context handoff)", () => {
    // ltx-video has no img2vid path on the server, so motion_tail is forced
    // to 0 — the client mirrors that. 241 frames @ clip=97, tail=0 →
    // effective=97, remainder=144, stageCount = 1 + ceil(144/97) = 1 + 2 = 3.
    const d = decideChainRouting(241, "ltx-video", "ltx-video-0.9.8-13b-dev:bf16");
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: 0,
      stageCount: 3,
    });
  });

  it("ignores caller-supplied motionTail for ltx-video (zeroed server-side)", () => {
    const d = decideChainRouting(300, "ltx-video", "ltx-video-0.9.8-13b-distilled:bf16", 17);
    expect(d).toMatchObject({ kind: "chain", motionTail: 0 });
  });

  it("rejects when motion tail is >= clip frames (only relevant for ltx2 chains)", () => {
    const d = decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 97);
    expect(d.kind).toBe("reject");
  });

  it("enforces the server's sixteen-stage chain ceiling", () => {
    expect(decideChainRouting(1297, "ltx2", "ltx-2-19b-distilled:fp8")).toMatchObject({
      kind: "chain",
      stageCount: MAX_CHAIN_STAGES,
    });
    expect(decideChainRouting(1305, "ltx2", "ltx-2-19b-distilled:fp8")).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("at most 1297 frames"),
    });
    expect(decideChainRouting(1552, "ltx-video", "ltx-video:bf16")).toMatchObject({
      kind: "chain",
      stageCount: MAX_CHAIN_STAGES,
    });
    expect(decideChainRouting(1553, "ltx-video", "ltx-video:bf16")).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("at most 1552 frames"),
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

  it("treats the 'ltx-2' family alias like 'ltx2' (chains distilled above the cap)", () => {
    // The server/manifest emit "ltx2", but the desktop capabilities layer also
    // accepts the "ltx-2" alias — chain routing must recognise it too.
    expect(decideChainRouting(241, "ltx-2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
  });

  it("normalizes whitespace and case in the family before matching", () => {
    expect(decideChainRouting(241, "  LTX2  ", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
    expect(decideChainRouting(241, "  LTX-2  ", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
  });

  it("chains through the 'ltx-2' family alias too", () => {
    // The alias canonicalizes to ltx2, which is chain-capable regardless of
    // whether the checkpoint name says "distilled".
    const d = decideChainRouting(489, "ltx-2", "ltx-2-19b:fp8");
    expect(d.kind).toBe("chain");
  });
});

describe("request-aware generation routing", () => {
  const request: GenerateRequest = {
    prompt: "a continuous crane shot",
    model: "ltx-2-19b:fp8",
    width: 768,
    height: 512,
    steps: 20,
    guidance: 3,
    batch_size: 8,
    output_format: "mp4",
    frames: 257,
    fps: 24,
    temporal_upscale: "x2",
  };

  it("keeps temporal x2 on ordinary generation up to the duration budget", () => {
    // Temporal x2 halves the stage-1 frames AND the stage-1 fps, so it renders
    // the same runtime — the ceiling is the plain fps-derived one, not double.
    expect(decideGenerateRequestRouting(request, "ltx2")).toEqual({ kind: "single" });
    expect(decideGenerateRequestRouting({ ...request, frames: 489 }, "ltx2")).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("484 frames"),
    });
  });

  it("moves the temporal x2 ceiling with fps", () => {
    expect(
      decideGenerateRequestRouting({ ...request, fps: 12, frames: 249 }, "ltx2"),
    ).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("244 frames"),
    });
    expect(decideGenerateRequestRouting({ ...request, fps: 12, frames: 241 }, "ltx2")).toEqual({
      kind: "single",
    });
  });

  it("estimates one sibling using the concrete render selected by safe routing", () => {
    const advancedSingle = {
      ...request,
      model: "ltx-2-19b-distilled:fp8",
      frames: 241,
      source_video: "VIDEO",
      keyframes: [{ frame: 0, image: "IMAGE" }],
      audio_file: "AUDIO",
      control_image: "CONTROL",
      control_model: "controlnet-canny-sd15",
      control_scale: 0.8,
    };
    delete advancedSingle.temporal_upscale;
    expect(buildGenerationEstimateRequest(advancedSingle, "ltx2")).toMatchObject({
      batch_size: 1,
      frames: 241,
    });
    const estimate = buildGenerationEstimateRequest(advancedSingle, "ltx2");
    expect(estimate).not.toHaveProperty("source_video");
    expect(estimate).not.toHaveProperty("keyframes");
    expect(estimate).not.toHaveProperty("audio_file");
    expect(estimate).not.toHaveProperty("control_image");
    expect(estimate).not.toHaveProperty("control_model");
    expect(estimate).not.toHaveProperty("control_scale");
    const compatibleChain: GenerateRequest = { ...advancedSingle };
    delete compatibleChain.source_video;
    delete compatibleChain.keyframes;
    delete compatibleChain.audio_file;
    expect(buildGenerationEstimateRequest(compatibleChain, "ltx2")).toMatchObject({
      batch_size: 1,
      frames: 97,
    });
    expect(buildGenerationEstimateRequest(request, "ltx2")).toMatchObject({
      batch_size: 1,
      frames: 257,
      temporal_upscale: "x2",
    });
  });
});

describe("automatic chain request projection", () => {
  const request: GenerateRequest = {
    prompt: "a lighthouse in a storm",
    negative_prompt: "text",
    model: "ltx-2.3-22b-distilled:fp8",
    width: 1536,
    height: 640,
    steps: 24,
    guidance: 3.5,
    seed: 42,
    batch_size: 1,
    output_format: "mp4",
    source_image: "source-b64",
    source_image_name: "source.png",
    strength: 0.7,
    frames: 241,
    fps: 24,
    enable_audio: true,
    cfg_plus: true,
    scheduler: "unipc",
    original_prompt: "a lighthouse at dusk",
    batch_id: "prepared-batch-1",
    batch_index: 2,
    batch_count: 3,
  };

  it("builds only fields accepted by the server auto-expand form", () => {
    const decision = decideChainRouting(request.frames, "ltx2", request.model);
    expect(decision.kind).toBe("chain");
    if (decision.kind !== "chain") throw new Error("expected chain routing");

    expect(buildAutoChainRequest(request, decision)).toEqual({
      model: request.model,
      prompt: request.prompt,
      total_frames: 241,
      clip_frames: 97,
      motion_tail_frames: 17,
      width: 1536,
      height: 640,
      fps: 24,
      seed: 42,
      steps: 24,
      guidance: 3.5,
      strength: 0.7,
      output_format: "mp4",
      source_image: "source-b64",
      enable_audio: true,
      original_prompt: "a lighthouse at dusk",
      batch_id: "prepared-batch-1",
      batch_index: 2,
      batch_count: 3,
    });
  });

  it("reports every advanced field that auto-expand cannot preserve", () => {
    expect(
      unsupportedAutoChainFields({
        ...request,
        loras: [
          { path: "/models/style.safetensors", scale: 0.8 },
          { path: "camera-control:dolly-in", scale: 1 },
        ],
        audio_file: "audio-b64",
        source_video: "video-b64",
        keyframes: [{ frame: 0, image: "frame-b64" }],
        pipeline: "retake",
        retake_range: { start_seconds: 1, end_seconds: 2 },
        spatial_upscale: "x2",
        temporal_upscale: "x2",
        guidance_overrides: { stg_scale: 1.5 },
      }),
    ).toEqual([
      "negative_prompt",
      "loras",
      "audio_file",
      "source_video",
      "keyframes",
      "pipeline",
      "retake_range",
      "spatial_upscale",
      "temporal_upscale",
      "guidance_overrides",
    ]);
  });

  it("keeps an in-budget long request single-shot when chaining would lose settings", () => {
    const decision = decideGenerateRequestRouting(
      {
        ...request,
        frames: 153,
        negative_prompt: "text",
        loras: [{ path: "/models/style.safetensors", scale: 0.8 }],
      },
      "ltx2",
    );

    expect(decision).toEqual({
      kind: "single",
      preservedAutoChainFields: ["negative_prompt", "loras"],
    });
  });

  it("rejects incompatible chaining only past the single-shot cap", () => {
    expect(
      decideGenerateRequestRouting({ ...request, frames: 489, negative_prompt: "text" }, "ltx2"),
    ).toEqual({
      kind: "reject",
      reason:
        "489 frames exceeds the 481-frame single-shot limit at 24 fps, and automatic chaining can’t preserve negative prompt. Reduce Frames, remove that option, or author a Sequence with compatible per-clip settings.",
    });
  });

  it("ignores empty optional values and reports singular LoRA compatibility", () => {
    expect(
      unsupportedAutoChainFields({
        ...request,
        negative_prompt: "  ",
        lora: { path: "/models/legacy.safetensors", scale: 1 },
      }),
    ).toEqual(["loras"]);
  });

  it("does not report guidance overrides when every wire value is unset", () => {
    expect(
      unsupportedAutoChainFields({
        ...request,
        guidance_overrides: {
          stg_scale: null,
          stg_blocks: [],
          rescale_scale: null,
          modality_scale: null,
          skip_step: null,
        },
      }),
    ).toEqual(["negative_prompt"]);

    expect(
      unsupportedAutoChainFields({
        ...request,
        guidance_overrides: { stg_scale: 0 },
      }),
    ).toEqual(["negative_prompt", "guidance_overrides"]);
  });
});
