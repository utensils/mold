import { describe, expect, it } from "vitest";
import {
  DEFAULT_SEQUENCE_CLIP_FRAMES,
  DEFAULT_VIDEO_FPS,
  defaultClipFrames,
  defaultSequenceStages,
  defaultVideoFps,
  formatFrameDuration,
  friendlySequenceError,
  modelSupportsSequence,
  modelsForOutput,
  sequenceDuration,
  sequenceFrameOptions,
  sequenceMotionTailFrames,
  sequenceValidation,
  transitionDescription,
  transitionLabel,
  type SequenceStage,
} from "./sequence";

describe("sequence authoring", () => {
  it("uses explicit server support when supplied", () => {
    expect(
      modelSupportsSequence({
        name: "ltx-2.3-22b-dev:fp8",
        family: "ltx2",
        supports_sequence: false,
      }),
    ).toBe(false);
    expect(
      modelSupportsSequence({
        name: "hf:example/one-stage",
        family: "ltx2",
        supports_sequence: true,
      }),
    ).toBe(true);
  });

  it("keeps older-server fallback conservative", () => {
    expect(
      modelSupportsSequence({
        name: "ltx-2-19b-dev:fp8",
        family: "ltx2",
      }),
    ).toBe(false);
    expect(
      modelSupportsSequence({
        name: "ltx-2-19b-distilled:fp8",
        family: "ltx2",
      }),
    ).toBe(true);
    expect(
      modelSupportsSequence({
        name: "hf:example/checkpoint",
        family: "ltx2",
      }),
    ).toBe(true);
    expect(
      modelSupportsSequence({
        name: "flux-dev:q4",
        family: "flux",
      }),
    ).toBe(false);
  });

  it("starts an explicit sequence with two clips", () => {
    expect(DEFAULT_SEQUENCE_CLIP_FRAMES).toBe(25);
    expect(defaultSequenceStages()).toEqual([
      { prompt: "", frames: 25, transition: "smooth" },
      { prompt: "", frames: 25, transition: "smooth" },
    ]);
    expect(defaultSequenceStages(65)).toEqual([
      { prompt: "", frames: 65, transition: "smooth" },
      { prompt: "", frames: 65, transition: "smooth" },
    ]);
  });

  it("uses context overlap only for model families that support handoff", () => {
    expect(
      sequenceMotionTailFrames({
        name: "ltx-video-0.9.8-2b-distilled:bf16",
        family: "ltx-video",
      }),
    ).toBe(0);
    expect(
      sequenceMotionTailFrames({
        name: "ltx-2.3-22b-distilled:fp8",
        family: "ltx2",
      }),
    ).toBe(17);
  });

  /**
   * Wan's carryover is a property of the checkpoint, not the family (#783):
   * it has no latent motion tail, so its handoff is last-frame image
   * conditioning, which only an image-conditioned checkpoint accepts.
   */
  it("reads wan's overlap from the checkpoint's own conditioning contract", () => {
    const wan = (source_image: string | null) =>
      sequenceMotionTailFrames({
        name: "wan22-t2v-a14b:q5",
        family: "wan",
        source_image,
      });
    expect(wan("optional")).toBe(17);
    expect(wan("required")).toBe(17);
    // A text-to-video checkpoint has no conditioning channel at all.
    expect(wan("unsupported")).toBe(0);
    // Unclassified is unknown, never an assumed handoff.
    expect(wan(null)).toBe(0);
  });

  it("offers only clip durations that exceed the active motion tail", () => {
    expect(sequenceFrameOptions(97, 17)).toEqual([
      25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
    ]);
    expect(sequenceFrameOptions(25, 0)).toEqual([9, 17, 25]);
  });

  /**
   * Wan's VAE compresses time by 4 where the LTX families compress by 8, so
   * its clip options are `4k+1`. Offering an `8k+1` option sends the request
   * into a 422 from the validator that owns the real rule.
   */
  it("offers wan clip durations on the 4k+1 grid", () => {
    expect(sequenceFrameOptions(25, 0, "wan")).toEqual([5, 9, 13, 17, 21, 25]);
    // The tail still filters, and every surviving option is on-grid.
    for (const frames of sequenceFrameOptions(121, 17, "wan")) {
      expect(frames).toBeGreaterThan(17);
      expect((frames - 1) % 4, `${frames} is off the 4k+1 grid`).toBe(0);
    }
    // An absent family keeps the LTX grid, so existing callers are unchanged.
    expect(sequenceFrameOptions(25, 0)).toEqual(
      sequenceFrameOptions(25, 0, "ltx2"),
    );
  });

  it("computes stitched duration using transition overlap", () => {
    const stages: SequenceStage[] = [
      { prompt: "one", frames: 97, transition: "smooth" },
      { prompt: "two", frames: 97, transition: "smooth" },
      { prompt: "three", frames: 97, transition: "fade", fade_frames: 8 },
    ];
    expect(sequenceDuration(stages, 24, 25)).toEqual({
      frames: 258,
      seconds: 258 / 24,
    });
  });

  it("requires two non-empty clips and respects server limits", () => {
    expect(
      sequenceValidation(
        [{ prompt: "only", frames: 97, transition: "smooth" }],
        { maxStages: 16, maxTotalFrames: 1552, motionTailFrames: 25 },
      ),
    ).toEqual(["Add at least two clips to make a sequence."]);

    expect(
      sequenceValidation(
        [
          { prompt: "one", frames: 97, transition: "smooth" },
          { prompt: " ", frames: 97, transition: "smooth" },
        ],
        { maxStages: 16, maxTotalFrames: 1552, motionTailFrames: 25 },
      ),
    ).toEqual(["Describe clip 2 before generating."]);
  });

  // The per-clip prompt gate is the only one in the codebase, so the optional
  // -prompt rule has to reach it through the limits the composer already
  // passes down — otherwise a conditioned LTX-2 sequence stays blocked on a
  // prompt the server would admit.
  it("allows blank clip prompts only when the limits mark them optional", () => {
    const stages: SequenceStage[] = [
      { prompt: "", frames: 97, transition: "smooth" },
      { prompt: " ", frames: 97, transition: "smooth" },
    ];
    expect(
      sequenceValidation(stages, {
        maxStages: 16,
        maxTotalFrames: 1552,
        motionTailFrames: 25,
        promptOptional: true,
      }),
    ).toEqual([]);
    expect(
      sequenceValidation(stages, {
        maxStages: 16,
        maxTotalFrames: 1552,
        motionTailFrames: 25,
        promptOptional: false,
      }),
    ).toEqual(["Describe clip 1 before generating."]);
  });

  it("still reports the other limits when clip prompts are optional", () => {
    expect(
      sequenceValidation(
        [
          { prompt: "", frames: 97, transition: "cut" },
          { prompt: "", frames: 97, transition: "cut" },
        ],
        {
          maxStages: 16,
          maxTotalFrames: 100,
          motionTailFrames: 25,
          promptOptional: true,
        },
      ),
    ).toEqual(["Reduce clip durations to 100 total frames or fewer."]);
  });

  it("uses the seam-pill transition vocabulary", () => {
    // Mold Studio spec §11: the seam control names transitions Smooth /
    // Cut / Fade; the zero-motion-tail fallback (LTX-Video) stays honest
    // as "Join" because no motion carries through.
    expect(transitionLabel("smooth")).toBe("Smooth");
    expect(transitionLabel("smooth", 0)).toBe("Join");
    expect(transitionLabel("cut")).toBe("Cut");
    expect(transitionLabel("fade")).toBe("Fade");
  });

  it("describes each transition for the seam editor rows", () => {
    expect(transitionDescription("smooth")).toBe("Motion carries through");
    expect(transitionDescription("smooth", 0)).toBe("Clips join end to end");
    expect(transitionDescription("cut")).toBe("Hard change of shot");
    expect(transitionDescription("fade")).toBe("Blend across the seam");
  });

  it("filters models by output mode", () => {
    const models = [
      { name: "flux-dev:q8", family: "flux" },
      { name: "ltx-video-0.9.6:bf16", family: "ltx-video" },
      { name: "ltx-2-19b-distilled:fp8", family: "ltx2" },
      { name: "ltx-2.3-22b-dev:fp8", family: "ltx2" },
    ];
    expect(modelsForOutput(models, "single")).toEqual(models);
    expect(modelsForOutput(models, "sequence").map((m) => m.name)).toEqual([
      "ltx-video-0.9.6:bf16",
      "ltx-2-19b-distilled:fp8",
    ]);
  });

  it("defaults clip frames from the model, then limits, then the floor", () => {
    const limits = { frames_per_clip_cap: 97, frames_per_clip_recommended: 97 };
    // Server-advertised model default wins.
    expect(defaultClipFrames({ default_frames: 97 }, limits, 17)).toBe(97);
    // LTX-Video ships 25.
    expect(
      defaultClipFrames(
        { default_frames: 25 },
        { frames_per_clip_cap: 97, frames_per_clip_recommended: 25 },
        0,
      ),
    ).toBe(25);
    // No model default → chain-limits recommendation.
    expect(
      defaultClipFrames(
        null,
        { frames_per_clip_cap: 97, frames_per_clip_recommended: 33 },
        17,
      ),
    ).toBe(33);
    // Nothing from the server → the conservative 25-frame floor.
    expect(defaultClipFrames(null, null, 17)).toBe(25);
    // Off-grid values snap DOWN onto the 8n+1 grid.
    expect(defaultClipFrames({ default_frames: 30 }, limits, 17)).toBe(25);
    // Values above the cap clamp to the cap.
    expect(defaultClipFrames({ default_frames: 500 }, limits, 17)).toBe(97);
    // Never at or below the motion tail — raised to the first valid option.
    expect(defaultClipFrames({ default_frames: 9 }, limits, 17)).toBe(25);
  });

  it("defaults fps from the model, then the current value, then the fallback", () => {
    // The server-advertised default wins — LTX-Video ships 30, LTX-2 24.
    expect(defaultVideoFps({ default_fps: 30 })).toBe(30);
    expect(defaultVideoFps({ default_fps: 24 })).toBe(24);
    // It wins over whatever the form is holding, exactly like steps/guidance.
    expect(defaultVideoFps({ default_fps: 30 }, 24)).toBe(30);
    // Nothing advertised (older server, image model) → the form value stands.
    expect(defaultVideoFps({}, 30)).toBe(30);
    expect(defaultVideoFps(null, 12)).toBe(12);
    // Nothing at all → the 24-fps fallback.
    expect(defaultVideoFps(null)).toBe(DEFAULT_VIDEO_FPS);
    expect(defaultVideoFps({ default_fps: null }, null)).toBe(24);
  });

  it("formats frame durations consistently across sequence surfaces", () => {
    expect(formatFrameDuration(97, 24)).toBe("97f · 4.0s");
    expect(formatFrameDuration(25, 30)).toBe("25f · 0.8s");
    expect(formatFrameDuration(97, 0)).toBe("97f · 4.0s");
  });

  it("turns GPU memory failures into actionable sequence recovery", () => {
    expect(
      friendlySequenceError(
        'DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")',
      ),
    ).toBe(
      "This sequence needs more GPU memory. Shorten the clip duration or reduce the size, then try again.",
    );
    expect(friendlySequenceError("model unavailable")).toBe(
      "model unavailable",
    );
    expect(
      friendlySequenceError("SSE request failed with HTTP 404", "plato"),
    ).toBe(
      "plato couldn’t find that model or job. It may have been removed or the host may have restarted.",
    );
  });
});
