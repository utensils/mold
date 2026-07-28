import { describe, expect, it } from "vitest";
import {
  DEFAULT_SEQUENCE_CLIP_FRAMES,
  defaultClipFrames,
  defaultSequenceStages,
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

  it("offers only clip durations that exceed the active motion tail", () => {
    expect(sequenceFrameOptions(97, 17)).toEqual([
      25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
    ]);
    expect(sequenceFrameOptions(25, 0)).toEqual([9, 17, 25]);
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

  it("uses the seam-pill transition vocabulary", () => {
    // Mold Studio spec §11: the seam control names transitions Smooth /
    // Cut / Fade; the zero-motion-tail fallback (LTX-Video) stays honest
    // as "Join clips" because no motion carries through.
    expect(transitionLabel("smooth")).toBe("Smooth");
    expect(transitionLabel("smooth", 0)).toBe("Join clips");
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
    expect(
      defaultClipFrames({ default_frames: 97 }, limits, 17),
    ).toBe(97);
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
      defaultClipFrames(null, { frames_per_clip_cap: 97, frames_per_clip_recommended: 33 }, 17),
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
  });
});
