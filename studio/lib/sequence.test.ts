import { describe, expect, it } from "vitest";
import {
  DEFAULT_SEQUENCE_CLIP_FRAMES,
  defaultSequenceStages,
  friendlySequenceError,
  modelSupportsSequence,
  sequenceDuration,
  sequenceFrameOptions,
  sequenceMotionTailFrames,
  sequenceValidation,
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

  it("uses plain-language transition labels", () => {
    expect(transitionLabel("smooth")).toBe("Continue motion");
    expect(transitionLabel("smooth", 0)).toBe("Join clips");
    expect(transitionLabel("cut")).toBe("Cut");
    expect(transitionLabel("fade")).toBe("Crossfade");
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
