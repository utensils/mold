import { describe, expect, it } from "vitest";
import {
  chainToToml,
  cycleTransition,
  durationSeconds,
  estimatedTotalFrames,
  frames8n1Error,
  isLtx2FrameCount,
  snapFrames,
} from "./chain";
import { chainFormToScript, newChainForm, newStage } from "./chainForm";
import type { ChainStage, TransitionMode } from "./api/types";

describe("frame-count validation", () => {
  it("accepts only 8n+1 counts", () => {
    for (const ok of [1, 9, 17, 25, 33, 97]) expect(isLtx2FrameCount(ok)).toBe(true);
    for (const bad of [0, 2, 8, 50, 96, 98, 100]) expect(isLtx2FrameCount(bad)).toBe(false);
  });

  it("returns the design-copy error for invalid counts", () => {
    expect(frames8n1Error(97)).toBeNull();
    expect(frames8n1Error(50)).toBe("Frames must be 8n+1 — try 97.");
  });

  it("snaps to the nearest valid count", () => {
    expect(snapFrames(50)).toBe(49);
    expect(snapFrames(52)).toBe(49);
    expect(snapFrames(54)).toBe(57);
    expect(snapFrames(0)).toBe(1);
    expect(snapFrames(97)).toBe(97);
  });
});

describe("estimatedTotalFrames", () => {
  const stages = (specs: [TransitionMode, number, number?][]): ChainStage[] =>
    specs.map(([transition, frames, fade]) => ({
      prompt: "x",
      frames,
      transition,
      ...(fade != null ? { fade_frames: fade } : {}),
    }));

  it("drops the motion tail on smooth continuations (matches chain.rs)", () => {
    // 97 + (97-25) + (97-25) = 241
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["smooth", 97],
          ["smooth", 97],
        ]),
        25,
      ),
    ).toBe(241);
  });

  it("keeps the whole clip after a cut", () => {
    // 97 + 97 (cut) + (97-25) (smooth) = 266
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["cut", 97],
          ["smooth", 97],
        ]),
        25,
      ),
    ).toBe(266);
  });

  it("nets minus fade_frames on a fade", () => {
    // 97 + 97 + (97-8) = 283
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["cut", 97],
          ["fade", 97, 8],
        ]),
        25,
      ),
    ).toBe(283);
  });

  it("computes duration at fps", () => {
    expect(durationSeconds(240, 24)).toBe(10);
    expect(durationSeconds(240, 0)).toBe(0);
  });
});

describe("cycleTransition", () => {
  it("cycles smooth → cut → fade → smooth", () => {
    expect(cycleTransition("smooth")).toBe("cut");
    expect(cycleTransition("cut")).toBe("fade");
    expect(cycleTransition("fade")).toBe("smooth");
  });
});

describe("chainToToml", () => {
  it("emits mold.chain.v1 with a [chain] table and one [[stage]] per stage", () => {
    const form = newChainForm();
    form.model = "ltx-2-19b-distilled:fp8";
    form.seed = "42";
    form.enableAudio = true;
    form.stages = [
      newStage("dawn over the sea"),
      { ...newStage("a storm rolls in"), transition: "fade", fadeFrames: 12 },
    ];
    const toml = chainToToml(chainFormToScript(form));

    expect(toml).toContain('schema = "mold.chain.v1"');
    expect(toml).toContain("[chain]");
    expect(toml).toContain('model = "ltx-2-19b-distilled:fp8"');
    expect(toml).toContain("seed = 42");
    expect(toml).toContain("enable_audio = true");
    expect(toml).toContain('output_format = "mp4"');
    // Two stages, first coerced to smooth.
    expect(toml.match(/\[\[stage\]\]/g)).toHaveLength(2);
    expect(toml).toContain('prompt = "dawn over the sea"');
    expect(toml).toContain('transition = "smooth"');
    expect(toml).toContain('transition = "fade"');
    expect(toml).toContain("fade_frames = 12");
  });

  it("omits seed and enable_audio when unset", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    const toml = chainToToml(chainFormToScript(form));
    expect(toml).not.toContain("seed =");
    expect(toml).not.toContain("enable_audio");
  });

  it("escapes quotes and newlines in prompts", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.stages = [newStage('a "quoted"\nprompt')];
    const toml = chainToToml(chainFormToScript(form));
    expect(toml).toContain('prompt = "a \\"quoted\\"\\nprompt"');
  });
});
