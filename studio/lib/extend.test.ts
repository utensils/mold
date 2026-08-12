import { describe, expect, it } from "vitest";

import {
  DEFAULT_EXTEND_OVERLAP_FRAMES,
  canOfferExtend,
  extendNewFrames,
  extendOverlapOptions,
  extendValidationError,
  familySupportsExtend,
  resolveExtendOverlapFrames,
  serverExtendOverlapDefault,
} from "./extend";
import { WAN_HANDOFF_DUPLICATED_FRAMES } from "./chainRouting";
import { sequenceFrameStep } from "./sequence";

describe("canOfferExtend", () => {
  it("requires both the advertised capability and a capable family", () => {
    expect(canOfferExtend({ family: "ltx2", supports_extend: true })).toBe(
      true,
    );
    expect(canOfferExtend({ family: "ltx-2", supports_extend: true })).toBe(
      true,
    );
    expect(canOfferExtend({ family: "ltx-video", supports_extend: true })).toBe(
      false,
    );
    expect(canOfferExtend({ family: "flux", supports_extend: true })).toBe(
      false,
    );
  });

  // An older server omits `supports_extend` entirely, so offering the control
  // would only produce a rejected request. Absence must read as "no".
  it("treats a model that does not advertise extend as unsupported", () => {
    expect(canOfferExtend({ family: "ltx2" })).toBe(false);
    expect(canOfferExtend({ family: "ltx2", supports_extend: null })).toBe(
      false,
    );
    expect(canOfferExtend({ family: "ltx2", supports_extend: false })).toBe(
      false,
    );
    expect(canOfferExtend(null)).toBe(false);
    expect(canOfferExtend(undefined)).toBe(false);
  });

  it("normalizes family case and whitespace", () => {
    expect(familySupportsExtend("  LTX2 ")).toBe(true);
    expect(familySupportsExtend(null)).toBe(false);
  });

  /**
   * Wan's continuation is a property of the checkpoint, not the family (#783):
   * the source clip's final frame becomes the continuation's image
   * conditioning, which only an image-conditioned checkpoint accepts. The
   * server already answers that per model (`catalog::extend_capable_model`),
   * so the advertised field is the authority — an LTX-2-only family gate hid
   * the control on every wan checkpoint that can genuinely continue.
   */
  it("lets the server's per-model answer decide for wan", () => {
    expect(canOfferExtend({ family: "wan", supports_extend: true })).toBe(true);
    expect(canOfferExtend({ family: "wan", supports_extend: false })).toBe(
      false,
    );
    expect(canOfferExtend({ family: "wan" })).toBe(false);
    expect(familySupportsExtend("wan")).toBe(true);
  });
});

describe("serverExtendOverlapDefault", () => {
  it("prefers the server's advertised default", () => {
    expect(
      serverExtendOverlapDefault({ extend_default_overlap_frames: 25 }),
    ).toBe(25);
  });

  it("falls back when the server says nothing useful", () => {
    expect(serverExtendOverlapDefault(null)).toBe(
      DEFAULT_EXTEND_OVERLAP_FRAMES,
    );
    expect(
      serverExtendOverlapDefault({ extend_default_overlap_frames: 0 }),
    ).toBe(DEFAULT_EXTEND_OVERLAP_FRAMES);
  });

  // The advertised default is per family on the server, so a wan checkpoint
  // still reports LTX-2's 17 — a value `wan/pipeline.rs`'s `extend_inner`
  // refuses outright. Offering it as the pre-selected choice would make every
  // untouched wan continuation fail.
  it("uses wan's single carried frame however the server advertises it", () => {
    expect(
      serverExtendOverlapDefault({
        family: "wan",
        extend_default_overlap_frames: 17,
      }),
    ).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    expect(serverExtendOverlapDefault({ family: "wan" })).toBe(
      WAN_HANDOFF_DUPLICATED_FRAMES,
    );
  });
});

/**
 * The clamp above is only worth anything if the clamped value is the one that
 * leaves the browser. Every surface renders `resolveExtendOverlapFrames` and
 * every request builder submits it, so what is on screen is what goes on the
 * wire — an untouched control cannot leave the field absent and inherit a host
 * default the user never saw (#783 review).
 */
describe("resolveExtendOverlapFrames", () => {
  it("submits wan's single carried frame when the control is untouched", () => {
    expect(
      resolveExtendOverlapFrames(null, {
        family: "wan",
        extend_default_overlap_frames: 17,
      }),
    ).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    expect(resolveExtendOverlapFrames(undefined, { family: "wan" })).toBe(
      WAN_HANDOFF_DUPLICATED_FRAMES,
    );
  });

  it("keeps the host's advertised default for the LTX families", () => {
    expect(
      resolveExtendOverlapFrames(null, {
        family: "ltx2",
        extend_default_overlap_frames: 25,
      }),
    ).toBe(25);
    expect(resolveExtendOverlapFrames(null, null)).toBe(
      DEFAULT_EXTEND_OVERLAP_FRAMES,
    );
  });

  it("prefers an explicit choice over either default", () => {
    expect(
      resolveExtendOverlapFrames(9, {
        family: "ltx2",
        extend_default_overlap_frames: 17,
      }),
    ).toBe(9);
    // A nonsensical stored value is not a choice; the display falls back and
    // so must the wire, or the two disagree about what was submitted.
    expect(resolveExtendOverlapFrames(0, { family: "wan" })).toBe(
      WAN_HANDOFF_DUPLICATED_FRAMES,
    );
  });
});

describe("extendOverlapOptions", () => {
  it("offers 8k+1 values strictly below the clip length", () => {
    expect(extendOverlapOptions(41)).toEqual([1, 9, 17, 25, 33]);
    expect(extendOverlapOptions(9)).toEqual([1]);
    expect(extendOverlapOptions(41, "ltx2")).toEqual([1, 9, 17, 25, 33]);
  });

  it("offers nothing when there is no room for new frames", () => {
    expect(extendOverlapOptions(1)).toEqual([]);
    expect(extendOverlapOptions(null)).toEqual([]);
  });

  /**
   * Wan has no latent motion tail: the continuation is seeded with the
   * source's final frame and re-renders exactly that one frame, so
   * `extend_inner` refuses anything larger rather than trimming good frames.
   * The 8k+1 ladder offered 9 / 17 / 25 …, every one of which fails.
   */
  it("collapses wan to the one frame its continuation carries", () => {
    expect(extendOverlapOptions(53, "wan")).toEqual([
      WAN_HANDOFF_DUPLICATED_FRAMES,
    ]);
    expect(extendOverlapOptions(1, "wan")).toEqual([]);
  });
});

describe("extendValidationError", () => {
  const valid = { overlapFrames: 17, frames: 97 };

  it("accepts a well-formed continuation", () => {
    expect(extendValidationError(valid)).toBeNull();
  });

  /// The overlap re-encodes through the VAE's 8x causal temporal grid.
  it("rejects an overlap off the latent grid", () => {
    expect(extendValidationError({ ...valid, overlapFrames: 12 })).toContain(
      "8k+1",
    );
    for (const overlapFrames of [1, 9, 17, 25]) {
      expect(extendValidationError({ ...valid, overlapFrames })).toBeNull();
    }
  });

  it("rejects an overlap that leaves no new frames", () => {
    expect(extendValidationError({ overlapFrames: 25, frames: 25 })).toContain(
      "less than the clip length",
    );
    expect(extendValidationError({ overlapFrames: 17, frames: 25 })).toBeNull();
  });

  it("rejects competing conditioning inputs, matching the server's order", () => {
    expect(extendValidationError({ ...valid, hasSourceVideo: true })).toContain(
      "source video",
    );
    expect(extendValidationError({ ...valid, hasSourceImage: true })).toContain(
      "source image",
    );
    expect(extendValidationError({ ...valid, hasKeyframes: true })).toContain(
      "keyframes",
    );
  });

  // Wan's grid rule is not LTX-2's: it accepts exactly one frame, so the
  // 8k+1 wording would name a rule its engine does not have.
  it("names wan's single-frame carryover instead of the 8k+1 grid", () => {
    expect(
      extendValidationError({ overlapFrames: 9, frames: 53, family: "wan" }),
    ).toContain("exactly 1 frame");
    expect(
      extendValidationError({ overlapFrames: 17, frames: 53, family: "wan" }),
    ).toContain("exactly 1 frame");
    expect(
      extendValidationError({ overlapFrames: 1, frames: 53, family: "wan" }),
    ).toBeNull();
  });

  /**
   * One rule, one derivation: `extendOverlapOptions` enumerates the grid from
   * `sequenceFrameStep` and the validator must reject off it from the same
   * authority, or a family whose VAE compresses time differently gets an
   * option list and an error message that disagree.
   */
  it("rejects off the same grid the options are enumerated from", () => {
    for (const family of [null, "ltx2", "ltx-2"]) {
      const step = sequenceFrameStep(family);
      for (const overlapFrames of extendOverlapOptions(97, family)) {
        expect(
          extendValidationError({ overlapFrames, frames: 97, family }),
        ).toBeNull();
      }
      const offGrid = step * 2;
      expect(
        extendValidationError({ overlapFrames: offGrid, frames: 97, family }),
      ).toContain(`${step}k+1`);
    }
  });

  it("uses the shared default when the overlap is unset", () => {
    expect(
      extendValidationError({ overlapFrames: null, frames: 97 }),
    ).toBeNull();
    expect(extendValidationError({ overlapFrames: null, frames: 9 })).toContain(
      "less than the clip length",
    );
  });
});

describe("extendNewFrames", () => {
  it("reports the frames actually appended to the source", () => {
    expect(extendNewFrames(97, 17)).toBe(80);
    expect(extendNewFrames(25, 17)).toBe(8);
    expect(extendNewFrames(97, null)).toBe(97 - DEFAULT_EXTEND_OVERLAP_FRAMES);
    expect(extendNewFrames(null, 17)).toBeNull();
  });

  it("never reports negative growth", () => {
    expect(extendNewFrames(9, 17)).toBe(0);
  });
});
