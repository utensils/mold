import { describe, expect, it } from "vitest";

import {
  DEFAULT_EXTEND_OVERLAP_FRAMES,
  canOfferExtend,
  extendNewFrames,
  extendOverlapOptions,
  extendValidationError,
  familySupportsExtend,
  serverExtendOverlapDefault,
} from "./extend";

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
});

describe("extendOverlapOptions", () => {
  it("offers 8k+1 values strictly below the clip length", () => {
    expect(extendOverlapOptions(41)).toEqual([1, 9, 17, 25, 33]);
    expect(extendOverlapOptions(9)).toEqual([1]);
  });

  it("offers nothing when there is no room for new frames", () => {
    expect(extendOverlapOptions(1)).toEqual([]);
    expect(extendOverlapOptions(null)).toEqual([]);
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
