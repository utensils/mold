import { describe, expect, it } from "vitest";

import { showsAlphaBed } from "./alphaMedia";

describe("showsAlphaBed", () => {
  it("draws the checkerboard for a print whose stored file carries alpha", () => {
    expect(showsAlphaBed({ metadata: { has_alpha: true } })).toBe(true);
  });

  it("draws it for a transparent-background request even before the file fact", () => {
    // A queue row / in-flight print only has the REQUEST; its result will be
    // transparent, so the bed belongs behind it from the first preview.
    expect(showsAlphaBed({ metadata: { transparent_background: true } })).toBe(
      true,
    );
  });

  it("keeps every other print on the plain media bed", () => {
    expect(showsAlphaBed({ metadata: {} })).toBe(false);
    expect(showsAlphaBed({ metadata: { has_alpha: false } })).toBe(false);
    expect(showsAlphaBed({ metadata: null })).toBe(false);
    expect(showsAlphaBed({})).toBe(false);
    expect(showsAlphaBed(null)).toBe(false);
    expect(showsAlphaBed(undefined)).toBe(false);
  });
});
