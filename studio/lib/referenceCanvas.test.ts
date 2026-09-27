import { describe, expect, it } from "vitest";

import {
  fitToTargetAreaTiesEven,
  referenceCanvasSize,
  roundHalfToEven,
} from "./referenceCanvas";

describe("roundHalfToEven", () => {
  it("rounds halves to the even neighbour like Python's round()", () => {
    expect(roundHalfToEven(0.5)).toBe(0);
    expect(roundHalfToEven(1.5)).toBe(2);
    expect(roundHalfToEven(2.5)).toBe(2);
    expect(roundHalfToEven(32.5)).toBe(32);
    expect(roundHalfToEven(33.5)).toBe(34);
    expect(roundHalfToEven(2.4999)).toBe(2);
    expect(roundHalfToEven(2.5001)).toBe(3);
  });
});

describe("fitToTargetAreaTiesEven", () => {
  // The SAME goldens `mold_core::validation`'s
  // `ties_even_fit_matches_upstream_calculate_dimensions` pins: diffusers
  // `calculate_dimensions(1024 * 1024, w / h)`
  // (`pipeline_qwenimage21.py:149-156`) under CPython 3.13. A client and the
  // engine may never land on different sides of a tie.
  const goldens: [[number, number], [number, number]][] = [
    [
      [4225, 4096],
      [1024, 1024],
    ],
    [
      [4096, 4225],
      [1024, 1024],
    ],
    [
      [1600, 900],
      [1376, 768],
    ],
    [
      [1024, 1024],
      [1024, 1024],
    ],
    [
      [3000, 2000],
      [1248, 832],
    ],
    [
      [1, 1],
      [1024, 1024],
    ],
    [
      [640, 480],
      [1184, 896],
    ],
    [
      [1080, 1920],
      [768, 1376],
    ],
  ];

  it("matches upstream calculate_dimensions on every golden", () => {
    for (const [[w, h], [ew, eh]] of goldens) {
      expect(
        fitToTargetAreaTiesEven(w, h, 1024 * 1024, 32),
        `${w}x${h}`,
      ).toEqual({ width: ew, height: eh });
    }
  });

  it("never produces a zero-size axis for a degenerate aspect", () => {
    expect(fitToTargetAreaTiesEven(100_000, 1, 1024 * 1024, 32).height).toBe(
      32,
    );
  });
});

describe("referenceCanvasSize", () => {
  const base = {
    canvas: "last-reference" as const,
    defaults: { width: 1024, height: 1024 },
    alignment: 32,
    intent: "model-default" as const,
  };

  it("sizes the default canvas from the LAST reference", () => {
    expect(
      referenceCanvasSize({
        ...base,
        references: [
          { width: 1024, height: 1024 },
          { width: 1600, height: 900 },
        ],
      }),
    ).toEqual({ width: 1376, height: 768 });
  });

  it("keeps the recipe default area, not the reference's own size", () => {
    expect(
      referenceCanvasSize({
        ...base,
        defaults: { width: 1024, height: 1024 },
        references: [{ width: 4000, height: 3000 }],
      }),
    ).toEqual({ width: 1184, height: 896 });
  });

  it("returns to the recipe default when the strip empties", () => {
    expect(referenceCanvasSize({ ...base, references: [] })).toEqual({
      width: 1024,
      height: 1024,
    });
  });

  it("never moves a canvas the user chose", () => {
    expect(
      referenceCanvasSize({
        ...base,
        intent: "manual",
        references: [{ width: 1600, height: 900 }],
      }),
    ).toBeNull();
  });

  it("does nothing for a recipe with no canvas rule (or an older server)", () => {
    expect(
      referenceCanvasSize({
        ...base,
        canvas: null,
        references: [{ width: 1600, height: 900 }],
      }),
    ).toBeNull();
  });

  it("waits rather than guessing when the last reference's size is unreadable", () => {
    expect(
      referenceCanvasSize({
        ...base,
        references: [{ width: 1600, height: 900 }, null],
      }),
    ).toBeNull();
  });
});
