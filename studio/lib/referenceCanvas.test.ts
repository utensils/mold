import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  clampCanvasToLimits,
  fitToTargetAreaTiesEven,
  lastReferenceCanvas,
  referenceCanvasSize,
  roundHalfToEven,
  stagedReferenceDimensions,
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

/** Qwen Image 2.1's advertised `resolution` bounds. */
const qwen21 = {
  alignment: 32,
  min_width: 64,
  min_height: 64,
  max_pixels: 2400 * 1792,
  max_axis_pixels: 2752,
};

describe("lastReferenceCanvas", () => {
  // The SAME goldens `mold_core::validation`'s
  // `last_reference_canvas_clamps_a_panorama_into_the_recipe` pins.
  const goldens: [[number, number], [number, number]][] = [
    [
      [8000, 1000],
      [2752, 320],
    ],
    [
      [1000, 8000],
      [320, 2752],
    ],
    [
      [7300, 1000],
      [2752, 384],
    ],
    [
      [100, 1],
      [2752, 64],
    ],
    [
      [100_000, 1],
      [2752, 64],
    ],
    [
      [1, 100_000],
      [64, 2752],
    ],
    [
      [1920, 1080],
      [1376, 768],
    ],
  ];
  it.each(goldens)("%j -> %j", ([w, h], [ew, eh]) => {
    expect(lastReferenceCanvas(w, h, qwen21)).toEqual({
      width: ew,
      height: eh,
    });
  });

  it("honours the pixel ceiling and leaves a fitting canvas alone", () => {
    expect(clampCanvasToLimits(2400, 1792, qwen21)).toEqual({
      width: 2400,
      height: 1792,
    });
    expect(clampCanvasToLimits(2752, 2752, qwen21)).toEqual({
      width: 2048,
      height: 2048,
    });
  });
});

describe("referenceCanvasSize", () => {
  const base = {
    canvas: "last-reference" as const,
    resolution: qwen21,
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

  it("keeps upstream's 1024x1024 area, not the reference's own size", () => {
    expect(
      referenceCanvasSize({
        ...base,
        references: [{ width: 4000, height: 3000 }],
      }),
    ).toEqual({ width: 1184, height: 896 });
  });

  it("clamps a panorama into the recipe's bounds", () => {
    expect(
      referenceCanvasSize({
        ...base,
        references: [{ width: 8000, height: 1000 }],
      }),
    ).toEqual({ width: 2752, height: 320 });
  });

  it("leaves the canvas alone with no references at all", () => {
    // A restored draft hydrating with an empty strip must keep its canvas.
    expect(referenceCanvasSize({ ...base, references: [] })).toBeNull();
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

describe("stagedReferenceDimensions", () => {
  it("prefers recorded sizes and falls back to the header", () => {
    // 1170x2532 PNG IHDR.
    const png = btoa(
      String.fromCharCode(
        0x89,
        0x50,
        0x4e,
        0x47,
        0x0d,
        0x0a,
        0x1a,
        0x0a,
        0x00,
        0x00,
        0x00,
        0x0d,
        0x49,
        0x48,
        0x44,
        0x52,
        0x00,
        0x00,
        0x04,
        0x92,
        0x00,
        0x00,
        0x09,
        0xe4,
      ),
    );
    expect(
      stagedReferenceDimensions([
        { base64: "ignored", width: 640, height: 480 },
        { data: png },
        { base64: "" },
      ]),
    ).toEqual([
      { width: 640, height: 480 },
      { width: 1170, height: 2532 },
      null,
    ]);
  });

  it("reads a rotated phone photo upright, over the picker's stored size", () => {
    // Pillow-written 96x48 pixels with EXIF Orientation = 6 — the file
    // `mold_core::reference_image` pins too.
    const relative =
      "crates/mold-core/testdata/reference_orientation/landscape_96x48_orientation6.jpg";
    let directory = process.cwd();
    while (!existsSync(resolve(directory, relative))) {
      if (dirname(directory) === directory) throw new Error(relative);
      directory = dirname(directory);
    }
    const bytes = readFileSync(resolve(directory, relative)).toString("base64");
    const [read] = stagedReferenceDimensions([
      { base64: bytes, width: 96, height: 48 },
    ]);
    expect(read).toEqual({ width: 48, height: 96 });
    expect(
      referenceCanvasSize({
        canvas: "last-reference",
        references: [read!],
        resolution: qwen21,
        intent: "model-default",
      }),
    ).toEqual(lastReferenceCanvas(48, 96, qwen21));
  });
});
