import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  fitToTargetAreaTiesEven,
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
        defaults: { width: 1024, height: 1024 },
        alignment: 32,
        intent: "model-default",
      }),
    ).toEqual(fitToTargetAreaTiesEven(48, 96, 1024 * 1024, 32));
  });
});
