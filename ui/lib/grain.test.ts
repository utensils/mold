import { describe, expect, it } from "vitest";
import { foldSeed, grainParams, mulberry32, tint } from "./grain";

describe("mulberry32 / foldSeed", () => {
  it("is deterministic for the same seed", () => {
    const a = mulberry32(foldSeed(4203968117));
    const b = mulberry32(foldSeed(4203968117));
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });

  it("differs across seeds — every job's grain is unique", () => {
    const a = mulberry32(foldSeed("42"));
    const b = mulberry32(foldSeed("43"));
    expect(a()).not.toBe(b());
  });
});

describe("grainParams", () => {
  it("is monotonic while developing: amplitude falls, cell grows, warmth rises", () => {
    let prev = grainParams(0, "developing");
    for (const p of [0.25, 0.5, 0.75, 1]) {
      const cur = grainParams(p, "developing");
      expect(cur.amplitude).toBeLessThan(prev.amplitude);
      expect(cur.cell).toBeGreaterThan(prev.cell);
      expect(cur.warmth).toBeGreaterThan(prev.warmth);
      prev = cur;
    }
  });

  it("latent is cold, coarse-free, and high-amplitude", () => {
    expect(grainParams(0, "latent")).toEqual({
      amplitude: 0.9,
      cell: 2,
      warmth: 0,
    });
  });

  it("fixed clears the grain; stopped keeps it visible and cold", () => {
    expect(grainParams(1, "fixed").amplitude).toBe(0);
    const stopped = grainParams(0.4, "stopped");
    expect(stopped.amplitude).toBeGreaterThan(0);
    expect(stopped.warmth).toBe(0);
  });

  it("clamps progress", () => {
    expect(grainParams(-1, "developing")).toEqual(grainParams(0, "developing"));
    expect(grainParams(2, "developing")).toEqual(grainParams(1, "developing"));
  });
});

describe("tint", () => {
  it("interpolates cold → warm and clamps", () => {
    expect(tint(0)).toEqual({ r: 147, g: 167, b: 176 });
    expect(tint(1)).toEqual({ r: 240, g: 138, b: 36 });
    expect(tint(5)).toEqual(tint(1));
  });

  it("stop-bath overrides warmth", () => {
    expect(tint(0.8, true)).toEqual({ r: 201, g: 79, b: 61 });
  });
});
