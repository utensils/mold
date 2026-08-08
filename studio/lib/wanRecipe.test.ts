import { describe, expect, it } from "vitest";
import { baseGenerationCapabilities } from "./generationCapabilities";
import {
  MAX_WAN_DISTILL_STRENGTH,
  distillStrengthError,
  emptyWanRecipe,
  sampleShiftError,
  wanRecipeCount,
  wanRecipeError,
  wanRecipeFromWire,
  wanRecipeToWire,
} from "./wanRecipe";

const fastTier = baseGenerationCapabilities(
  "wan",
  "wan22-t2v-a14b:q5",
).wanRecipe;
const qualityTier = baseGenerationCapabilities(
  "wan",
  "wan22-t2v-a14b:q8",
).wanRecipe;
const offFamily = baseGenerationCapabilities("ltx2").wanRecipe;

describe("wanRecipeToWire", () => {
  it("omits every field while the controls are untouched", () => {
    // The load-bearing invariant: an empty control keeps the tier's own shift
    // and distill strengths, which only happens while the field is absent.
    expect(wanRecipeToWire(emptyWanRecipe(), fastTier)).toEqual({});
    expect(wanRecipeCount(emptyWanRecipe())).toBe(0);
  });

  it("carries only the fields the user touched", () => {
    expect(
      wanRecipeToWire({ ...emptyWanRecipe(), sampleShift: 12 }, fastTier),
    ).toEqual({ sample_shift: 12 });
    expect(
      wanRecipeToWire(
        { sampleShift: 5, distillStrengthHigh: 1.8, distillStrengthLow: 1 },
        fastTier,
      ),
    ).toEqual({
      sample_shift: 5,
      distill_strength_high: 1.8,
      distill_strength_low: 1,
    });
  });

  it("drops distill strengths on a tier that ships no distill", () => {
    // The engine rejects a strength addressed at an empty slot, so the quality
    // tier keeps the shift and loses the pair rather than sending a 422.
    expect(
      wanRecipeToWire(
        { sampleShift: 8, distillStrengthHigh: 2, distillStrengthLow: 2 },
        qualityTier,
      ),
    ).toEqual({ sample_shift: 8 });
  });

  it("drops everything off-family", () => {
    expect(
      wanRecipeToWire(
        { sampleShift: 8, distillStrengthHigh: 2, distillStrengthLow: 2 },
        offFamily,
      ),
    ).toEqual({});
  });

  it("omits a value the server would reject", () => {
    // The surface's submit gate reports these through `wanRecipeError`; the
    // serializer never smuggles an out-of-band value onto the wire.
    expect(
      wanRecipeToWire(
        {
          sampleShift: 0,
          distillStrengthHigh: 0,
          distillStrengthLow: MAX_WAN_DISTILL_STRENGTH + 1,
        },
        fastTier,
      ),
    ).toEqual({});
  });
});

describe("wanRecipe validation", () => {
  it("accepts any finite positive flow shift", () => {
    expect(sampleShiftError(null)).toBeNull();
    expect(sampleShiftError(12)).toBeNull();
    expect(sampleShiftError(0.5)).toBeNull();
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(sampleShiftError(bad)).toBe(
        "Flow shift must be a positive number.",
      );
    }
  });

  it("holds distill strengths to the accepted band", () => {
    expect(distillStrengthError(null, "High-noise")).toBeNull();
    expect(
      distillStrengthError(MAX_WAN_DISTILL_STRENGTH, "High-noise"),
    ).toBeNull();
    expect(distillStrengthError(0, "High-noise")).toBe(
      "High-noise distill strength must be greater than 0.",
    );
    expect(distillStrengthError(4.5, "Low-noise")).toBe(
      "Low-noise distill strength must be at most 4.",
    );
  });

  it("reports the first offending control for the submit gate", () => {
    expect(wanRecipeError(emptyWanRecipe())).toBeNull();
    expect(wanRecipeError(null)).toBeNull();
    expect(
      wanRecipeError({
        sampleShift: -2,
        distillStrengthHigh: 9,
        distillStrengthLow: null,
      }),
    ).toBe("Flow shift must be a positive number.");
    expect(
      wanRecipeError({
        sampleShift: 5,
        distillStrengthHigh: 9,
        distillStrengthLow: null,
      }),
    ).toBe("High-noise distill strength must be at most 4.");
  });
});

describe("wanRecipeFromWire", () => {
  it("round-trips through saved metadata", () => {
    const state = {
      sampleShift: 12,
      distillStrengthHigh: 1.8,
      distillStrengthLow: 0.9,
    };
    expect(wanRecipeFromWire(wanRecipeToWire(state, fastTier))).toEqual(state);
    expect(wanRecipeCount(state)).toBe(3);
  });

  it("restores an absent or partial record as untouched controls", () => {
    expect(wanRecipeFromWire(null)).toEqual(emptyWanRecipe());
    expect(wanRecipeFromWire({})).toEqual(emptyWanRecipe());
    expect(wanRecipeFromWire({ sample_shift: 7 })).toEqual({
      sampleShift: 7,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });
});
