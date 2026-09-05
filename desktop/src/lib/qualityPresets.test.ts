import { describe, expect, it } from "vitest";
import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";
import { activeQualityPreset, qualityPresets } from "./qualityPresets";

function steps(overrides: Partial<IntegerControl> = {}): IntegerControl {
  return {
    default: 25,
    min: 8,
    max: 50,
    step: 1,
    mode: "adjustable",
    recommended: [13, 25, 38],
    ...overrides,
  };
}

describe("qualityPresets", () => {
  it("takes Draft, Good and Best from the recipe's own recommended ladder", () => {
    expect(qualityPresets(steps())).toEqual([
      { key: "draft", label: "Draft", steps: 13 },
      { key: "good", label: "Good", steps: 25 },
      { key: "best", label: "Best", steps: 38 },
    ]);
  });

  it("never reaches past the ladder to the control's own floor and ceiling", () => {
    const rows = qualityPresets(steps());
    expect(rows.map((row) => row.steps)).not.toContain(8);
    expect(rows.map((row) => row.steps)).not.toContain(50);
  });

  it("drops a row the ladder cannot distinguish", () => {
    expect(qualityPresets(steps({ default: 2, recommended: [2, 2, 4] }))).toEqual([
      { key: "draft", label: "Draft", steps: 2 },
      { key: "best", label: "Best", steps: 4 },
    ]);
  });

  it("offers nothing for a recipe that pins its steps, or whose ladder has one rung", () => {
    expect(qualityPresets(steps({ mode: "fixed", default: 4, recommended: [4] }))).toEqual([]);
    expect(qualityPresets(steps({ default: 20, recommended: [20] }))).toEqual([]);
    expect(qualityPresets(null)).toEqual([]);
  });

  /** A host older than the ladder advertises no `recommended` at all; the
   *  rows stay away rather than inventing a ladder of their own. */
  it("offers nothing when the recipe advertises no ladder", () => {
    const { recommended: _dropped, ...withoutLadder } = steps();
    expect(qualityPresets(withoutLadder)).toEqual([]);
  });
});

describe("activeQualityPreset", () => {
  it("reads the row the current step count sits on", () => {
    const presets = qualityPresets(steps());
    expect(activeQualityPreset(presets, 38)).toBe("best");
    expect(activeQualityPreset(presets, 25)).toBe("good");
  });

  it("lights nothing for a step count between the rows", () => {
    expect(activeQualityPreset(qualityPresets(steps()), 31)).toBeNull();
  });
});
