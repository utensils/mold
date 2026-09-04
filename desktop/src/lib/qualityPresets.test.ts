import { describe, expect, it } from "vitest";
import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";
import { activeQualityPreset, qualityPresets } from "./qualityPresets";

function steps(overrides: Partial<IntegerControl> = {}): IntegerControl {
  return { default: 28, min: 8, max: 50, step: 1, mode: "adjustable", ...overrides };
}

describe("qualityPresets", () => {
  it("takes Draft, Good and Best from the recipe's own floor, default and ceiling", () => {
    expect(qualityPresets(steps())).toEqual([
      { key: "draft", label: "Draft", steps: 8 },
      { key: "good", label: "Good", steps: 28 },
      { key: "best", label: "Best", steps: 50 },
    ]);
  });

  it("drops a row the recipe cannot distinguish", () => {
    expect(qualityPresets(steps({ min: 8, default: 8, max: 40 }))).toEqual([
      { key: "draft", label: "Draft", steps: 8 },
      { key: "best", label: "Best", steps: 40 },
    ]);
  });

  it("offers nothing for a recipe that pins its steps, or that has no range", () => {
    expect(qualityPresets(steps({ mode: "fixed", min: 8, default: 8, max: 8 }))).toEqual([]);
    expect(qualityPresets(steps({ min: 20, default: 20, max: 20 }))).toEqual([]);
    expect(qualityPresets(null)).toEqual([]);
  });
});

describe("activeQualityPreset", () => {
  it("reads the row the current step count sits on", () => {
    const presets = qualityPresets(steps());
    expect(activeQualityPreset(presets, 50)).toBe("best");
    expect(activeQualityPreset(presets, 28)).toBe("good");
  });

  it("lights nothing for a step count between the rows", () => {
    expect(activeQualityPreset(qualityPresets(steps()), 31)).toBeNull();
  });
});
