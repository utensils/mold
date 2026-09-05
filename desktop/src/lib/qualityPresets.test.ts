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

  it("offers nothing for a recipe that pins its steps", () => {
    expect(qualityPresets(steps({ mode: "fixed", default: 4, recommended: [4] }))).toEqual([]);
    expect(qualityPresets(null)).toEqual([]);
  });

  /**
   * A host older than the ladder advertises one rung, or none at all — every
   * 0.27.x server does, and a remote-only style would lose the group entirely.
   * An absent additive field means an OLDER SERVER, never "no choice offered"
   * (the supports_strength lesson), so the client stands in with the profile's
   * own formula: half, default, one and a half times, clamped into the
   * control's bounds — never the raw floor or ceiling.
   */
  it("stands in for a host older than the ladder with the profile's own formula", () => {
    expect(qualityPresets(steps({ default: 20, recommended: [20] })).map((p) => p.steps)).toEqual([
      10, 20, 30,
    ]);
    const { recommended: _dropped, ...withoutLadder } = steps({ default: 4, min: 1, max: 100 });
    expect(qualityPresets(withoutLadder).map((p) => p.steps)).toEqual([2, 4, 6]);
    // Clamped into the control's bounds, then deduped, exactly as the host does.
    expect(qualityPresets(steps({ default: 9, min: 8, max: 12, recommended: [9] }))).toEqual([
      { key: "draft", label: "Draft", steps: 8 },
      { key: "good", label: "Good", steps: 9 },
      { key: "best", label: "Best", steps: 12 },
    ]);
    expect(qualityPresets(steps({ default: 4, min: 4, max: 4, recommended: [4] }))).toEqual([]);
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
