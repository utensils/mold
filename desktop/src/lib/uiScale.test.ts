import { describe, expect, it } from "vitest";

import { nextUiScale, normalizeUiScale, UI_SCALE_LEVELS } from "./uiScale";

describe("UI scale", () => {
  it("uses a bounded, readable set of whole-app zoom levels", () => {
    expect(UI_SCALE_LEVELS).toEqual([80, 90, 100, 110, 120, 130]);
    expect(normalizeUiScale(5)).toBe(80);
    expect(normalizeUiScale(500)).toBe(130);
    expect(normalizeUiScale(114)).toBe(110);
  });

  it("steps, resets, and clamps at both ends", () => {
    expect(nextUiScale(100, "in")).toBe(110);
    expect(nextUiScale(100, "out")).toBe(90);
    expect(nextUiScale(120, "reset")).toBe(100);
    expect(nextUiScale(130, "in")).toBe(130);
    expect(nextUiScale(80, "out")).toBe(80);
  });
});
