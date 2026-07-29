import { describe, expect, it } from "vitest";
import { PANEL_LIMITS, clampPanelWidth, dragWidth, normalizePanelWidth } from "./panelResize";

describe("PANEL_LIMITS", () => {
  it("keeps each panel's default inside its own min/max range", () => {
    for (const limits of Object.values(PANEL_LIMITS)) {
      expect(limits.min).toBeLessThanOrEqual(limits.def);
      expect(limits.def).toBeLessThanOrEqual(limits.max);
    }
  });

  it("fits all five Generate aspect controls on one row at the default width", () => {
    const inspectorPadding = 18 * 2;
    const aspectControls = 52 * 5;
    const aspectGaps = 7 * 4;
    expect(PANEL_LIMITS.generateParams.def - inspectorPadding).toBeGreaterThanOrEqual(
      aspectControls + aspectGaps,
    );
  });
});

describe("clampPanelWidth", () => {
  it.each([
    ["navRail", 100, 160], // below min
    ["navRail", 160, 160], // at min
    ["navRail", 250, 250], // in range
    ["navRail", 320, 320], // at max
    ["navRail", 9999, 320], // above max
    ["generateParams", 10, 280],
    ["generateParams", 280, 280],
    ["generateParams", 333, 333],
    ["generateParams", 480, 480],
    ["generateParams", 4000, 480],
    ["historyDrawer", 10, 420],
    ["historyDrawer", 680, 680],
    ["historyDrawer", 4000, 960],
  ] as const)("clamps %s width %d to %d", (panel, px, expected) => {
    expect(clampPanelWidth(panel, px)).toBe(expected);
  });

  it("rounds fractional widths to whole pixels", () => {
    expect(clampPanelWidth("navRail", 200.4)).toBe(200);
    expect(clampPanelWidth("navRail", 200.6)).toBe(201);
    expect(clampPanelWidth("generateParams", 320.5)).toBe(321);
  });
});

describe("dragWidth", () => {
  it("grows a right-edge handle with +dx (nav rail)", () => {
    expect(dragWidth("navRail", 208, 30, "right")).toBe(238);
    expect(dragWidth("navRail", 208, -20, "right")).toBe(188);
  });

  it("grows a left-edge handle with -dx (generate inspector)", () => {
    // The inspector sits on the right side of the window: dragging its left
    // edge leftwards (negative dx) makes the panel wider.
    expect(dragWidth("generateParams", 320, -40, "left")).toBe(360);
    expect(dragWidth("generateParams", 320, 25, "left")).toBe(295);
    expect(dragWidth("historyDrawer", 620, -80, "left")).toBe(700);
  });

  it("clamps the dragged width to the panel's limits", () => {
    expect(dragWidth("navRail", 208, -500, "right")).toBe(160);
    expect(dragWidth("navRail", 208, 500, "right")).toBe(320);
    expect(dragWidth("generateParams", 320, -500, "left")).toBe(480);
    expect(dragWidth("generateParams", 320, 500, "left")).toBe(280);
  });
});

describe("normalizePanelWidth", () => {
  it.each([null, undefined, Number.NaN, Infinity, -Infinity, "300", {}, true])(
    "falls back to the default for %o",
    (raw) => {
      expect(normalizePanelWidth("navRail", raw)).toBe(PANEL_LIMITS.navRail.def);
      expect(normalizePanelWidth("generateParams", raw)).toBe(PANEL_LIMITS.generateParams.def);
      expect(normalizePanelWidth("historyDrawer", raw)).toBe(PANEL_LIMITS.historyDrawer.def);
    },
  );

  it("clamps out-of-range persisted values", () => {
    expect(normalizePanelWidth("navRail", 1)).toBe(PANEL_LIMITS.navRail.min);
    expect(normalizePanelWidth("navRail", 100000)).toBe(PANEL_LIMITS.navRail.max);
    expect(normalizePanelWidth("generateParams", 1)).toBe(PANEL_LIMITS.generateParams.min);
    expect(normalizePanelWidth("generateParams", 100000)).toBe(PANEL_LIMITS.generateParams.max);
    expect(normalizePanelWidth("historyDrawer", 1)).toBe(PANEL_LIMITS.historyDrawer.min);
    expect(normalizePanelWidth("historyDrawer", 100000)).toBe(PANEL_LIMITS.historyDrawer.max);
  });

  it("passes valid persisted widths through (rounded)", () => {
    expect(normalizePanelWidth("navRail", 260)).toBe(260);
    expect(normalizePanelWidth("generateParams", 400.6)).toBe(401);
    expect(normalizePanelWidth("historyDrawer", 700.4)).toBe(700);
  });
});
