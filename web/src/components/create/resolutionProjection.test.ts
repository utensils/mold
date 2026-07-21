import { describe, expect, it } from "vitest";
import { dimsForMp } from "@ui/lib/resolution";
import { aspectIdForDims, projectResolution } from "./resolutionProjection";

describe("aspectIdForDims", () => {
  it("recognizes the canonical shapes from their exact projections", () => {
    const square = dimsForMp(1, 1);
    expect(aspectIdForDims(square.width, square.height)).toBe("square");
    const portrait = dimsForMp(1, 3 / 4);
    expect(aspectIdForDims(portrait.width, portrait.height)).toBe("portrait");
    const landscape = dimsForMp(1, 4 / 3);
    expect(aspectIdForDims(landscape.width, landscape.height)).toBe(
      "landscape",
    );
  });

  it("returns null for a custom size", () => {
    expect(aspectIdForDims(1000, 300)).toBeNull();
    expect(aspectIdForDims(0, 0)).toBeNull();
  });
});

describe("projectResolution", () => {
  it("round-trips the canonical grid without flagging custom", () => {
    for (const mp of [0.5, 1, 2]) {
      const { width, height } = dimsForMp(mp, 1);
      const proj = projectResolution(width, height);
      expect(proj.aspectId).toBe("square");
      expect(proj.mp).toBe(mp);
      expect(proj.isCustom).toBe(false);
    }
  });

  it("flags a custom ratio as custom", () => {
    const proj = projectResolution(1000, 300);
    expect(proj.aspectId).toBeNull();
    expect(proj.isCustom).toBe(true);
  });

  it("flags an off-grid 1:1 size as custom with no shape highlighted", () => {
    // Square ratio but not a canonical MP projection (1024 / 1408 / 704).
    const proj = projectResolution(1200, 1200);
    expect(proj.aspectId).toBeNull();
    expect(proj.isCustom).toBe(true);
  });
});
