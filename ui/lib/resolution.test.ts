import { describe, expect, it } from "vitest";
import {
  ASPECTS,
  RESOLUTIONS,
  dimsForMp,
  dimsLabel,
  nearestMp,
  swatchDims,
} from "./resolution";

describe("dimsForMp", () => {
  it("resolves 1 MP square to 1024×1024 (spec example)", () => {
    expect(dimsForMp(1, 1)).toEqual({ width: 1024, height: 1024 });
  });

  it("snaps every canonical shape × step to the server's /16 grid", () => {
    for (const aspect of ASPECTS) {
      for (const step of RESOLUTIONS) {
        const { width, height } = dimsForMp(step.mp, aspect.ratio);
        expect(width % 16, `${step.label} ${aspect.label} width`).toBe(0);
        expect(height % 16, `${step.label} ${aspect.label} height`).toBe(0);
        expect(width).toBeGreaterThanOrEqual(64);
        expect(height).toBeGreaterThanOrEqual(64);
      }
    }
  });

  it("tracks the megapixel budget within grid tolerance", () => {
    for (const aspect of ASPECTS) {
      const { width, height } = dimsForMp(1.8, aspect.ratio);
      expect((width * height) / 1e6).toBeGreaterThan(1.5);
      expect((width * height) / 1e6).toBeLessThanOrEqual(1.8);
    }
  });

  it("formats the label with the × separator", () => {
    expect(dimsLabel(1, 1)).toBe("1024×1024");
  });
});

describe("nearestMp", () => {
  it("maps 1024×1024 to the 1 MP step", () => {
    expect(nearestMp(1024, 1024)).toBe(1);
  });
  it("maps 704×704 to the 0.5 MP step", () => {
    expect(nearestMp(704, 704)).toBe(0.5);
  });
  it("maps 1328×1328 to the 1.8 MP step", () => {
    expect(nearestMp(1328, 1328)).toBe(1.8);
  });
});

describe("swatchDims", () => {
  it("uses the prototype's exact boxes for the five canonical shapes", () => {
    expect(swatchDims(1)).toEqual({ width: 24, height: 24 });
    expect(swatchDims(3 / 4)).toEqual({ width: 18, height: 24 });
    expect(swatchDims(4 / 3)).toEqual({ width: 24, height: 18 });
    expect(swatchDims(16 / 9)).toEqual({ width: 27, height: 15 });
    expect(swatchDims(9 / 16)).toEqual({ width: 15, height: 27 });
  });

  it("keeps arbitrary ratios proportional", () => {
    const { width, height } = swatchDims(2);
    expect(width / height).toBeCloseTo(2, 0);
  });
});
