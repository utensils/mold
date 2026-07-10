import { describe, expect, it } from "vitest";
import { fitAspectRatio } from "./fitAspectRatio";

describe("fitAspectRatio", () => {
  it.each([
    [1200, 600, 1, 1, 600, 600],
    [1200, 600, 2, 3, 400, 600],
    [800, 600, 16, 9, 800, 450],
    [320, 180, 9, 16, 101, 180],
  ])(
    "keeps %sx%s content inside a %sx%s region",
    (containerWidth, containerHeight, contentWidth, contentHeight, width, height) => {
      const fitted = fitAspectRatio(containerWidth, containerHeight, contentWidth, contentHeight);

      expect(fitted).toEqual({ width, height });
      expect(fitted.width).toBeLessThanOrEqual(containerWidth);
      expect(fitted.height).toBeLessThanOrEqual(containerHeight);
    },
  );

  it("collapses safely until the preview region has measurable space", () => {
    expect(fitAspectRatio(0, 600, 1, 1)).toEqual({ width: 0, height: 0 });
    expect(fitAspectRatio(800, 0, 1, 1)).toEqual({ width: 0, height: 0 });
  });
});
