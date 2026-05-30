import { describe, expect, it } from "vitest";
import {
  describeSourceFit,
  maskPaddingRectangles,
  resolveSourceFitTransform,
} from "./sourceFit";

describe("source fit policies", () => {
  it("defaults to pad repaint, preserving requested dimensions and repainting added pixels", () => {
    const transform = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "pad-repaint" },
    );

    expect(transform).toMatchObject({
      outputWidth: 1024,
      outputHeight: 1024,
      drawWidth: 1024,
      drawHeight: 768,
      offsetX: 0,
      offsetY: 128,
      maskPaddedPixels: true,
    });
    expect(maskPaddingRectangles(transform)).toEqual([
      { x: 0, y: 0, width: 1024, height: 128 },
      { x: 0, y: 896, width: 1024, height: 128 },
    ]);
  });

  it("supports crop fill with explicit side alignment", () => {
    const transform = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "crop-fill", alignX: "left", alignY: "center" },
    );

    expect(transform).toMatchObject({
      outputWidth: 1024,
      outputHeight: 1024,
      drawWidth: 1365,
      drawHeight: 1024,
      offsetX: 0,
      offsetY: 0,
      maskPaddedPixels: false,
    });
  });

  it("describes prefit upscaler policies for submit preprocessing", () => {
    expect(
      describeSourceFit({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "pad-repaint" },
      }),
    ).toContain("real-esrgan-x2plus:fp16");
  });
});
