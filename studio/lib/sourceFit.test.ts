import { describe, expect, it } from "vitest";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  describeSourceFit,
  maskPaddingRectangles,
  parseSourceFitPolicy,
  resolveSourceFitTransform,
  sourceFitPolicyForMode,
} from "./sourceFit";

it("keeps crop fill as the cross-surface source-image default", () => {
  expect(defaultSourceFitPolicy()).toEqual({ mode: "crop-fill" });
});

describe("sourceFitPolicyForMode", () => {
  it("seeds upscale-then-fit with crop fill even when repaint is supported", () => {
    expect(
      sourceFitPolicyForMode("upscale-then-fit", {
        supportsMask: true,
        upscalerModel: "real-esrgan-x4plus:fp16",
      }),
    ).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
  });

  it("builds maskless sequence policies without unrepaintable padding", () => {
    expect(
      sourceFitPolicyForMode("upscale-then-fit", {
        supportsMask: false,
        upscalerModel: "real-esrgan-x4plus:fp16",
      }),
    ).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
    expect(
      sourceFitPolicyForMode("pad-repaint", { supportsMask: false }),
    ).toEqual({ mode: "crop-fill", alignX: "center", alignY: "center" });
  });
});

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

  it("pads left/right for a tall source and masks the side bands", () => {
    const transform = resolveSourceFitTransform(
      { width: 480, height: 640 },
      { width: 1024, height: 1024 },
      { mode: "pad-repaint" },
    );

    expect(transform).toMatchObject({
      drawWidth: 768,
      drawHeight: 1024,
      offsetX: 128,
      offsetY: 0,
      maskPaddedPixels: true,
    });
    expect(maskPaddingRectangles(transform)).toEqual([
      { x: 0, y: 0, width: 128, height: 1024 },
      { x: 896, y: 0, width: 128, height: 1024 },
    ]);
  });

  it("pad-fit letterboxes without masking the padded pixels", () => {
    const transform = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "pad-fit" },
    );

    expect(transform).toMatchObject({
      drawWidth: 1024,
      drawHeight: 768,
      offsetX: 0,
      offsetY: 128,
      maskPaddedPixels: false,
    });
    expect(maskPaddingRectangles(transform)).toEqual([]);
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

  it("crop fill defaults to centered overflow with negative offsets", () => {
    const transform = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "crop-fill" },
    );

    expect(transform.drawWidth).toBe(1365);
    expect(transform.drawHeight).toBe(1024);
    expect(transform.offsetX).toBe(-171);
    expect(transform.offsetY).toBe(0);
    expect(Object.is(transform.offsetY, -0)).toBe(false);
    expect(maskPaddingRectangles(transform)).toEqual([]);
  });

  it("lanczos-resize stretches to the exact target with no padding", () => {
    const transform = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "lanczos-resize" },
    );

    expect(transform).toEqual({
      outputWidth: 1024,
      outputHeight: 1024,
      drawWidth: 1024,
      drawHeight: 1024, // full stretch: drawHeight === outputHeight
      offsetX: 0,
      offsetY: 0,
      maskPaddedPixels: false,
    });
  });

  it("upscale-then-fit delegates the geometry to its nested fit policy", () => {
    const nested = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      { mode: "pad-repaint" },
    );
    const viaUpscale = resolveSourceFitTransform(
      { width: 640, height: 480 },
      { width: 1024, height: 1024 },
      {
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "pad-repaint" },
      },
    );
    expect(viaUpscale).toEqual(nested);
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

  describe("coerceSourceFitForMaskless (video img2img — no repaint mask possible)", () => {
    it("maps pad-repaint to centered crop-fill, outer and nested", () => {
      expect(coerceSourceFitForMaskless({ mode: "pad-repaint" })).toEqual({
        mode: "crop-fill",
        alignX: "center",
        alignY: "center",
      });
      expect(
        coerceSourceFitForMaskless({
          mode: "upscale-then-fit",
          upscalerModel: "real-esrgan-x2plus:fp16",
          fit: { mode: "pad-repaint" },
        }),
      ).toEqual({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
      });
    });

    it("passes maskless-safe policies through untouched", () => {
      const cropFill = { mode: "crop-fill", alignX: "left" } as const;
      expect(coerceSourceFitForMaskless(cropFill)).toBe(cropFill);
      const padFit = { mode: "pad-fit" } as const;
      expect(coerceSourceFitForMaskless(padFit)).toBe(padFit);
      const lanczos = { mode: "lanczos-resize" } as const;
      expect(coerceSourceFitForMaskless(lanczos)).toBe(lanczos);
    });
  });
});

describe("parseSourceFitPolicy (metadata provenance restore)", () => {
  it("accepts every wire-shaped policy", () => {
    expect(parseSourceFitPolicy({ mode: "pad-repaint" })).toEqual({
      mode: "pad-repaint",
    });
    expect(
      parseSourceFitPolicy({
        mode: "crop-fill",
        alignX: "left",
        alignY: "top",
      }),
    ).toEqual({ mode: "crop-fill", alignX: "left", alignY: "top" });
    expect(parseSourceFitPolicy({ mode: "lanczos-resize" })).toEqual({
      mode: "lanczos-resize",
    });
    expect(
      parseSourceFitPolicy({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x4plus:fp16",
        fit: { mode: "crop-fill" },
      }),
    ).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill" },
    });
  });

  it("rejects malformed provenance instead of poisoning the form", () => {
    expect(parseSourceFitPolicy(null)).toBeNull();
    expect(parseSourceFitPolicy("crop-fill")).toBeNull();
    expect(parseSourceFitPolicy({ mode: "teleport" })).toBeNull();
    expect(
      parseSourceFitPolicy({ mode: "crop-fill", alignX: "sideways" }),
    ).toBeNull();
    expect(
      parseSourceFitPolicy({
        mode: "upscale-then-fit",
        upscalerModel: 4,
        fit: { mode: "pad-fit" },
      }),
    ).toBeNull();
    expect(
      parseSourceFitPolicy({
        mode: "upscale-then-fit",
        upscalerModel: "u",
        fit: {
          mode: "upscale-then-fit",
          upscalerModel: "u",
          fit: { mode: "pad-fit" },
        },
      }),
    ).toBeNull();
  });
});
