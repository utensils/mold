import { describe, expect, it } from "vitest";
import {
  canvasMatchesSourceResolution,
  resolveSourceResolution,
  sourceResolutionStatus,
} from "./sourceResolution";

const qwen = {
  family: "qwen-image-edit",
  max_pixels: 1_800_000,
  dimension_alignment: 16,
};

describe("source resolution", () => {
  it("keeps an in-bounds aligned custom resolution exactly", () => {
    const result = resolveSourceResolution({ width: 896, height: 1152 }, qwen);

    expect(result.output).toEqual({ width: 896, height: 1152 });
    expect(result.reason).toBe("exact");
    expect(result.fitsModel).toBe(true);
    expect(sourceResolutionStatus(result)).toEqual({
      label: "Source",
      detail: "Matches source · 896×1152",
    });
  });

  it("downscales without changing aspect materially or exceeding the model cap", () => {
    const result = resolveSourceResolution({ width: 4032, height: 3024 }, qwen);

    expect(result.output).toEqual({ width: 1536, height: 1152 });
    expect(result.output.width).toBeLessThan(4032);
    expect(result.output.height).toBeLessThan(3024);
    expect(result.output.width * result.output.height).toBeLessThanOrEqual(
      1_800_000,
    );
    expect(result.output.width / result.output.height).toBeCloseTo(4 / 3, 2);
    expect(result.reason).toBe("downscaled");
    expect(sourceResolutionStatus(result)).toEqual({
      label: "Downscaled",
      detail: "4032×3024 → 1536×1152 · 1.8 MP limit, 16 px aligned",
    });
  });

  it("only rounds an in-bounds unaligned source downward", () => {
    const result = resolveSourceResolution({ width: 1179, height: 786 }, qwen);

    expect(result.output).toEqual({ width: 1168, height: 784 });
    expect(result.reason).toBe("aligned");
    expect(result.output.width).toBeLessThanOrEqual(result.source.width);
    expect(result.output.height).toBeLessThanOrEqual(result.source.height);
  });

  it("honors a model-specific 32 px alignment", () => {
    const result = resolveSourceResolution(
      { width: 1215, height: 703 },
      {
        family: "ltx2",
        max_pixels: 1_800_000,
        dimension_alignment: 32,
      },
    );

    expect(result.output).toEqual({ width: 1184, height: 672 });
    expect(result.output.width % 32).toBe(0);
    expect(result.output.height % 32).toBe(0);
  });

  it("uses the video-family alignment before a live model contract arrives", () => {
    const result = resolveSourceResolution(
      { width: 1215, height: 703 },
      "ltx2",
    );

    expect(result.output).toEqual({ width: 1184, height: 672 });
    expect(result.alignment).toBe(32);
  });

  it("uses the minimum valid canvas only when a source dimension is too small", () => {
    const result = resolveSourceResolution({ width: 32, height: 47 }, qwen);

    expect(result.output).toEqual({ width: 64, height: 64 });
    expect(result.reason).toBe("minimum");
    expect(result.minimumDimension).toBe(64);
  });

  it("reports an impossible contradictory model contract", () => {
    const result = resolveSourceResolution(
      { width: 1000, height: 1000 },
      {
        family: "broken",
        max_pixels: 3000,
        dimension_alignment: 16,
      },
    );

    expect(result.output).toEqual({ width: 64, height: 64 });
    expect(result.fitsModel).toBe(false);
  });

  it("falls back to the shared ceiling and alignment for a family string", () => {
    const result = resolveSourceResolution(
      { width: 2000, height: 1000 },
      "flux",
    );

    expect(result.maxPixels).toBe(1_800_000);
    expect(result.alignment).toBe(16);
    expect(result.output.width * result.output.height).toBeLessThanOrEqual(
      result.maxPixels,
    );
  });

  it("detects source-follow versus a manual canvas override", () => {
    const result = resolveSourceResolution({ width: 896, height: 1152 }, qwen);

    expect(
      canvasMatchesSourceResolution({ width: 896, height: 1152 }, result),
    ).toBe(true);
    expect(
      canvasMatchesSourceResolution({ width: 1024, height: 1024 }, result),
    ).toBe(false);
  });

  it("treats recommended dimensions as presets rather than custom-canvas bounds", () => {
    const result = resolveSourceResolution(
      { width: 3000, height: 500 },
      {
        family: "custom",
        max_pixels: 1_800_000,
        dimension_alignment: 16,
        recommended_dimensions: [
          { width: 1024, height: 1024 },
          { width: 1344, height: 768 },
        ],
      },
    );

    expect(result.maxPixels).toBe(1_800_000);
    expect(result.output).toEqual({ width: 2992, height: 496 });
    expect(result.output).not.toEqual({ width: 1344, height: 768 });
  });

  it("uses the shared admission ceiling as the old-host fallback", () => {
    const result = resolveSourceResolution(
      { width: 2000, height: 1000 },
      {
        family: "flux",
        max_pixels: null,
        recommended_dimensions: [],
        dimension_alignment: null,
      },
    );

    expect(result.maxPixels).toBe(1_800_000);
    expect(result.output).toEqual({ width: 1888, height: 944 });
  });
});
