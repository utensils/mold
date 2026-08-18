import { describe, expect, it } from "vitest";
import {
  canvasMatchesSourceResolution,
  resolveDefaultSourceResolution,
  resolveSourceCanvasTransition,
  resolveSourceConditioningTarget,
  resolveSourceResolution,
  sourceConditioningLimitLabel,
  sourceResolutionStatus,
} from "./sourceResolution";

const qwen = {
  family: "qwen-image-edit",
  max_pixels: 1_800_000,
  dimension_alignment: 16,
};

describe("source resolution", () => {
  it("applies the automatic preset to a fresh attachment", () => {
    const source = resolveSourceResolution({ width: 1080, height: 1920 }, qwen);
    expect(
      resolveSourceCanvasTransition({
        current: { width: 1024, height: 1024 },
        previousSource: null,
        previousAutomatic: null,
        source,
        automatic: { width: 928, height: 1664 },
        replaced: true,
        mode: "automatic",
      }),
    ).toEqual({ width: 928, height: 1664 });
  });

  it("preserves an arbitrary manual canvas during an asynchronous restore", () => {
    const source = resolveSourceResolution({ width: 640, height: 384 }, qwen);
    expect(
      resolveSourceCanvasTransition({
        current: { width: 1024, height: 1024 },
        previousSource: null,
        previousAutomatic: null,
        source,
        automatic: { width: 1664, height: 928 },
        replaced: true,
        mode: "manual",
        preserveReplacement: true,
      }),
    ).toBeNull();
  });

  it("preserves explicit Source intent when source and automatic sizes had tied", () => {
    const previous = resolveSourceResolution(
      { width: 1024, height: 1024 },
      qwen,
    );
    const portrait = resolveSourceResolution(
      { width: 896, height: 1600 },
      qwen,
    );
    expect(
      resolveSourceCanvasTransition({
        current: { width: 1024, height: 1024 },
        previousSource: previous,
        previousAutomatic: { width: 1024, height: 1024 },
        source: portrait,
        automatic: { width: 576, height: 1024 },
        replaced: true,
        mode: "source",
      }),
    ).toEqual(portrait.output);
  });

  it("keeps automatic and explicit Source choices live across model changes", () => {
    const previousSource = resolveSourceResolution(
      { width: 1080, height: 1920 },
      qwen,
    );
    const nextSource = resolveSourceResolution(
      { width: 1080, height: 1920 },
      { ...qwen, dimension_alignment: 32 },
    );
    const transition = {
      previousSource,
      source: nextSource,
      automatic: { width: 720, height: 1280 },
      replaced: false,
    };

    expect(
      resolveSourceCanvasTransition({
        ...transition,
        current: { width: 928, height: 1664 },
        previousAutomatic: { width: 928, height: 1664 },
        mode: "automatic" as const,
      }),
    ).toEqual({ width: 720, height: 1280 });
    expect(
      resolveSourceCanvasTransition({
        ...transition,
        current: previousSource.output,
        previousAutomatic: { width: 928, height: 1664 },
        mode: "source" as const,
      }),
    ).toEqual(nextSource.output);
  });

  it("defaults a source to the model preset with the closest aspect ratio", () => {
    const model = {
      family: "flux",
      default_width: 1024,
      default_height: 1024,
      recommended_dimensions: [
        { width: 1024, height: 1024 },
        { width: 1024, height: 576 },
        { width: 576, height: 1024 },
      ],
      dimension_alignment: 16,
    };

    expect(
      resolveDefaultSourceResolution({ width: 1080, height: 1920 }, model),
    ).toEqual({ width: 576, height: 1024 });
    expect(
      resolveDefaultSourceResolution({ width: 1920, height: 1080 }, model),
    ).toEqual({ width: 1024, height: 576 });
  });

  it("uses the recipe default area to choose a tier within the closest aspect", () => {
    const model = {
      family: "flux",
      default_width: 1024,
      default_height: 1024,
      recommended_dimensions: [
        { width: 768, height: 768 },
        { width: 1024, height: 1024 },
      ],
      dimension_alignment: 16,
    };

    expect(
      resolveDefaultSourceResolution({ width: 2000, height: 2000 }, model),
    ).toEqual({ width: 1024, height: 1024 });
  });

  it("falls back to the model-safe source canvas when no preset is advertised", () => {
    const model = {
      family: "custom",
      max_pixels: 1_800_000,
      dimension_alignment: 16,
      recommended_dimensions: [],
    };

    expect(
      resolveDefaultSourceResolution({ width: 1179, height: 786 }, model),
    ).toEqual({ width: 1168, height: 784 });
  });

  it("caps a Qwen conditioning canvas independently from its larger output", () => {
    expect(
      resolveSourceConditioningTarget({ width: 1328, height: 1328 }, qwen),
    ).toEqual({ width: 1024, height: 1024 });
    expect(sourceConditioningLimitLabel(qwen)).toBe("1 MP");
  });

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

    expect(result.output).toEqual({ width: 1168, height: 880 });
    expect(result.output.width).toBeLessThan(4032);
    expect(result.output.height).toBeLessThan(3024);
    expect(result.output.width * result.output.height).toBeLessThanOrEqual(
      1024 * 1024,
    );
    expect(result.output.width / result.output.height).toBeCloseTo(4 / 3, 1);
    expect(result.reason).toBe("downscaled");
    expect(sourceResolutionStatus(result)).toEqual({
      label: "Downscaled",
      detail: "4032×3024 → 1168×880 · 1 MP limit, 16 px aligned",
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

  it("snaps sources onto a host-advertised 32 px grid the family fallback would miss", () => {
    // wan22-ti2v-5b: the family-wide fallback is 16, but the host advertises
    // dimension_alignment: 32 for this model (its 2.2 VAE's real grid). A
    // 1280x720 source is %16-but-not-%32 and must land on the /32 grid the
    // server admits instead of a canvas it rejects.
    const result = resolveSourceResolution(
      { width: 1280, height: 720 },
      {
        family: "wan",
        max_pixels: 1_800_000,
        dimension_alignment: 32,
      },
    );

    expect(result.alignment).toBe(32);
    expect(result.output).toEqual({ width: 1280, height: 704 });
    expect(result.output.width % 32).toBe(0);
    expect(result.output.height % 32).toBe(0);
    expect(result.fitsModel).toBe(true);
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

  it("clamps an extreme aspect ratio to the model's axis span, not just its pixel budget", () => {
    // 8000x640 is 5.1 MP — comfortably inside a composing LTX-2 checkpoint's
    // 8.9 MP budget once scaled, and its long edge would still be far outside
    // the 4096 px span the composition can hold. The pixel budget alone
    // produced a canvas the server then rejected.
    const result = resolveSourceResolution(
      { width: 8000, height: 640 },
      {
        family: "ltx2",
        max_pixels: 4_096 * 2_176,
        max_axis_pixels: 4_096,
        recommended_dimensions: [],
        dimension_alignment: 32,
      },
    );

    expect(
      Math.max(result.output.width, result.output.height),
    ).toBeLessThanOrEqual(4_096);
    expect(result.output.width * result.output.height).toBeLessThanOrEqual(
      4_096 * 2_176,
    );
    expect(result.fitsModel).toBe(true);
  });

  it("keeps a single-pass LTX-2 source inside the trained span", () => {
    const result = resolveSourceResolution(
      { width: 4000, height: 1000 },
      {
        family: "ltx2",
        max_pixels: 1_920 * 1_088,
        max_axis_pixels: 2_048,
        recommended_dimensions: [],
        dimension_alignment: 32,
      },
    );

    expect(
      Math.max(result.output.width, result.output.height),
    ).toBeLessThanOrEqual(2_048);
    expect(result.fitsModel).toBe(true);
  });
});
