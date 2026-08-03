import { describe, expect, it } from "vitest";
import {
  dimensionAlignmentForFamily,
  LTX2_MAX_GENERATION_PIXELS,
  MAX_GENERATION_PIXELS,
  maxAxisPixelsForFamily,
  maxPixelsForFamily,
  megapixelLabel,
  presetsForFamily,
  presetsForModel,
} from "./resolutions";

describe("shared resolution contract", () => {
  it("keeps every cross-surface fallback runnable under the server ceiling", () => {
    for (const family of [
      "sd15",
      "sdxl",
      "sd3",
      "flux",
      "flux2",
      "z-image",
      "qwen-image",
      "qwen-image-edit",
      "wuerstchen",
      "ltx-video",
      "ltx2",
    ]) {
      // Both the ceiling and the grid are family-specific: LTX-2 renders on
      // a /32 grid up to its own raised limit, everything else on /16 up to
      // the shared one.
      const alignment = dimensionAlignmentForFamily(family);
      const maxPixels = maxPixelsForFamily(family);
      const axisLimit = maxAxisPixelsForFamily(family);
      for (const preset of presetsForFamily(family)) {
        expect(preset.width % alignment, `${family} ${preset.label}`).toBe(0);
        expect(preset.height % alignment, `${family} ${preset.label}`).toBe(0);
        expect(
          preset.width * preset.height,
          `${family} ${preset.label}`,
        ).toBeLessThanOrEqual(maxPixels);
        if (axisLimit !== null) {
          expect(
            Math.max(preset.width, preset.height),
            `${family} ${preset.label}`,
          ).toBeLessThanOrEqual(axisLimit);
        }
      }
    }
  });

  it("prefers exact server-advertised buckets and filters invalid rows", () => {
    expect(
      presetsForModel({
        family: "flux2",
        max_pixels: 1_800_000,
        dimension_alignment: 16,
        recommended_dimensions: [
          { width: 1328, height: 1328 },
          { width: 1408, height: 1408 },
          { width: 1001, height: 1001 },
        ],
      }),
    ).toEqual([
      {
        label: "1:1 · 1328×1328",
        aspect: "1:1",
        width: 1328,
        height: 1328,
      },
    ]);
    expect(megapixelLabel(1328, 1328)).toBe("1.8 MP");
  });

  it("keeps the runnable LTX-2 landscape bucket shared by every client", () => {
    expect(presetsForFamily("ltx2")).toContainEqual({
      label: "16:9 · 1216×704",
      aspect: "16:9",
      width: 1216,
      height: 704,
    });
  });

  it("offers upstream's 1080p pair and a 9:16 portrait for ltx2 only", () => {
    const ltx2 = presetsForFamily("ltx2").map((p) => [p.width, p.height]);
    expect(ltx2).toContainEqual([1920, 1088]);
    expect(ltx2).toContainEqual([1088, 1920]);
    expect(ltx2).toContainEqual([704, 1216]);

    // ltx-video shares the VAE grid but not the raised ceiling.
    const ltxVideo = presetsForFamily("ltx-video").map((p) => [
      p.width,
      p.height,
    ]);
    expect(ltxVideo).not.toContainEqual([1920, 1088]);
  });

  it("filters 1080p out when an older host still advertises 1.8 MP", () => {
    const presets = presetsForModel({
      family: "ltx2",
      max_pixels: MAX_GENERATION_PIXELS,
      dimension_alignment: 32,
      recommended_dimensions: [
        { width: 1216, height: 704 },
        { width: 1920, height: 1088 },
      ],
    });
    expect(presets.map((p) => [p.width, p.height])).toEqual([[1216, 704]]);
  });

  it("accepts the ltx-2 family alias, not just the canonical spelling", () => {
    // `chainRouting` canonicalizes `ltx-2` -> `ltx2`, so an exact-string test
    // here would silently hand an LTX-2 caller the shared 1.8 MP limit.
    for (const alias of ["ltx-2", "LTX2", " ltx2 ", "Ltx-2"]) {
      expect(maxPixelsForFamily(alias), alias).toBe(LTX2_MAX_GENERATION_PIXELS);
      expect(maxAxisPixelsForFamily(alias), alias).toBe(2048);
      expect(dimensionAlignmentForFamily(alias), alias).toBe(32);
    }
  });

  it("exposes family-aware ceilings and grids", () => {
    expect(maxPixelsForFamily("ltx2")).toBe(LTX2_MAX_GENERATION_PIXELS);
    expect(maxPixelsForFamily("flux")).toBe(MAX_GENERATION_PIXELS);
    expect(dimensionAlignmentForFamily("ltx2")).toBe(32);
    expect(dimensionAlignmentForFamily("flux")).toBe(16);
    expect(maxAxisPixelsForFamily("ltx2")).toBe(2048);
    expect(maxAxisPixelsForFamily("flux")).toBeNull();
  });
});
