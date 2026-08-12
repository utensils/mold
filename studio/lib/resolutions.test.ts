import { describe, expect, it } from "vitest";
import {
  dimensionAlignmentForFamily,
  LTX2_COMPOSED_MAX_AXIS_PIXELS,
  LTX2_COMPOSED_MAX_GENERATION_PIXELS,
  LTX2_MAX_AXIS_PIXELS,
  LTX2_MAX_GENERATION_PIXELS,
  MAX_GENERATION_PIXELS,
  maxAxisPixelsForFamily,
  maxAxisPixelsForModel,
  maxPixelsForFamily,
  maxPixelsForModel,
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
      "wan",
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

  it("keeps the conservative legacy Wan buckets instead of FLUX image shapes", () => {
    // Wan had no entry, so `presetsForFamily` returned the FLUX list: a video
    // family offered 1024x1024 image buckets on any host that does not
    // advertise `recommended_dimensions`. TI2V-5B is deliberately absent:
    // its /32 pair is checkpoint-specific and cannot be inferred from a bare
    // family string.
    const wan = presetsForFamily("wan").map((preset) => [
      preset.width,
      preset.height,
    ]);
    expect(wan).toEqual([
      [832, 480],
      [480, 832],
      [1280, 720],
      [720, 1280],
    ]);
    expect(wan).not.toContainEqual([1024, 1024]);
  });

  it("uses the advertised checkpoint contract for Wan TI2V-5B", () => {
    // A bare legacy family cannot safely guess TI2V-5B. The per-model path
    // still exposes the exact /32 pair when the old host advertises it.
    const wan = presetsForFamily("wan");
    expect(wan.some((preset) => preset.width % 32 !== 0)).toBe(true);
    expect(
      presetsForModel({
        family: "wan",
        dimension_alignment: 32,
        recommended_dimensions: [{ width: 1280, height: 704 }],
      }).map((preset) => [preset.width, preset.height]),
    ).toEqual([[1280, 704]]);
  });

  it("keeps the runnable LTX-2 landscape bucket shared by every client", () => {
    expect(presetsForFamily("ltx2")).toContainEqual({
      label: "19:11 · 1216×704",
      aspect: "19:11",
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

  it("keeps the family fallback at the single-pass span", () => {
    // A host that predates `max_axis_pixels` cannot render a composed shape,
    // so the fallback must never guess the composed ceiling for it.
    expect(maxAxisPixelsForFamily("ltx2")).toBe(LTX2_MAX_AXIS_PIXELS);
    expect(maxPixelsForFamily("ltx2")).toBe(LTX2_MAX_GENERATION_PIXELS);
    for (const preset of presetsForFamily("ltx2")) {
      expect(
        Math.max(preset.width, preset.height),
        preset.label,
      ).toBeLessThanOrEqual(LTX2_MAX_AXIS_PIXELS);
    }
  });

  it("takes the per-model ceiling over the family fallback", () => {
    const composing = {
      family: "ltx2",
      max_pixels: LTX2_COMPOSED_MAX_GENERATION_PIXELS,
      max_axis_pixels: LTX2_COMPOSED_MAX_AXIS_PIXELS,
    };
    expect(maxAxisPixelsForModel(composing)).toBe(
      LTX2_COMPOSED_MAX_AXIS_PIXELS,
    );
    expect(maxPixelsForModel(composing)).toBe(
      LTX2_COMPOSED_MAX_GENERATION_PIXELS,
    );
    // Same family, single-pass checkpoint.
    const singlePass = {
      family: "ltx2",
      max_pixels: LTX2_MAX_GENERATION_PIXELS,
      max_axis_pixels: LTX2_MAX_AXIS_PIXELS,
    };
    expect(maxAxisPixelsForModel(singlePass)).toBe(LTX2_MAX_AXIS_PIXELS);
    // A host that advertises neither falls back to the family span.
    expect(maxAxisPixelsForModel({ family: "ltx2" })).toBe(
      LTX2_MAX_AXIS_PIXELS,
    );
    expect(maxAxisPixelsForModel("ltx2")).toBe(LTX2_MAX_AXIS_PIXELS);
    expect(maxAxisPixelsForModel(null)).toBeNull();
  });

  it("filters advertised presets on the axis span as well as the pixel budget", () => {
    const advertised = [
      { width: 1920, height: 1088 },
      { width: 3840, height: 2176 },
    ];
    // 3840x2176 is 8.4 MP — inside the composed pixel budget and outside the
    // single-pass span at the same time, so a pixel-only filter would offer
    // it to a checkpoint that rejects it.
    expect(
      presetsForModel({
        family: "ltx2",
        dimension_alignment: 32,
        max_pixels: LTX2_COMPOSED_MAX_GENERATION_PIXELS,
        max_axis_pixels: LTX2_MAX_AXIS_PIXELS,
        recommended_dimensions: advertised,
      }).map((preset) => [preset.width, preset.height]),
    ).toEqual([[1920, 1088]]);
    expect(
      presetsForModel({
        family: "ltx2",
        dimension_alignment: 32,
        max_pixels: LTX2_COMPOSED_MAX_GENERATION_PIXELS,
        max_axis_pixels: LTX2_COMPOSED_MAX_AXIS_PIXELS,
        recommended_dimensions: advertised,
      }).map((preset) => [preset.width, preset.height]),
    ).toEqual([
      [1920, 1088],
      [3840, 2176],
    ]);
  });

  it("narrows a composing model's ceiling when a single-pass pipeline is chosen", () => {
    const composing = {
      family: "ltx2",
      max_pixels: LTX2_COMPOSED_MAX_GENERATION_PIXELS,
      max_axis_pixels: LTX2_COMPOSED_MAX_AXIS_PIXELS,
    };
    // The advertised ceiling assumes the server's default pipeline, which
    // refines. An explicit non-refining choice denoises the requested shape
    // once, so the client must not offer what the server will reject.
    for (const refining of [
      undefined,
      null,
      "two-stage",
      "two-stage-hq",
      "distilled",
      "ic-lora",
      "keyframe",
      "a2-vid",
    ]) {
      expect(maxAxisPixelsForModel(composing, refining), String(refining)).toBe(
        LTX2_COMPOSED_MAX_AXIS_PIXELS,
      );
      expect(maxPixelsForModel(composing, refining), String(refining)).toBe(
        LTX2_COMPOSED_MAX_GENERATION_PIXELS,
      );
    }
    for (const singlePass of ["one-stage", "retake", "lip-dub"]) {
      expect(maxAxisPixelsForModel(composing, singlePass), singlePass).toBe(
        LTX2_MAX_AXIS_PIXELS,
      );
      expect(maxPixelsForModel(composing, singlePass), singlePass).toBe(
        LTX2_MAX_GENERATION_PIXELS,
      );
    }
    // A single-pass checkpoint is never widened by a refining pipeline.
    const singlePassModel = {
      family: "ltx2",
      max_pixels: LTX2_MAX_GENERATION_PIXELS,
      max_axis_pixels: LTX2_MAX_AXIS_PIXELS,
    };
    expect(maxAxisPixelsForModel(singlePassModel, "two-stage")).toBe(
      LTX2_MAX_AXIS_PIXELS,
    );
    // Non-LTX-2 families have no axis limit either way.
    expect(maxAxisPixelsForModel({ family: "flux" }, "one-stage")).toBeNull();
    expect(maxPixelsForModel({ family: "flux" }, "one-stage")).toBe(
      MAX_GENERATION_PIXELS,
    );
  });
});
