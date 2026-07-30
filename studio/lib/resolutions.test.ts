import { describe, expect, it } from "vitest";
import {
  MAX_GENERATION_PIXELS,
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
      for (const preset of presetsForFamily(family)) {
        expect(preset.width % 16, `${family} ${preset.label}`).toBe(0);
        expect(preset.height % 16, `${family} ${preset.label}`).toBe(0);
        expect(
          preset.width * preset.height,
          `${family} ${preset.label}`,
        ).toBeLessThanOrEqual(MAX_GENERATION_PIXELS);
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
});
