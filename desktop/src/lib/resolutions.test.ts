import { describe, expect, it } from "vitest";
import { aspectRatioLabel, matchPreset, orientationLabel, presetsForFamily } from "./resolutions";

describe("resolution presets", () => {
  it("every preset in every family is a multiple of 16", () => {
    for (const family of [
      "sd15",
      "sdxl",
      "flux",
      "flux2",
      "sd3",
      "zimage",
      "qwen-image",
      "qwen-image-edit",
      "wuerstchen",
      "ltx-video",
      "ltx2",
      "unknown-future-family",
    ]) {
      for (const preset of presetsForFamily(family)) {
        expect(preset.width % 16, `${family} ${preset.label} width`).toBe(0);
        expect(preset.height % 16, `${family} ${preset.label} height`).toBe(0);
      }
    }
  });

  it("keeps SD 1.5 near its 512 training size and SDXL near 1 MP", () => {
    expect(presetsForFamily("sd15").every((r) => r.width * r.height <= 640 * 832)).toBe(true);
    expect(presetsForFamily("sdxl").some((r) => r.width === 1024 && r.height === 1024)).toBe(true);
  });

  it("unknown families do not fabricate FLUX presets", () => {
    expect(presetsForFamily("brand-new")).toEqual([]);
  });

  it("matchPreset finds exact matches and returns null for custom sizes", () => {
    expect(matchPreset(1024, 1024, "flux")?.aspect).toBe("1:1");
    expect(matchPreset(1000, 1000, "flux")).toBeNull();
  });

  it("exposes Qwen Image aspect presets for generation and editing", () => {
    for (const family of ["qwen-image", "qwen-image-edit"]) {
      expect(matchPreset(1328, 1328, family)?.aspect).toBe("1:1");
      expect(matchPreset(1664, 928, family)?.aspect).toBe("≈16:9");
      expect(matchPreset(928, 1664, family)?.aspect).toBe("≈9:16");
      expect(presetsForFamily(family)).toHaveLength(7);
    }
  });

  it("exposes the runtime-qualified Z-Image 1024 tier", () => {
    expect(matchPreset(1280, 720, "z-image")?.aspect).toBe("16:9");
    expect(matchPreset(720, 1280, "z-image")?.aspect).toBe("9:16");
    expect(presetsForFamily("z-image")).toHaveLength(11);
  });

  it("describes preset and custom aspect ratios with their orientation", () => {
    expect(aspectRatioLabel(1328, 1328, "qwen-image-edit")).toBe("1:1");
    expect(aspectRatioLabel(1584, 1056, "qwen-image")).toBe("3:2");
    expect(aspectRatioLabel(1200, 800, "flux")).toBe("3:2");
    expect(orientationLabel(1200, 800)).toBe("Landscape");
    expect(orientationLabel(800, 1200)).toBe("Portrait");
    expect(orientationLabel(800, 800)).toBe("Square");
  });
});
