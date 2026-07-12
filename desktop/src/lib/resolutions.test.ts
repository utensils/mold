import { describe, expect, it } from "vitest";
import { matchPreset, presetsForFamily } from "./resolutions";

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

  it("unknown families fall back to the modern 1 MP list", () => {
    expect(presetsForFamily("brand-new")).toEqual(presetsForFamily("flux"));
  });

  it("matchPreset finds exact matches and returns null for custom sizes", () => {
    expect(matchPreset(1024, 1024, "flux")?.aspect).toBe("1:1");
    expect(matchPreset(1000, 1000, "flux")).toBeNull();
  });
});
