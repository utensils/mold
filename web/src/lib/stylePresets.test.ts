import { describe, expect, it } from "vitest";
import {
  STYLE_ANGLES,
  STYLE_PRESETS,
  angleForIndex,
  stylePresetById,
  stylePresetExtras,
} from "./stylePresets";

describe("stylePresets", () => {
  it("ships the eight prototype presets in chip order", () => {
    expect(STYLE_PRESETS.map((p) => p.id)).toEqual([
      "photoreal",
      "cinematic",
      "portrait",
      "illustration",
      "anime",
      "3d",
      "watercolor",
      "product",
    ]);
  });

  it("keeps the exact extras strings", () => {
    expect(stylePresetExtras("cinematic")).toBe(
      "cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    expect(stylePresetExtras("3d")).toBe(
      "octane render, subsurface scattering, studio HDRI lighting",
    );
    expect(stylePresetExtras("product")).toBe(
      "product photography, seamless backdrop, softbox lighting",
    );
  });

  it("returns null for a null or unknown preset id", () => {
    expect(stylePresetExtras(null)).toBeNull();
    expect(stylePresetExtras("does-not-exist")).toBeNull();
    expect(stylePresetById(null)).toBeNull();
    expect(stylePresetById("nope")).toBeNull();
  });

  it("resolves a preset by id", () => {
    expect(stylePresetById("anime")).toEqual({
      id: "anime",
      name: "Anime",
      extras: "anime key art, cel shading, vibrant palette",
    });
  });

  it("cycles the five angles, wrapping past the end", () => {
    expect(STYLE_ANGLES).toHaveLength(5);
    expect(angleForIndex(0)).toBe("at golden hour");
    expect(angleForIndex(4)).toBe("backlit with warm rim light");
    expect(angleForIndex(5)).toBe("at golden hour");
    expect(angleForIndex(6)).toBe("in moody overcast light");
  });
});
