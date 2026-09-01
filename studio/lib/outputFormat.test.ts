import { describe, expect, it } from "vitest";
import { hunyuan3dRecipe, sdxlRecipe } from "./generationProfile.testFixtures";
import { coerceOutputFormatForRecipe } from "./outputFormat";

describe("coerceOutputFormatForRecipe", () => {
  it("pins a raster format to glb on a mesh recipe", () => {
    expect(
      coerceOutputFormatForRecipe(hunyuan3dRecipe(), "hunyuan3d", "png"),
    ).toBe("glb");
    expect(
      coerceOutputFormatForRecipe(hunyuan3dRecipe(), "hunyuan3d", "webp"),
    ).toBe("glb");
  });

  it("pins an absent format to glb on a mesh recipe", () => {
    expect(
      coerceOutputFormatForRecipe(hunyuan3dRecipe(), "hunyuan3d", undefined),
    ).toBe("glb");
    expect(
      coerceOutputFormatForRecipe(hunyuan3dRecipe(), "hunyuan3d", null),
    ).toBe("glb");
  });

  it("keeps an advertised mesh format on a mesh recipe", () => {
    expect(
      coerceOutputFormatForRecipe(hunyuan3dRecipe(), "hunyuan3d", "glb"),
    ).toBe("glb");
  });

  it("leaves an advertised raster format alone on a raster recipe", () => {
    expect(coerceOutputFormatForRecipe(sdxlRecipe(), "sdxl", "jpeg")).toBe(
      "jpeg",
    );
    expect(coerceOutputFormatForRecipe(sdxlRecipe(), "sdxl", undefined)).toBe(
      undefined,
    );
  });

  it("restores the recipe default when a mesh format lingers on a raster recipe", () => {
    expect(coerceOutputFormatForRecipe(sdxlRecipe(), "sdxl", "glb")).toBe(
      sdxlRecipe().capabilities.output.default_format,
    );
  });

  it("falls back to the legacy family rule when no recipe is advertised", () => {
    expect(
      coerceOutputFormatForRecipe(null, "hunyuan3d", "png", ["png", "jpeg"]),
    ).toBe("glb");
    expect(
      coerceOutputFormatForRecipe(null, "sdxl", "png", ["png", "jpeg"]),
    ).toBe("png");
    expect(
      coerceOutputFormatForRecipe(null, "sdxl", "glb", ["png", "jpeg"]),
    ).toBe("png");
    expect(coerceOutputFormatForRecipe(null, "ltx2", "gif")).toBe("gif");
  });
});
