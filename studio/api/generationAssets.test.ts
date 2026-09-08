import { describe, expect, it } from "vitest";
import { generationAssetLabel, generationAssetPath } from "./generationAssets";

describe("generation assets", () => {
  it("addresses an asset beneath its owning print without path ambiguity", () => {
    expect(generationAssetPath("chair one.glb", "base/color")).toBe(
      "/api/gallery/assets/chair%20one.glb/base%2Fcolor",
    );
  });

  it("gives PBR maps stable user-facing names", () => {
    const asset = {
      asset_id: "base_color",
      role: "base_color",
      display_name: "chair-base-color.png",
      media_type: "image/png",
      size_bytes: 12,
      sha256: "abc",
    };
    expect(generationAssetLabel(asset)).toBe("Download base color map");
    expect(
      generationAssetLabel({
        ...asset,
        asset_id: "metallic_roughness",
        role: "metallic_roughness",
      }),
    ).toBe("Download metallic-roughness map");
    expect(
      generationAssetLabel({ ...asset, asset_id: "normal", role: "normal" }),
    ).toBe("Download normal map");
  });
});
