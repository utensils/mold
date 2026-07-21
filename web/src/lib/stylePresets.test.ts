import { describe, expect, it } from "vitest";
import {
  STYLE_ANGLES,
  STYLE_PRESETS,
  angleForIndex,
  composeStyle,
  mergeStyleNegative,
  stylePresetById,
  stylePresetLabel,
  styleHint,
} from "./stylePresets";
import * as kit from "@ui/lib/stylePresets";

describe("stylePresets", () => {
  it("is the shared kit table, not a web-only fork", () => {
    // Same entries, same order — the only addition is the `name` alias below.
    expect(STYLE_PRESETS.map((preset) => preset.id)).toEqual(
      kit.STYLE_PRESETS.map((preset) => preset.id),
    );
    for (const [index, preset] of STYLE_PRESETS.entries()) {
      expect(preset).toMatchObject(kit.STYLE_PRESETS[index]!);
    }
    expect(composeStyle).toBe(kit.composeStyle);
    expect(styleHint).toBe(kit.styleHint);
    expect(mergeStyleNegative).toBe(kit.mergeStyleNegative);
  });

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

  it("keeps `name` as an alias of the kit label for existing consumers", () => {
    for (const preset of STYLE_PRESETS) {
      expect(preset.name).toBe(preset.label);
    }
    expect(stylePresetById("anime")?.name).toBe("Anime");
    expect(stylePresetLabel("anime")).toBe("Anime");
  });

  it("composes the kit's positive template, not the old bare extras", () => {
    expect(
      composeStyle("a heron", "cinematic", { supportsNegativePrompt: false })
        .prompt,
    ).toBe(
      "cinematic film still of a heron, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    // Append-style presets still comma-join.
    expect(
      composeStyle("a red sneaker", "product", {
        supportsNegativePrompt: false,
      }).prompt,
    ).toBe(
      "a red sneaker, product photography, seamless backdrop, softbox lighting",
    );
  });

  it("carries the kit's curated negatives", () => {
    expect(
      mergeStyleNegative("text", "cinematic", { supportsNegativePrompt: true }),
    ).toBe("text, anime, cartoon, graphic, washed out");
  });

  it("tolerates a null preset id (the form stores null for no style)", () => {
    expect(stylePresetById(null)).toBeNull();
    expect(stylePresetById("nope")).toBeNull();
    expect(stylePresetLabel(null)).toBe("None");
  });

  it("ignores retired style ids", () => {
    expect(stylePresetById("photographic")).toBeNull();
    expect(stylePresetById("3d-render")).toBeNull();
  });

  it("cycles the five angles, wrapping past the end", () => {
    expect(STYLE_ANGLES).toHaveLength(5);
    expect(angleForIndex(0)).toBe("at golden hour");
    expect(angleForIndex(4)).toBe("backlit with warm rim light");
    expect(angleForIndex(5)).toBe("at golden hour");
    expect(angleForIndex(6)).toBe("in moody overcast light");
  });
});
