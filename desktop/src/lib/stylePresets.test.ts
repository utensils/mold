import { describe, expect, it } from "vitest";
import {
  STYLE_ANGLES,
  STYLE_PRESETS,
  angleForIndex,
  composeStyle,
  composeStylePrompt,
  styleHint,
  stylePresetLabel,
} from "./stylePresets";

describe("style presets (shared kit surface)", () => {
  it("exposes the eight canonical looks in chip order", () => {
    expect(STYLE_PRESETS.map((preset) => preset.id)).toEqual([
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

  it("labels canonical and legacy desktop ids, falling back to None", () => {
    expect(stylePresetLabel("cinematic")).toBe("Cinematic");
    // Every id the pre-kit desktop table persisted keeps resolving.
    expect(stylePresetLabel("photographic")).toBe("Photoreal");
    expect(stylePresetLabel("3d-render")).toBe("3D Render");
    expect(stylePresetLabel("neon")).toBe("Cinematic");
    expect(stylePresetLabel("minimal")).toBe("Illustration");
    expect(stylePresetLabel("")).toBe("None");
    expect(stylePresetLabel("nope")).toBe("None");
  });

  it("templates the prompt into looks that read as a prefix", () => {
    expect(composeStylePrompt("a lighthouse", "cinematic")).toBe(
      "cinematic film still of a lighthouse, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
  });

  it("appends looks without a placeholder and resolves legacy ids", () => {
    expect(composeStylePrompt("a gadget", "product")).toBe(
      "a gadget, product photography, seamless backdrop, softbox lighting",
    );
    expect(composeStylePrompt("a lighthouse", "photographic")).toBe(
      composeStylePrompt("a lighthouse", "photoreal"),
    );
  });

  it("returns the prompt unchanged for no preset or an empty prompt", () => {
    expect(composeStylePrompt("a lighthouse", "")).toBe("a lighthouse");
    expect(composeStylePrompt("a lighthouse", "unknown")).toBe("a lighthouse");
    expect(composeStylePrompt("   ", "cinematic")).toBe("   ");
  });

  it("merges the preset negative through composeStyle only when the family allows it", () => {
    const merged = composeStyle("a lighthouse", "cinematic", {
      supportsNegativePrompt: true,
      negative: "text, watermark, anime",
    });
    // User fragments first, preset fragments appended, exact duplicates dropped.
    expect(merged.negative).toBe("text, watermark, anime, cartoon, graphic, washed out");

    const gated = composeStyle("a lighthouse", "cinematic", {
      supportsNegativePrompt: false,
      negative: "text",
    });
    expect(gated.negative).toBe("text");
  });

  it("describes the look as natural language for the server expansion directive", () => {
    expect(styleHint("cinematic")).toBe(
      "Cinematic look — cinematic film still, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    expect(styleHint("")).toBeNull();
    expect(styleHint("unknown")).toBeNull();
  });

  it("keeps the shared variation angles and their wrapping cycle", () => {
    expect(STYLE_ANGLES).toHaveLength(5);
    expect(angleForIndex(0)).toBe("at golden hour");
    expect(angleForIndex(5)).toBe("at golden hour");
    expect(angleForIndex(6)).toBe("in moody overcast light");
  });
});
