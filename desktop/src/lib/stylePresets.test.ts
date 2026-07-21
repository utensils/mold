import { describe, expect, it } from "vitest";
import { STYLE_PRESETS, composeStylePrompt, stylePresetLabel } from "./stylePresets";

describe("style presets", () => {
  it("exposes eight selectable looks", () => {
    expect(STYLE_PRESETS).toHaveLength(8);
    expect(new Set(STYLE_PRESETS.map((p) => p.id)).size).toBe(8);
  });

  it("labels the active preset, falling back to None", () => {
    expect(stylePresetLabel("cinematic")).toBe("Cinematic");
    expect(stylePresetLabel("")).toBe("None");
    expect(stylePresetLabel("nope")).toBe("None");
  });

  it("composes the extra fragment onto the prompt at submit", () => {
    expect(composeStylePrompt("a lighthouse", "cinematic")).toBe(
      "a lighthouse, cinematic lighting, film grain, shallow depth of field",
    );
  });

  it("returns the prompt unchanged for no preset or an empty prompt", () => {
    expect(composeStylePrompt("a lighthouse", "")).toBe("a lighthouse");
    expect(composeStylePrompt("a lighthouse", "unknown")).toBe("a lighthouse");
    expect(composeStylePrompt("   ", "cinematic")).toBe("   ");
  });
});
