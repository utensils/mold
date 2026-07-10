import { describe, expect, it } from "vitest";
import {
  ENGINE_KEY_SCHEMAS,
  ENV_KNOB_SCHEMAS,
  matchesSearch,
  schemaFor,
  sectionForConfigKey,
} from "./settingsSchema";

describe("settings schema", () => {
  it("routes curated keys to their sections", () => {
    expect(sectionForConfigKey("models_dir")).toBe("engine");
    expect(sectionForConfigKey("default_model")).toBe("generation");
    expect(sectionForConfigKey("expand.temperature")).toBe("expansion");
  });

  it("unknown keys fall through to advanced — future engine keys must surface", () => {
    expect(sectionForConfigKey("runpod.api_key")).toBe("advanced");
    expect(sectionForConfigKey("some.future.key")).toBe("advanced");
  });

  it("tui keys never surface in a desktop app", () => {
    expect(sectionForConfigKey("tui.theme")).toBeNull();
  });

  it("every env knob names a real MOLD_ variable and needs a restart", () => {
    for (const knob of ENV_KNOB_SCHEMAS) {
      expect(knob.key).toMatch(/^env\.MOLD_[A-Z_]+$/);
      expect(knob.needsEngineRestart).toBe(true);
      expect(knob.section).toBe("performance");
    }
  });

  it("select editors always carry options", () => {
    for (const s of [...ENGINE_KEY_SCHEMAS, ...ENV_KNOB_SCHEMAS]) {
      if (s.editor === "select" && s.key !== "default_model") {
        expect(s.options?.length, s.key).toBeGreaterThan(0);
      }
    }
  });

  it("schemaFor resolves both engine keys and env knobs", () => {
    expect(schemaFor("embed_metadata")?.editor).toBe("toggle");
    expect(schemaFor("env.MOLD_VAE_TILED")?.editor).toBe("select");
    expect(schemaFor("nope")).toBeNull();
  });
});

describe("matchesSearch", () => {
  const item = { key: "env.MOLD_VAE_TILED", label: "Tiled VAE decode", help: "auto retries" };

  it("matches key, label, and help case-insensitively", () => {
    expect(matchesSearch("vae", item)).toBe(true);
    expect(matchesSearch("TILED", item)).toBe(true);
    expect(matchesSearch("retries", item)).toBe(true);
    expect(matchesSearch("flash", item)).toBe(false);
  });

  it("empty query matches everything", () => {
    expect(matchesSearch("  ", item)).toBe(true);
  });
});
