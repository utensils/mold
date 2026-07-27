import { describe, expect, it } from "vitest";
import {
  ACCORDION_SECTIONS,
  ENGINE_KEY_SCHEMAS,
  ENV_KNOB_SCHEMAS,
  matchesSearch,
  schemaFor,
  schemasForSection,
  sectionForConfigKey,
  sectionMatchesSearch,
  SECTIONS,
} from "./settingsSchema";

describe("settings schema", () => {
  it("routes curated keys to their sections", () => {
    expect(sectionForConfigKey("models_dir")).toBe("hosts");
    expect(sectionForConfigKey("default_model")).toBe("generation");
    expect(sectionForConfigKey("expand.temperature")).toBe("expansion");
  });

  it("names the first section Hosts (the default) and keeps Updates", () => {
    expect(SECTIONS[0]).toMatchObject({ id: "hosts", label: "Hosts" });
    expect(SECTIONS.find((section) => section.id === "updates")?.label).toBe("Updates");
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

  it("marks output_dir as startup-only with actionable CLI copy", () => {
    const output = schemaFor("output_dir");
    expect(output?.liveReadOnly).toBe(true);
    expect(output?.help).toContain("mold config set output_dir <path>");
    expect(output?.help).toContain("restart");
  });
});

describe("Settings accordion sections", () => {
  it("lists the collapsible 'All settings' region in order, without top-card sections", () => {
    expect(ACCORDION_SECTIONS.map((s) => s.id)).toEqual([
      "performance",
      "generation",
      "expansion",
      "accounts",
      "profiles",
      "advanced",
    ]);
    // Appearance / Updates / About / Hosts are top cards, not accordions.
    for (const id of ["app", "updates", "about", "hosts"]) {
      expect(ACCORDION_SECTIONS.some((s) => s.id === id)).toBe(false);
    }
  });

  it("gives every group an icon and concise summary for the shared Settings treatment", () => {
    for (const section of ACCORDION_SECTIONS) {
      expect(section.icon, section.id).toBeTruthy();
      expect(section.summary, section.id).toBeTruthy();
    }
  });

  it("collects the curated schemas that belong to a section", () => {
    expect(schemasForSection("expansion").map((s) => s.key)).toContain("expand.temperature");
    expect(schemasForSection("performance").every((s) => s.section === "performance")).toBe(true);
  });
});

describe("sectionMatchesSearch", () => {
  const expansion = ACCORDION_SECTIONS.find((s) => s.id === "expansion")!;
  const accounts = ACCORDION_SECTIONS.find((s) => s.id === "accounts")!;
  const advanced = ACCORDION_SECTIONS.find((s) => s.id === "advanced")!;

  it("matches a section by a curated key it owns", () => {
    expect(sectionMatchesSearch("temperature", expansion)).toBe(true);
    expect(sectionMatchesSearch("temperature", accounts)).toBe(false);
  });

  it("matches keyword-only sections that carry no curated key", () => {
    expect(sectionMatchesSearch("civitai", accounts)).toBe(true);
    expect(sectionMatchesSearch("token", accounts)).toBe(true);
  });

  it("matches Advanced against a raw engine row key", () => {
    expect(sectionMatchesSearch("runpod", advanced, ["runpod.api_key"])).toBe(true);
    expect(sectionMatchesSearch("runpod", advanced, [])).toBe(false);
  });

  it("an empty query matches every section", () => {
    expect(sectionMatchesSearch("  ", expansion)).toBe(true);
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
