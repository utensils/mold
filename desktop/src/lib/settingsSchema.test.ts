import { describe, expect, it } from "vitest";
import {
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

  it("leads with Look and ends with Updates & about, in the lexicon", () => {
    expect(SECTIONS[0]).toMatchObject({ id: "app", label: "Look" });
    expect(SECTIONS.at(-1)).toMatchObject({ id: "updates", label: "Updates & about" });
    expect(SECTIONS.find((section) => section.id === "hosts")?.label).toBe("Machines");
    expect(SECTIONS.find((section) => section.id === "library")?.label).toBe("My images & trash");
    expect(SECTIONS.find((section) => section.id === "expansion")?.label).toBe("Write more for me");
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
    expect(schemaFor("scheduler.replan_debounce_ms")).toMatchObject({
      editor: "number",
      min: 0,
      max: 30000,
      needsEngineRestart: true,
    });
    expect(schemaFor("nope")).toBeNull();
  });

  it("marks output_dir as startup-only with actionable CLI copy", () => {
    const output = schemaFor("output_dir");
    expect(output?.liveReadOnly).toBe(true);
    expect(output?.help).toContain("mold config set output_dir <path>");
    expect(output?.help).toContain("restart");
  });
});

describe("Settings sections", () => {
  it("lists every section once, each with a summary sentence", () => {
    const ids = SECTIONS.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const section of SECTIONS) {
      expect(section.summary, section.id).toBeTruthy();
    }
  });

  it("finds the saved-media section using the user's save-location language", () => {
    const media = SECTIONS.find((section) => section.id === "media")!;
    expect(sectionMatchesSearch("save location", media)).toBe(true);
    expect(sectionMatchesSearch("default save location", media)).toBe(true);
  });

  it("owns the Library trash-retention setting with a Forever option", () => {
    expect(sectionForConfigKey("gallery.trash_retention_days")).toBe("library");
    const schema = schemaFor("gallery.trash_retention_days")!;
    expect(schema.editor).toBe("select");
    expect(schema.label).toBe("Keep deleted prints for");
    expect(schema.options?.map((o) => o.value)).toEqual(["1", "7", "30", "90", "365", "0"]);
    expect(schema.options?.find((o) => o.value === "0")?.label).toBe("Forever");
    expect(schema.options?.find((o) => o.value === "30")?.label).toBe("30 days");
    expect(schema.help).toContain("0 keeps them until you empty the trash");
    const library = SECTIONS.find((section) => section.id === "library")!;
    expect(library.label).toBe("My images & trash");
    expect(sectionMatchesSearch("trash", library)).toBe(true);
    expect(sectionMatchesSearch("retention", library)).toBe(true);
    expect(sectionMatchesSearch("collections", library)).toBe(true);
  });

  it("collects the curated schemas that belong to a section", () => {
    expect(schemasForSection("expansion").map((s) => s.key)).toContain("expand.temperature");
    expect(schemasForSection("performance").every((s) => s.section === "performance")).toBe(true);
  });
});

describe("sectionMatchesSearch", () => {
  const expansion = SECTIONS.find((s) => s.id === "expansion")!;
  const accounts = SECTIONS.find((s) => s.id === "accounts")!;
  const advanced = SECTIONS.find((s) => s.id === "advanced")!;

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
