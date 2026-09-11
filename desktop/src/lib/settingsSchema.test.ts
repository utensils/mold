import { readFileSync } from "node:fs";
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

const COMMANDS_RS = "../../src-tauri/src/commands.rs";

describe("settings schema", () => {
  it("routes curated keys to their sections", () => {
    expect(sectionForConfigKey("default_model")).toBe("generation");
    expect(sectionForConfigKey("expand.temperature")).toBe("expansion");
  });

  it("leads with Look and ends with Updates & about, in the lexicon", () => {
    expect(SECTIONS[0]).toMatchObject({ id: "app", label: "Look" });
    expect(SECTIONS.at(-1)).toMatchObject({ id: "updates", label: "Updates & about" });
    expect(SECTIONS.find((section) => section.id === "hosts")?.label).toBe("Machines");
    expect(SECTIONS.find((section) => section.id === "library")?.label).toBe("My images & trash");
    expect(SECTIONS.find((section) => section.id === "expansion")?.label).toBe("Write more for me");
    expect(SECTIONS.find((section) => section.id === "styles")?.label).toBe("Styles & disk");
  });

  it("keeps the mock's relative order for every section the mock names", () => {
    // docs/design/mold-studio-desktop.dc.html `settingsNav`. The four extra
    // sections slot between them; the shared nine never trade places.
    const mock = [
      "app",
      "generation",
      "hosts",
      "styles",
      "licenses",
      "library",
      "pairing",
      "performance",
      "updates",
    ];
    const ids = SECTIONS.map((section) => section.id);
    expect(ids.filter((id) => mock.includes(id))).toEqual(mock);
  });

  it("gives the disk its own section: styles and finished pictures on this machine", () => {
    expect(sectionForConfigKey("models_dir")).toBe("styles");
    expect(sectionForConfigKey("output_dir")).toBe("styles");
    expect(schemasForSection("styles").map((s) => s.key)).toEqual(["models_dir", "output_dir"]);
    expect(schemasForSection("hosts")).toEqual([]);
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
      // MOLD_FLUX2_* carry a digit, so the name is not letters-and-underscores.
      expect(knob.key).toMatch(/^env\.MOLD_[A-Z0-9_]+$/);
      expect(knob.needsEngineRestart).toBe(true);
      expect(knob.section).toBe("performance");
    }
  });

  // A knob the Performance section offers but `apply_engine_environment` never
  // copies into the engine's process is a control that silently does nothing —
  // and a variable the bridge copies with no row is a knob the app pretends it
  // cannot set. The two lists must be EQUAL, not merely nested: asserting one
  // direction only is how `MOLD_RESERVE_VRAM_MB` stayed engine-shaping,
  // documented and unofferable for a whole campaign. Read the Rust rather than
  // restating its list here: a list maintained by hand in two languages is
  // exactly the drift this test exists to catch.
  it("the env knobs and the Tauri side's ENGINE_ENV_KEYS are the same set", () => {
    // The path goes through a variable because Vite rewrites a literal first
    // argument to `new URL(..., import.meta.url)` into an asset URL, which
    // readFileSync then refuses as "must be of scheme file".
    const commandsPath = COMMANDS_RS;
    const commands = readFileSync(new URL(commandsPath, import.meta.url), "utf8");
    const block = commands.match(/pub const ENGINE_ENV_KEYS: &\[&str\] = &\[([\s\S]*?)\];/);
    expect(block, "ENGINE_ENV_KEYS not found in commands.rs").not.toBeNull();
    const allowlisted = [...block![1].matchAll(/"([A-Z0-9_]+)"/g)].map((m) => m[1]);
    expect(allowlisted.length).toBeGreaterThan(0);
    const offered = ENV_KNOB_SCHEMAS.map((knob) => knob.key.replace(/^env\./, ""));
    expect([...offered].sort()).toEqual([...allowlisted].sort());
  });

  // The per-family defaults are the campaign's whole point: a user reading
  // "Automatic" has to be told it is not one answer for every model.
  it("names the per-family default on the two backend knobs", () => {
    for (const key of ["env.MOLD_ATTN", "env.MOLD_CONV"]) {
      const knob = schemaFor(key)!;
      expect(knob.options?.[0]?.value, key).toBe("");
      expect(knob.options?.[0]?.label, key).toMatch(/per family/i);
      expect(knob.help, key).toMatch(/FLUX/);
      expect(knob.help, key).toMatch(/Wan|video/i);
    }
  });

  // `1` is an accepted value that resolves identically to unset — the budget
  // overrides an explicit keep on a card that cannot afford it (#276). Offering
  // it as "force on" would be a promise the engine does not keep.
  it("does not promise that keeping the FLUX transformer overrides the budget", () => {
    const knob = schemaFor("env.MOLD_FLUX_KEEP_TRANSFORMER")!;
    expect(knob.options?.map((o) => o.value)).toEqual(["", "1", "0"]);
    expect(knob.options?.find((o) => o.value === "1")?.label).toContain("same as automatic");
    expect(knob.options?.find((o) => o.value === "0")?.label).toMatch(/drop/i);
  });

  it("offers MOLD_KEEP_TE_RAM as the tri-state it became", () => {
    const knob = schemaFor("env.MOLD_KEEP_TE_RAM")!;
    expect(knob.options?.map((o) => o.value)).toEqual(["", "1", "0"]);
    expect(knob.options?.[0]?.label).toMatch(/^Automatic/);
    expect(knob.help).toMatch(/8 GB/);
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

  it("finds a section by a word that appears only in its summary sentence", () => {
    const look = SECTIONS.find((section) => section.id === "app")!;
    expect(sectionMatchesSearch("interface size", look)).toBe(true);
    expect(
      sectionMatchesSearch(
        "interface size",
        SECTIONS.find((s) => s.id === "media")!,
      ),
    ).toBe(false);
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
    expect(schema.label).toBe("Keep deleted pictures for");
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
