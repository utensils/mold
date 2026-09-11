import { describe, expect, it } from "vitest";
import {
  ENGINE_KEY_SCHEMAS,
  ENV_KNOB_SCHEMAS,
  matchesSearch,
  schemasForSection,
  PER_STYLE_FIELDS,
  SECTIONS,
  groupPerStyleRows,
  parsePerStyleKey,
  schemaFor,
  sectionForConfigKey,
  sectionMatchesSearch,
  sectionsForSurface,
  type SectionId,
} from "./settingsSchema";

describe("SECTIONS", () => {
  it("names every section in the lexicon, in the order the page scrolls", () => {
    expect(SECTIONS.map((s) => s.id)).toEqual([
      "app",
      "generation",
      "expansion",
      "hosts",
      "styles",
      "licenses",
      "library",
      "media",
      "pairing",
      "performance",
      "accounts",
      "cloud",
      "styleDefaults",
      "profiles",
      "advanced",
      "updates",
    ]);
    expect(SECTIONS.map((s) => s.label)).toEqual([
      "Look",
      "Defaults for new images",
      "Write more for me",
      "Machines",
      "Styles & disk",
      "Style licences",
      "My images & trash",
      "Saving pictures & clips",
      "Phone pairing",
      "Speed & memory",
      "Accounts & tokens",
      "Cloud GPUs",
      "Per-style defaults",
      "Profiles",
      "Advanced",
      "Updates & about",
    ]);
  });

  it("declares which shells render each section", () => {
    for (const section of SECTIONS) {
      expect(section.surfaces.length, section.id).toBeGreaterThan(0);
    }
    // Saving pictures & clips is a native file-system concern: a browser tab
    // has no default save location to offer.
    expect(sectionsForSurface("web").map((s) => s.id)).not.toContain("media");
    expect(sectionsForSurface("desktop").map((s) => s.id)).toContain("media");
  });

  it("keeps the surface lists in schema order", () => {
    const order = SECTIONS.map((s) => s.id);
    for (const surface of ["web", "desktop"] as const) {
      const ids = sectionsForSurface(surface).map((s) => s.id);
      expect(
        [...ids].sort((a, b) => order.indexOf(a) - order.indexOf(b)),
      ).toEqual(ids);
    }
  });
});

describe("sectionForConfigKey", () => {
  it("routes a per-style override to Per-style defaults", () => {
    expect(sectionForConfigKey("models.flux-dev:q4.lora_scale")).toBe(
      "styleDefaults",
    );
    expect(sectionForConfigKey("models.sd1.5.default_steps")).toBe(
      "styleDefaults",
    );
  });

  it("keeps the TUI's own preferences out of a graphical surface", () => {
    expect(sectionForConfigKey("tui.theme")).toBeNull();
  });

  it("sends a curated key to its section and an unknown one to Advanced", () => {
    expect(sectionForConfigKey("expand.enabled")).toBe("expansion");
    expect(sectionForConfigKey("runpod.api_key")).toBe("cloud");
    expect(sectionForConfigKey("logging.level")).toBe("updates");
    expect(sectionForConfigKey("server_port")).toBe("performance");
    expect(sectionForConfigKey("queue.held_retention_days")).toBe(
      "performance",
    );
    expect(sectionForConfigKey("generate.auto_tag_title")).toBe("library");
    expect(sectionForConfigKey("umt5_variant")).toBe("generation");
    expect(sectionForConfigKey("definitely.not.a.key")).toBe("advanced");
  });
});

describe("parsePerStyleKey", () => {
  /* Style names carry dots and colons, so the field is what follows the LAST
   * dot — the same rule as `mold_core::config_keys::parse_model_key`. Splitting
   * on the first dot reads `sd1` as the style and `5.default_steps` as a field
   * nothing renders. */
  it("splits on the last dot, so a dotted or tagged style name survives", () => {
    expect(parsePerStyleKey("models.flux-dev:q4.lora_scale")).toEqual({
      style: "flux-dev:q4",
      field: "lora_scale",
    });
    expect(parsePerStyleKey("models.sd1.5.default_steps")).toEqual({
      style: "sd1.5",
      field: "default_steps",
    });
  });

  it("answers for nothing that is not a per-style key", () => {
    expect(parsePerStyleKey("models_dir")).toBeNull();
    expect(parsePerStyleKey("models.flux-dev")).toBeNull();
    expect(parsePerStyleKey("models..lora")).toBeNull();
    expect(parsePerStyleKey("models.flux-dev.")).toBeNull();
    expect(parsePerStyleKey("expand.enabled")).toBeNull();
  });
});

describe("groupPerStyleRows", () => {
  const row = (key: string) => ({ key });

  it("gives one group per style, styles alphabetical", () => {
    const groups = groupPerStyleRows([
      row("models.z-image.lora"),
      row("models.flux-dev:q4.default_steps"),
      row("models.z-image.default_steps"),
    ]);
    expect(groups.map((g) => g.style)).toEqual(["flux-dev:q4", "z-image"]);
  });

  it("orders a style's fields the way `mold config list` does, not alphabetically", () => {
    const groups = groupPerStyleRows([
      row("models.z-image.lora_scale"),
      row("models.z-image.default_steps"),
      row("models.z-image.default_guidance"),
    ]);
    expect(groups[0]?.rows.map((r) => r.key)).toEqual([
      "models.z-image.default_steps",
      "models.z-image.default_guidance",
      "models.z-image.lora_scale",
    ]);
  });

  it("keeps a field this client does not know, after the ones it does", () => {
    const groups = groupPerStyleRows([
      row("models.z-image.transformer"),
      row("models.z-image.default_steps"),
    ]);
    expect(groups[0]?.rows.map((r) => r.key)).toEqual([
      "models.z-image.default_steps",
      "models.z-image.transformer",
    ]);
  });

  it("ignores every row that is not a per-style override", () => {
    expect(
      groupPerStyleRows([row("models_dir"), row("expand.enabled")]),
    ).toEqual([]);
  });
});

describe("PER_STYLE_FIELDS", () => {
  it("is the eight fields a style may override", () => {
    expect(PER_STYLE_FIELDS).toEqual([
      "default_steps",
      "default_guidance",
      "default_width",
      "default_height",
      "scheduler",
      "negative_prompt",
      "lora",
      "lora_scale",
    ]);
  });
});

describe("the curated schemas", () => {
  it("says how long the trash keeps a picture, in the words the control offers", () => {
    // Moved from web's second schema: the option reads "Forever", so the help
    // may not say "0".
    const schema = schemaFor("gallery.trash_retention_days");
    expect(schema?.help).toContain(
      "Forever keeps them until you empty the trash",
    );
    expect(schema?.help).not.toContain("0 keeps");
  });

  it("edits a rented machine's key as a secret, never as text", () => {
    expect(schemaFor("runpod.api_key")?.editor).toBe("secret");
    expect(schemaFor("lambda.api_key")?.editor).toBe("secret");
  });

  it("picks a folder for every folder key, and types a file path", () => {
    for (const key of ["models_dir", "output_dir", "logging.dir"]) {
      expect(schemaFor(key)?.editor, key).toBe("path");
    }
    // The injected picker chooses FOLDERS; a private key is a file, and a
    // folder chooser with no way to type would make it unsettable.
    expect(schemaFor("lambda.ssh_private_key_path")?.editor).toBe("text");
  });

  it("asks for an engine restart only where the value is read at boot", () => {
    expect(schemaFor("server_port")?.needsEngineRestart).toBe(true);
    expect(schemaFor("logging.level")?.needsEngineRestart).toBe(true);
    // RunPod and Lambda are read per request, so nothing has to restart.
    expect(schemaFor("runpod.default_gpu")?.needsEngineRestart).toBeUndefined();
  });

  it("carries no duplicate key", () => {
    const keys = ENGINE_KEY_SCHEMAS.map((s) => s.key);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("sectionMatchesSearch", () => {
  const section = (id: SectionId) => SECTIONS.find((s) => s.id === id)!;

  it("matches every section on an empty query", () => {
    for (const s of SECTIONS)
      expect(sectionMatchesSearch("  ", s), s.id).toBe(true);
  });

  it("matches a label, a summary, a keyword, and a curated key alike", () => {
    expect(sectionMatchesSearch("look", section("app"))).toBe(true);
    expect(sectionMatchesSearch("runpod", section("cloud"))).toBe(true);
    expect(sectionMatchesSearch("expand.top_p", section("expansion"))).toBe(
      true,
    );
    expect(sectionMatchesSearch("expand.top_p", section("cloud"))).toBe(false);
  });

  /* Advanced used to be the only section holding raw rows; Per-style defaults
   * now holds far more of them, so the raw evidence is per section. */
  it("matches a raw key against the section that actually renders it", () => {
    const raw = {
      advanced: ["future.key"],
      styleDefaults: ["models.z-image.lora"],
    };
    expect(sectionMatchesSearch("future", section("advanced"), raw)).toBe(true);
    expect(sectionMatchesSearch("future", section("styleDefaults"), raw)).toBe(
      false,
    );
    expect(sectionMatchesSearch("z-image", section("styleDefaults"), raw)).toBe(
      true,
    );
    expect(sectionMatchesSearch("z-image", section("advanced"), raw)).toBe(
      false,
    );
  });
});

describe("settings schema", () => {
  it("routes curated keys to their sections", () => {
    expect(sectionForConfigKey("default_model")).toBe("generation");
    expect(sectionForConfigKey("expand.temperature")).toBe("expansion");
  });

  it("leads with Look and ends with Updates & about, in the lexicon", () => {
    expect(SECTIONS[0]).toMatchObject({ id: "app", label: "Look" });
    expect(SECTIONS.at(-1)).toMatchObject({
      id: "updates",
      label: "Updates & about",
    });
    expect(SECTIONS.find((section) => section.id === "hosts")?.label).toBe(
      "Machines",
    );
    expect(SECTIONS.find((section) => section.id === "library")?.label).toBe(
      "My images & trash",
    );
    expect(SECTIONS.find((section) => section.id === "expansion")?.label).toBe(
      "Write more for me",
    );
    expect(SECTIONS.find((section) => section.id === "styles")?.label).toBe(
      "Styles & disk",
    );
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
    expect(schemasForSection("styles").map((s) => s.key)).toEqual([
      "models_dir",
      "output_dir",
    ]);
    expect(schemasForSection("hosts")).toEqual([]);
  });

  it("unknown keys fall through to advanced — a key newer than the client must surface", () => {
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

  it("finds a section by a word that appears only in its summary sentence", () => {
    const look = SECTIONS.find((section) => section.id === "app")!;
    expect(sectionMatchesSearch("light or dark", look)).toBe(true);
    expect(
      sectionMatchesSearch(
        "light or dark",
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
    expect(schema.options?.map((o) => o.value)).toEqual([
      "1",
      "7",
      "30",
      "90",
      "365",
      "0",
    ]);
    expect(schema.options?.find((o) => o.value === "0")?.label).toBe("Forever");
    expect(schema.options?.find((o) => o.value === "30")?.label).toBe(
      "30 days",
    );
    expect(schema.help).toContain(
      "Forever keeps them until you empty the trash",
    );
    const library = SECTIONS.find((section) => section.id === "library")!;
    expect(library.label).toBe("My images & trash");
    expect(sectionMatchesSearch("trash", library)).toBe(true);
    expect(sectionMatchesSearch("retention", library)).toBe(true);
    expect(sectionMatchesSearch("collections", library)).toBe(true);
  });

  it("collects the curated schemas that belong to a section", () => {
    expect(schemasForSection("expansion").map((s) => s.key)).toContain(
      "expand.temperature",
    );
    expect(
      schemasForSection("performance").every(
        (s) => s.section === "performance",
      ),
    ).toBe(true);
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

  it("matches Advanced against a raw engine row key it renders", () => {
    expect(
      sectionMatchesSearch("future", advanced, {
        advanced: ["some.future.key"],
      }),
    ).toBe(true);
    expect(sectionMatchesSearch("future", advanced, {})).toBe(false);
  });

  it("an empty query matches every section", () => {
    expect(sectionMatchesSearch("  ", expansion)).toBe(true);
  });
});

describe("matchesSearch", () => {
  const item = {
    key: "env.MOLD_VAE_TILED",
    label: "Tiled VAE decode",
    help: "auto retries",
  };

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
