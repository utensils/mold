import { describe, expect, it } from "vitest";
import {
  ENGINE_KEY_SCHEMAS,
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

  it("picks a folder for every path key", () => {
    for (const key of [
      "models_dir",
      "output_dir",
      "logging.dir",
      "lambda.ssh_private_key_path",
    ]) {
      expect(schemaFor(key)?.editor, key).toBe("path");
    }
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
