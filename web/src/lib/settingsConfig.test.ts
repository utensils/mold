import { describe, expect, it } from "vitest";
import {
  CONFIG_SCHEMAS,
  CONFIG_SECTIONS,
  canResetConfig,
  schemaForRow,
} from "./settingsConfig";
import {
  RETENTION_OPTIONS,
  retentionLabel,
} from "@studio/lib/libraryOrganization";

describe("settingsConfig · Library", () => {
  it("places the Library section between Generation and Prompt expansion", () => {
    const generation = CONFIG_SECTIONS.indexOf("Generation");
    const library = CONFIG_SECTIONS.indexOf("Library");
    const expansion = CONFIG_SECTIONS.indexOf("Prompt expansion");
    expect(library).toBe(generation + 1);
    expect(expansion).toBe(library + 1);
  });

  it("describes gallery.trash_retention_days as a numeric select over the shared retention choices", () => {
    const schema = schemaForRow({
      key: "gallery.trash_retention_days",
      value: 30,
      source: "db",
    });
    expect(schema.section).toBe("Library");
    expect(schema.editor).toBe("select");
    expect(schema.valueType).toBe("number");
    expect(schema.options).toEqual(RETENTION_OPTIONS.map(String));
    expect(schema.optionLabels?.["0"]).toBe("Forever");
    expect(schema.optionLabels?.["30"]).toBe(retentionLabel(30));
    expect(schema.help).toBe(
      "Prints moved to the trash are deleted forever after this long. Forever keeps them until you empty the trash.",
    );
    expect(CONFIG_SCHEMAS.some((s) => s.key === schema.key)).toBe(true);
  });

  it("lets the gallery.* keys reset to their fallback", () => {
    expect(canResetConfig("gallery.trash_retention_days")).toBe(true);
    expect(canResetConfig("gallery.future_key")).toBe(true);
    expect(canResetConfig("models_dir")).toBe(false);
  });
});
