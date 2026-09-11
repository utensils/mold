import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/api/config", () => ({
  fetchConfig: vi.fn(() => Promise.resolve([])),
  fetchProfiles: vi.fn(() => Promise.resolve({ profiles: [], active: "default" })),
  setConfig: vi.fn(() => Promise.resolve()),
  resetConfig: vi.fn(() => Promise.resolve()),
  setProfile: vi.fn(() => Promise.resolve()),
}));

import { useSettingsConfigStore } from "./settingsConfig";
import type { ConfigRow } from "../lib/api/types";

function row(key: string, value: ConfigRow["value"] = "x"): ConfigRow {
  return { key, value, source: "db", env_var: null, restart_required: false };
}

beforeEach(() => {
  setActivePinia(createPinia());
});

/*
 * The Settings page asks the store for a section's rows rather than naming
 * keys twice. Cloud GPUs renders every `runpod.*`/`lambda.*` row the engine
 * reports, Per-style defaults renders `models.<style>.<field>`, and the shell's
 * search needs the raw keys each section is actually drawing — Advanced's
 * unknown keys AND the per-style ones, which are raw rows too.
 */
describe("settingsConfig rowsForSection", () => {
  it("returns the curated rows a section owns, in the schema's order", () => {
    const config = useSettingsConfigStore();
    config.rows = [
      row("runpod.endpoint"),
      row("runpod.api_key"),
      row("expand.model"),
      row("lambda.api_key"),
    ];
    expect(config.rowsForSection("cloud").map((r) => r.key)).toEqual([
      "runpod.api_key",
      "runpod.endpoint",
      "lambda.api_key",
    ]);
    expect(config.rowsForSection("expansion").map((r) => r.key)).toEqual(["expand.model"]);
  });

  it("keeps a key this build has never heard of, after the ones it knows", () => {
    const config = useSettingsConfigStore();
    config.rows = [row("zz.future_key"), row("aa.another_future_key")];
    // Both route to Advanced, and neither has a schema position — so they sort
    // by key rather than vanishing.
    expect(config.rowsForSection("advanced").map((r) => r.key)).toEqual([
      "aa.another_future_key",
      "zz.future_key",
    ]);
  });

  it("never surfaces a tui.* row in any section", () => {
    const config = useSettingsConfigStore();
    config.rows = [row("tui.theme"), row("models_dir")];
    for (const id of ["advanced", "styles", "app"] as const) {
      expect(config.rowsForSection(id).map((r) => r.key), id).not.toContain("tui.theme");
    }
    expect(config.rowsForSection("styles").map((r) => r.key)).toEqual(["models_dir"]);
  });
});

describe("settingsConfig perStyleRows", () => {
  it("collects every models.<style>.<field> row and leaves the rest alone", () => {
    const config = useSettingsConfigStore();
    config.rows = [
      row("models.sd1.5.default_steps", 28),
      row("models.flux-dev:q4.lora_scale", 0.8),
      row("models_dir"),
      row("default_steps", 20),
    ];
    expect(config.perStyleRows.map((r) => r.key)).toEqual([
      "models.sd1.5.default_steps",
      "models.flux-dev:q4.lora_scale",
    ]);
  });

  it("keeps those rows out of Advanced — 104 of them is the whole page", () => {
    const config = useSettingsConfigStore();
    config.rows = [row("models.sd1.5.default_steps", 28), row("zz.future_key")];
    expect(config.advancedRows.map((r) => r.key)).toEqual(["zz.future_key"]);
  });
});

describe("settingsConfig rawKeysBySection", () => {
  it("reports the raw keys each section draws, keyed by that section", () => {
    const config = useSettingsConfigStore();
    config.rows = [
      row("zz.future_key"),
      row("models.sd1.5.default_steps", 28),
      row("models.nebula.negative_prompt"),
      row("models_dir"),
    ];
    // Advanced's rows are keys no schema knows; the per-style ones belong to
    // Per-style defaults, so searching a style name finds that section instead
    // of an Advanced list it is no longer in.
    expect(config.rawKeysBySection).toEqual({
      advanced: ["zz.future_key"],
      styleDefaults: ["models.sd1.5.default_steps", "models.nebula.negative_prompt"],
    });
  });

  it("reports empty lists on a current engine", () => {
    const config = useSettingsConfigStore();
    config.rows = [row("models_dir"), row("expand.model")];
    expect(config.rawKeysBySection).toEqual({ advanced: [], styleDefaults: [] });
  });
});
