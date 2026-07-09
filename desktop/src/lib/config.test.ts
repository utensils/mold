import { describe, expect, it } from "vitest";
import { groupConfigRows, isRowLocked, provenance, tabForKey } from "./config";
import type { ConfigRow, ConfigSource } from "./api/types";

const row = (key: string, source: ConfigSource = "db"): ConfigRow => ({ key, value: "x", source });

describe("tabForKey", () => {
  it("routes generation prefixes to the generation tab", () => {
    expect(tabForKey("expand.model")).toBe("generation");
    expect(tabForKey("generate.steps")).toBe("generation");
  });
  it("routes bootstrap keys to the engine tab", () => {
    expect(tabForKey("models_dir")).toBe("engine");
    expect(tabForKey("server.port")).toBe("engine");
  });
  it("skips tui keys", () => {
    expect(tabForKey("tui.theme")).toBeNull();
  });
  it("sends everything else to advanced", () => {
    expect(tabForKey("logging.level")).toBe("advanced");
    expect(tabForKey("runpod.api_key")).toBe("advanced");
  });
});

describe("isRowLocked", () => {
  it("locks only env-sourced rows", () => {
    expect(isRowLocked(row("k", "env"))).toBe(true);
    expect(isRowLocked(row("k", "db"))).toBe(false);
    expect(isRowLocked(row("k", "file"))).toBe(false);
    expect(isRowLocked(row("k", "default"))).toBe(false);
  });
});

describe("provenance", () => {
  it("maps each source to its glyph and label", () => {
    expect(provenance("db")).toEqual({ glyph: "⌂", label: "db" });
    expect(provenance("file")).toEqual({ glyph: "⛁", label: "file" });
    expect(provenance("env")).toEqual({ glyph: "⚿", label: "env" });
    expect(provenance("default")).toEqual({ glyph: "·", label: "default" });
  });
});

describe("groupConfigRows", () => {
  it("buckets by tab, skips tui, and orders env → file → db → default then key", () => {
    const rows: ConfigRow[] = [
      row("generate.steps", "db"),
      row("generate.width", "env"),
      row("expand.model", "file"),
      row("tui.theme", "db"),
      row("models_dir", "file"),
      row("logging.level", "default"),
      row("runpod.token", "db"),
    ];
    const groups = groupConfigRows(rows);

    // tui skipped entirely.
    expect(groups.generation.map((r) => r.key)).toEqual([
      "generate.width", // env first
      "expand.model", // file next
      "generate.steps", // db last
    ]);
    expect(groups.engine.map((r) => r.key)).toEqual(["models_dir"]);
    expect(groups.advanced.map((r) => r.key)).toEqual(["runpod.token", "logging.level"]);
  });

  it("returns empty buckets for no rows", () => {
    expect(groupConfigRows([])).toEqual({ engine: [], generation: [], advanced: [] });
  });
});
