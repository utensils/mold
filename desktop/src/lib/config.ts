/**
 * Pure helpers for the Settings config surface: which tab a key belongs to,
 * whether a row is env-locked, and grouping rows into tabs with a stable
 * provenance ordering. The tab mapping mirrors mold's config key prefixes
 * (`expand.*` / `generate.*` are generation prefs; `models_dir` / ports are
 * engine bootstrap; `tui.*` is desktop-irrelevant and skipped; the rest is
 * advanced).
 */
import type { ConfigRow, ConfigSource } from "./api/types";

export type ConfigTab = "engine" | "generation" | "advanced";

/** Precedence order used for both the provenance tag and the sort. */
const SOURCE_ORDER: ConfigSource[] = ["env", "file", "db", "default"];

export interface ProvenanceTag {
  glyph: string;
  label: string;
}

export function provenance(source: ConfigSource): ProvenanceTag {
  switch (source) {
    case "db":
      return { glyph: "⌂", label: "db" };
    case "file":
      return { glyph: "⛁", label: "file" };
    case "env":
      return { glyph: "⚿", label: "env" };
    default:
      return { glyph: "·", label: "default" };
  }
}

/** Env-resolved rows are locked — the environment wins over any stored value. */
export function isRowLocked(row: ConfigRow): boolean {
  return row.source === "env";
}

/** Tab a config key belongs to, or null when it should not surface (tui.*). */
export function tabForKey(key: string): ConfigTab | null {
  if (key.startsWith("tui.")) return null;
  if (key.startsWith("expand.") || key.startsWith("generate.")) return "generation";
  if (key === "models_dir" || key.includes("port")) return "engine";
  return "advanced";
}

/**
 * Bucket config rows into tabs, skipping tui.*. Within a tab, rows sort by
 * source precedence (env → file → db → default) so env-locked rows surface
 * first, then alphabetically by key. Pure.
 */
export function groupConfigRows(rows: ConfigRow[]): Record<ConfigTab, ConfigRow[]> {
  const groups: Record<ConfigTab, ConfigRow[]> = { engine: [], generation: [], advanced: [] };
  for (const row of rows) {
    const tab = tabForKey(row.key);
    if (tab) groups[tab].push(row);
  }
  const bySourceThenKey = (a: ConfigRow, b: ConfigRow) => {
    const d = SOURCE_ORDER.indexOf(a.source) - SOURCE_ORDER.indexOf(b.source);
    return d !== 0 ? d : a.key.localeCompare(b.key);
  };
  for (const tab of Object.keys(groups) as ConfigTab[]) groups[tab].sort(bySourceThenKey);
  return groups;
}
