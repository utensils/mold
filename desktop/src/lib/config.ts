/**
 * Pure helpers for the Settings config surface: which tab a key belongs to,
 * whether a row is env-locked, and grouping rows into tabs with a stable
 * provenance ordering. The tab mapping mirrors mold's config key prefixes
 * (`expand.*` / `generate.*` are generation prefs; `models_dir` / ports are
 * engine bootstrap; the rest is advanced).
 */
import type { ConfigRow, ConfigSource } from "./api/types";

/** Provenance tagging is the shared kit's — one glyph vocabulary for every
 *  surface that renders a config row. */
export { isRowLocked, provenance, type ProvenanceTag } from "@studio/api/config";

export type ConfigTab = "engine" | "generation" | "advanced";

/** Precedence order used for both the provenance tag and the sort. */
const SOURCE_ORDER: ConfigSource[] = ["env", "file", "db", "default"];

/** Tab a config key belongs to. Every key surfaces somewhere. */
export function tabForKey(key: string): ConfigTab {
  if (key.startsWith("expand.") || key.startsWith("generate.")) return "generation";
  if (key === "models_dir" || key.includes("port")) return "engine";
  return "advanced";
}

/**
 * Bucket config rows into tabs. Within a tab, rows sort by
 * source precedence (env → file → db → default) so env-locked rows surface
 * first, then alphabetically by key. Pure.
 */
export function groupConfigRows(rows: ConfigRow[]): Record<ConfigTab, ConfigRow[]> {
  const groups: Record<ConfigTab, ConfigRow[]> = { engine: [], generation: [], advanced: [] };
  for (const row of rows) {
    groups[tabForKey(row.key)].push(row);
  }
  const bySourceThenKey = (a: ConfigRow, b: ConfigRow) => {
    const d = SOURCE_ORDER.indexOf(a.source) - SOURCE_ORDER.indexOf(b.source);
    return d !== 0 ? d : a.key.localeCompare(b.key);
  };
  for (const tab of Object.keys(groups) as ConfigTab[]) groups[tab].sort(bySourceThenKey);
  return groups;
}
