/**
 * Option building for the ControlNet model picker — a port of the web SPA's
 * GenerateParamsPanel logic (`builtInControlNetModels` +
 * `catalogControlNetOptions` + the trailing unknown-current option), collapsed
 * into one pure function so the desktop select and its tests share it.
 *
 * Sources, in render order:
 *  1. Installed models with family `controlnet` (from `/api/models`), sorted
 *     downloaded-first then by name; not-downloaded entries stay listed but
 *     disabled with a " (missing)" hint. Option value = model name.
 *  2. Catalog-installed control-net entries (`GET /api/catalog/installed`),
 *     kept only when installed with a resolvable weights file. Option value =
 *     `primary_path`. Deduped against (1) by name and against each other by
 *     value.
 *  3. A trailing option for a current form value nothing else covers (e.g.
 *     restored from gallery metadata) so the select never silently drops it.
 */
import type { CatalogEntry, ModelEntry } from "./api/types";

export interface ControlNetOption {
  /** What ships as `control_model` on the wire. */
  value: string;
  label: string;
  disabled: boolean;
}

export function buildControlNetOptions(
  installed: readonly Pick<ModelEntry, "name" | "family" | "downloaded">[],
  catalog: readonly Pick<CatalogEntry, "name" | "installed" | "primary_path">[],
  current: string,
): ControlNetOption[] {
  const builtIn = installed
    .filter((m) => m.family === "controlnet")
    .slice()
    .sort((a, b) => Number(b.downloaded) - Number(a.downloaded) || a.name.localeCompare(b.name))
    .map<ControlNetOption>((m) => ({
      value: m.name,
      label: m.downloaded ? m.name : `${m.name} (missing)`,
      disabled: !m.downloaded,
    }));

  const names = new Set(builtIn.map((o) => o.value));
  const values = new Set(names);
  const options = builtIn;
  for (const entry of catalog) {
    if (!entry.installed || !entry.primary_path) continue;
    if (names.has(entry.name) || values.has(entry.primary_path)) continue;
    names.add(entry.name);
    values.add(entry.primary_path);
    options.push({ value: entry.primary_path, label: entry.name, disabled: false });
  }

  const trimmed = current.trim();
  if (trimmed && !values.has(trimmed)) {
    options.push({ value: trimmed, label: trimmed, disabled: false });
  }
  return options;
}
