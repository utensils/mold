/**
 * Pure view logic for the catalog detail drawer: the itemized "what will
 * this pull actually download" list (primary weights files + shared
 * companions with per-file sizes and a computed total) and the
 * Pull-vs-Repair action state. Framework-free so the contracts are unit
 * testable (project test rule).
 */
import type { CatalogEntry } from "./api/types";

export interface DownloadContentItem {
  /** Stable render key (`primary:<dest>` / `companion:<name>`). */
  key: string;
  label: string;
  kind: string;
  sizeBytes: number | null;
}

/**
 * Itemize everything a pull fetches: the primary `download_recipe.files`
 * plus each shared companion. Older servers omit both fields — the list is
 * simply empty and the drawer hides the section.
 */
export function buildDownloadContents(
  entry: Pick<CatalogEntry, "download_recipe" | "companion_details">,
): DownloadContentItem[] {
  const primary = (entry.download_recipe?.files ?? []).map((file) => ({
    key: `primary:${file.dest}`,
    label: file.dest.split("/").pop() || "Primary model",
    kind: "primary",
    sizeBytes: file.size_bytes ?? null,
  }));
  const companions = (entry.companion_details ?? []).map((companion) => ({
    key: `companion:${companion.name}`,
    label: companion.name,
    kind: companion.kind ?? "component",
    sizeBytes: companion.size_bytes ?? null,
  }));
  return [...primary, ...companions];
}

export interface DownloadContentsTotal {
  /** Sum of the known per-file sizes; null when nothing reports one. */
  bytes: number | null;
  /**
   * `true` only when every item reported a size, so `bytes` is the exact
   * total. `false` means some items lack a size and `bytes` is a lower bound
   * (the drawer prefixes it with "≥").
   */
  complete: boolean;
}

/**
 * Total of the known per-file sizes. When some items omit their size the sum
 * is a lower bound, not the true total — `complete` distinguishes the two so
 * callers can present the partial sum honestly.
 */
export function downloadContentsTotalBytes(items: DownloadContentItem[]): DownloadContentsTotal {
  const total = items.reduce((sum, item) => sum + (item.sizeBytes ?? 0), 0);
  const complete = items.every((item) => item.sizeBytes != null);
  return { bytes: total > 0 ? total : null, complete };
}

/**
 * Installed entries repair (the catalog download re-fetches only missing
 * files server-side); everything else pulls.
 */
export function catalogActionLabel(entry: Pick<CatalogEntry, "installed">): "Pull" | "Repair" {
  return entry.installed ? "Repair" : "Pull";
}

/**
 * Phases `>= 6` are catalog packages no shipped engine can run yet. Older
 * servers don't report a phase — treat their entries as downloadable, same
 * as the search path does.
 */
export function canDownloadEntry(entry: Pick<CatalogEntry, "engine_phase">): boolean {
  return entry.engine_phase == null || entry.engine_phase <= 5;
}
