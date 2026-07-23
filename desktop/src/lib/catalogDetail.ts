/**
 * Pure view logic for the catalog detail drawer: the itemized "what will
 * this pull actually download" list (primary weights files + shared
 * companions with per-file sizes and a computed total) and the
 * Pull-vs-Repair action state. Framework-free so the contracts are unit
 * testable (project test rule).
 */
import { modelSource } from "./modelSource";
import { modelDisplayName } from "./models";
import type { CatalogEntry, ModelEntry } from "./api/types";

/**
 * Present an installed model (`GET /api/models` row) in the drawer's entry
 * shape, so the Installed shelf and host pages open the same detail drawer
 * as the catalog. Catalog installs are already named by their catalog id
 * (`cv:…` / `hf:…`), and manifest models resolve by name — either way the
 * drawer's `GET /api/catalog/:id` enrichment and component listing key off
 * `id`, and `installed: true` flips the action to Repair. Source follows
 * the same classification as the row glyphs (`modelSource`), and catalog
 * ids keep their upstream repo / version id from the name itself.
 */
export function installedModelToEntry(m: ModelEntry): CatalogEntry {
  const catalogNamed = m.name.startsWith("cv:") || m.name.startsWith("hf:");
  const sourceId = catalogNamed ? m.name.slice(3) : m.hf_repo || null;
  const hfRepo = m.name.startsWith("hf:") ? m.name.slice(3) : m.hf_repo || null;
  const displayName = modelDisplayName(m);
  return {
    id: m.name,
    source: modelSource(m),
    source_id: sourceId,
    // `name` stays the raw identifier — load/unload/remove and dedup key
    // off it. The human title rides along as additive `display_name`.
    name: m.name,
    display_name: displayName !== m.name ? displayName : null,
    family: m.family,
    kind: "checkpoint",
    nsfw: false,
    installed: true,
    size_bytes: m.size_gb > 0 ? Math.round(m.size_gb * 1_000_000_000) : null,
    page_url: hfRepo ? `https://huggingface.co/${hfRepo}` : null,
    description: m.description || null,
    thumbnail_url: null,
  };
}

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

/** Older servers don't report support explicitly; keep those entries pullable. */
export function canDownloadEntry(entry: Pick<CatalogEntry, "supported">): boolean {
  return entry.supported == null || entry.supported;
}
