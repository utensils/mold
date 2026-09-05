/**
 * Pure view logic for the catalog detail drawer: the itemized "what will
 * this pull actually download" list (primary weights files + shared
 * companions with per-file sizes and a computed total) and the
 * Pull-vs-Repair action state. Framework-free so the contracts are unit
 * testable (project test rule).
 */
import { modelSource } from "./modelSource";
import { modelDisplayName } from "./models";
import { modelKindValue } from "@studio/lib/modelMetadata";
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
  const description = m.description?.trim() ?? "";
  const repeatsDisplayName =
    description.length > 0 &&
    (description.localeCompare(displayName.trim(), undefined, { sensitivity: "accent" }) === 0 ||
      description.toLocaleLowerCase().startsWith(`${displayName.trim().toLocaleLowerCase()} by `));
  const modality = m.modality?.trim();
  return {
    id: m.name,
    source: modelSource(m),
    source_id: sourceId,
    // `name` stays the raw identifier — load/unload/remove and dedup key
    // off it. The human title rides along as additive `display_name`.
    name: m.name,
    display_name: displayName !== m.name ? displayName : null,
    family: m.family,
    kind: modelKindValue({ kind: m.kind ?? null, family: m.family }),
    ...(modality ? { modality } : {}),
    nsfw: m.nsfw ?? null,
    installed: true,
    size_bytes: m.size_gb > 0 ? Math.round(m.size_gb * 1_000_000_000) : null,
    page_url: hfRepo ? `https://huggingface.co/${hfRepo}` : null,
    description: description && !repeatsDisplayName ? description : null,
    thumbnail_url: null,
  };
}

function nonEmptyText(
  detail: string | null | undefined,
  summary: string | null | undefined,
): string | null {
  if (detail?.trim()) return detail;
  if (summary?.trim()) return summary;
  return detail ?? summary ?? null;
}

function nonEmptyList<T>(detail: T[] | null | undefined, summary: T[] | null | undefined): T[] {
  if (detail?.length) return detail;
  if (summary?.length) return summary;
  return detail ?? summary ?? [];
}

function nonZeroCount(
  detail: number | null | undefined,
  summary: number | null | undefined,
): number | null {
  if (detail != null && detail > 0) return detail;
  if (summary != null && summary > 0) return summary;
  return detail ?? summary ?? null;
}

/**
 * Overlay a fetched catalog detail onto the clicked/search summary without
 * letting sparse/default detail fields erase metadata the summary already
 * knew. The clicked thumbnail intentionally remains authoritative because
 * upstream detail endpoints may select a different preview for the same
 * model version.
 */
export function mergeCatalogSummaryDetail(
  summary: CatalogEntry,
  detail: CatalogEntry,
): CatalogEntry {
  return {
    ...summary,
    ...detail,
    installed: summary.installed || detail.installed,
    nsfw: summary.nsfw === true || detail.nsfw === true ? true : (detail.nsfw ?? summary.nsfw),
    author: nonEmptyText(detail.author, summary.author),
    description: nonEmptyText(detail.description, summary.description),
    license: nonEmptyText(detail.license, summary.license),
    page_url: nonEmptyText(detail.page_url, summary.page_url),
    thumbnail_url: nonEmptyText(summary.thumbnail_url, detail.thumbnail_url),
    tags: nonEmptyList(detail.tags, summary.tags),
    trained_words: nonEmptyList(detail.trained_words, summary.trained_words),
    download_count: nonZeroCount(detail.download_count, summary.download_count),
    likes: nonZeroCount(detail.likes, summary.likes),
    rating: detail.rating ?? summary.rating ?? null,
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
 * An entry already on the machine repairs (the catalog download re-fetches
 * only the missing files server-side); everything else is a fresh download.
 */
export function catalogNeedsRepair(entry: Pick<CatalogEntry, "installed">): boolean {
  return !!entry.installed;
}

/** Older servers don't report support explicitly; keep those entries pullable. */
export function canDownloadEntry(entry: Pick<CatalogEntry, "supported">): boolean {
  return entry.supported == null || entry.supported;
}
