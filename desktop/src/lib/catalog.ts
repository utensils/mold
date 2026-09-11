/**
 * Catalog entry presentation for the desktop app.
 *
 * The SIZE/FETCH accounting, the acquisition label and the catalog-id helpers
 * now live in `@studio/lib/catalogLabel` so web, desktop and the phone read one
 * source; they are re-exported here so every existing desktop import keeps
 * working. What stays below is desktop-only: the model-page link and the
 * installed-first ordering.
 */
export {
  catalogFetchCaption,
  catalogIdentityKey,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  isCatalogId,
} from "@studio/lib/catalogLabel";
export type { CatalogSizeInfo } from "@studio/lib/catalogLabel";

import type { CatalogEntry } from "./api/types";

/**
 * Human-facing model page for a catalog entry, or null when none exists.
 *
 * New servers send `page_url` on the wire; prefer it. For older servers the
 * HF page is recoverable from the repo id (`source_id`, falling back to the
 * `hf:`-stripped catalog id), except for `hf:companion/*` pseudo-ids whose
 * path is a curated bundle name, not a repo. Civitai model pages need the
 * parent model id which the old wire never carried — no link beats a 404.
 */
export function catalogPageUrl(
  entry: Pick<CatalogEntry, "id" | "source_id" | "page_url">,
): string | null {
  if (entry.page_url) return entry.page_url;
  if (!entry.id.startsWith("hf:")) return null;
  const repo = entry.source_id || entry.id.slice("hf:".length);
  if (!entry.source_id && repo.startsWith("companion/")) return null;
  return `https://huggingface.co/${repo}`;
}

/**
 * Installed entries surface first (stable within each group) so "what do I
 * already have?" is answered at the top of the catalog instead of scattered
 * through the ranking.
 */
export function sortInstalledFirst<T extends { installed?: boolean }>(entries: T[]): T[] {
  const installed = entries.filter((e) => e.installed);
  const available = entries.filter((e) => !e.installed);
  return [...installed, ...available];
}
