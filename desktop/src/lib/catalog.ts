/**
 * SIZE vs FETCH accounting for catalog entries, rendered honestly.
 *
 * The `/api/catalog/search` wire gives the primary weights (`size_bytes`) and
 * each shared component's size (`companion_details[].size_bytes`), but it does
 * NOT report which companions are already on disk. So the honest figures we can
 * compute are:
 *   - SIZE  = primary weights only.
 *   - FETCH = weights + every shared component = the full download, worst case.
 * FETCH ≥ SIZE always (matches the project's established SIZE/FETCH semantics).
 * We can't claim "the rest is already downloaded" without per-companion
 * presence, so the caption says what's actually true: FETCH is the total pull
 * including shared components.
 */
import type { CatalogEntry } from "./api/types";
import { formatGB } from "./format";

export interface CatalogSizeInfo {
  /** Primary model weights, bytes. */
  weightsBytes: number | null;
  /** Weights + all shared components, bytes — the full download. */
  fetchBytes: number | null;
  /** True when shared components make the fetch larger than the weights. */
  differs: boolean;
}

export function catalogSizeInfo(entry: CatalogEntry): CatalogSizeInfo {
  const weights = entry.size_bytes ?? null;
  const companions = (entry.companion_details ?? []).reduce(
    (sum, c) => sum + (c.size_bytes ?? 0),
    0,
  );
  const fetch = weights != null ? weights + companions : companions > 0 ? companions : null;
  const differs = weights != null && fetch != null && fetch !== weights;
  return { weightsBytes: weights, fetchBytes: fetch, differs };
}

/** "SIZE 23.9 GB · FETCH 31.2 GB" when they differ, else "SIZE 23.9 GB". */
export function catalogSizeLabel(info: CatalogSizeInfo): string {
  const size = info.weightsBytes != null ? formatGB(info.weightsBytes) : "—";
  if (!info.differs || info.fetchBytes == null) return `SIZE ${size}`;
  return `SIZE ${size} · FETCH ${formatGB(info.fetchBytes)}`;
}

/** Caption shown under the size line only when a fetch adds shared components. */
export function catalogFetchCaption(info: CatalogSizeInfo): string | null {
  if (!info.differs || info.fetchBytes == null) return null;
  return `${formatGB(info.fetchBytes)} to download, including shared components`;
}

/** Pull button label — the number the user will actually download. */
export function catalogPullLabel(info: CatalogSizeInfo): string {
  const bytes = info.fetchBytes ?? info.weightsBytes;
  return bytes != null ? `Pull · ${formatGB(bytes)}` : "Pull";
}

/** Catalog ids look like `cv:8001` / `hf:author/model`; plain names don't. */
export function isCatalogId(id: string): boolean {
  return id.startsWith("cv:") || id.startsWith("hf:");
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
