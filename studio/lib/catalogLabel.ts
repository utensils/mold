/**
 * SIZE vs FETCH accounting for catalog entries, rendered honestly — shared by
 * every Studio shell so a style's download total reads the same on web, on the
 * desktop app and on the phone.
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
 *
 * The entry types are STRUCTURAL: web's `CatalogEntryWire` and desktop's
 * `CatalogEntry` both satisfy them, and neither shell's wire type is imported
 * here — `studio/` is the lower layer.
 */

/** The least a catalog row needs for the size accounting. */
export interface CatalogSizeEntry {
  size_bytes?: number | null;
  companion_details?: readonly { size_bytes?: number | null }[] | null;
}

/** The least a catalog row needs for a stable upstream identity. */
export interface CatalogIdentityEntry {
  source: string;
  source_id?: string | null;
}

/** Decimal units, one decimal place — matches mold's shared byte formatting.
 *  Module-private on purpose: this is the catalog's own rounding, not a second
 *  public byte formatter beside `formatBytes`. */
function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export interface CatalogSizeInfo {
  /** Primary model weights, bytes. */
  weightsBytes: number | null;
  /** Weights + all shared components, bytes — the full download. */
  fetchBytes: number | null;
  /** True when shared components make the fetch larger than the weights. */
  differs: boolean;
}

export function catalogSizeInfo(entry: CatalogSizeEntry): CatalogSizeInfo {
  const weights = entry.size_bytes ?? null;
  const companions = (entry.companion_details ?? []).reduce(
    (sum, c) => sum + (c.size_bytes ?? 0),
    0,
  );
  const fetch =
    weights != null ? weights + companions : companions > 0 ? companions : null;
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

/**
 * The acquisition button's label — the number the user will actually download.
 * The VERB is the caller's: every Studio surface passes the lexicon's "Get it",
 * and the bare default exists only so a caller cannot end up with no word at
 * all.
 */
export function catalogPullLabel(info: CatalogSizeInfo, verb = "Pull"): string {
  const bytes = info.fetchBytes ?? info.weightsBytes;
  return bytes != null ? `${verb} · ${formatGB(bytes)}` : verb;
}

/** Catalog ids look like `cv:8001` / `hf:author/model`; plain names don't. */
export function isCatalogId(id: string): boolean {
  return id.startsWith("cv:") || id.startsWith("hf:");
}

/**
 * Stable upstream identity for rows whose local ids differ across server
 * generations. Human-facing `name` is deliberately excluded: titles are
 * neither unique nor immutable and must never merge unrelated models.
 */
export function catalogIdentityKey(entry: CatalogIdentityEntry): string | null {
  const source = entry.source.trim().toLocaleLowerCase();
  const sourceId = entry.source_id?.trim();
  return source && sourceId ? `${source}:${sourceId}` : null;
}
