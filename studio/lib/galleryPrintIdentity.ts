export interface GalleryPrintIdentityInput {
  filename: string;
  timestamp: number;
  size_bytes?: number | null;
  metadata_synthetic?: boolean;
  metadata?: {
    seed?: number | null;
    model?: string | null;
  } | null;
}

/**
 * Identity matches only count as one print when the rows were written around
 * the same time. Mirrors land within seconds of their origin, while a genuine
 * regeneration that happens to reuse a seed and byte length must stay distinct.
 */
export const GALLERY_IDENTITY_WINDOW_SECS = 3600;

/**
 * Synthetic legacy rows recorded seed 0 as "unknown". Auto-save filenames
 * retain the real seed, so recover it only from that trusted filename shape.
 */
export function galleryIdentitySeed(
  item: GalleryPrintIdentityInput,
): number | null {
  if (item.metadata_synthetic) {
    const match = /-(\d+)-(\d+)(?:-(?:original|upscaled))?\.[a-z0-9]+$/i.exec(
      item.filename,
    );
    return match ? Number(match[1]) : null;
  }
  return item.metadata?.seed ?? null;
}

function modelIdentitySlug(model: string | null | undefined): string {
  return (model ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Cross-host identity beyond filename. Mirrored copies are byte-identical, so
 * seed + exact byte size + model joins legacy copies whose filenames diverged.
 */
export function galleryPrintIdentity(
  item: GalleryPrintIdentityInput,
): string | null {
  const size = item.size_bytes;
  const seed = galleryIdentitySeed(item);
  if (!size || seed == null) return null;
  return `${seed}:${size}:${modelIdentitySlug(item.metadata?.model)}`;
}

export function sameLogicalGalleryPrint(
  a: GalleryPrintIdentityInput,
  b: GalleryPrintIdentityInput,
): boolean {
  if (a.filename === b.filename) return true;
  const identity = galleryPrintIdentity(a);
  return (
    identity !== null &&
    identity === galleryPrintIdentity(b) &&
    Math.abs(a.timestamp - b.timestamp) <= GALLERY_IDENTITY_WINDOW_SECS
  );
}

export interface LogicalGalleryPrintGroup<T> {
  /** The first copy in the input order, used as the rendered print. */
  representative: T;
  /** Every physical copy, including the representative. */
  copies: T[];
}

/**
 * Collapse physical copies from any number of hosts into logical prints.
 * Callers control which copy represents a group by ordering the input first.
 */
export function groupLogicalGalleryPrints<T extends GalleryPrintIdentityInput>(
  items: readonly T[],
): LogicalGalleryPrintGroup<T>[] {
  const groups: LogicalGalleryPrintGroup<T>[] = [];
  const byFilename = new Map<string, LogicalGalleryPrintGroup<T>>();
  const byIdentity = new Map<string, LogicalGalleryPrintGroup<T>[]>();

  for (const item of items) {
    const identity = galleryPrintIdentity(item);
    let group = byFilename.get(item.filename);
    if (!group && identity) {
      group = byIdentity
        .get(identity)
        ?.find((candidate) =>
          candidate.copies.some(
            (copy) =>
              Math.abs(copy.timestamp - item.timestamp) <=
              GALLERY_IDENTITY_WINDOW_SECS,
          ),
        );
    }

    if (!group) {
      group = { representative: item, copies: [] };
      groups.push(group);
    }
    group.copies.push(item);
    byFilename.set(item.filename, group);
    if (identity) {
      const candidates = byIdentity.get(identity) ?? [];
      if (!candidates.includes(group)) candidates.push(group);
      byIdentity.set(identity, candidates);
    }
  }

  return groups;
}
