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
