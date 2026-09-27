/**
 * Whether a print is drawn over the checkerboard alpha bed (`.ms-alpha-bed`).
 *
 * `OutputMetadata.has_alpha` is a fact about the stored FILE — the encoder
 * sets it when any pixel is below full opacity, which also covers an edit of
 * a transparent reference rendered with the toggle off. `transparent_background`
 * is the REQUEST, so a queue row or an in-flight preview (whose metadata IS the
 * request) gets the bed before the file exists. Every other print keeps the
 * plain media bed, so an opaque picture never sits on a pattern it does not
 * need.
 */
export interface AlphaMetadata {
  has_alpha?: boolean | null;
  transparent_background?: boolean | null;
}

export function showsAlphaBed(
  item: { metadata?: AlphaMetadata | null } | null | undefined,
): boolean {
  const metadata = item?.metadata;
  if (!metadata) return false;
  return (
    metadata.has_alpha === true || metadata.transparent_background === true
  );
}
