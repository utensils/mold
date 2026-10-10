/** Ordered, uncropped rows. Metadata owns geometry; decoding never reflows it.
 * A local greedy break keeps completed rows stable when pagination appends.
 * Choose the closer height either side of the target, allowing at most 1.5x
 * enlargement. The final incomplete row stays at target height, left aligned.
 * Keep this contract in sync with MoldClient.JustifiedLayout.
 */
export interface DimensionedMedia {
  metadata: { width?: number | null; height?: number | null };
}
export interface LaidOutItem<T> {
  item: T;
  index: number;
  x: number;
  width: number;
  height: number;
}
export interface JustifiedRow<T> {
  items: LaidOutItem<T>[];
  height: number;
  top: number;
}
export function aspectOf(image: DimensionedMedia): number {
  const { width, height } = image.metadata;
  const ratio = (width ?? 0) / (height ?? 0);
  return width != null &&
    height != null &&
    width > 0 &&
    height > 0 &&
    Number.isFinite(ratio) &&
    ratio > 0
    ? ratio
    : 1;
}
export function layoutJustifiedRows<T extends DimensionedMedia>(
  images: readonly T[],
  containerWidth: number,
  targetHeight = 180,
  gap = 2,
): JustifiedRow<T>[] {
  if (
    !Number.isFinite(containerWidth) ||
    containerWidth <= 0 ||
    !Number.isFinite(targetHeight) ||
    targetHeight <= 0
  )
    return [];
  gap = Math.max(
    0,
    Math.min(Number.isFinite(gap) ? gap : 2, containerWidth / 2),
  );
  const rows: JustifiedRow<T>[] = [];
  let start = 0,
    sum = 0,
    top = 0;
  const flush = (end: number, justify: boolean) => {
    if (end <= start) return;
    const fit = Math.max(0, containerWidth - gap * (end - start - 1)) / sum;
    const height = justify ? fit : Math.min(targetHeight, fit);
    let x = 0;
    const items = images.slice(start, end).map((item, offset) => {
      const width = aspectOf(item) * height;
      const tile = { item, index: start + offset, x, width, height };
      x += width + gap;
      return tile;
    });
    rows.push({ items, height, top });
    top += height + gap;
    start = end;
    sum = 0;
  };
  for (let i = 0; i < images.length; i++) {
    const ratio = aspectOf(images[i]!);
    // A run of very thin portraits can fill the width with seams alone.
    // Close before that happens, retaining positive image area and ratios.
    if (i > start && gap * (i - start) >= containerWidth) flush(i, false);
    const fit = (containerWidth - gap * (i - start)) / (sum + ratio);
    if (i > start && fit <= targetHeight) {
      const previous = (containerWidth - gap * (i - start - 1)) / sum;
      if (
        previous <= targetHeight * 1.5 &&
        Math.abs(previous - targetHeight) < Math.abs(fit - targetHeight)
      )
        flush(i, true);
    }
    sum += ratio;
    if (sum * targetHeight + gap * (i - start) >= containerWidth)
      flush(i + 1, true);
  }
  flush(images.length, false);
  return rows;
}
/** Binary search by actual row offsets, so scrolling costs O(log rows). */
export function justifiedWindow<T>(
  rows: readonly JustifiedRow<T>[],
  viewportStart: number,
  viewportSize: number,
  overscan = 2,
) {
  const lower = (y: number) => {
    let lo = 0,
      hi = rows.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const row = rows[mid]!;
      if (row.top + row.height < y) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const first = lower(Math.max(0, viewportStart));
  const last = lower(Math.max(0, viewportStart) + Math.max(0, viewportSize));
  return {
    start: Math.max(0, first - overscan),
    end: Math.min(rows.length, last + 1 + overscan),
    first,
    last,
  };
}
