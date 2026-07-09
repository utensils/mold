import type { GalleryImage } from "../api/types";

/**
 * Justified rows, like contact sheets: every row is filled edge-to-edge by
 * scaling a target row height; the final row keeps the target height.
 */
export interface LaidOutItem {
  item: GalleryImage;
  width: number;
  height: number;
}

export interface JustifiedRow {
  items: LaidOutItem[];
  height: number;
}

export function aspectOf(image: GalleryImage): number {
  const { width, height } = image.metadata;
  if (!width || !height) return 1;
  return width / height;
}

export function layoutJustifiedRows(
  images: GalleryImage[],
  containerWidth: number,
  targetHeight = 180,
  gap = 8,
): JustifiedRow[] {
  if (containerWidth <= 0) return [];
  const rows: JustifiedRow[] = [];
  let row: GalleryImage[] = [];
  let rowAspect = 0;

  const flush = (justify: boolean) => {
    if (row.length === 0) return;
    const gaps = gap * (row.length - 1);
    const height = justify
      ? Math.min(targetHeight * 1.5, (containerWidth - gaps) / rowAspect)
      : targetHeight;
    rows.push({
      height,
      items: row.map((item) => ({
        item,
        width: aspectOf(item) * height,
        height,
      })),
    });
    row = [];
    rowAspect = 0;
  };

  for (const image of images) {
    row.push(image);
    rowAspect += aspectOf(image);
    const gaps = gap * (row.length - 1);
    if (rowAspect * targetHeight + gaps >= containerWidth) flush(true);
  }
  flush(false);
  return rows;
}
