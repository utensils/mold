import type { GalleryImage } from "../api/types";

/** True only for the enlarged output, never its separately saved original. */
export function isUpscaledImage(item: GalleryImage): boolean {
  if (item.format === "mp4" || /\.mp4$/i.test(item.filename)) return false;
  if (/-upscaled\.[a-z0-9]+$/i.test(item.filename)) return true;

  const metadata = item.metadata;
  if (!metadata.upscale_model || !metadata.generation_width || !metadata.generation_height) {
    return false;
  }
  return metadata.width > metadata.generation_width || metadata.height > metadata.generation_height;
}
