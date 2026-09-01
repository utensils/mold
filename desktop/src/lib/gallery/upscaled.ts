import type { GalleryImage } from "../api/types";

/**
 * True only for the enlarged output, never its separately saved original.
 *
 * Stills and Framewise videos answer the same question from the same
 * metadata: the server records `upscale_model` beside the source's
 * `generation_width`/`generation_height`, and the print is upscaled when its
 * own dimensions exceed those. A generated video that merely carried an
 * `upscale_model` request setting has equal dimensions and stays unbadged.
 *
 * The server-issued filenames are authority too: stills publish as
 * `<stem>-upscaled.<ext>` and Framewise videos as
 * `<stem>-framewise-upscaled-<job>.mp4`. The latter matters when the
 * level 5.2 fit lands the published frame back at the source size, where the
 * dimension rule alone would answer false.
 */
export function isUpscaledImage(item: GalleryImage): boolean {
  if (/-upscaled\.[a-z0-9]+$/i.test(item.filename)) return true;
  if (/-framewise-upscaled-[0-9a-f]+\.mp4$/i.test(item.filename)) return true;

  const metadata = item.metadata;
  if (!metadata.upscale_model || !metadata.generation_width || !metadata.generation_height) {
    return false;
  }
  return metadata.width > metadata.generation_width || metadata.height > metadata.generation_height;
}
