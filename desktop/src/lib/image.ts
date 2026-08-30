import { blobToBase64 } from "@studio/lib/base64";
import type { GalleryImage } from "./api/types";

export { blobToBase64 };

/**
 * Read a `File` (drag-drop or <input type=file>) to base64 with no data-URI
 * prefix — the shape mold-core expects for `source_image` / `mask_image` /
 * `control_image` on the wire. Works in WKWebView and a plain browser.
 *
 * Native desktop chooser selection is read by the Rust backend; this remains
 * the portable path for drag-and-drop and the browser development surface.
 */
export function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file);
}

/** Object URL for a base64 payload so a `<img>` can preview it. */
export function base64ToDataUrl(b64: string, mime = "image/png"): string {
  return `data:${mime};base64,${b64}`;
}

/**
 * True for the still-image formats the engine accepts as `source_image` /
 * `mask_image` / keyframe conditioning: PNG and JPEG only. The gallery also
 * holds WebP/GIF/APNG/MP4 outputs, which the generate endpoints reject — so the
 * image picker filters its grid with this to avoid forwarding a pick that
 * would only fail at generation time.
 */
export function isStillImageFile(filename: string): boolean {
  return /\.(png|jpe?g)$/i.test(filename.trim());
}

/**
 * Gallery metadata is an independent authority from the stored filename.
 * Require both to describe a still image so a legacy/mislabelled video row
 * cannot enter an image-only source picker merely because its poster or
 * filename ends in `.png`.
 */
export function isStillImageGalleryItem(
  item: Pick<GalleryImage, "filename" | "format" | "metadata">,
): boolean {
  if (!isStillImageFile(item.filename)) return false;
  const format = item.format?.toLowerCase();
  if (format && format !== "png" && format !== "jpeg") return false;
  return !item.metadata.frames && !item.metadata.video_frames;
}
