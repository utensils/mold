import { blobToBase64 } from "@studio/lib/base64";
import type { GalleryImage } from "./api/types";
import {
  imageInputFormatForName,
  imageInputFormatOfBase64,
  LEGACY_REFERENCE_IMAGE_FORMATS,
  referenceImageMimeTypes,
  type ImageInputFormat,
} from "@studio/lib/referenceImagesProfile";

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

/** Object URL for a base64 payload so a `<img>` can preview it. Without an
 * explicit type the container is read from the payload's own first bytes
 * (a WebP reference is labelled WebP), falling back to PNG. */
export function base64ToDataUrl(b64: string, mime?: string): string {
  const resolved = mime ?? referenceImageMimeTypes([imageInputFormatOfBase64(b64) ?? "png"])[0]!;
  return `data:${resolved};base64,${b64}`;
}

/**
 * The still containers the phone hands to Photos verbatim: every still output
 * format mold writes. A transparent Qwen Image 2.1 print can be a WebP still,
 * and its alpha lives in the original bytes, so none of these is re-encoded.
 */
export const PHOTO_SAVE_FORMATS: readonly ImageInputFormat[] = ["png", "jpeg", "webp"];

/**
 * True for the still-image formats the engine accepts as `source_image` /
 * `mask_image` / keyframe conditioning: PNG and JPEG only. The gallery also
 * holds WebP/GIF/APNG/MP4 outputs, which the generate endpoints reject — so the
 * image picker filters its grid with this to avoid forwarding a pick that
 * would only fail at generation time.
 */
export function isStillImageFile(
  filename: string,
  formats: readonly ImageInputFormat[] = LEGACY_REFERENCE_IMAGE_FORMATS,
): boolean {
  // `formats` is the recipe's advertised `reference_images.formats` for a
  // reference strip (Qwen Image 2.1 adds WebP); every other caller keeps the
  // PNG/JPEG pair source images are admitted as.
  const format = imageInputFormatForName(filename);
  return format !== null && formats.includes(format);
}

/**
 * Gallery metadata is an independent authority from the stored filename.
 * Require both to describe a still image so a legacy/mislabelled video row
 * cannot enter an image-only source picker merely because its poster or
 * filename ends in `.png`.
 */
export function isStillImageGalleryItem(
  item: Pick<GalleryImage, "filename" | "format" | "metadata">,
  formats: readonly ImageInputFormat[] = LEGACY_REFERENCE_IMAGE_FORMATS,
): boolean {
  if (!isStillImageFile(item.filename, formats)) return false;
  const format = item.format?.toLowerCase();
  if (format && !(formats as readonly string[]).includes(format)) return false;
  // A still WebP and an animated one share the container; the frame counts
  // tell them apart, exactly as for a mislabelled video row.
  return !item.metadata.frames && !item.metadata.video_frames;
}
