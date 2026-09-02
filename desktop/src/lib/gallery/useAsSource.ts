import { blobToBase64 } from "@studio/lib/base64";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import {
  appendMinimaxH3GalleryImageReference,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  setMinimaxH3GalleryImageFirstFrame,
} from "@studio/lib/minimaxH3Authoring";
import { attachPickedImage, attachPickedVideo } from "../sourceAttachment";
import { isAudioItem, isVideoItem } from "./media";
import type { GenerateForm } from "../generateForm";
import type { GalleryImage } from "../api/types";
import type { MergedPrint } from "../../stores/gallery";

/**
 * "Use as source" — ONE rule, for every desktop surface that shows a print.
 *
 * The Library tile menu, the Lightbox, and the History drawer all offer the
 * same item on the same prints, so the decision of what a print becomes (LTX
 * source video, an H3 first frame or ordered reference, or the img2img source
 * image) lives here rather than once per view. Callers supply the bytes
 * reader for their surface (`readGalleryMediaBlob` bound to the gallery
 * authority) and own the toast, the panel they close, and the route.
 *
 * Deliberately does NOT touch `composer.prefill`: GenerateView's prefill
 * watcher would clobber the source that was just attached.
 */
export type UseAsSourceOutcome = { ok: true; message: string } | { ok: false; error: string };

/** An audio print has no pixels and conditions nothing — the menus disable
 *  the item, and this is the rule they read. */
export function canUseGalleryEntryAsSource(item: GalleryImage): boolean {
  return !isAudioItem(item);
}

/** The MIME type an H3 reference records, preferring what the read reported. */
function galleryImageMimeType(item: GalleryImage, declared: string): string {
  const mime = declared.split(";", 1)[0]!.trim().toLowerCase();
  if (mime.startsWith("image/")) return mime;
  const format = (item.format ?? item.filename.split(".").pop() ?? "")
    .toLowerCase()
    .replace("jpg", "jpeg");
  return format ? `image/${format}` : "application/octet-stream";
}

export async function applyGalleryEntryAsSource(
  entry: MergedPrint,
  form: GenerateForm,
  readBlob: (entry: MergedPrint) => Promise<Blob>,
): Promise<UseAsSourceOutcome> {
  const item = entry.item;
  if (!canUseGalleryEntryAsSource(item)) {
    return { ok: false, error: "An audio print cannot be used as a source." };
  }
  try {
    const blob = await readBlob(entry);
    const base64 = await blobToBase64(blob);
    if (isVideoItem(item)) {
      attachPickedVideo(form, { filename: item.filename, base64 });
      return { ok: true, message: "Loaded as source video" };
    }
    const h3Task = minimaxH3TaskForModel(form.model);
    if (h3Task) {
      const dimensions = imageDimensionsFromBase64(base64) ?? {
        width: item.metadata.width,
        height: item.metadata.height,
      };
      const image = {
        filename: item.filename,
        mimeType: galleryImageMimeType(item, blob.type),
        width: dimensions.width,
        height: dimensions.height,
        data: base64,
      };
      const result =
        h3Task === "ref2va"
          ? await appendMinimaxH3GalleryImageReference(form.h3Authoring, image)
          : setMinimaxH3GalleryImageFirstFrame(form.h3Authoring, image);
      if (!result.ok) return { ok: false, error: result.error };
      form.h3Authoring = result.state;
      return {
        ok: true,
        message: h3Task === "ref2va" ? "Added as ordered reference" : "Loaded as source",
      };
    }
    if (isMinimaxH3Identity(form.family, form.model)) {
      return {
        ok: false,
        error: "Choose an explicit MiniMax H3 FL2VA or Ref2VA model before adding a source.",
      };
    }
    attachPickedImage(form, { filename: item.filename, base64 });
    return { ok: true, message: "Loaded as source" };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}
