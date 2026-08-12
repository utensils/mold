import { blobToBase64 } from "@studio/lib/base64";
import { apiFetch, apiFetchTo } from "../api/client";
import { localMediaPath, mediaPath } from "./media";
import type { MergedPrint } from "../../stores/gallery";

export interface GalleryMediaAuthority {
  mediaSourceOf(sourceKey: string): "host" | "local";
  targetOf(sourceKey: string): { baseUrl: string; apiKey: string | null } | null;
}

/**
 * Read the original bytes behind one unified-gallery row.
 *
 * This is the authority boundary shared by the gallery picker, Library
 * actions, and Lightbox actions. Host-backed rows use the exact origin's
 * authenticated API target. Native-only This-Mac rows use the restricted
 * `mold-local://` protocol that already rendered their preview. Never fall
 * through from a native row to the primary HTTP connection: that can return
 * an error body which only fails later as an undecodable "image".
 */
export async function readGalleryMediaBlob(
  entry: MergedPrint,
  gallery: GalleryMediaAuthority,
): Promise<Blob> {
  let response: Response;
  if (gallery.mediaSourceOf(entry.sourceKey) === "local") {
    response = await fetch(localMediaPath(entry.item.filename));
  } else {
    const path = mediaPath(entry.item.filename);
    const target = gallery.targetOf(entry.sourceKey);
    response = await (target ? apiFetchTo(target, path) : apiFetch(path));
  }
  if (!response.ok) {
    throw new Error(`Could not read ${entry.item.filename} (HTTP ${response.status})`);
  }
  return response.blob();
}

export async function readGalleryMediaBase64(
  entry: MergedPrint,
  gallery: GalleryMediaAuthority,
): Promise<string> {
  return blobToBase64(await readGalleryMediaBlob(entry, gallery));
}
