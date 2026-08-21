import { DOWNLOAD_FALLBACK_STEM, downloadFileName } from "@studio/lib/fileUnder";

/**
 * The name a Save / Export / Download hands to the file dialog. The gallery
 * filename itself never changes — this is only the name handed to the media
 * folder (`ipc.saveGalleryMedia(target, filename, outputFilename)` /
 * `ipc.saveMediaBytes(outputFilename, …)`), whose Tauri side already resolves
 * collisions by appending `-2`, `-3`, ….
 *
 * The grammar is `mold-core`'s `download_file_name`, mirrored by
 * `@studio/lib/fileUnder`: `{title-slug}__{model}__s{seed}.{ext}`, with each
 * segment dropped when it has nothing to contribute. A gallery filename is an
 * identity (timestamped, never renamed); a download name is a LABEL read by a
 * human in a Downloads folder, so it leads with the title and carries the two
 * things that tell two prints apart — the model and the seed.
 *
 * A row with no usable metadata at all (a synthesized entry, an import the
 * server never enriched) would slug to the library's bare `print` stem, which
 * is useless in a folder; those keep today's gallery filename instead.
 */
export interface SaveNameEntry {
  filename: string;
  /** Row title (editable authority); `null`/absent = untitled. */
  title?: string | null;
  /** Creation-time provenance: the title fallback plus the model and seed the
   * download name is built from. */
  metadata?: {
    title?: string | null;
    model?: string | null;
    seed?: number | string | null;
  } | null;
}

export interface SaveNameOptions {
  /** Appended to the stem, before the extension (e.g. `-upscaled`). */
  suffix?: string;
  /** Overrides the print's own extension (e.g. a video export format). */
  extension?: string;
}

function splitFilename(filename: string): { stem: string; ext: string } {
  const dot = filename.lastIndexOf(".");
  if (dot <= 0) return { stem: filename, ext: "" };
  return { stem: filename.slice(0, dot), ext: filename.slice(dot + 1) };
}

export function suggestedSaveName(entry: SaveNameEntry, options: SaveNameOptions = {}): string {
  const { stem: originalStem, ext: originalExt } = splitFilename(entry.filename);
  const title = entry.title?.trim() || entry.metadata?.title?.trim() || "";
  // The extension is applied here, after the optional suffix, so pass none.
  const labelled = downloadFileName({
    title: title || null,
    model: entry.metadata?.model ?? "",
    seed: entry.metadata?.seed ?? null,
    ext: "",
  });
  const stem = labelled === DOWNLOAD_FALLBACK_STEM ? originalStem : labelled;
  const named = `${stem}${options.suffix ?? ""}`;
  const ext = options.extension?.replace(/^\./, "") || originalExt;
  return ext ? `${named}.${ext}` : named;
}
