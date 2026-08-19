import { titleSlug } from "@studio/lib/libraryOrganization";

/**
 * Title-based save names (Library organization, D5): Save locally / Save
 * image / Export / upscale saves suggest `<title-slug>.<ext>` when a print
 * has a title, and today's gallery filename otherwise. The gallery filename
 * itself never changes — this is only the name handed to the media folder
 * (`ipc.saveGalleryMedia(target, filename, outputFilename)` /
 * `ipc.saveMediaBytes(outputFilename, …)`), whose Tauri side already resolves
 * collisions by appending `-2`, `-3`, ….
 */
export interface SaveNameEntry {
  filename: string;
  /** Row title (editable authority); `null`/absent = untitled. */
  title?: string | null;
  /** Creation-time title fallback for rows the server has not enriched. */
  metadata?: { title?: string | null } | null;
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

/** `<slug>.<ext>` for a titled print, else the gallery filename (with the
 * optional suffix / extension applied to either). */
export function suggestedSaveName(entry: SaveNameEntry, options: SaveNameOptions = {}): string {
  const { stem: originalStem, ext: originalExt } = splitFilename(entry.filename);
  const title = entry.title?.trim() || entry.metadata?.title?.trim() || "";
  const slug = title ? titleSlug(title) : null;
  const stem = `${slug ?? originalStem}${options.suffix ?? ""}`;
  const ext = options.extension?.replace(/^\./, "") || originalExt;
  return ext ? `${stem}.${ext}` : stem;
}
