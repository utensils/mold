export type VideoExportFormat = "gif" | "apng" | "webp";
export type GifPlayback = "loop" | "bounce";
export type GifRepeat = "forever" | "once";

export interface VideoExportOptions {
  format: VideoExportFormat;
  playback: GifPlayback;
  repeat: GifRepeat;
  max_dimension: number | null;
  fps: number | null;
}

export interface VideoExportCapabilities {
  formats: VideoExportFormat[];
  gif_playback: GifPlayback[];
  gif_repeat: GifRepeat[];
}

export const DEFAULT_VIDEO_EXPORT_CAPABILITIES: VideoExportCapabilities = {
  formats: ["gif", "apng"],
  gif_playback: ["loop", "bounce"],
  gif_repeat: ["forever", "once"],
};

export function videoExportPath(filename: string): string {
  return `/api/gallery/export/${encodeURIComponent(filename)}`;
}

export function videoExportFilename(
  filename: string,
  format: VideoExportFormat,
): string {
  const stem = filename.replace(/\.[^.]+$/, "") || "mold-video";
  return `${stem}.${format === "apng" ? "png" : format}`;
}

/** Download an exported animation. This is deliberately the browser default:
 * desktop Web Share support must not turn a local save into a share sheet. */
export function downloadVideoExport(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

/** Open the native iOS share sheet for an exported animation, falling back to
 * a file download on older WebKit builds. Callers must opt into this path. */
export async function shareVideoExport(
  blob: Blob,
  filename: string,
): Promise<"shared" | "saved" | "cancelled"> {
  const file = new File([blob], filename, { type: blob.type });
  if (
    typeof navigator !== "undefined" &&
    typeof navigator.share === "function" &&
    typeof navigator.canShare === "function" &&
    navigator.canShare({ files: [file] })
  ) {
    try {
      await navigator.share({ files: [file], title: filename });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError")
        return "cancelled";
      throw error;
    }
    return "shared";
  }

  downloadVideoExport(blob, filename);
  return "saved";
}
