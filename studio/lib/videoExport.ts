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

/** Save in browsers/desktop, or open the native iOS share sheet when WebKit
 * exposes file sharing. The latter lets the user choose Photos, Files, or any
 * installed destination instead of silently assuming one library. */
export async function saveVideoExport(
  blob: Blob,
  filename: string,
): Promise<"shared" | "saved"> {
  const file = new File([blob], filename, { type: blob.type });
  if (
    typeof navigator !== "undefined" &&
    typeof navigator.share === "function" &&
    typeof navigator.canShare === "function" &&
    navigator.canShare({ files: [file] })
  ) {
    await navigator.share({ files: [file], title: filename });
    return "shared";
  }

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
  return "saved";
}
