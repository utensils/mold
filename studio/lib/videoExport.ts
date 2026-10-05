export type VideoExportFormat = "gif" | "apng" | "webp";
export type GifPlayback = "loop" | "bounce";
export type GifRepeat = "forever" | "once";

export interface VideoExportOptions {
  format: VideoExportFormat;
  playback: GifPlayback;
  repeat: GifRepeat;
  max_dimension: number | null;
  fps: number | null;
  /**
   * Mesh turntables only: render the object over nothing instead of the
   * poster's slate backdrop. Omitted entirely for a video re-encode, whose
   * frames already exist — the host refuses the key there rather than
   * ignoring it.
   */
  transparent?: boolean;
  pause_ms?: number;
  frames?: number;
}

export interface VideoExportCapabilities {
  formats: VideoExportFormat[];
  gif_playback: GifPlayback[];
  gif_repeat: GifRepeat[];
  gif_pause?: GifPauseControl;
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

/** Extra dwell only; GIF frame cadence remains positive and FPS-derived. */
export interface GifPauseControl {
  min: number;
  max: number;
  step: number;
  default: number;
}
export function validGifPause(
  value: GifPauseControl | undefined,
): value is GifPauseControl {
  return (
    !!value &&
    [value.min, value.max, value.step, value.default].every(
      Number.isSafeInteger,
    ) &&
    value.min >= 0 &&
    value.max <= 5000 &&
    value.max >= value.min &&
    value.step > 0 &&
    value.step % 10 === 0 &&
    value.min % 10 === 0 &&
    value.default >= value.min &&
    value.default <= value.max &&
    (value.default - value.min) % value.step === 0
  );
}
export function turntableFrameLimit(
  edge: number,
  transparent: boolean,
): number {
  return Math.max(
    8,
    Math.min(
      180,
      Math.floor((256 * 1024 * 1024) / (edge * edge * (transparent ? 4 : 3))),
    ),
  );
}
