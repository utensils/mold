// Matches `mold_core::OutputFormat` on the wire (lowercase strings).
export type OutputFormat = "png" | "jpeg" | "gif" | "apng" | "webp" | "mp4";

export type Scheduler =
  | "default"
  | "ddim"
  | "euler-ancestral"
  | "unipc"
  | { ddim: unknown }
  | { "euler-ancestral": unknown }
  | { unipc: unknown };

export interface OutputMetadata {
  prompt: string;
  negative_prompt?: string | null;
  original_prompt?: string | null;
  model: string;
  seed: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  strength?: number | null;
  scheduler?: Scheduler | null;
  lora?: string | null;
  lora_scale?: number | null;
  frames?: number | null;
  fps?: number | null;
  version: string;
}

export interface GalleryImage {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  metadata_synthetic?: boolean;
}

export type MediaKind = "image" | "animated" | "video";

export const VIDEO_FORMATS: ReadonlyArray<OutputFormat> = ["mp4"];
export const ANIMATED_FORMATS: ReadonlyArray<OutputFormat> = [
  "gif",
  "apng",
  "webp",
];

export function mediaKind(
  fmt: OutputFormat | null | undefined,
  filename: string,
): MediaKind {
  const resolved = fmt ?? inferFormatFromName(filename);
  if (resolved && VIDEO_FORMATS.includes(resolved)) return "video";
  if (resolved && ANIMATED_FORMATS.includes(resolved)) return "animated";
  return "image";
}

// Mirror of `mold_core::GalleryCapabilities`.
export interface GalleryCapabilities {
  can_delete: boolean;
}

// Mirror of `mold_core::ServerCapabilities`.
export interface ServerCapabilities {
  gallery: GalleryCapabilities;
}

export function inferFormatFromName(filename: string): OutputFormat | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".mp4")) return "mp4";
  if (lower.endsWith(".gif")) return "gif";
  if (lower.endsWith(".apng")) return "apng";
  if (lower.endsWith(".webp")) return "webp";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "jpeg";
  if (lower.endsWith(".png")) return "png";
  return null;
}
