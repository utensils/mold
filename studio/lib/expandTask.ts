/**
 * Browser-safe mirror of `mold_core::ExpandTask::for_generation`.
 *
 * The expansion endpoint needs only the semantic conditioning contract, never
 * the attached media itself. Structural fields let web, desktop, and iPhone
 * pass their already-pruned generation request without shell-specific glue.
 */
export type ExpandTask =
  | "text-to-image"
  | "text-to-video"
  | "image-to-video"
  | "video-to-video"
  | "retake"
  | "keyframe-interpolation"
  | "audio-driven-video"
  | "text-to-audio";

export interface ExpansionTaskRequest {
  source_image?: unknown;
  source_video?: unknown;
  source_video_path?: string | null;
  extend_video?: unknown;
  extend_video_path?: string | null;
  audio_file?: unknown;
  audio_file_path?: string | null;
  keyframes?: readonly unknown[] | null;
  pipeline?: string | null;
  retake_range?: unknown;
}

function presentPath(path: string | null | undefined): boolean {
  return Boolean(path?.trim());
}

export function expansionTaskForRequest(
  family: string | null | undefined,
  request: ExpansionTaskRequest,
): ExpandTask {
  const normalized = (family ?? "").trim().toLowerCase();
  if (
    !["ltx2", "ltx-2", "ltx-video", "wan", "wan2.1", "wan2.2"].includes(
      normalized,
    )
  ) {
    return "text-to-image";
  }
  switch (request.pipeline) {
    case "t2a":
      return "text-to-audio";
    case "retake":
      return "retake";
    case "keyframe":
      return "keyframe-interpolation";
    case "a2-vid":
    case "lip-dub":
      return "audio-driven-video";
    case null:
    case undefined:
      // Mirrors the engine's implicit-pipeline priority.
      if (request.retake_range) return "retake";
      if (request.audio_file || presentPath(request.audio_file_path)) {
        return "audio-driven-video";
      }
      if ((request.keyframes?.length ?? 0) > 1) {
        return "keyframe-interpolation";
      }
      break;
  }
  if (
    request.source_video ||
    presentPath(request.source_video_path) ||
    request.extend_video ||
    presentPath(request.extend_video_path)
  ) {
    return "video-to-video";
  }
  if (request.source_image) return "image-to-video";
  return "text-to-video";
}
