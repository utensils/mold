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
  | "reference-to-audio-video"
  | "text-to-audio";

export interface ExpansionTaskRequest {
  model?: string | null;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  clip_frames?: number | null;
  enable_audio?: boolean | null;
  negative_prompt?: string | null;
  lora?: string | null;
  loras?: readonly { path?: string | null }[] | null;
  edit_images?: readonly unknown[] | null;
  id_images?: readonly unknown[] | null;
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
  references?: readonly unknown[] | null;
  frames?: number | null;
  /** Face-identity reference (#1224). It never changes the expansion TASK —
   * a face photo does not make a text-to-image print an img2img one — but it
   * IS conditioning media, so `conditioningFingerprint` reads it. */
  id_image?: unknown;
}

function presentPath(path: string | null | undefined): boolean {
  return Boolean(path?.trim());
}

export function expansionTaskForRequest(
  family: string | null | undefined,
  request: ExpansionTaskRequest,
): ExpandTask {
  const normalized = (family ?? "").trim().toLowerCase();
  const h3 = ["minimax-h3", "minimax_h3", "minimaxh3"].includes(normalized);
  if (
    !h3 &&
    !["ltx2", "ltx-2", "ltx-video", "wan", "wan2.1", "wan2.2"].includes(
      normalized,
    )
  ) {
    return "text-to-image";
  }
  if (h3 && (request.references?.length ?? 0) > 0) {
    return "reference-to-audio-video";
  }
  if (h3) {
    const first = Boolean(request.source_image);
    const last = (request.keyframes?.length ?? 0) > 0;
    if (first && last) return "keyframe-interpolation";
    if (first || last) return "image-to-video";
    return "text-to-video";
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
  // A single-frame Wan render with no conditioning is a still (#798):
  // prompt work is image-style visual description, not chronological shot
  // direction. Deliberately after the source checks — a source-conditioned
  // one-frame request keeps its source-preserving contract. Mirrors
  // `mold_core::ExpandTask::for_conditioning`.
  if (
    ["wan", "wan2.1", "wan2.2"].includes(normalized) &&
    request.frames === 1
  ) {
    return "text-to-image";
  }
  return "text-to-video";
}

/** Mirrors `mold_core::ExpandReferenceRole`. */
export type ExpandReferenceRole =
  | "first-frame"
  | "last-frame"
  | "keyframe"
  | "source"
  | "identity"
  | "edit"
  | "reference";

/** Mirrors `mold_core::ExpandReference`: structure only, never bytes. */
export interface ExpandReference {
  kind: "image" | "video" | "audio";
  has_audio?: boolean;
  role?: ExpandReferenceRole;
}

/**
 * Mirrors `mold_core::ExpandContext`. Additive on `/api/expand` and
 * `/api/remix`; the server renders it after the model's prompting guide so
 * the expander knows the identity, canvas, clip length, and ordered
 * references. Duration is never sent: it is `frames / fps`.
 */
export interface ExpandContext {
  model?: string;
  width?: number;
  height?: number;
  frames?: number;
  fps?: number;
  clip_frames?: number;
  negative_prompt_supported?: boolean;
  audio?: boolean;
  references?: ExpandReference[];
  loras?: string[];
}

function loraStem(path: string | null | undefined): string | null {
  const trimmed = path?.trim();
  if (!trimmed) return null;
  const base = trimmed.split(/[\\/]/).pop() ?? trimmed;
  return base.replace(/\.(safetensors|gguf|pt|bin)$/i, "");
}

/**
 * Browser-safe mirror of `mold_core::ExpandContext::for_generation`.
 *
 * Reads the already-built generation request so web, desktop, and iPhone
 * send the same facts without shell-specific glue. Reference order follows
 * the request: MiniMax H3 ordered references first, then the source frame,
 * edit images, identity images, source video, and conditioning audio.
 */
export function expansionContextForRequest(
  family: string | null | undefined,
  request: ExpansionTaskRequest,
): ExpandContext {
  const normalized = (family ?? "").trim().toLowerCase();
  const videoFamily = [
    "ltx2",
    "ltx-2",
    "ltx-video",
    "wan",
    "wan2.1",
    "wan2.2",
    "minimax-h3",
    "minimax_h3",
    "minimaxh3",
  ].includes(normalized);
  const references: ExpandReference[] = [];
  for (const reference of request.references ?? []) {
    const row = reference as { kind?: string; has_audio?: boolean } | null;
    const kind = row?.kind;
    if (kind !== "image" && kind !== "video" && kind !== "audio") continue;
    references.push({
      kind,
      ...(kind === "video" && row?.has_audio ? { has_audio: true } : {}),
      role: "reference",
    });
  }
  const keyframes = request.keyframes?.length ?? 0;
  if (keyframes > 1) {
    for (let index = 0; index < keyframes; index += 1) {
      references.push({ kind: "image", role: "keyframe" });
    }
  } else if (request.source_image) {
    references.push({
      kind: "image",
      role: videoFamily
        ? "first-frame"
        : normalized === "qwen-image-edit"
          ? "edit"
          : "source",
    });
  }
  for (const _ of request.edit_images ?? []) {
    references.push({ kind: "image", role: "edit" });
  }
  const identities = request.id_images?.length ?? (request.id_image ? 1 : 0);
  for (let index = 0; index < identities; index += 1) {
    references.push({ kind: "image", role: "identity" });
  }
  if (
    request.source_video ||
    presentPath(request.source_video_path) ||
    request.extend_video ||
    presentPath(request.extend_video_path)
  ) {
    references.push({ kind: "video", role: "source" });
  }
  if (request.audio_file || presentPath(request.audio_file_path)) {
    references.push({ kind: "audio", role: "source" });
  }
  const loras: string[] = [];
  const single = loraStem(request.lora);
  if (single) loras.push(single);
  for (const lora of request.loras ?? []) {
    const stem = loraStem(lora?.path);
    if (stem) loras.push(stem);
  }
  const context: ExpandContext = {};
  const model = request.model?.trim();
  if (model) context.model = model;
  if (request.width && request.width > 0) context.width = request.width;
  if (request.height && request.height > 0) context.height = request.height;
  if (request.frames != null && request.frames > 0) {
    context.frames = request.frames;
  }
  if (request.fps != null && request.fps > 0) context.fps = request.fps;
  if (request.clip_frames != null && request.clip_frames > 0) {
    context.clip_frames = request.clip_frames;
  }
  if (typeof request.enable_audio === "boolean") {
    context.audio = request.enable_audio;
  }
  if (references.length > 0) context.references = references;
  if (loras.length > 0) context.loras = loras;
  return context;
}
