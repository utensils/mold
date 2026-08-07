import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "./generationReferences";
import {
  isModelAccessRestricted,
  type ModelAccessCapabilityRecord,
} from "./modelAccess";

export const MINIMAX_H3_FIXED_FPS = 24;
export const MINIMAX_H3_MIN_FRAMES = 124;
export const MINIMAX_H3_MAX_FRAMES = 362;
export const MINIMAX_H3_FRAME_STEP = 17;
export const MINIMAX_H3_FRAME_OFFSET = 5;
export const MINIMAX_H3_MAX_REFERENCES = 12;
export const MINIMAX_H3_MAX_REFERENCE_IMAGES = 9;
export const MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3;
export const MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3;
export const MINIMAX_H3_MIN_REFERENCE_DURATION_MS = 2_000;
export const MINIMAX_H3_MAX_REFERENCE_DURATION_MS = 15_000;
export const MINIMAX_H3_MAX_REFERENCE_VIDEO_MS = 15_000;
export const MINIMAX_H3_MAX_REFERENCE_AUDIO_MS = 15_000;

export const MINIMAX_H3_RESYNTHESIS_TITLE =
  "Reference-guided semantic resynthesis";
export const MINIMAX_H3_RESYNTHESIS_GUIDANCE =
  "References guide identity, relationships, motion, timing, and sound in a newly synthesized shot. This is not pixel-aligned video editing, so there is no denoise-strength control.";
export const MINIMAX_H3_PROMPT_PLACEHOLDER =
  "Describe the new synchronized shot…";

export type MinimaxH3Task = "fl2va" | "ref2va";

/** Browser-owned first/last-frame authority. The raw bytes live in IndexedDB
 * when a draft/template is persisted; only the small descriptor is allowed in
 * localStorage. */
export interface MinimaxH3BoundaryImage {
  filename: string;
  mimeType: string;
  width: number;
  height: number;
  /** Raw base64 without a data-URI prefix. */
  data: string;
  sha256?: string | null;
  draftId?: string;
}

/** UI wrapper keeps draft-storage identity outside the public generation
 * reference wire contract. */
export interface MinimaxH3ReferenceDraft {
  reference: GenerationReference;
  draftId?: string;
}

export interface MinimaxH3AuthoringState {
  firstFrame: MinimaxH3BoundaryImage | null;
  lastFrame: MinimaxH3BoundaryImage | null;
  references: MinimaxH3ReferenceDraft[];
}

export interface MinimaxH3AuthoringCapabilities {
  task: MinimaxH3Task;
  runtimeAvailable: boolean;
  fixedFps: number;
  minFrames: number;
  maxFrames: number;
  frameStep: number;
  frameOffset: number;
  synchronizedAudio: true;
  audioDisableSupported: false;
}

export interface MinimaxH3ModelIdentity {
  name: string;
  family?: string | null;
  /** Additive future server field. Explicit false is authoritative; omission
   * remains compatible with pre-field servers, whose model list itself is a
   * runnable-model boundary. */
  runtime_available?: boolean | null;
}

function normalize(value: string): string {
  return value.trim().toLowerCase().replaceAll("_", "-");
}

export function isMinimaxH3Family(family: string | null | undefined): boolean {
  const value = normalize(family ?? "").replaceAll("-", "");
  return value === "minimaxh3";
}

/** Resolve only the released, explicit task partitions. An opaque catalog ID
 * must carry a future advertised task instead of being guessed from family. */
export function minimaxH3TaskForModel(
  model: string | null | undefined,
): MinimaxH3Task | null {
  const value = normalize(model ?? "");
  if (value === "minimax-h3-ref2va" || value.startsWith("minimax-h3-ref2va:")) {
    return "ref2va";
  }
  if (
    value === "minimax-h3" ||
    value === "minimaxh3" ||
    value === "minimax-h3-fl2va" ||
    value.startsWith("minimax-h3-fl2va:")
  ) {
    return "fl2va";
  }
  return null;
}

/** Family metadata can be absent in durable gallery/request snapshots. Exact
 * released model identities still carry enough authority to recover the H3
 * contract without guessing opaque catalog IDs. */
export function isMinimaxH3Identity(
  family: string | null | undefined,
  model: string | null | undefined,
): boolean {
  return isMinimaxH3Family(family) || minimaxH3TaskForModel(model) !== null;
}

export function emptyMinimaxH3AuthoringState(): MinimaxH3AuthoringState {
  return { firstFrame: null, lastFrame: null, references: [] };
}

export function cloneMinimaxH3AuthoringState(
  state: MinimaxH3AuthoringState | null | undefined,
): MinimaxH3AuthoringState {
  return state
    ? (JSON.parse(JSON.stringify(state)) as MinimaxH3AuthoringState)
    : emptyMinimaxH3AuthoringState();
}

/** Small localStorage-safe projection. IndexedDB/template consumers persist
 * the removed bytes separately and restore them before submission. */
export function stripMinimaxH3AuthoringMedia(
  state: MinimaxH3AuthoringState | null | undefined,
): MinimaxH3AuthoringState {
  const clone = cloneMinimaxH3AuthoringState(state);
  if (clone.firstFrame) clone.firstFrame.data = "";
  if (clone.lastFrame) clone.lastFrame.data = "";
  clone.references = clone.references.map((draft) => ({
    ...draft,
    reference: {
      ...draft.reference,
      media: { authority: "descriptor" },
    } as GenerationReference,
  }));
  return clone;
}

export function minimaxH3AuthoringCapabilities(
  model: MinimaxH3ModelIdentity,
  serverCapabilities?: ModelAccessCapabilityRecord | null,
): MinimaxH3AuthoringCapabilities | null {
  const task = minimaxH3TaskForModel(model.name);
  if (!isMinimaxH3Identity(model.family, model.name) || !task) return null;
  const restricted = isModelAccessRestricted(serverCapabilities, {
    model: model.name,
    family: model.family,
  });
  return {
    task,
    runtimeAvailable: model.runtime_available !== false && !restricted,
    fixedFps: MINIMAX_H3_FIXED_FPS,
    minFrames: MINIMAX_H3_MIN_FRAMES,
    maxFrames: MINIMAX_H3_MAX_FRAMES,
    frameStep: MINIMAX_H3_FRAME_STEP,
    frameOffset: MINIMAX_H3_FRAME_OFFSET,
    synchronizedAudio: true,
    audioDisableSupported: false,
  };
}

export function minimaxH3Mode(
  task: MinimaxH3Task,
  state: MinimaxH3AuthoringState,
):
  | "text-to-audio-video"
  | "first-frame-to-audio-video"
  | "last-frame-to-audio-video"
  | "first-and-last-frame-to-audio-video"
  | "reference-to-audio-video" {
  if (task === "ref2va") return "reference-to-audio-video";
  if (state.firstFrame && state.lastFrame)
    return "first-and-last-frame-to-audio-video";
  if (state.firstFrame) return "first-frame-to-audio-video";
  if (state.lastFrame) return "last-frame-to-audio-video";
  return "text-to-audio-video";
}

export interface MinimaxH3ReferenceBudget {
  total: number;
  images: number;
  videos: number;
  audios: number;
  videoDurationMs: number;
  audioDurationMs: number;
  errors: string[];
}

function referenceDurationError(
  reference: number,
  label: string,
  durationMs: number,
): string | null {
  if (
    durationMs >= MINIMAX_H3_MIN_REFERENCE_DURATION_MS &&
    durationMs <= MINIMAX_H3_MAX_REFERENCE_DURATION_MS
  ) {
    return null;
  }
  return `Reference ${reference} ${label} must be 2–15 seconds.`;
}

/** UI-side mirror of the server's one-based Ref2VA budget errors. The server
 * remains authoritative and content-sniffs every upload. */
export function minimaxH3ReferenceBudget(
  drafts: readonly MinimaxH3ReferenceDraft[],
): MinimaxH3ReferenceBudget {
  let images = 0;
  let videos = 0;
  let audios = 0;
  let videoDurationMs = 0;
  let audioDurationMs = 0;
  const errors: string[] = [];

  drafts.forEach(({ reference }, index) => {
    const oneBased = index + 1;
    if (reference.kind === "image") {
      images += 1;
      return;
    }
    if (reference.kind === "video") {
      videos += 1;
      videoDurationMs += reference.duration_ms;
      const durationError = referenceDurationError(
        oneBased,
        "video duration",
        reference.duration_ms,
      );
      if (durationError) errors.push(durationError);
      if (reference.has_audio) {
        const duration = reference.audio_duration_ms;
        if (duration == null) {
          errors.push(
            `Reference ${oneBased} soundtrack duration is required for a video with audio.`,
          );
        } else {
          audioDurationMs += duration;
          const soundtrackError = referenceDurationError(
            oneBased,
            "soundtrack duration",
            duration,
          );
          if (soundtrackError) errors.push(soundtrackError);
        }
      } else if (reference.audio_duration_ms != null) {
        errors.push(
          `Reference ${oneBased} soundtrack duration is only valid when the video has audio.`,
        );
      }
      return;
    }
    audios += 1;
    audioDurationMs += reference.duration_ms;
    const durationError = referenceDurationError(
      oneBased,
      "audio duration",
      reference.duration_ms,
    );
    if (durationError) errors.push(durationError);
  });

  if (drafts.length === 0) {
    errors.push("Add at least one image or video reference.");
  }
  if (drafts.length > MINIMAX_H3_MAX_REFERENCES) {
    errors.push(`Use at most ${MINIMAX_H3_MAX_REFERENCES} references total.`);
  }
  if (images > MINIMAX_H3_MAX_REFERENCE_IMAGES) {
    errors.push(
      `Use at most ${MINIMAX_H3_MAX_REFERENCE_IMAGES} image references.`,
    );
  }
  if (videos > MINIMAX_H3_MAX_REFERENCE_VIDEOS) {
    errors.push(
      `Use at most ${MINIMAX_H3_MAX_REFERENCE_VIDEOS} video references.`,
    );
  }
  if (audios > MINIMAX_H3_MAX_REFERENCE_AUDIOS) {
    errors.push(
      `Use at most ${MINIMAX_H3_MAX_REFERENCE_AUDIOS} standalone audio references.`,
    );
  }
  if (images + videos === 0 && drafts.length > 0) {
    errors.push(
      "Audio references require at least one image or video reference.",
    );
  }
  if (videoDurationMs > MINIMAX_H3_MAX_REFERENCE_VIDEO_MS) {
    errors.push("Combined reference video must be 15 seconds or less.");
  }
  if (audioDurationMs > MINIMAX_H3_MAX_REFERENCE_AUDIO_MS) {
    errors.push(
      "Combined standalone audio and video soundtracks must be 15 seconds or less.",
    );
  }

  return {
    total: drafts.length,
    images,
    videos,
    audios,
    videoDurationMs,
    audioDurationMs,
    errors,
  };
}

export function minimaxH3ReferenceName(
  reference: GenerationReference,
  index: number,
): string {
  return (
    reference.provenance?.name?.trim() ||
    `${reference.kind[0]!.toUpperCase()}${reference.kind.slice(1)} ${index + 1}`
  );
}

export function minimaxH3ReferenceDurationMs(
  reference: GenerationReference,
): number | null {
  return reference.kind === "image" ? null : reference.duration_ms;
}

export function moveMinimaxH3Reference(
  references: readonly MinimaxH3ReferenceDraft[],
  from: number,
  to: number,
): MinimaxH3ReferenceDraft[] {
  if (
    from < 0 ||
    to < 0 ||
    from >= references.length ||
    to >= references.length ||
    from === to
  ) {
    return [...references];
  }
  const next = [...references];
  const [entry] = next.splice(from, 1);
  next.splice(to, 0, entry!);
  return next;
}

export function minimaxH3ReferenceNeedsMedia(
  draft: MinimaxH3ReferenceDraft,
): boolean {
  return draft.reference.media.authority === "descriptor";
}

export function minimaxH3AuthoringError(
  family: string | null | undefined,
  model: string,
  state: MinimaxH3AuthoringState | null | undefined,
): string | null {
  if (!isMinimaxH3Identity(family, model)) return null;
  const task = minimaxH3TaskForModel(model);
  if (!task)
    return "MiniMax H3 requires an explicit FL2VA or Ref2VA model partition.";
  const value = state ?? emptyMinimaxH3AuthoringState();
  if (task === "fl2va") {
    if (value.firstFrame && !value.firstFrame.data) {
      return `Reattach first frame ${value.firstFrame.filename} before generating.`;
    }
    if (value.lastFrame && !value.lastFrame.data) {
      return `Reattach last frame ${value.lastFrame.filename} before generating.`;
    }
    return null;
  }
  const budget = minimaxH3ReferenceBudget(value.references);
  if (budget.errors[0]) return budget.errors[0];
  const missing = value.references.findIndex(minimaxH3ReferenceNeedsMedia);
  if (missing >= 0) {
    return `Reattach reference ${missing + 1} (${minimaxH3ReferenceName(value.references[missing]!.reference, missing)}) before generating.`;
  }
  return null;
}

type H3Request = {
  frames?: number | null;
};

const FOREIGN_H3_FIELDS = [
  "negative_prompt",
  "scheduler",
  "cfg_plus",
  "mask_image",
  "control_image",
  "control_model",
  "control_scale",
  "loras",
  "lora",
  "upscale_model",
  "gif_preview",
  "enable_audio",
  "audio_file",
  "audio_file_path",
  "source_video",
  "source_video_path",
  "extend_video",
  "extend_video_path",
  "extend_overlap_frames",
  "pipeline",
  "ic_lora_control",
  "retake_range",
  "spatial_upscale",
  "temporal_upscale",
  "guidance_overrides",
] as const;

/** Final shared H3 request projection. This is deliberately pure so web,
 * desktop, iPhone, prepared expansion, and retry can snapshot exactly the
 * same object. */
export function serializeMinimaxH3Authoring<T extends H3Request>(
  request: T,
  family: string | null | undefined,
  model: string,
  state: MinimaxH3AuthoringState,
): T {
  if (!isMinimaxH3Identity(family, model)) return { ...request };
  const task = minimaxH3TaskForModel(model);

  const frames = Math.min(
    MINIMAX_H3_MAX_FRAMES,
    Math.max(
      MINIMAX_H3_MIN_FRAMES,
      MINIMAX_H3_FRAME_OFFSET +
        Math.round(
          (Number(request.frames ?? MINIMAX_H3_MIN_FRAMES) -
            MINIMAX_H3_FRAME_OFFSET) /
            MINIMAX_H3_FRAME_STEP,
        ) *
          MINIMAX_H3_FRAME_STEP,
    ),
  );
  const next: H3Request & Record<string, unknown> = {
    ...request,
    frames,
    fps: MINIMAX_H3_FIXED_FPS,
    batch_size: 1,
    guidance: 0,
    strength: 1,
    output_format: "mp4",
  };
  for (const field of FOREIGN_H3_FIELDS) delete next[field];
  delete next.edit_images;

  // Unknown H3 partitions are not submit-ready, but estimates/persistence
  // must still stay inside the family's fixed AV contract and must not leak
  // an unrelated model's conditioning fields while validation explains the
  // missing explicit task.
  if (!task) {
    delete next.source_image;
    delete next.source_image_name;
    delete next.keyframes;
    delete next.references;
    return next as T;
  }

  if (task === "ref2va") {
    delete next.source_image;
    delete next.source_image_name;
    delete next.keyframes;
    next.references = state.references.map(({ reference }) =>
      JSON.parse(JSON.stringify(reference)),
    );
  } else {
    delete next.references;
    if (state.firstFrame) {
      next.source_image = state.firstFrame.data;
      next.source_image_name = state.firstFrame.filename;
    } else {
      delete next.source_image;
      delete next.source_image_name;
    }
    next.keyframes = state.lastFrame
      ? [{ frame: frames - 1, image: state.lastFrame.data }]
      : undefined;
    if (!state.lastFrame) delete next.keyframes;
  }
  return next as T;
}

/** Rebuild redacted reference rows from durable gallery provenance. The
 * descriptor authority makes missing media explicit: Recreate can display the
 * exact order but cannot queue until the original bytes are reattached. */
export function minimaxH3ReferenceDraftsFromMetadata(
  metadata: readonly GenerationReferenceMetadata[] | null | undefined,
): MinimaxH3ReferenceDraft[] {
  return (metadata ?? []).map(
    ({ index: _index, prepared_shape: _shape, ...row }) => {
      const media = { authority: "descriptor" as const };
      const provenance = { name: row.name ?? null, sha256: row.sha256 };
      if (row.kind === "image") {
        return {
          reference: {
            kind: "image",
            media,
            provenance,
            mime_type: row.mime_type,
            width: row.width ?? 0,
            height: row.height ?? 0,
          },
        };
      }
      if (row.kind === "video") {
        return {
          reference: {
            kind: "video",
            media,
            provenance,
            mime_type: row.mime_type,
            width: row.width ?? 0,
            height: row.height ?? 0,
            frame_count: row.frame_count ?? null,
            duration_ms: row.duration_ms ?? 0,
            fps: row.fps ?? 0,
            has_audio: row.has_audio ?? false,
            audio_duration_ms: row.audio_duration_ms ?? null,
            audio_sample_count: row.audio_sample_count ?? null,
            audio_sample_rate: row.audio_sample_rate ?? null,
            audio_channels: row.audio_channels ?? null,
          },
        };
      }
      return {
        reference: {
          kind: "audio",
          media,
          provenance,
          mime_type: row.mime_type,
          duration_ms: row.duration_ms ?? 0,
          sample_rate: row.sample_rate ?? 0,
          channels: row.channels ?? 0,
          sample_count: row.sample_count ?? null,
        },
      };
    },
  );
}

/** Recreate-safe FL2VA opening-frame provenance. Gallery metadata carries a
 * display name and exact digest, never payload bytes, so this deliberately
 * returns a reattach-required descriptor. */
export function minimaxH3BoundaryFromSourceMetadata(
  name: string | null | undefined,
  sha256: string | null | undefined,
): MinimaxH3BoundaryImage | null {
  if (!name?.trim() && !sha256?.trim()) return null;
  return {
    filename: name?.trim() || "First frame",
    mimeType: "image/*",
    width: 0,
    height: 0,
    data: "",
    sha256: sha256?.trim() || null,
  };
}
