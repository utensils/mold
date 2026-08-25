import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "./generationReferences";
import {
  isModelAccessRestricted,
  type ModelAccessCapabilityRecord,
} from "./modelAccess";
import { imageDimensionsFromBase64 } from "./imageDimensions";
import { MINIMAX_H3_REVIEWED_COMPACT_STEPS } from "./minimaxH3Inventory";
import { readFileBase64 } from "./fileBase64";
import {
  canonicalMinimaxH3ModelName,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  MINIMAX_H3_FL2VA_COMFY as FL2VA_COMFY_BASE,
  MINIMAX_H3_REF2VA_COMFY as REF2VA_COMFY_BASE,
  type MinimaxH3Task,
} from "./minimaxH3Identity";

export {
  canonicalMinimaxH3ModelName,
  isMinimaxH3Family,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  MINIMAX_H3_FL2VA_COMFY,
  MINIMAX_H3_FL2VA_COMFY_NVFP4,
  MINIMAX_H3_FL2VA_COMFY_TURBO_4STEP_768P,
  MINIMAX_H3_FL2VA_COMFY_TURBO_8STEP,
  MINIMAX_H3_FL2VA_OFFICIAL,
  MINIMAX_H3_REF2VA_COMFY,
  MINIMAX_H3_REF2VA_COMFY_NVFP4,
  MINIMAX_H3_REF2VA_OFFICIAL,
  type MinimaxH3Task,
} from "./minimaxH3Identity";

export const MINIMAX_H3_FIXED_FPS = 24;
// Mirrors `mold_core::minimax_h3::REVIEWED_COMPACT_FRAMES` — the compact
// stack's DEFAULT clip length, and only that. It was the exact length the
// runtime rendered, because the runtime envelope validated `frames` by
// equality; the envelope is minted per request now and a compact tag takes
// the family grid, so this seeds a form rather than gating one.
export const MINIMAX_H3_REVIEWED_COMPACT_FRAMES = 124;
// Mirrors `mold_core::minimax_h3::MIN_FRAMES` — the family floor, derived
// from the model card's 4-second minimum at the fixed 24 fps.
export const MINIMAX_H3_MIN_FRAMES = 107;
// Mirrors `mold_core::minimax_h3::MAX_FRAMES`. 345, not 362: the next grid
// value is 15.083 s at the family's fixed 24 fps, which the diffusers path
// rejects. This is the fallback for a server that does not advertise its own
// `max_frames`; an advertised value always wins.
export const MINIMAX_H3_MAX_FRAMES = 345;
export const MINIMAX_H3_FRAME_STEP = 17;
export const MINIMAX_H3_FRAME_OFFSET = 5;
// Mirrors `mold_core::minimax_h3::COMPACT_MIN_STEPS` /
// `COMPACT_MAX_STEPS` — the base compact tag's step range. A reviewed Turbo
// tier keeps its distilled adapter's exact count instead
// (`MINIMAX_H3_REVIEWED_COMPACT_STEPS`).
export const MINIMAX_H3_COMPACT_MIN_STEPS = 2;
export const MINIMAX_H3_COMPACT_MAX_STEPS = 50;
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
  /** The step range this identity accepts. A reviewed Turbo tier reports its
   * distilled adapter's exact count on both bounds, because that count is the
   * schedule's length rather than a preference. */
  minSteps: number;
  maxSteps: number;
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
  generation_profile?: { profile_hash?: string | null } | null;
}

export function emptyMinimaxH3AuthoringState(): MinimaxH3AuthoringState {
  return { firstFrame: null, lastFrame: null, references: [] };
}

export interface MinimaxH3GalleryImageSource {
  filename: string;
  mimeType: string;
  width: number;
  height: number;
  /** Raw base64 without a data-URI prefix. */
  data: string;
  sha256?: string | null;
}

/** Surface-picker shape shared by desktop, web, and iPhone. Their established
 * pickers all return base64 + a filename, while dimensions and MIME may be
 * present when the picker already knows them. */
export interface MinimaxH3PickedImageSource {
  filename: string;
  base64: string;
  mimeType?: string | null;
  width?: number | null;
  height?: number | null;
}

export type MinimaxH3GalleryImageResult =
  | {
      ok: true;
      state: MinimaxH3AuthoringState;
      /** One-based ordered reference position; null for an FL2VA boundary. */
      reference: number | null;
    }
  | { ok: false; error: string };

function validateGalleryImageSource(
  image: MinimaxH3GalleryImageSource,
): string | null {
  if (!image.filename.trim()) return "The gallery image needs a filename.";
  const mimeType = image.mimeType.split(";", 1)[0]!.trim().toLowerCase();
  if (!mimeType.startsWith("image/")) {
    return "Only gallery images can be used as MiniMax H3 visual references.";
  }
  if (
    !Number.isInteger(image.width) ||
    image.width <= 0 ||
    !Number.isInteger(image.height) ||
    image.height <= 0
  ) {
    return "The gallery image dimensions could not be read.";
  }
  if (!image.data.trim()) return "The gallery image is empty.";
  return null;
}

/** Append one gallery print to the authoritative Ref2VA order as inline media.
 * The returned state is shallowly immutable: existing potentially-large media
 * strings are retained without a JSON-clone, while the ordered array and new
 * row are fresh objects for Vue reactivity. */
export function appendMinimaxH3GalleryImageReference(
  state: MinimaxH3AuthoringState | null | undefined,
  image: MinimaxH3GalleryImageSource,
): MinimaxH3GalleryImageResult {
  const invalid = validateGalleryImageSource(image);
  if (invalid) return { ok: false, error: invalid };
  const current = state ?? emptyMinimaxH3AuthoringState();
  const budget = minimaxH3ReferenceBudget(current.references);
  if (budget.total >= MINIMAX_H3_MAX_REFERENCES) {
    return {
      ok: false,
      error: `Use at most ${MINIMAX_H3_MAX_REFERENCES} references total.`,
    };
  }
  if (budget.images >= MINIMAX_H3_MAX_REFERENCE_IMAGES) {
    return {
      ok: false,
      error: `Use at most ${MINIMAX_H3_MAX_REFERENCE_IMAGES} image references.`,
    };
  }
  const mimeType = image.mimeType.split(";", 1)[0]!.trim().toLowerCase();
  const reference: GenerationReference = {
    kind: "image",
    media: { authority: "inline", data: image.data },
    provenance: {
      name: image.filename.trim(),
      ...(image.sha256 ? { sha256: image.sha256.toLowerCase() } : {}),
    },
    mime_type: mimeType,
    width: image.width,
    height: image.height,
  };
  const references = [...current.references, { reference }];
  return {
    ok: true,
    state: { ...current, references },
    reference: references.length,
  };
}

/** The two FL2VA boundary slots share one setter contract. */
export type MinimaxH3BoundaryEndpoint = "firstFrame" | "lastFrame";

/** Use one gallery print as an FL2VA boundary. Kept beside the
 * ordered-reference helper so every Library surface validates identical image
 * facts before writing dedicated H3 authoring state. */
export function setMinimaxH3GalleryImageBoundary(
  state: MinimaxH3AuthoringState | null | undefined,
  endpoint: MinimaxH3BoundaryEndpoint,
  image: MinimaxH3GalleryImageSource,
): MinimaxH3GalleryImageResult {
  const invalid = validateGalleryImageSource(image);
  if (invalid) return { ok: false, error: invalid };
  const current = state ?? emptyMinimaxH3AuthoringState();
  return {
    ok: true,
    state: {
      ...current,
      [endpoint]: {
        filename: image.filename.trim(),
        mimeType: image.mimeType.split(";", 1)[0]!.trim().toLowerCase(),
        width: image.width,
        height: image.height,
        data: image.data,
        ...(image.sha256 ? { sha256: image.sha256.toLowerCase() } : {}),
      },
    },
    reference: null,
  };
}

export function setMinimaxH3GalleryImageFirstFrame(
  state: MinimaxH3AuthoringState | null | undefined,
  image: MinimaxH3GalleryImageSource,
): MinimaxH3GalleryImageResult {
  return setMinimaxH3GalleryImageBoundary(state, "firstFrame", image);
}

/** Normalize an image from any existing surface picker into the one FL2VA
 * boundary contract. This replaces three H3-only file readers without making
 * the shared authoring state depend on a desktop, web, or native picker type. */
export function setMinimaxH3PickedImageBoundary(
  state: MinimaxH3AuthoringState | null | undefined,
  endpoint: MinimaxH3BoundaryEndpoint,
  image: MinimaxH3PickedImageSource,
): MinimaxH3GalleryImageResult {
  const decoded = imageDimensionsFromBase64(image.base64);
  const width = image.width ?? decoded?.width ?? 0;
  const height = image.height ?? decoded?.height ?? 0;
  const extension = image.filename.trim().toLowerCase();
  const mimeType =
    image.mimeType?.split(";", 1)[0]?.trim().toLowerCase() ||
    (extension.endsWith(".jpg") || extension.endsWith(".jpeg")
      ? "image/jpeg"
      : "image/png");
  return setMinimaxH3GalleryImageBoundary(state, endpoint, {
    filename: image.filename,
    mimeType,
    width,
    height,
    data: image.base64,
  });
}

export function setMinimaxH3PickedImageFirstFrame(
  state: MinimaxH3AuthoringState | null | undefined,
  image: MinimaxH3PickedImageSource,
): MinimaxH3GalleryImageResult {
  return setMinimaxH3PickedImageBoundary(state, "firstFrame", image);
}

/** Read a picked or dropped File into an FL2VA boundary. All surfaces route
 * their file wells through this so a file and a gallery pick produce
 * identical boundary facts. */
export async function setMinimaxH3BoundaryFile(
  state: MinimaxH3AuthoringState | null | undefined,
  endpoint: MinimaxH3BoundaryEndpoint,
  file: File,
): Promise<MinimaxH3GalleryImageResult> {
  if (!file.type.toLowerCase().startsWith("image/")) {
    return { ok: false, error: "FL2VA endpoints must be still images." };
  }
  let base64: string;
  try {
    base64 = await readFileBase64(file);
  } catch (reason) {
    return {
      ok: false,
      error: reason instanceof Error ? reason.message : String(reason),
    };
  }
  if (!imageDimensionsFromBase64(base64)) {
    return { ok: false, error: "Use a PNG or JPEG image for FL2VA endpoints." };
  }
  return setMinimaxH3PickedImageBoundary(state, endpoint, {
    filename: file.name,
    base64,
    mimeType: file.type,
  });
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
    generation_profile_sha256: model.generation_profile?.profile_hash ?? null,
  });
  // A reviewed Turbo tier's count is its distilled adapter's schedule length,
  // so both bounds are that number; the base tag takes the compact range.
  const canonical = canonicalMinimaxH3ModelName(model.name) ?? model.name;
  const reviewedSteps = MINIMAX_H3_REVIEWED_COMPACT_STEPS[canonical];
  const turboSteps =
    reviewedSteps != null &&
    canonical !== FL2VA_COMFY_BASE &&
    canonical !== REF2VA_COMFY_BASE
      ? reviewedSteps
      : null;
  return {
    task,
    runtimeAvailable: model.runtime_available !== false && !restricted,
    fixedFps: MINIMAX_H3_FIXED_FPS,
    minFrames: MINIMAX_H3_MIN_FRAMES,
    maxFrames: MINIMAX_H3_MAX_FRAMES,
    frameStep: MINIMAX_H3_FRAME_STEP,
    frameOffset: MINIMAX_H3_FRAME_OFFSET,
    minSteps: turboSteps ?? MINIMAX_H3_COMPACT_MIN_STEPS,
    maxSteps: turboSteps ?? MINIMAX_H3_COMPACT_MAX_STEPS,
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
  requireFirstFrame = false,
): string | null {
  if (!isMinimaxH3Identity(family, model)) return null;
  const task = minimaxH3TaskForModel(model);
  if (!task)
    return "MiniMax H3 requires an explicit FL2VA or Ref2VA model partition.";
  const value = state ?? emptyMinimaxH3AuthoringState();
  if (task === "fl2va") {
    if (requireFirstFrame && !value.firstFrame) {
      return "This reviewed MiniMax H3 runtime requires a first frame.";
    }
    if (requireFirstFrame && value.lastFrame) {
      return "This reviewed MiniMax H3 runtime accepts only one first-frame endpoint.";
    }
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
  model?: string;
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

  // Snap onto the family grid and clamp to the family bounds. The floor is
  // `MIN_FRAMES`, not the default clip length: a compact tag renders the whole
  // `17n+5` range now, so clamping up to 124 would silently lengthen a clip
  // the user asked to be shorter.
  const frames = Math.min(
    MINIMAX_H3_MAX_FRAMES,
    Math.max(
      MINIMAX_H3_MIN_FRAMES,
      MINIMAX_H3_FRAME_OFFSET +
        Math.round(
          (Number(request.frames ?? MINIMAX_H3_REVIEWED_COMPACT_FRAMES) -
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
  const canonicalModel = canonicalMinimaxH3ModelName(model);
  if (canonicalModel) next.model = canonicalModel;
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
      ? [
          {
            frame: frames - 1,
            image: state.lastFrame.data,
            name: state.lastFrame.filename,
          },
        ]
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

/** A staged single-source image as the Create surfaces hold one — web's
 * `SourceImageState`, desktop's `sourceImage` + sidecar fields. Everything but
 * the bytes is optional so either shape bridges without adapters. */
export interface StagedSourceImageLike {
  base64: string;
  filename?: string | null;
  width?: number | null;
  height?: number | null;
  mime?: string | null;
  sha256?: string | null;
  draftId?: string;
}

/** Model-switch bridge: carry a staged single-source image into the H3
 * first/last-frame authority so switching into FL2VA keeps the picture.
 * Bytes are required — a descriptor without payload has nothing to render. */
export function minimaxH3BoundaryFromStagedImage(
  staged: StagedSourceImageLike | null | undefined,
): MinimaxH3BoundaryImage | null {
  const data = staged?.base64 ?? "";
  if (!data.trim()) return null;
  const boundary: MinimaxH3BoundaryImage = {
    filename: staged?.filename?.trim() || "First frame",
    mimeType: staged?.mime?.trim() || "image/*",
    width: staged?.width ?? 0,
    height: staged?.height ?? 0,
    data,
    sha256: staged?.sha256?.trim() || null,
  };
  if (staged?.draftId) boundary.draftId = staged.draftId;
  return boundary;
}

/** Reverse bridge: promote an H3 boundary back into the staged-image shape
 * when leaving FL2VA for a single-source model. A bytes-less reattach
 * descriptor (gallery provenance) must never fill a source well — the well
 * would look populated while the request had nothing to send. */
export function stagedImageFromMinimaxH3Boundary(
  boundary: MinimaxH3BoundaryImage | null | undefined,
): (StagedSourceImageLike & { base64: string; filename: string }) | null {
  if (!boundary?.data?.trim()) return null;
  const staged: StagedSourceImageLike & { base64: string; filename: string } = {
    base64: boundary.data,
    filename: boundary.filename || "First frame",
    width: boundary.width || null,
    height: boundary.height || null,
    mime: boundary.mimeType === "image/*" ? null : boundary.mimeType || null,
    sha256: boundary.sha256 ?? null,
  };
  if (boundary.draftId) staged.draftId = boundary.draftId;
  return staged;
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

export interface MinimaxH3KeyframeMetadataLike {
  frame: number;
  name?: string | null;
  sha256: string;
}

/** Recreate-safe FL2VA closing-frame provenance. Prefer the exact final-frame
 * entry; accept one singleton only when the closing role is unambiguous. */
export function minimaxH3ClosingBoundaryFromMetadata(
  frames: number | null | undefined,
  keyframes: readonly MinimaxH3KeyframeMetadataLike[] | null | undefined,
): MinimaxH3BoundaryImage | null {
  if (!keyframes?.length) return null;
  const finalFrame = frames == null ? null : frames - 1;
  const closing =
    finalFrame == null
      ? keyframes.length === 1
        ? keyframes[0]!
        : null
      : (keyframes.find((keyframe) => keyframe.frame === finalFrame) ?? null);
  if (!closing) return null;
  return {
    filename: closing.name?.trim() || "Last frame",
    mimeType: "image/*",
    width: 0,
    height: 0,
    data: "",
    sha256: closing.sha256.trim() || null,
  };
}
