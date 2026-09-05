import {
  DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
  type SequenceTransition,
} from "@ui/lib/seam";
import { FALLBACK_VIDEO_FPS } from "@ui/lib/duration";

// One owner for the wan seam size and the per-model routing clip size;
// `./chainRouting` computes the routing that applies them, so they live there
// and authoring reads them.
import {
  LTX2_DEFAULT_CLIP_FRAMES,
  WAN_HANDOFF_DUPLICATED_FRAMES,
  wanRoutingClipFrames,
} from "./chainRouting";

// The seam vocabulary is presentational and lives beside the SeamPill /
// SeamEditor primitives in ui/; re-exported here so studio consumers keep
// one import surface for sequence logic.
export {
  DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
  transitionDescription,
  transitionLabel,
  type SequenceTransition,
} from "@ui/lib/seam";
export { formatFrameDuration } from "@ui/lib/duration";

export interface SequenceModel {
  name: string;
  family: string;
  supports_sequence?: boolean | null;
  /**
   * The checkpoint's advertised source-image contract. Wan reads it to decide
   * whether the seam can carry context at all (#783): it has no latent motion
   * tail, so its smooth handoff is last-frame image conditioning, which only
   * an image-conditioned checkpoint accepts. Absent reads as "unknown" and
   * takes the conservative independent-clip path.
   *
   * Typed as the raw wire string, matching `ModelInfoExtended`: an
   * unrecognized value is not a handoff, which is what the conservative branch
   * already yields.
   */
  source_image?: string | null;
}

export interface SequenceStage {
  prompt: string;
  frames: number;
  transition: SequenceTransition;
  fade_frames?: number | null;
}

export interface SequenceLimits {
  maxStages: number;
  maxTotalFrames: number;
  maxFramesPerClip?: number;
  frameStep?: number;
  frameOffset?: number;
  motionTailFrames: number;
  /**
   * Whether a clip may be left undescribed — `promptOptional` from
   * `./promptRequirement` for the draft's family and conditioning. Absent
   * reads as "required", which is every image family and unconditioned
   * text-to-video.
   */
  promptOptional?: boolean;
  /**
   * The caller's words for one piece and for the whole: web and the phone
   * say `clip` / `sequence` (the default); desktop says `scene` / `clip`,
   * because there the whole thing IS the clip and one word, scene, names
   * every piece on that surface. Only the sentences change.
   */
  wording?: SequenceWording;
}

export interface SequenceWording {
  piece: string;
  whole: string;
}

export const DEFAULT_SEQUENCE_WORDING: SequenceWording = {
  piece: "clip",
  whole: "sequence",
};

// Verified on private UAT host's 48 GB L40S at the catalog model's 1216×704 defaults.
// The server's per-clip cap/recommendation describes a format limit, not a
// promise that the largest clip fits the active GPU at every resolution.
export const DEFAULT_SEQUENCE_CLIP_FRAMES = 25;

export function sequenceMotionTailFrames(
  model:
    Pick<SequenceModel, "name" | "family" | "source_image"> | null | undefined,
): number {
  const family = model?.family.trim().toLowerCase();
  // LTX-Video renders independent clips: nothing crosses the seam, so there is
  // no tail to trim.
  if (family === "ltx-video") return 0;
  // Wan's answer is per checkpoint, not per family. A text-to-video checkpoint
  // has no conditioning channel at all, so offering it a tail would promise
  // continuity it cannot produce; an unclassified checkpoint is "unknown" and
  // takes the same conservative path.
  if (family === "wan") {
    const source = model?.source_image;
    return source === "required" || source === "optional"
      ? WAN_HANDOFF_DUPLICATED_FRAMES
      : 0;
  }
  return DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES;
}

/**
 * The frame-count grid a family's clips must sit on.
 *
 * Wan's VAE compresses time by 4 where the LTX families compress by 8, so its
 * valid counts are `4k+1`, not `8k+1`. Offering an off-grid option sends the
 * request straight into a 422 from the validator that owns the real rule.
 */
export function sequenceFrameStep(family: string | null | undefined): number {
  return family?.trim().toLowerCase() === "wan" ? 4 : 8;
}

export function sequenceFrameOptions(
  framesPerClipCap: number,
  motionTailFrames: number,
  family?: string | null,
): number[] {
  const step = sequenceFrameStep(family);
  const options: number[] = [];
  for (let frames = step + 1; frames <= framesPerClipCap; frames += step) {
    if (frames > motionTailFrames) options.push(frames);
  }
  return options;
}

export function modelSupportsSequence(
  model: SequenceModel | null | undefined,
): boolean {
  if (!model) return false;
  if (typeof model.supports_sequence === "boolean")
    return model.supports_sequence;
  // Fallback for a server that does not advertise the field. Wan and
  // LTX-Video chain for every checkpoint in the family; what varies for wan is
  // the carryover, not whether it can render clips at all.
  if (model.family === "ltx-video" || model.family === "wan") return true;
  if (model.family !== "ltx2" && model.family !== "ltx-2") return false;

  const name = model.name.toLowerCase();
  return (
    (name.includes("distilled") ||
      name.startsWith("cv:") ||
      name.startsWith("hf:")) &&
    !name.includes("-dev")
  );
}

export function defaultSequenceStages(
  frames = DEFAULT_SEQUENCE_CLIP_FRAMES,
): SequenceStage[] {
  return [
    { prompt: "", frames, transition: "smooth" },
    { prompt: "", frames, transition: "smooth" },
  ];
}

export function friendlySequenceError(
  error: string,
  hostName?: string | null,
): string {
  const normalized = error.toLowerCase();
  if (
    normalized.includes("cuda_error_out_of_memory") ||
    normalized.includes("out of memory")
  ) {
    return "This sequence needs more GPU memory. Shorten the clip duration or reduce the size, then try again.";
  }
  return describeTransportError(error, hostName);
}

export function sequenceDuration(
  stages: readonly SequenceStage[],
  fps: number,
  motionTailFrames: number,
): { frames: number; seconds: number } {
  const frames = stages.reduce((total, stage, index) => {
    if (index === 0) return total + stage.frames;
    if (stage.transition === "smooth") {
      return total + Math.max(0, stage.frames - motionTailFrames);
    }
    if (stage.transition === "fade") {
      return total + Math.max(0, stage.frames - (stage.fade_frames ?? 8));
    }
    return total + stage.frames;
  }, 0);
  return { frames, seconds: fps > 0 ? frames / fps : 0 };
}

export function sequenceValidation(
  stages: readonly SequenceStage[],
  limits: SequenceLimits,
): string[] {
  const { piece, whole } = limits.wording ?? DEFAULT_SEQUENCE_WORDING;
  if (stages.length < 2)
    return [`Add at least two ${piece}s to make a ${whole}.`];
  if (!limits.promptOptional) {
    const empty = stages.findIndex((stage) => !stage.prompt.trim());
    if (empty >= 0)
      return [`Describe ${piece} ${empty + 1} before generating.`];
  }
  if (stages.length > limits.maxStages) {
    return [`Reduce the ${whole} to ${limits.maxStages} ${piece}s or fewer.`];
  }
  if (limits.maxFramesPerClip != null) {
    const cap = limits.maxFramesPerClip;
    const oversized = stages.findIndex((stage) => stage.frames > cap);
    if (oversized >= 0) {
      return [`Reduce ${piece} ${oversized + 1} to ${cap} frames or fewer.`];
    }
  }
  if (limits.frameStep != null) {
    const step = limits.frameStep;
    const offset = limits.frameOffset ?? 1;
    const offGrid = stages.findIndex(
      (stage) => stage.frames % step !== offset % step,
    );
    if (offGrid >= 0) {
      return [
        `Change ${piece} ${offGrid + 1} to the ${step}k+${offset} frame grid.`,
      ];
    }
  }
  const total = sequenceDuration(stages, 1, limits.motionTailFrames).frames;
  if (total > limits.maxTotalFrames) {
    return [
      `Reduce ${piece} durations to ${limits.maxTotalFrames} total frames or fewer.`,
    ];
  }
  return [];
}

export type OutputMode = "single" | "sequence";

/**
 * Filter the installed-model list for the active Output mode: Sequence
 * shows only chain-capable video models; One shot passes everything
 * through. Every surface's picker consumes this so the filtering rule
 * can't drift.
 */
export function modelsForOutput<M extends SequenceModel>(
  models: readonly M[],
  output: OutputMode,
): M[] {
  if (output !== "sequence") return [...models];
  return models.filter((model) => modelSupportsSequence(model));
}

/** Frame rate used when nothing else is known — older servers and image
 * models omit the additive `/api/models.default_fps`. */
export const DEFAULT_VIDEO_FPS = FALLBACK_VIDEO_FPS;

/**
 * Frame rate for a selected video model: the model's own server-advertised
 * default (`/api/models.default_fps` — LTX-Video ships 30, LTX-2 24), then
 * whatever the form already holds, then the 24-fps fallback.
 *
 * Every surface applies this on model selection exactly as it applies
 * `default_steps` / `default_guidance`, so the sequence composer's duration
 * note, the Advanced video summary, and the submitted request all agree with
 * the model. Governs one-shot video and sequences alike.
 */
export function defaultVideoFps(
  model: { default_fps?: number | null } | null | undefined,
  current?: number | null,
): number {
  return model?.default_fps ?? current ?? DEFAULT_VIDEO_FPS;
}

/**
 * Largest clip a model renders as ONE generation.
 *
 * `frames_per_clip_cap` from `/api/capabilities/chain-limits` is the server's
 * answer, but a server predating the per-model fix advertises the FAMILY's
 * single-request ceiling instead — 481 LTX-2 frames at 24 fps — which is a
 * clip the one-shot auto-chain router would have split into five. Bounding it
 * by the model's own routing clip size (`./chainRouting`) keeps every picker
 * locked to what actually renders as one clip, on old and new hosts alike, and
 * a server advertising something SMALLER (the duration budget still binds
 * below 97 at very low fps) still wins.
 *
 * A family with no routing clip size adds no bound of its own.
 */
export function sequenceClipFrameCap(
  model:
    | {
        name?: string | null;
        family?: string | null;
        default_frames?: number | null;
      }
    | null
    | undefined,
  limits:
    | { frames_per_clip_cap: number; frames_per_clip_recommended?: number }
    | null
    | undefined,
): number {
  const routing = routingClipFrames(model, limits);
  const advertised =
    limits?.frames_per_clip_cap ?? routing ?? LTX2_DEFAULT_CLIP_FRAMES;
  return routing === null ? advertised : Math.min(advertised, routing);
}

/**
 * The model's own routing clip size, mirroring
 * `mold_core::chain::routing_clip_frames`. `null` = the family publishes none,
 * so nothing bounds the server's advertised cap.
 */
function routingClipFrames(
  model:
    | {
        name?: string | null;
        family?: string | null;
        default_frames?: number | null;
      }
    | null
    | undefined,
  limits: { frames_per_clip_recommended?: number } | null | undefined,
): number | null {
  switch (model?.family?.trim().toLowerCase()) {
    case "ltx2":
    case "ltx-2":
    case "ltx-video":
      return LTX2_DEFAULT_CLIP_FRAMES;
    case "wan":
      // Wan's routing size is the checkpoint's own recorded default over the
      // family floor (`mold_core::chain::wan_default_clip_frames`). The
      // browser has no manifest, but it has the same default on
      // `/api/models.default_frames` and on chain-limits'
      // `frames_per_clip_recommended`, which old and new servers both send.
      return wanRoutingClipFrames(
        model?.name ?? "",
        model?.default_frames ?? limits?.frames_per_clip_recommended ?? null,
      );
    default:
      return null;
  }
}

/**
 * Default frames for a NEW clip: the model's own server-advertised default
 * (`/api/models.default_frames` — LTX-2 ships 97, LTX-Video 25), then the
 * chain-limits recommendation, then the conservative 25-frame floor; the
 * result is clamped to the per-clip cap, snapped DOWN onto the 8n+1 grid,
 * and raised to the first valid option strictly greater than the motion
 * tail. The cap is a format limit, not a promise the largest clip fits the
 * active GPU — which is exactly why default/recommended win over cap.
 */
export function defaultClipFrames(
  model:
    | {
        name?: string | null;
        default_frames?: number | null;
        family?: string | null;
      }
    | null
    | undefined,
  limits:
    | { frames_per_clip_cap: number; frames_per_clip_recommended: number }
    | null
    | undefined,
  motionTailFrames: number,
): number {
  const step = sequenceFrameStep(model?.family);
  // The same cap the picker offers, so a default can never exceed its own
  // options — including on a host that still advertises the family ceiling.
  const cap =
    limits || model?.family
      ? sequenceClipFrameCap(model, limits)
      : Number.MAX_SAFE_INTEGER;
  const preferred =
    model?.default_frames ??
    limits?.frames_per_clip_recommended ??
    DEFAULT_SEQUENCE_CLIP_FRAMES;
  let frames = Math.min(preferred, cap);
  if (frames > 1) frames -= (frames - 1) % step;
  while (frames <= motionTailFrames) frames += step;
  return Math.max(frames, step + 1);
}
import { describeTransportError } from "./errors";
