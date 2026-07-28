import {
  DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
  type SequenceTransition,
} from "@ui/lib/seam";

// The seam vocabulary is presentational and lives beside the SeamPill /
// SeamEditor primitives in ui/; re-exported here so studio consumers keep
// one import surface for sequence logic.
export {
  DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
  transitionDescription,
  transitionLabel,
  type SequenceTransition,
} from "@ui/lib/seam";

export interface SequenceModel {
  name: string;
  family: string;
  supports_sequence?: boolean | null;
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
  motionTailFrames: number;
}

// Verified on Plato's 48 GB L40S at the catalog model's 1216×704 defaults.
// The server's per-clip cap/recommendation describes a format limit, not a
// promise that the largest clip fits the active GPU at every resolution.
export const DEFAULT_SEQUENCE_CLIP_FRAMES = 25;

export function sequenceMotionTailFrames(
  model: Pick<SequenceModel, "name" | "family"> | null | undefined,
): number {
  return model?.family.trim().toLowerCase() === "ltx-video"
    ? 0
    : DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES;
}

export function sequenceFrameOptions(
  framesPerClipCap: number,
  motionTailFrames: number,
): number[] {
  const options: number[] = [];
  for (let frames = 9; frames <= framesPerClipCap; frames += 8) {
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
  if (model.family === "ltx-video") return true;
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

export function friendlySequenceError(error: string): string {
  const normalized = error.toLowerCase();
  if (
    normalized.includes("cuda_error_out_of_memory") ||
    normalized.includes("out of memory")
  ) {
    return "This sequence needs more GPU memory. Shorten the clip duration or reduce the size, then try again.";
  }
  return error;
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
  if (stages.length < 2) return ["Add at least two clips to make a sequence."];
  const empty = stages.findIndex((stage) => !stage.prompt.trim());
  if (empty >= 0) return [`Describe clip ${empty + 1} before generating.`];
  if (stages.length > limits.maxStages) {
    return [`Reduce the sequence to ${limits.maxStages} clips or fewer.`];
  }
  const total = sequenceDuration(stages, 1, limits.motionTailFrames).frames;
  if (total > limits.maxTotalFrames) {
    return [
      `Reduce clip durations to ${limits.maxTotalFrames} total frames or fewer.`,
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
export const DEFAULT_VIDEO_FPS = 24;

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
 * Default frames for a NEW clip: the model's own server-advertised default
 * (`/api/models.default_frames` — LTX-2 ships 97, LTX-Video 25), then the
 * chain-limits recommendation, then the conservative 25-frame floor; the
 * result is clamped to the per-clip cap, snapped DOWN onto the 8n+1 grid,
 * and raised to the first valid option strictly greater than the motion
 * tail. The cap is a format limit, not a promise the largest clip fits the
 * active GPU — which is exactly why default/recommended win over cap.
 */
export function defaultClipFrames(
  model: { default_frames?: number | null } | null | undefined,
  limits:
    | { frames_per_clip_cap: number; frames_per_clip_recommended: number }
    | null
    | undefined,
  motionTailFrames: number,
): number {
  const cap = limits?.frames_per_clip_cap ?? Number.MAX_SAFE_INTEGER;
  const preferred =
    model?.default_frames ??
    limits?.frames_per_clip_recommended ??
    DEFAULT_SEQUENCE_CLIP_FRAMES;
  let frames = Math.min(preferred, cap);
  if (frames > 1) frames -= (frames - 1) % 8;
  while (frames <= motionTailFrames) frames += 8;
  return Math.max(frames, 9);
}
