export type SequenceTransition = "smooth" | "cut" | "fade";

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

export function transitionLabel(transition: SequenceTransition): string {
  if (transition === "smooth") return "Continue motion";
  if (transition === "fade") return "Crossfade";
  return "Cut";
}
