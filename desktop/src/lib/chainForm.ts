/**
 * Reactive state for the chain composer and pure builders that project it onto
 * the wire `ChainRequest` (POST body) and `ChainScript` (TOML / edit view).
 * Kept separate from `chain.ts` (pure math/TOML) so the composer's editable
 * shape lives in one place, like `generateForm.ts` for single generations.
 */
import type { ChainRequest, ChainScript, ChainStage, TransitionMode } from "./api/types";
import { DEFAULT_FADE_FRAMES, DEFAULT_MOTION_TAIL_FRAMES, DEFAULT_FPS } from "./chain";

export interface ChainStageForm {
  prompt: string;
  frames: number;
  transition: TransitionMode;
  fadeFrames: number;
  negativePrompt: string;
}

export interface ChainForm {
  model: string;
  width: number;
  height: number;
  fps: number;
  /** Empty string = no explicit seed. */
  seed: string;
  steps: number;
  guidance: number;
  strength: number;
  motionTailFrames: number;
  enableAudio: boolean;
  stages: ChainStageForm[];
}

export function newStage(prompt = ""): ChainStageForm {
  return {
    prompt,
    frames: 97,
    transition: "smooth",
    fadeFrames: DEFAULT_FADE_FRAMES,
    negativePrompt: "",
  };
}

export function newChainForm(): ChainForm {
  return {
    model: "",
    width: 1216,
    height: 704,
    fps: DEFAULT_FPS,
    seed: "",
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    motionTailFrames: DEFAULT_MOTION_TAIL_FRAMES,
    enableAudio: false,
    stages: [newStage()],
  };
}

/** Project a form stage onto a wire `ChainStage`. Stage 0's transition is
 * always coerced to smooth (the server does this too — nothing precedes it). */
function stageToWire(stage: ChainStageForm, idx: number): ChainStage {
  const transition: TransitionMode = idx === 0 ? "smooth" : stage.transition;
  const wire: ChainStage = { prompt: stage.prompt.trim(), frames: stage.frames, transition };
  if (transition === "fade") wire.fade_frames = stage.fadeFrames;
  if (stage.negativePrompt.trim()) wire.negative_prompt = stage.negativePrompt.trim();
  return wire;
}

export function chainFormToRequest(form: ChainForm): ChainRequest {
  const seed = form.seed.trim() === "" ? undefined : Number(form.seed);
  const req: ChainRequest = {
    model: form.model,
    stages: form.stages.map(stageToWire),
    motion_tail_frames: form.motionTailFrames,
    width: form.width,
    height: form.height,
    fps: form.fps,
    steps: form.steps,
    guidance: form.guidance,
    strength: form.strength,
    output_format: "mp4",
  };
  if (seed !== undefined && Number.isFinite(seed)) req.seed = seed;
  if (form.enableAudio) req.enable_audio = true;
  return req;
}

export function chainFormToScript(form: ChainForm): ChainScript {
  const seed = form.seed.trim() === "" ? null : Number(form.seed);
  return {
    schema: "mold.chain.v1",
    chain: {
      model: form.model,
      width: form.width,
      height: form.height,
      fps: form.fps,
      seed: seed != null && Number.isFinite(seed) ? seed : null,
      steps: form.steps,
      guidance: form.guidance,
      strength: form.strength,
      motion_tail_frames: form.motionTailFrames,
      output_format: "mp4",
      ...(form.enableAudio ? { enable_audio: true } : {}),
    },
    stage: form.stages.map(stageToWire),
  };
}
