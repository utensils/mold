/**
 * Sequence form model shared by desktop, web, and iPhone Create.
 *
 * The clip list is the ONLY sequence-specific state; shared generation
 * params (model / shape / detail / seed / fps / …) stay in each surface's
 * existing generate form and are read at submit time via
 * [`buildChainRequest`]. That split is load-bearing: it is what fixes the
 * old web bug where the sequence composer kept stale private copies of
 * width/steps/guidance and silently ignored the inspector.
 */

import { createUuid } from "./id";
import type {
  ChainRequestWire,
  ChainScript,
  ChainStageWire,
} from "./api/chainTypes";
import { imageDimensionsFromBase64 } from "./imageDimensions";
import type { SequenceTransition } from "./sequence";
import type { SourceFitPolicy } from "./sourceFit";
import {
  CAMERA_MOTION_PRESETS,
  cameraMotionLabel,
  isCameraMotionPreset,
} from "./cameraMotion";

export interface SequenceClipSourceImage {
  filename: string;
  /** Original source raster dimensions, when decoded by the picker/loader. */
  width?: number;
  height?: number;
  /** IndexedDB key used to restore the payload without localStorage quota risk. */
  draftId?: string;
  /** Raw base64 payload; stripped before persistence (quota), kept in memory. */
  base64: string | null;
}

export interface SequenceClipForm {
  /** Stable client id — keys the rail, maps clips onto job stages. */
  id: string;
  prompt: string;
  frames: number;
  /** Transition INTO this clip; clip 0's value is coerced to smooth on the wire. */
  transition: SequenceTransition;
  fadeFrames: number;
  negativePrompt: string;
  /** Camera-control preset id or an explicit server-local `.safetensors` path. */
  cameraControl: string | null;
  sourceImage: SequenceClipSourceImage | null;
}

export interface SequenceSharedParams {
  model: string;
  family: string;
  width: number;
  height: number;
  fps: number;
  steps: number;
  guidance: number;
  strength: number;
  /** Client-side preprocessing policy for an opening image. */
  sourceFitPolicy?: SourceFitPolicy;
  /** Preferred source pre-upscaler when the policy requests one. */
  upscalerModel?: string;
  /** Decimal string; "" means random (matches the generate forms). */
  seed: string;
}

export const DEFAULT_FADE_FRAMES = 8;

export function newSequenceClip(frames: number): SequenceClipForm {
  return {
    id: createUuid(),
    prompt: "",
    frames,
    transition: "smooth",
    fadeFrames: DEFAULT_FADE_FRAMES,
    negativePrompt: "",
    cameraControl: null,
    sourceImage: null,
  };
}

function serializeStage(
  clip: SequenceClipForm,
  idx: number,
  openingImage: SequenceClipSourceImage | null,
  sourceField: "source_image" | "source_image_b64",
): ChainStageWire & ChainScript["stages"][number] {
  const transition: SequenceTransition = idx === 0 ? "smooth" : clip.transition;
  const stage: ChainStageWire & ChainScript["stages"][number] = {
    prompt: clip.prompt,
    frames: clip.frames,
    transition,
  };
  if (idx > 0 && transition === "fade") stage.fade_frames = clip.fadeFrames;
  if (clip.negativePrompt.trim()) stage.negative_prompt = clip.negativePrompt;
  const cameraControl = clip.cameraControl?.trim();
  if (cameraControl) {
    const preset = isCameraMotionPreset(cameraControl);
    stage.loras = [
      {
        path: preset ? `camera-control:${cameraControl}` : cameraControl,
        scale: 1,
        name: preset ? cameraMotionLabel(cameraControl) : "Camera motion",
      },
    ];
  }
  const sourceImage = idx === 0 ? openingImage : clip.sourceImage;
  if (sourceImage?.base64) stage[sourceField] = sourceImage.base64;
  return stage;
}

export function sequenceOpeningImageError(
  openingImage: SequenceClipSourceImage | null,
  restoring = false,
): string | null {
  if (!openingImage || openingImage.base64) return null;
  return restoring
    ? `Restoring opening image ${openingImage.filename}…`
    : `Reattach opening image ${openingImage.filename} before generating.`;
}

/** Build the `POST /api/chain-jobs` request from the LIVE shared params. */
export function buildChainRequest(
  shared: SequenceSharedParams,
  clips: readonly SequenceClipForm[],
  opts: {
    motionTailFrames: number;
    enableAudio: boolean;
    openingImage?: SequenceClipSourceImage | null;
  },
): ChainRequestWire {
  const openingError = sequenceOpeningImageError(opts.openingImage ?? null);
  if (openingError) throw new Error(openingError);
  const seed = shared.seed.trim() === "" ? undefined : Number(shared.seed);
  const req: ChainRequestWire = {
    model: shared.model,
    stages: clips.map((clip, idx) =>
      serializeStage(clip, idx, opts.openingImage ?? null, "source_image"),
    ),
    motion_tail_frames: opts.motionTailFrames,
    width: shared.width,
    height: shared.height,
    fps: shared.fps,
    steps: shared.steps,
    guidance: shared.guidance,
    strength: shared.strength,
    output_format: "mp4",
  };
  if (seed !== undefined && Number.isFinite(seed)) req.seed = seed;
  if (opts.enableAudio) req.enable_audio = true;
  return req;
}

/** Load a job's effective script (retakes/amends applied) into editable clips. */
export function chainScriptToClips(script: ChainScript): {
  clips: SequenceClipForm[];
  shared: Partial<SequenceSharedParams>;
  enableAudio: boolean;
  motionTailFrames: number | null;
  openingImage: SequenceClipSourceImage | null;
} {
  let openingImage: SequenceClipSourceImage | null = null;
  const clips = script.stages.map((stage, idx) => {
    const clip = newSequenceClip(stage.frames ?? 97);
    clip.prompt = stage.prompt;
    clip.transition = idx === 0 ? "smooth" : (stage.transition ?? "smooth");
    clip.fadeFrames = stage.fade_frames ?? DEFAULT_FADE_FRAMES;
    clip.negativePrompt = stage.negative_prompt ?? "";
    const cameraLora = stage.loras?.find(
      (lora) =>
        lora.path.startsWith("camera-control:") ||
        lora.name === "Camera motion" ||
        Boolean(
          lora.name &&
          CAMERA_MOTION_PRESETS.some((preset) => preset.label === lora.name),
        ),
    );
    if (cameraLora) {
      const alias = cameraLora.path.replace(/^camera-control:/, "");
      const namedPreset = CAMERA_MOTION_PRESETS.find(
        (preset) => preset.label === cameraLora.name,
      )?.id;
      clip.cameraControl = isCameraMotionPreset(alias)
        ? alias
        : (namedPreset ?? cameraLora.path);
    }
    if (stage.source_image_b64) {
      const dimensions = imageDimensionsFromBase64(stage.source_image_b64);
      const sourceImage = {
        filename: stage.source_image_path ?? "opening image",
        base64: stage.source_image_b64,
        ...(dimensions ?? {}),
      };
      if (idx === 0) openingImage = sourceImage;
      else clip.sourceImage = sourceImage;
    }
    return clip;
  });
  const chain = script.chain;
  const shared: Partial<SequenceSharedParams> = { model: chain.model };
  if (chain.width != null) shared.width = chain.width;
  if (chain.height != null) shared.height = chain.height;
  if (chain.fps != null) shared.fps = chain.fps;
  if (chain.steps != null) shared.steps = chain.steps;
  if (chain.guidance != null) shared.guidance = chain.guidance;
  if (chain.strength != null) shared.strength = chain.strength;
  if (chain.seed != null) shared.seed = String(chain.seed);
  return {
    clips,
    shared,
    enableAudio: chain.enable_audio === true,
    motionTailFrames: chain.motion_tail_frames ?? null,
    openingImage,
  };
}

/** Project the form to a `mold.chain.v1` script (TOML export, amend body). */
export function clipsToChainScript(
  shared: SequenceSharedParams,
  clips: readonly SequenceClipForm[],
  opts: {
    motionTailFrames: number;
    enableAudio: boolean;
    openingImage?: SequenceClipSourceImage | null;
  },
): ChainScript {
  return {
    schema: "mold.chain.v1",
    chain: {
      model: shared.model,
      motion_tail_frames: opts.motionTailFrames,
      width: shared.width,
      height: shared.height,
      fps: shared.fps,
      seed: shared.seed.trim() === "" ? null : shared.seed,
      steps: shared.steps,
      guidance: shared.guidance,
      strength: shared.strength,
      enable_audio: opts.enableAudio ? true : null,
    },
    stages: clips.map((clip, idx) =>
      serializeStage(clip, idx, opts.openingImage ?? null, "source_image_b64"),
    ),
  };
}

export type ClipRenderPlan = "cached" | "rerender" | "new";

export interface StageInvalidationPlan {
  /** First stage that must re-render, or null when the edit is boundary-only
   * (transition/fade-length tweaks, trailing removals) or a no-op. */
  firstDirtyStage: number | null;
  perClip: ClipRenderPlan[];
}

/** Render identity — the inputs that change a stage's PIXELS. Transition
 * type participates only through `usesCarry` (smooth carries motion in;
 * cut↔fade swaps are finalize-only under raw segments). Mirrors the
 * server's amend invalidation rule for display; the server recomputes and
 * is authoritative. */
function renderIdentity(clip: SequenceClipForm, idx: number): string {
  const usesCarry = idx > 0 && clip.transition === "smooth";
  return JSON.stringify([
    clip.prompt,
    clip.frames,
    clip.negativePrompt,
    clip.cameraControl,
    clip.sourceImage?.base64 ?? clip.sourceImage?.filename ?? null,
    usesCarry,
  ]);
}

/**
 * Compute which clips of an edit session stay cached vs re-render, given
 * the loaded job's baseline clips and how many leading stages the job has
 * actually completed.
 */
export function stageInvalidation(
  baseline: readonly SequenceClipForm[],
  current: readonly SequenceClipForm[],
  opts: { chainLevelDirty: boolean; completedStages: number },
): StageInvalidationPlan {
  let firstDirty: number | null = null;
  if (opts.chainLevelDirty) {
    firstDirty = 0;
  } else {
    const overlap = Math.min(baseline.length, current.length);
    for (let i = 0; i < overlap; i += 1) {
      const base = baseline[i];
      const cur = current[i];
      if (!base || !cur) break;
      if (renderIdentity(base, i) !== renderIdentity(cur, i)) {
        firstDirty = i;
        break;
      }
    }
    if (firstDirty === null && current.length > baseline.length) {
      firstDirty = baseline.length;
    }
  }

  const cachedCeiling = Math.min(
    firstDirty ?? current.length,
    Math.max(0, opts.completedStages),
  );
  const perClip = current.map((_, i): ClipRenderPlan => {
    if (i >= baseline.length) return "new";
    return i < cachedCeiling ? "cached" : "rerender";
  });
  // A clip past a dirty point that is ALSO past the baseline is "new"
  // (renders either way); the map above already handles it.
  const anyRender = perClip.some((p) => p !== "cached");
  return {
    firstDirtyStage: anyRender ? firstDirty : null,
    perClip,
  };
}
