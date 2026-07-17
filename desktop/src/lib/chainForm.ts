/**
 * Reactive state for the chain composer and pure builders that project it onto
 * the wire `ChainRequest` (POST body) and `ChainScript` (TOML / edit view).
 * Kept separate from `chain.ts` (pure math/TOML) so the composer's editable
 * shape lives in one place, like `generateForm.ts` for single generations.
 */
import { parse as parseToml } from "smol-toml";
import type { ChainRequest, ChainScript, ChainStage, TransitionMode } from "./api/types";
import { DEFAULT_FADE_FRAMES, DEFAULT_MOTION_TAIL_FRAMES, DEFAULT_FPS } from "./chain";

export interface ChainStageForm {
  prompt: string;
  frames: number;
  transition: TransitionMode;
  fadeFrames: number;
  negativePrompt: string;
  /** Per-stage starting image, base64 (no data-URI prefix). Null = none. */
  sourceImage: string | null;
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
  /**
   * Chain-level starting image, base64. Sugar for "seed the film from this
   * still": projected onto `stages[0].source_image` when stage 0 has no image
   * of its own (the server ignores top-level `source_image` in canonical
   * `stages` form, so it must ride on stage 0).
   */
  startImage: string | null;
  stages: ChainStageForm[];
}

export function newStage(prompt = ""): ChainStageForm {
  return {
    prompt,
    frames: 97,
    transition: "smooth",
    fadeFrames: DEFAULT_FADE_FRAMES,
    negativePrompt: "",
    sourceImage: null,
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
    startImage: null,
    stages: [newStage()],
  };
}

/** Project a form stage onto a wire `ChainStage`. Stage 0's transition is
 * always coerced to smooth (the server does this too — nothing precedes it).
 * The chain-level `startImage` rides on stage 0, unless stage 0 carries its
 * own (more specific) image. */
function stageToWire(stage: ChainStageForm, idx: number, startImage: string | null): ChainStage {
  const transition: TransitionMode = idx === 0 ? "smooth" : stage.transition;
  const wire: ChainStage = { prompt: stage.prompt.trim(), frames: stage.frames, transition };
  if (transition === "fade") wire.fade_frames = stage.fadeFrames;
  if (stage.negativePrompt.trim()) wire.negative_prompt = stage.negativePrompt.trim();
  const image = stage.sourceImage || (idx === 0 ? startImage : null);
  if (image) wire.source_image = image;
  return wire;
}

export function chainFormToRequest(form: ChainForm): ChainRequest {
  const seed = form.seed.trim() === "" ? undefined : Number(form.seed);
  const req: ChainRequest = {
    model: form.model,
    stages: form.stages.map((s, i) => stageToWire(s, i, form.startImage)),
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

function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}
function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}
function asTransition(v: unknown): TransitionMode {
  return v === "cut" || v === "fade" ? v : "smooth";
}

/**
 * Extract a stage's starting image from its TOML table. Mirrors
 * `mold_core::chain_toml::resolve_stage_source_image`: `source_image_b64`
 * (authoring form) and canonical `source_image` are both accepted, at most
 * one image field may be set, and `source_image_path` is rejected — the
 * composer has no script directory to resolve it against.
 */
function stageSourceImage(s: Record<string, unknown>, idx: number): string | null {
  const fields = ["source_image", "source_image_path", "source_image_b64"].filter(
    (k) => s[k] != null,
  );
  if (fields.length > 1) {
    throw new Error(
      `Stage ${idx + 1}: set at most one of source_image, source_image_path, source_image_b64.`,
    );
  }
  if (s.source_image_path != null) {
    throw new Error(
      `Stage ${idx + 1} uses source_image_path, which needs the script's folder to resolve. ` +
        "Inline the image as source_image_b64 to open it here.",
    );
  }
  const b64 = s.source_image_b64 ?? s.source_image;
  return typeof b64 === "string" && b64.length > 0 ? b64 : null;
}

/**
 * Parse a `mold.chain.v1` TOML document into a `ChainForm` — the inverse of
 * `chainFormToScript` + `chainToToml`. Pure and defensive: unknown fields are
 * ignored and missing ones fall back to `newChainForm` defaults. Rejects any
 * `schema` other than `mold.chain.v1`; an absent `schema` is accepted (the
 * server injects it too). Throws on malformed TOML or a missing `[chain]`.
 */
export function tomlToChainForm(toml: string): ChainForm {
  let doc: Record<string, unknown>;
  try {
    doc = parseToml(toml) as Record<string, unknown>;
  } catch (e) {
    throw new Error(`Couldn't parse the TOML: ${e instanceof Error ? e.message : String(e)}`);
  }

  const schema = doc.schema;
  if (schema != null && schema !== "mold.chain.v1") {
    throw new Error(`Unsupported chain schema "${String(schema)}" (expected mold.chain.v1).`);
  }
  const chain = doc.chain;
  if (chain == null || typeof chain !== "object") {
    throw new Error("Chain TOML is missing its [chain] table.");
  }
  const c = chain as Record<string, unknown>;
  const defaults = newChainForm();

  const rawStages = Array.isArray(doc.stage) ? (doc.stage as Record<string, unknown>[]) : [];
  const stages: ChainStageForm[] = rawStages.map((s, idx) => ({
    prompt: str(s.prompt),
    frames: num(s.frames, 97),
    transition: idx === 0 ? "smooth" : asTransition(s.transition),
    fadeFrames: num(s.fade_frames, DEFAULT_FADE_FRAMES),
    negativePrompt: str(s.negative_prompt),
    sourceImage: stageSourceImage(s, idx),
  }));

  return {
    model: str(c.model),
    width: num(c.width, defaults.width),
    height: num(c.height, defaults.height),
    fps: num(c.fps, defaults.fps),
    seed: typeof c.seed === "number" ? String(c.seed) : "",
    steps: num(c.steps, defaults.steps),
    guidance: num(c.guidance, defaults.guidance),
    strength: num(c.strength, defaults.strength),
    motionTailFrames: num(c.motion_tail_frames, DEFAULT_MOTION_TAIL_FRAMES),
    enableAudio: c.enable_audio === true,
    // TOML has no chain-level image — a start image round-trips on stage 0.
    startImage: null,
    stages: stages.length > 0 ? stages : defaults.stages,
  };
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
    stage: form.stages.map((s, i) => stageToWire(s, i, form.startImage)),
  };
}
