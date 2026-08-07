import {
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
} from "./minimaxH3Authoring";

export { isMinimaxH3Family } from "./minimaxH3Authoring";

export type GenerationScheduler =
  "default" | "ddim" | "euler-ancestral" | "unipc";

export type SourceImageMode =
  | "single"
  | "qwen-edit"
  | "references"
  | "h3-boundaries"
  | "ordered-references";

export interface BaseGenerationCapabilities {
  supportsNegativePrompt: boolean;
  guidanceAdjustable: boolean;
  fixedGuidance: number | null;
  supportsScheduler: boolean;
  schedulerOptions: GenerationScheduler[];
  supportsCfgPlus: boolean;
  supportsVideo: boolean;
  supportsAudio: boolean;
  supportsLora: boolean;
  supportsControlNet: boolean;
  sourceImageMode: SourceImageMode;
  supportsMask: boolean;
  forcesBatchSizeOne: boolean;
}

export interface AdvertisedGuidanceCapabilities {
  adjustable: boolean;
  supports_negative_prompt: boolean;
  fixed_scale?: number | null;
}

const SCHEDULER_OPTIONS: GenerationScheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "unipc",
];

const NO_NEGATIVE_PROMPT_FAMILIES = new Set([
  "flux",
  "flux2",
  "flux.2",
  "flux-2",
  "z-image",
  "qwen-image",
  "qwen_image",
  "qwen-image-edit",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);

const SCHEDULER_FAMILIES = new Set([
  "sd15",
  "sd1.5",
  "stable-diffusion-1.5",
  "sdxl",
]);

const CFG_PLUS_FAMILIES = new Set(["sd3", "sd3.5"]);
const VIDEO_FAMILIES = new Set([
  "ltx-video",
  "ltx2",
  "ltx-2",
  "wan",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);
/**
 * Generated audio is LTX-2's alone. Wan is a video family with no audio
 * branch at all — its checkpoints ship no audio VAE and no vocoder — so it
 * belongs in `VIDEO_FAMILIES` and emphatically not here. Listing it would
 * surface an audio toggle whose request the server rejects before denoising.
 */
const AUDIO_FAMILIES = new Set([
  "ltx2",
  "ltx-2",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);
/**
 * The families with a bespoke advanced-controls panel (LTX-2's guidance
 * overrides, STG, spatial/temporal rungs). Wan needs only the generic video
 * controls — steps, guidance, negative — so it stays out.
 */
const ADVANCED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2"]);
/**
 * Video families whose engine accepts a still image as conditioning.
 *
 * This is deliberately its own set rather than a synonym for
 * `ADVANCED_VIDEO_FAMILIES`, even though the two held the same members until
 * now. Desktop derived the source-image well from the advanced-video flag
 * (`!supportsVideo || supportsAdvancedVideo`), which was indistinguishable
 * from correct while LTX-2 was the only image-conditioned video family. Wan is
 * the first that is image-conditioned *without* the advanced panel, and under
 * that derivation its well vanished — including for `wan22-i2v-a14b`, which
 * cannot generate at all without a source image.
 *
 * Plain `ltx-video` stays out: its engine has no image-to-video path and would
 * silently ignore an attached image.
 */
const IMAGE_CONDITIONED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2", "wan"]);
const CONTROLNET_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5"]);

export const MAX_LORA_STACK = 4;

/**
 * Families whose engines can merge an adapter. Mirrors the `workflows.lora`
 * flag each family declares in `crates/mold-inference/src/batch.rs`; a family
 * missing here has its LoRA control hidden on every surface even though the
 * server would accept the request.
 *
 * Wan reaches its adapters two different ways depending on the checkpoint's
 * container — merged into the weights for bf16/fp8 safetensors, applied as a
 * parallel low-rank branch for GGUF — but that is invisible from here: the
 * request shape is identical either way.
 */
export const LORA_CAPABLE_FAMILIES = [
  "flux",
  "flux2",
  "ltx2",
  "sd15",
  "sd3",
  "sdxl",
  "qwen-image",
  "qwen-image-edit",
  "wan",
  "z-image",
] as const;

export function baseGenerationCapabilities(
  family: string,
  model = "",
  pipeline?: string | null,
  advertisedGuidance?: AdvertisedGuidanceCapabilities | null,
): BaseGenerationCapabilities {
  const normalized = family.trim().toLowerCase();
  const qwenEdit = isQwenImageEditFamily(normalized);
  const flux2Dev = isFlux2DevModel(model);
  const h3 = isMinimaxH3Identity(normalized, model);
  const h3Ref2va = h3 && isMinimaxH3Ref2vaModel(model);
  const schedulerOptions = SCHEDULER_FAMILIES.has(normalized)
    ? SCHEDULER_OPTIONS.slice()
    : [];
  const normalizedModel = model.trim().toLowerCase();
  const ltx =
    normalized === "ltx-video" ||
    normalized === "ltx2" ||
    normalized === "ltx-2";
  const fixedLtxPipeline = [
    "distilled",
    "ic-lora",
    "retake",
    "lip-dub",
  ].includes(pipeline ?? "");
  const inferredFixedGuidance =
    ltx &&
    (fixedLtxPipeline || (!pipeline && normalizedModel.includes("distilled")));
  const advertisedDefault = !pipeline ? advertisedGuidance : null;
  const fixedGuidance = h3
    ? true
    : advertisedDefault
      ? !advertisedDefault.adjustable
      : inferredFixedGuidance;
  return {
    supportsNegativePrompt:
      advertisedDefault?.supports_negative_prompt ??
      (!NO_NEGATIVE_PROMPT_FAMILIES.has(normalized) && !fixedGuidance),
    guidanceAdjustable: !fixedGuidance,
    fixedGuidance: fixedGuidance
      ? h3
        ? 0
        : (advertisedDefault?.fixed_scale ?? 1)
      : null,
    supportsScheduler: schedulerOptions.length > 0,
    schedulerOptions,
    supportsCfgPlus: CFG_PLUS_FAMILIES.has(normalized),
    supportsVideo: h3 || VIDEO_FAMILIES.has(normalized),
    supportsAudio: h3 || AUDIO_FAMILIES.has(normalized),
    supportsLora:
      !flux2Dev &&
      (LORA_CAPABLE_FAMILIES as readonly string[]).includes(normalized),
    supportsControlNet: CONTROLNET_FAMILIES.has(normalized),
    sourceImageMode: h3Ref2va
      ? "ordered-references"
      : h3
        ? "h3-boundaries"
        : flux2Dev
          ? "references"
          : qwenEdit
            ? "qwen-edit"
            : "single",
    supportsMask: !h3 && !qwenEdit && !flux2Dev,
    forcesBatchSizeOne: h3 || qwenEdit,
  };
}

export function isAdvancedVideoFamily(family: string): boolean {
  return ADVANCED_VIDEO_FAMILIES.has(family.trim().toLowerCase());
}

/**
 * Whether a video family's engine reads a still source image.
 *
 * Surfaces gate the source-image control on this, never on
 * `isAdvancedVideoFamily` — the two answer different questions and only
 * coincided while LTX-2 was the sole image-conditioned video family.
 */
export function isImageConditionedVideoFamily(family: string): boolean {
  return IMAGE_CONDITIONED_VIDEO_FAMILIES.has(family.trim().toLowerCase());
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

export function isFlux2DevModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized.includes("flux2-dev") || normalized.includes("flux.2-dev");
}

export function isMinimaxH3Ref2vaModel(model: string): boolean {
  return minimaxH3TaskForModel(model) === "ref2va";
}
