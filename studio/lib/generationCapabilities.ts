export type GenerationScheduler =
  "default" | "ddim" | "euler-ancestral" | "unipc";

export type SourceImageMode = "single" | "qwen-edit" | "references";

export interface BaseGenerationCapabilities {
  supportsNegativePrompt: boolean;
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
]);

const SCHEDULER_FAMILIES = new Set([
  "sd15",
  "sd1.5",
  "stable-diffusion-1.5",
  "sdxl",
]);

const CFG_PLUS_FAMILIES = new Set(["sd3", "sd3.5"]);
const VIDEO_FAMILIES = new Set(["ltx-video", "ltx2", "ltx-2"]);
const AUDIO_FAMILIES = new Set(["ltx2", "ltx-2"]);
const ADVANCED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2"]);
const CONTROLNET_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5"]);

export const MAX_LORA_STACK = 4;

export const LORA_CAPABLE_FAMILIES = [
  "flux",
  "flux2",
  "ltx2",
  "sd15",
  "sd3",
  "sdxl",
  "qwen-image",
  "qwen-image-edit",
  "z-image",
] as const;

export function baseGenerationCapabilities(
  family: string,
  model = "",
): BaseGenerationCapabilities {
  const normalized = family.trim().toLowerCase();
  const qwenEdit = isQwenImageEditFamily(normalized);
  const flux2Dev = isFlux2DevModel(model);
  const schedulerOptions = SCHEDULER_FAMILIES.has(normalized)
    ? SCHEDULER_OPTIONS.slice()
    : [];
  return {
    supportsNegativePrompt: !NO_NEGATIVE_PROMPT_FAMILIES.has(normalized),
    supportsScheduler: schedulerOptions.length > 0,
    schedulerOptions,
    supportsCfgPlus: CFG_PLUS_FAMILIES.has(normalized),
    supportsVideo: VIDEO_FAMILIES.has(normalized),
    supportsAudio: AUDIO_FAMILIES.has(normalized),
    supportsLora:
      !flux2Dev &&
      (LORA_CAPABLE_FAMILIES as readonly string[]).includes(normalized),
    supportsControlNet: CONTROLNET_FAMILIES.has(normalized),
    sourceImageMode: flux2Dev
      ? "references"
      : qwenEdit
        ? "qwen-edit"
        : "single",
    supportsMask: !qwenEdit && !flux2Dev,
    forcesBatchSizeOne: qwenEdit,
  };
}

export function isAdvancedVideoFamily(family: string): boolean {
  return ADVANCED_VIDEO_FAMILIES.has(family.trim().toLowerCase());
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

export function isFlux2DevModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized.includes("flux2-dev") || normalized.includes("flux.2-dev");
}
