import type { Scheduler } from "../types";
import { LORA_CAPABLE_FAMILIES } from "../types";

export type SourceImageMode = "single" | "qwen-edit" | "references";

export interface GenerationCapabilities {
  supportsNegativePrompt: boolean;
  supportsScheduler: boolean;
  schedulerOptions: Scheduler[];
  supportsCfgPlus: boolean;
  supportsVideo: boolean;
  supportsAudio: boolean;
  supportsLora: boolean;
  supportsControlNet: boolean;
  sourceImageMode: SourceImageMode;
  supportsMask: boolean;
  forcesBatchSizeOne: boolean;
}

const SCHEDULER_OPTIONS: Scheduler[] = [
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
const CONTROLNET_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5"]);

export function generationCapabilitiesForFamily(
  family: string,
  model = "",
): GenerationCapabilities {
  const normalized = family.trim().toLowerCase();
  const qwenEdit = isQwenImageEditFamily(normalized);
  const referenceEdit = qwenEdit || isFlux2DevModel(model);
  const schedulerOptions = SCHEDULER_FAMILIES.has(normalized)
    ? SCHEDULER_OPTIONS
    : [];
  return {
    supportsNegativePrompt: !NO_NEGATIVE_PROMPT_FAMILIES.has(normalized),
    supportsScheduler: schedulerOptions.length > 0,
    schedulerOptions,
    supportsCfgPlus: CFG_PLUS_FAMILIES.has(normalized),
    supportsVideo: VIDEO_FAMILIES.has(normalized),
    supportsAudio: AUDIO_FAMILIES.has(normalized),
    supportsLora:
      !isFlux2DevModel(model) &&
      (LORA_CAPABLE_FAMILIES as readonly string[]).includes(normalized),
    supportsControlNet: CONTROLNET_FAMILIES.has(normalized),
    sourceImageMode: isFlux2DevModel(model)
      ? "references"
      : qwenEdit
        ? "qwen-edit"
        : "single",
    supportsMask: !referenceEdit,
    // Qwen edit always has a target image. FLUX.2 Dev only requires one
    // output when references are actually attached; serializers and controls
    // apply that request-sensitive lock.
    forcesBatchSizeOne: qwenEdit,
  };
}

export function schedulerOptionsForFamily(family: string): Scheduler[] {
  return generationCapabilitiesForFamily(family).schedulerOptions.slice();
}

export function isVideoFamily(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsVideo;
}

export function supportsNegativePrompt(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsNegativePrompt;
}

export function supportsScheduler(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsScheduler;
}

export function supportsLora(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsLora;
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

export function isFlux2DevModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized.includes("flux2-dev") || normalized.includes("flux.2-dev");
}
