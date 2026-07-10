/**
 * Per-family generation capability matrix.
 *
 * Logic ported verbatim from the web SPA's `generateCapabilities.ts`
 * (families, scheduler options, CFG++ gate, negative-prompt gate, LoRA gate,
 * ControlNet gate, qwen-edit mode, batch-size lock). Two desktop additions:
 *   - `supportsImg2img` — whether the SourceImageWell should render at all.
 *     True for every non-video image family; video families (ltx) condition on
 *     source video / keyframes instead, which land in M5.
 *   - `pruneRequestForFamily` — strips request fields the target family does
 *     not support, applied on model change so a leftover value never ships.
 *
 * Keep the LoRA-capable list in sync with the web `LORA_CAPABLE_FAMILIES`,
 * `mold-tui/src/model_info.rs::capabilities_for_family`, and the server-side
 * gate in `mold-core/src/validation.rs`.
 */
import type { GenerateRequest, OutputFormat, Scheduler } from "./api/types";

export type SourceImageMode = "single" | "qwen-edit";

export interface GenerationCapabilities {
  supportsNegativePrompt: boolean;
  supportsScheduler: boolean;
  schedulerOptions: Scheduler[];
  supportsCfgPlus: boolean;
  supportsVideo: boolean;
  supportsAudio: boolean;
  supportsLora: boolean;
  supportsControlNet: boolean;
  supportsImg2img: boolean;
  sourceImageMode: SourceImageMode;
  supportsMask: boolean;
  forcesBatchSizeOne: boolean;
  /** LTX-2 only — the pipeline/keyframe/upscale/retake surface. `ltx-video`
   * is a plain video family and does NOT get these. */
  supportsAdvancedVideo: boolean;
}

const SCHEDULER_OPTIONS: Scheduler[] = ["default", "ddim", "euler-ancestral", "unipc"];

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

const SCHEDULER_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5", "sdxl"]);

const CFG_PLUS_FAMILIES = new Set(["sd3", "sd3.5"]);
const VIDEO_FAMILIES = new Set(["ltx-video", "ltx2", "ltx-2"]);
const AUDIO_FAMILIES = new Set(["ltx2", "ltx-2"]);
/** LTX-2 only — the advanced pipeline/keyframe/upscale/retake surface. */
const ADVANCED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2"]);
const CONTROLNET_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5"]);

/** Mirrors web `LORA_CAPABLE_FAMILIES`. */
const LORA_CAPABLE_FAMILIES = new Set([
  "flux",
  "flux2",
  "ltx2",
  "sd15",
  "sd3",
  "sdxl",
  "qwen-image",
  "qwen-image-edit",
  "z-image",
]);

/** Soft ceiling on stacked LoRAs — matches web `MAX_LORA_STACK`. */
export const MAX_LORA_STACK = 4;

export function generationCapabilitiesForFamily(family: string): GenerationCapabilities {
  const normalized = family.trim().toLowerCase();
  const qwenEdit = isQwenImageEditFamily(normalized);
  const supportsVideo = VIDEO_FAMILIES.has(normalized);
  const schedulerOptions = SCHEDULER_FAMILIES.has(normalized) ? SCHEDULER_OPTIONS.slice() : [];
  return {
    supportsNegativePrompt: !NO_NEGATIVE_PROMPT_FAMILIES.has(normalized),
    supportsScheduler: schedulerOptions.length > 0,
    schedulerOptions,
    supportsCfgPlus: CFG_PLUS_FAMILIES.has(normalized),
    supportsVideo,
    supportsAudio: AUDIO_FAMILIES.has(normalized),
    supportsLora: LORA_CAPABLE_FAMILIES.has(normalized),
    supportsControlNet: CONTROLNET_FAMILIES.has(normalized),
    // The SourceImageWell handles still-image img2img only; video families use
    // source-video / keyframe conditioning (M5), so gate the well off there.
    supportsImg2img: !supportsVideo,
    sourceImageMode: qwenEdit ? "qwen-edit" : "single",
    supportsMask: !qwenEdit && !supportsVideo,
    forcesBatchSizeOne: qwenEdit,
    supportsAdvancedVideo: ADVANCED_VIDEO_FAMILIES.has(normalized),
  };
}

/** LTX-2 advanced video gate: pipeline mode, keyframes, spatial/temporal
 * upscale, retake range, and source video. `ltx-video` returns false. */
export function supportsAdvancedVideo(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsAdvancedVideo;
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

export function supportsLora(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsLora;
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

/** Output-format options for a family, most-preferred first (the UI default). */
export function outputFormatsForFamily(family: string): OutputFormat[] {
  return isVideoFamily(family) ? ["mp4", "gif", "apng", "webp"] : ["png", "jpeg", "webp"];
}

export function defaultOutputFormat(family: string): OutputFormat {
  return outputFormatsForFamily(family)[0]!;
}

/**
 * Drop request fields the target family does not support. Pure — returns a new
 * object, never mutates the input. Applied whenever the selected model (hence
 * family) changes so a value set for one family never leaks into a request for
 * another (e.g. a scheduler chosen under SDXL must not ship with FLUX).
 */
export function pruneRequestForFamily(req: GenerateRequest, family: string): GenerateRequest {
  const caps = generationCapabilitiesForFamily(family);
  const next: GenerateRequest = { ...req };

  if (!caps.supportsNegativePrompt) delete next.negative_prompt;
  if (!caps.supportsScheduler) delete next.scheduler;
  if (!caps.supportsCfgPlus) delete next.cfg_plus;

  if (caps.forcesBatchSizeOne) next.batch_size = 1;

  if (!caps.supportsImg2img || caps.sourceImageMode === "qwen-edit") {
    delete next.source_image;
    delete next.strength;
  }
  if (!caps.supportsMask) delete next.mask_image;
  if (!caps.supportsControlNet) {
    delete next.control_image;
    delete next.control_model;
    delete next.control_scale;
  }
  if (!caps.supportsLora) {
    delete next.loras;
    delete next.lora;
  }
  if (!caps.supportsVideo) {
    delete next.frames;
    delete next.fps;
  }
  if (!caps.supportsAudio) delete next.enable_audio;
  if (!caps.supportsAdvancedVideo) {
    delete next.audio_file;
    delete next.source_video;
    delete next.keyframes;
    delete next.pipeline;
    delete next.retake_range;
    delete next.spatial_upscale;
    delete next.temporal_upscale;
  }

  // Keep the output format valid for the family (png stays out of video, etc.).
  const formats = outputFormatsForFamily(family);
  if (next.output_format && !formats.includes(next.output_format)) {
    next.output_format = formats[0]!;
  }

  return next;
}
