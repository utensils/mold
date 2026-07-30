/**
 * Per-family generation capability matrix.
 *
 * Logic ported verbatim from the web SPA's `generateCapabilities.ts`
 * (families, scheduler options, CFG++ gate, negative-prompt gate, LoRA gate,
 * ControlNet gate, qwen-edit mode, batch-size lock). Two desktop additions:
 *   - `supportsImg2img` — whether the SourceImageWell should render at all.
 *     True for every non-video image family AND for LTX-2, whose engine
 *     stages `source_image` as frame-0 conditioning (image-to-video, with
 *     `strength`). Plain `ltx-video` stays false — that engine has no
 *     img2vid path and would silently ignore the image.
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
    // LTX-2 accepts a still source_image as frame-0 conditioning (img2video);
    // ltx-video's engine has no img2vid path, so only the advanced-video
    // families get the well among video families.
    supportsImg2img: !supportsVideo || ADVANCED_VIDEO_FAMILIES.has(normalized),
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

  // qwen-edit requests carry `edit_images` (ordered: target first, then
  // references) and NEVER `source_image`/`strength`; every other family is the
  // exact inverse. The sanitizer used to strip the image entirely for
  // qwen-edit — keep `edit_images` intact there (P7 regression flip).
  if (!caps.supportsImg2img || caps.sourceImageMode === "qwen-edit") {
    delete next.source_image;
    delete next.strength;
  }
  if (!caps.supportsImg2img || caps.sourceImageMode !== "qwen-edit") {
    delete next.edit_images;
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
    delete next.ic_lora_control;
    delete next.retake_range;
    delete next.spatial_upscale;
    delete next.temporal_upscale;
    // The `camera-control:<preset>` virtual lora alias only resolves on the
    // LTX-2 engine — strip it so it never leaks into another family's stack.
    if (next.loras) {
      const kept = next.loras.filter((l) => !l.path.startsWith("camera-control:"));
      if (kept.length) next.loras = kept;
      else delete next.loras;
    }
  }

  // Keep the output format valid for the family (png stays out of video, etc.).
  const formats = outputFormatsForFamily(family);
  if (next.output_format && !formats.includes(next.output_format)) {
    next.output_format = formats[0]!;
  }

  return next;
}
