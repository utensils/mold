/**
 * Per-family generation capability matrix.
 *
 * Shared family policy lives in `@studio/lib/generationCapabilities`. Two
 * desktop additions:
 *   - `supportsImg2img` — whether the SourceImageWell should render at all.
 *     True for every non-video image family AND for the video families whose
 *     engine reads a still source image, which the shared kit names via
 *     `isImageConditionedVideoFamily`. Plain `ltx-video` stays false — that
 *     engine has no img2vid path and would silently ignore the image.
 *   - `pruneRequestForFamily` — strips request fields the target family does
 *     not support, applied on model change so a leftover value never ships.
 *
 * Keep the shared LoRA-capable list in sync with
 * `mold-tui/src/model_info.rs::capabilities_for_family` and the server-side
 * gate in `mold-core/src/validation.rs`.
 */
import {
  baseGenerationCapabilities,
  isAdvancedVideoFamily,
  isFlux2DevModel,
  isImageConditionedVideoFamily,
  isQwenImageEditFamily,
  MAX_LORA_STACK,
  type BaseGenerationCapabilities,
} from "@studio/lib/generationCapabilities";
import type { GenerateRequest, OutputFormat, Scheduler } from "./api/types";

export type { SourceImageMode } from "@studio/lib/generationCapabilities";
export { isFlux2DevModel, isQwenImageEditFamily, MAX_LORA_STACK };

export interface GenerationCapabilities extends Omit<
  BaseGenerationCapabilities,
  "schedulerOptions"
> {
  schedulerOptions: Scheduler[];
  supportsImg2img: boolean;
  /** LTX-2 only — the pipeline/keyframe/upscale/retake surface. `ltx-video`
   * is a plain video family and does NOT get these. */
  supportsAdvancedVideo: boolean;
}

export function generationCapabilitiesForFamily(
  family: string,
  model = "",
  pipeline?: string | null,
  advertisedGuidance?: Parameters<typeof baseGenerationCapabilities>[3],
): GenerationCapabilities {
  const shared = baseGenerationCapabilities(family, model, pipeline, advertisedGuidance);
  const supportsAdvancedVideo = isAdvancedVideoFamily(family);
  return {
    ...shared,
    // Image conditioning is its own capability, not a consequence of having
    // the advanced-video panel. Deriving it from `supportsAdvancedVideo` was
    // correct only while LTX-2 was the sole image-conditioned video family:
    // Wan is video, is not advanced-video, and reads a source image — and
    // `wan22-i2v-a14b` cannot generate without one.
    supportsImg2img:
      !shared.supportsVideo || isImageConditionedVideoFamily(family),
    supportsMask: shared.supportsMask && !shared.supportsVideo,
    supportsAdvancedVideo,
  };
}

/** LTX-2 advanced video gate: pipeline mode, keyframes, spatial/temporal
 * upscale, retake range, and source video. `ltx-video` returns false. */
export function supportsAdvancedVideo(family: string): boolean {
  return isAdvancedVideoFamily(family);
}

export function schedulerOptionsForFamily(family: string): Scheduler[] {
  return generationCapabilitiesForFamily(family).schedulerOptions.slice();
}

export function isVideoFamily(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsVideo;
}

/** Output-format options for a family, most-preferred first (the UI default). */
export function outputFormatsForFamily(family: string): OutputFormat[] {
  // `wav` is deliberately absent: it is valid only for LTX-2's audio-only
  // `t2a` pipeline, which sets the format itself. Offering it as a free choice
  // would let a video request pick a container the server rejects.
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
export function pruneRequestForFamily(
  req: GenerateRequest,
  family: string,
  model = "",
): GenerateRequest {
  const caps = generationCapabilitiesForFamily(family, model);
  const next: GenerateRequest = { ...req };

  if (!caps.supportsNegativePrompt) delete next.negative_prompt;
  if (!caps.supportsScheduler) delete next.scheduler;
  if (!caps.supportsCfgPlus) delete next.cfg_plus;

  if (
    caps.forcesBatchSizeOne ||
    (caps.sourceImageMode === "references" && (next.edit_images?.length ?? 0) > 0)
  ) {
    next.batch_size = 1;
  }

  // qwen-edit requests carry `edit_images` (ordered: target first, then
  // references) and NEVER `source_image`/`strength`; every other family is the
  // exact inverse. The sanitizer used to strip the image entirely for
  // qwen-edit — keep `edit_images` intact there (P7 regression flip).
  if (!caps.supportsImg2img || caps.sourceImageMode !== "single") {
    delete next.source_image;
    delete next.strength;
  }
  if (!caps.supportsImg2img || caps.sourceImageMode === "single") {
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
    delete next.guidance_overrides;
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
