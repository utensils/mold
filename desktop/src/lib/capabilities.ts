/**
 * Per-family generation capability matrix.
 *
 * Shared family policy lives in `@studio/lib/generationCapabilities`. Two
 * desktop additions:
 *   - `supportsImg2img` — whether the SourceImageWell should render at all.
 *     An alias for the shared `supportsSourceImage`: the selected model's own
 *     advertised contract when the server has one, the family answer
 *     otherwise. Plain `ltx-video` stays false — that engine has no img2vid
 *     path and would silently ignore the image.
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
  isMinimaxH3Family,
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
  advertisedSourceImage?: string | null,
  advertisedRecipe?: Parameters<typeof baseGenerationCapabilities>[5],
): GenerationCapabilities {
  const shared = baseGenerationCapabilities(
    family,
    model,
    pipeline,
    advertisedGuidance,
    advertisedSourceImage,
    advertisedRecipe,
  );
  const supportsAdvancedVideo = isAdvancedVideoFamily(family);
  return {
    ...shared,
    // Image conditioning is its own capability, not a consequence of having
    // the advanced-video panel, and since #772 it is per model rather than
    // per family: the three wan checkpoints split three ways and only the
    // server can tell them apart. The family answer behind
    // `supportsSourceImage` is what an older server still gets.
    supportsImg2img: shared.supportsSourceImage,
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
export function outputFormatsForFamily(
  family: string,
  recipe?: Parameters<typeof baseGenerationCapabilities>[5],
): OutputFormat[] {
  if (recipe) {
    return baseGenerationCapabilities(family, "", null, null, null, recipe)
      .outputFormats as OutputFormat[];
  }
  // `wav` is deliberately absent: it is valid only for LTX-2's audio-only
  // `t2a` pipeline, which sets the format itself. Offering it as a free choice
  // would let a video request pick a container the server rejects.
  if (isMinimaxH3Family(family)) return ["mp4"];
  return isVideoFamily(family) ? ["mp4", "gif", "apng", "webp"] : ["png", "jpeg", "webp"];
}

export function defaultOutputFormat(
  family: string,
  recipe?: Parameters<typeof baseGenerationCapabilities>[5],
): OutputFormat {
  return outputFormatsForFamily(family, recipe)[0]!;
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
  advertisedSourceImage?: string | null,
): GenerateRequest {
  const caps = generationCapabilitiesForFamily(family, model, null, null, advertisedSourceImage);
  const next: GenerateRequest = { ...req };

  // MiniMax H3 has its own final serializer in `minimaxH3Authoring`; do not
  // project its first/last boundaries or ordered references through the
  // legacy source/edit inverse below.
  if (isMinimaxH3Family(family)) {
    if (caps.sourceImageMode === "ordered-references") {
      delete next.source_image;
      delete next.source_image_name;
      delete next.edit_images;
      delete next.keyframes;
    } else {
      delete next.edit_images;
      delete next.references;
    }
    delete next.negative_prompt;
    delete next.scheduler;
    delete next.cfg_plus;
    delete next.sample_shift;
    delete next.distill_strength_high;
    delete next.distill_strength_low;
    delete next.mask_image;
    delete next.control_image;
    delete next.control_model;
    delete next.control_scale;
    delete next.loras;
    delete next.lora;
    delete next.upscale_model;
    next.batch_size = 1;
    next.guidance = 0;
    next.strength = 1;
    next.output_format = "mp4";
    return next;
  }

  if (!caps.supportsNegativePrompt) delete next.negative_prompt;
  // The UNet schedulers and wan's sample solvers share the field but are
  // disjoint on the server, so a solver that survived a family change is
  // rejected, not ignored.
  if (!caps.supportsScheduler || !caps.schedulerOptions.includes(next.scheduler ?? "default")) {
    delete next.scheduler;
  }
  if (!caps.supportsCfgPlus) delete next.cfg_plus;
  if (!caps.wanRecipe.supported) {
    delete next.sample_shift;
  }
  if (!caps.wanRecipe.supportsDistillStrength) {
    delete next.distill_strength_high;
    delete next.distill_strength_low;
  }

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
  // Wan reaches the keyframes field too — its first/last-frame layout rides
  // the LTX-2 contract — so the keyframe strip is its own question, not part
  // of the advanced-video panel's.
  if (!caps.supportsAdvancedVideo && !caps.supportsEndFrame) {
    delete next.keyframes;
  }
  if (!caps.supportsAdvancedVideo) {
    delete next.audio_file;
    delete next.source_video;
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
