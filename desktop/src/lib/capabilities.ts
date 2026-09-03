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
  isMinimaxH3Family,
  isQwenImageEditFamily,
  MAX_LORA_STACK,
  sourceImageModeForReferences,
  type BaseGenerationCapabilities,
  type ReferenceImagesCapabilities,
} from "@studio/lib/generationCapabilities";
import { conditioningForRequest } from "@studio/lib/sourceMediaPlan";
import {
  recipeIsCanvasless,
  type GenerationRecipeProfile,
  type MeshCapabilitiesProfile,
  type PromptRequirement,
} from "@studio/lib/generationProfile";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import { coerceOutputFormatForRecipe, type OutputFormatRecipe } from "@studio/lib/outputFormat";
import type { GenerateRequest, OutputFormat, Scheduler } from "./api/types";

export type { SourceImageMode } from "@studio/lib/generationCapabilities";
export { isQwenImageEditFamily, MAX_LORA_STACK };

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

/**
 * The recipe-derived facts the request builder needs after the model row is
 * out of scope. Snapshotted onto the form when a model or pipeline is
 * applied, exactly like `guidanceCapabilities` and `sourceImageCapability`,
 * so `buildRequest(form)` and `pruneRequestForFamily` read the advertised
 * contract rather than the family heuristic. `null` is an older host that
 * advertises no recipe, where the legacy rules answer.
 */
export interface RecipeCapabilitiesSnapshot {
  outputFormats: OutputFormat[];
  defaultOutputFormat: OutputFormat;
  promptMode: PromptRequirement;
  supportsStrength: boolean;
  /** The recipe renders no pixel canvas (a 3-D mesh). */
  canvasless: boolean;
  /** The recipe's 3-D controls, or `null` when `mesh` is refused. */
  mesh: MeshCapabilitiesProfile | null;
  /**
   * The recipe's ordered-reference contract (`GenerateRequest.edit_images`),
   * or `null` where it takes none.
   *
   * Without this the request builders — which take only the form — fell
   * through to `legacyReferenceImages`, whose answer for FLUX.2 [klein] is
   * deliberately `null` (an older host has no Klein reference engine). The
   * References strip rendered from the recipe and the wire never carried what
   * the user put in it.
   */
  referenceImages: ReferenceImagesCapabilities | null;
}

export function recipeCapabilitiesSnapshot(
  recipe: GenerationRecipeProfile | null | undefined,
  family = "",
  model = "",
  pipeline: string | null = null,
  advertisedSourceImage?: string | null,
): RecipeCapabilitiesSnapshot | null {
  if (!recipe) return null;
  const caps = baseGenerationCapabilities(
    family,
    model,
    pipeline,
    null,
    advertisedSourceImage,
    recipe,
  );
  return {
    outputFormats: caps.outputFormats as OutputFormat[],
    defaultOutputFormat: caps.defaultOutputFormat as OutputFormat,
    promptMode: caps.promptMode,
    supportsStrength: caps.supportsStrength,
    canvasless: recipeIsCanvasless(recipe),
    mesh: caps.mesh ?? null,
    referenceImages: caps.referenceImages,
  };
}

/**
 * The capabilities a FORM-ONLY caller reads — the request builders, the
 * request pruner, and the submit-time source-fit preprocessors, none of which
 * still have the model row in hand.
 *
 * It is `generationCapabilitiesForFamily` plus the recipe snapshot the form
 * already carries, so those callers resolve the reference contract (and the
 * layout it projects) from exactly what the wells rendered from. Without the
 * snapshot — an older host that advertises no recipe — the answer is
 * byte-identical to the family derivation, which is the whole point of the
 * `null`.
 *
 * H3's two tasks own their own layouts and their own serializer, so their
 * mode is never overridden here.
 */
export function generationCapabilitiesForForm(
  family: string,
  model = "",
  pipeline: string | null = null,
  advertisedGuidance?: Parameters<typeof baseGenerationCapabilities>[3],
  advertisedSourceImage?: string | null,
  snapshot?: RecipeCapabilitiesSnapshot | null,
): GenerationCapabilities {
  const caps = generationCapabilitiesForFamily(
    family,
    model,
    pipeline,
    advertisedGuidance,
    advertisedSourceImage,
  );
  if (!snapshot || isMinimaxH3Family(family)) return caps;
  return {
    ...caps,
    referenceImages: snapshot.referenceImages,
    referenceImagesReason: snapshot.referenceImages ? null : caps.referenceImagesReason,
    sourceImageMode: sourceImageModeForReferences(snapshot.referenceImages),
  };
}

/** The snapshot as the shared format coercion reads a recipe. */
export function asOutputFormatRecipe(
  snapshot: RecipeCapabilitiesSnapshot | null | undefined,
): OutputFormatRecipe | null {
  if (!snapshot) return null;
  return {
    capabilities: {
      mesh: snapshot.mesh,
      output: {
        formats: snapshot.outputFormats,
        default_format: snapshot.defaultOutputFormat,
      },
    },
  };
}

/**
 * The format a request for this family/recipe may carry: the shared rule
 * (`@studio/lib/outputFormat`) applied against the snapshot when one is
 * known and the family list otherwise.
 */
export function coerceFormOutputFormat(
  format: OutputFormat | null | undefined,
  family: string,
  snapshot: RecipeCapabilitiesSnapshot | null | undefined,
): OutputFormat | undefined {
  return coerceOutputFormatForRecipe(
    asOutputFormatRecipe(snapshot),
    family,
    format,
    outputFormatsForFamily(family),
  );
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
  // A mesh family stores binary glTF and nothing else; OBJ/STL/PLY are
  // gallery exports, never generation targets.
  if (isMeshFamily(family)) return ["glb"];
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
  recipe?: RecipeCapabilitiesSnapshot | null,
): GenerateRequest {
  const caps = generationCapabilitiesForForm(
    family,
    model,
    null,
    null,
    advertisedSourceImage,
    recipe,
  );
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

  // qwen-edit requests carry `edit_images` (ordered: target first, then
  // references) and NEVER `source_image`/`strength`; a single-source family is
  // the exact inverse; an EXCLUSIVE recipe (Klein) is whichever the request
  // itself carries, so the pruner asks the same shared question the request
  // builder asked rather than re-deriving one from the mode. The sanitizer
  // used to strip the image entirely for qwen-edit — keep `edit_images`
  // intact there (P7 regression flip).
  const conditioning = conditioningForRequest(caps.sourceImageMode, {
    hasSource: Boolean(next.source_image),
    referenceCount: next.edit_images?.length ?? 0,
    // A request is already resolved: only one of the two can be on the wire,
    // and a stale pair prefers the references the edit families ship.
    lastWrite: (next.edit_images?.length ?? 0) > 0 ? "references" : null,
  });

  if (caps.forcesBatchSizeOne || conditioning === "references") {
    next.batch_size = 1;
  }

  // An empty single-source request keeps its strength (it is a form value on
  // an ordinary img2img family, not conditioning); only a request whose
  // conditioning is REFERENCES loses the source pair.
  if (
    !caps.supportsImg2img ||
    (conditioning !== "source" &&
      caps.sourceImageMode !== "single" &&
      !(caps.sourceImageMode === "single-or-references" && conditioning === "none"))
  ) {
    delete next.source_image;
    delete next.strength;
  }
  if (!caps.supportsImg2img || conditioning !== "references") {
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

  // Keep the output format valid for the recipe (png stays out of video, a
  // mesh recipe is pinned to glb, a glb never rides a raster recipe).
  const format = coerceFormOutputFormat(next.output_format, family, recipe);
  if (format === undefined) delete next.output_format;
  else next.output_format = format;

  // The 3-D controls are refused at admission on a recipe with no mesh block,
  // and a canvasless recipe reads neither strength nor a repaint mask; a
  // legacy host without a recipe gets the family rule.
  const meshRecipe = recipe ? recipe.mesh !== null : isMeshFamily(family);
  if (!meshRecipe) delete next.mesh;
  const canvasless = recipe ? recipe.canvasless : isMeshFamily(family);
  if (canvasless) {
    delete next.strength;
    delete next.mask_image;
    delete next.source_fit;
  }
  if (recipe && !recipe.supportsStrength) delete next.strength;

  return next;
}
