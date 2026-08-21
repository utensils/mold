import { generationCapabilitiesForFamily } from "./capabilities";
import type { GenerateForm } from "./generateForm";
import { isCameraMotionPreset } from "@studio/lib/cameraMotion";
import {
  dimensionAlignmentForFamily,
  maxAxisPixelsForModel,
  maxPixelsForModel,
  type ModelResolutionContract,
} from "@studio/lib/resolutions";
import { guidanceOverridesError } from "@studio/lib/guidanceOverrides";
import { identityValidationError } from "@studio/lib/identityConditioning";
import { isMinimaxH3Family } from "@studio/lib/minimaxH3Authoring";
import { sourceImageValidationError } from "@studio/lib/sourceImageCapability";
import { submitsExtend } from "@studio/lib/extend";
import { wanRecipeError } from "@studio/lib/wanRecipe";
import { minimaxH3TaskForModel } from "@studio/lib/minimaxH3Authoring";
import {
  effectiveGenerationRecipe,
  floatControlError,
  generationRecipeSelectionError,
  integerControlError,
  resolutionProfileFinding,
} from "@studio/lib/generationProfile";

export const MAX_INLINE_GENERATION_MEDIA_BYTES = 64 * 1024 * 1024;
// JSON base64 expands bytes by roughly 4/3 and the server body limit is 64
// MiB. Keep the whole phone conditioning payload below 45 MiB so the request
// has room for JSON/scalars without a late extractor rejection.
export const MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES = 45 * 1024 * 1024;
export const MOBILE_MEDIA_BUDGET_ERROR =
  "Combined generation media must be 45 MiB or smaller on this phone.";

export type InlineGenerationMediaField =
  | "sourceImage"
  | "identityImage"
  | "maskImage"
  | "controlImage"
  | "imageAttachments"
  | "endFrame"
  | "sourceVideo"
  | "extendVideo"
  | "keyframes"
  | "audioFile"
  | "h3FirstFrame"
  | "h3LastFrame"
  | "h3References";

export function decodedBase64Bytes(value: string | null | undefined): number {
  if (!value) return 0;
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((value.length * 3) / 4) - padding);
}

export function inlineGenerationMediaBytes(
  form: GenerateForm,
  exclude: InlineGenerationMediaField | null = null,
): number {
  let total = 0;
  if (exclude !== "sourceImage") total += decodedBase64Bytes(form.sourceImage);
  // The identity photo rides the same JSON body as every other inline input,
  // so it spends the same budget even though it is never fitted.
  if (exclude !== "identityImage") total += decodedBase64Bytes(form.identityImage?.base64);
  if (exclude !== "maskImage") total += decodedBase64Bytes(form.maskImage);
  if (exclude !== "controlImage") total += decodedBase64Bytes(form.controlImage);
  if (exclude !== "imageAttachments") {
    total += form.imageAttachments.reduce((sum, image) => sum + decodedBase64Bytes(image), 0);
  }
  if (exclude !== "endFrame" && form.endFrame) {
    // A first/last-frame render ships each still exactly once: the pair
    // travels as `keyframes` and `source_image` stays off the wire (the
    // engine refuses both together), so the opening still is already counted
    // by the `sourceImage` line above.
    total += decodedBase64Bytes(form.endFrame.base64);
  }
  if (exclude !== "sourceVideo") total += decodedBase64Bytes(form.sourceVideo?.base64);
  if (exclude !== "extendVideo") total += decodedBase64Bytes(form.extendVideo?.base64);
  if (exclude !== "audioFile") total += decodedBase64Bytes(form.audioFile?.base64);
  if (exclude !== "keyframes") {
    total += form.keyframes.reduce(
      (sum, keyframe) => sum + decodedBase64Bytes(keyframe.image.base64),
      0,
    );
  }
  const h3Task = minimaxH3TaskForModel(form.model);
  if (h3Task === "fl2va" && exclude !== "h3FirstFrame") {
    total += decodedBase64Bytes(form.h3Authoring?.firstFrame?.data);
  }
  if (h3Task === "fl2va" && exclude !== "h3LastFrame") {
    total += decodedBase64Bytes(form.h3Authoring?.lastFrame?.data);
  }
  if (h3Task === "ref2va" && exclude !== "h3References") {
    total += (form.h3Authoring?.references ?? []).reduce((sum, draft) => {
      const media = draft.reference.media;
      return sum + (media.authority === "inline" ? decodedBase64Bytes(media.data) : 0);
    }, 0);
  }
  return total;
}

export function mobileMediaBudgetValidationError(form: GenerateForm): string | null {
  return inlineGenerationMediaBytes(form) <= MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES
    ? null
    : MOBILE_MEDIA_BUDGET_ERROR;
}

export function stepsValidationError(value: number): string | null {
  return Number.isInteger(value) && value >= 1 && value <= 100
    ? null
    : "Steps must be a whole number from 1 to 100.";
}

export function profileStepsValidationError(
  value: number,
  contract?: ModelResolutionContract | null,
  pipeline?: string | null,
): string | null {
  const selectionError = generationRecipeSelectionError(contract, pipeline);
  if (selectionError) return selectionError;
  const control = effectiveGenerationRecipe(contract, pipeline)?.steps;
  if (!control) return stepsValidationError(value);
  return integerControlError("Steps", value, control);
}

export function guidanceValidationError(value: number): string | null {
  return Number.isFinite(value) && value >= 0 && value <= 100
    ? null
    : "Guidance must be from 0 to 100.";
}

export function profileGuidanceValidationError(
  value: number,
  contract?: ModelResolutionContract | null,
  pipeline?: string | null,
): string | null {
  const selectionError = generationRecipeSelectionError(contract, pipeline);
  if (selectionError) return selectionError;
  const control = effectiveGenerationRecipe(contract, pipeline)?.guidance;
  if (!control) return guidanceValidationError(value);
  return floatControlError("Guidance", value, control);
}

export function fpsValidationError(value: number): string | null {
  return Number.isInteger(value) && value >= 1 && value <= 60
    ? null
    : "FPS must be a whole number from 1 to 60.";
}

/**
 * `contract` is the selected model's advertised resolution limits. Supply it
 * wherever the model row is in scope: the ceiling and the pixel grid are
 * family-specific, and hard-coding 1.8 MP / 16 px rejected LTX-2 shapes the
 * server accepts while admitting off-grid ones it does not.
 */
export function resolutionValidationError(
  width: number,
  height: number,
  contract?: ModelResolutionContract | null,
  pipeline?: string | null,
): string | null {
  // The server is the authority on whether a size renders: every profile and
  // legacy limit is an advisory (see `resolutionValidationWarning`). Only
  // input that cannot form a coherent request blocks here.
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    return "Width and height must be whole numbers.";
  }
  const selectionError = generationRecipeSelectionError(contract, pipeline);
  if (selectionError) return selectionError;
  const recipe = effectiveGenerationRecipe(contract, pipeline);
  const finding = resolutionProfileFinding(width, height, recipe?.resolution);
  return finding?.level === "block" ? finding.message : null;
}

/**
 * `availablePresetIds` is what the selected host advertises for this
 * checkpoint. Supply it wherever the fetched list is in scope; the submit gate
 * has no list and omits it, letting the host's own 422 be the authority.
 *
 * This used to gate on `!form.model.includes("ltx-2.3")`, which is unsound in
 * both directions — an opaque `cv:` / `hf:` ID for an LTX-2.3 checkpoint
 * contains no architecture, so the check passed a preset that cannot resolve
 * and rejected one on a 19B install reached the same way.
 */
/** Advisory counterpart to {@link resolutionValidationError}: every size
 * constraint — profile minimums, alignment, span, pixel budget, bucket
 * membership, and the legacy no-recipe family limits — reports here. Never a
 * blocker: the request submits and the server's own refusal is authoritative. */
export function resolutionValidationWarning(
  width: number,
  height: number,
  contract?: ModelResolutionContract | null,
  pipeline?: string | null,
): string | null {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    return null; // already blocked as malformed input
  }
  const recipe = effectiveGenerationRecipe(contract, pipeline);
  if (recipe) {
    const finding = resolutionProfileFinding(width, height, recipe.resolution);
    return finding?.level === "warn" ? finding.message : null;
  }
  // Legacy hosts without a generation profile: the family constants that used
  // to hard-block now advise.
  if (width < 64 || height < 64) {
    return "This model expects at least 64 × 64 pixels — the server may reject this size.";
  }
  const alignment = contract?.dimension_alignment ?? dimensionAlignmentForFamily(contract?.family);
  if (width % alignment !== 0 || height % alignment !== 0) {
    return `This model expects multiples of ${alignment} — the server may reject this size.`;
  }
  const axisLimit = maxAxisPixelsForModel(contract, pipeline);
  if (axisLimit && Math.max(width, height) > axisLimit) {
    return `${width} × ${height} exceeds the ${axisLimit}px span this model can hold — the server may reject it.`;
  }
  const maxPixels = maxPixelsForModel(contract, pipeline);
  if (width * height > maxPixels) {
    return `${width} × ${height} is ${((width * height) / 1_000_000).toFixed(1)} MP — above this model's ${(maxPixels / 1_000_000).toFixed(1)} MP guideline; the server may reject it.`;
  }
  return null;
}

export function cameraControlValidationError(
  form: GenerateForm,
  availablePresetIds?: readonly string[],
): string | null {
  if (!generationCapabilitiesForFamily(form.family).supportsAdvancedVideo) return null;
  if (form.cameraControl === null) return null;
  const value = form.cameraControl.trim();
  if (!value) return "Choose a camera motion or enter a custom .safetensors path.";
  if (value.endsWith(".safetensors")) return null;
  if (availablePresetIds) {
    if (availablePresetIds.includes(value)) return null;
    return isCameraMotionPreset(value)
      ? "This host has no camera-motion preset by that name for the selected model. Use a custom .safetensors path."
      : "Custom camera motion must be a .safetensors path on the selected host.";
  }
  if (isCameraMotionPreset(value)) return null;
  return "Custom camera motion must be a .safetensors path on the selected host.";
}

/**
 * The per-model source-image contract (#772) plus wan's first/last-frame
 * pairing (#779). H3 is excluded: its boundary images have their own
 * authoring validator, which names the missing one precisely.
 *
 * A continuation counts as carrying source (#783), the same reading admission
 * takes through `mold_core::validation::request_carries_source_frames` — its
 * first frames come from the tail of the clip being continued. Without it the
 * Continue-a-video control a Wan I2V checkpoint now offers would be
 * unsubmittable, refused for the very contract that makes the checkpoint
 * extend-capable. `submitsExtend` applies the family gate `buildRequest`
 * applies, so a staged clip the wire drops satisfies nothing.
 */
export function sourceConditioningValidationError(
  form: GenerateForm,
  options: { ignoreUnsupportedStagedSource?: boolean } = {},
): string | null {
  if (isMinimaxH3Family(form.family)) return null;
  const caps = generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    form.guidanceCapabilities,
    form.sourceImageCapability,
  );
  // Source media is intentionally parked across model switches so changing
  // back restores the user's draft. Validate the request the selected model
  // will actually receive, not that retained UI state: buildRequest drops a
  // parked image when the checkpoint advertises source images as unsupported.
  // Continuations remain separate because their tail frames really do travel
  // on the wire and must still satisfy/refuse the advertised contract.
  const hasSourceImage =
    (!options.ignoreUnsupportedStagedSource || caps.supportsSourceImage) &&
    (caps.sourceImageMode === "single"
      ? Boolean(form.sourceImage)
      : form.imageAttachments.length > 0);
  return sourceImageValidationError({
    capability: caps.sourceImageCapability,
    hasSourceImage,
    isExtend: submitsExtend({
      family: form.family,
      extendVideo: form.extendVideo,
    }),
    hasEndFrame: caps.supportsEndFrame && Boolean(form.endFrame),
    frames: caps.supportsVideo ? form.frames : null,
    model: form.model,
  });
}

/**
 * Why this form's face-identity partition (#1224) would be refused, in the
 * server's own order — or `null` when it is valid, including "not used".
 *
 * Every rule lives in `@studio/lib/identityConditioning`; this only resolves
 * the two inputs that are desktop-shaped. `hasSourceImage` is the source the
 * REQUEST will carry, not the retained UI state: source media is parked
 * across model switches so switching back restores the draft, and a parked
 * image a checkpoint drops in `buildRequest` must not refuse an identity
 * photo it never travels with — the same reading
 * {@link sourceConditioningValidationError} takes.
 *
 * A checkpoint that cannot take an identity photo reports NOTHING: the whole
 * partition parks, off the wire and out of the inspector, so blocking Generate
 * on it would refuse a print the server would happily render.
 */
export function identityConditioningValidationError(form: GenerateForm): string | null {
  const caps = generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    form.guidanceCapabilities,
    form.sourceImageCapability,
  );
  const hasSourceImage =
    caps.supportsImg2img &&
    (caps.sourceImageMode === "single"
      ? Boolean(form.sourceImage)
      : form.imageAttachments.length > 0);
  return identityValidationError({
    supported: form.identitySupported === true,
    image: form.identityImage,
    weight: form.identityWeight,
    startStep: form.identityStartStep,
    steps: form.steps,
    hasLora: form.loras.length > 0,
    hasSourceImage,
  });
}

export function audioOutputValidationError(form: GenerateForm): string | null {
  const caps = generationCapabilitiesForFamily(form.family);
  return caps.supportsAudio && form.enableAudio && form.outputFormat !== "mp4"
    ? "Generated audio requires MP4 output."
    : null;
}

function keyframePositionValidationError(form: GenerateForm): string | null {
  if (form.keyframes.length === 0) return null;
  if (
    form.keyframes.some(
      (keyframe) =>
        !Number.isInteger(keyframe.frame) || keyframe.frame < 0 || keyframe.frame >= form.frames,
    )
  ) {
    return `Keyframe positions must be whole numbers from 0 to ${Math.max(0, form.frames - 1)}.`;
  }
  return new Set(form.keyframes.map((keyframe) => keyframe.frame)).size === form.keyframes.length
    ? null
    : "Each keyframe must use a unique frame position.";
}

/**
 * Wan flow shift and distill strengths. Out-of-band values are dropped by the
 * serializer rather than smuggled onto the wire, so the surface has to report
 * them or the render would silently fall back to the tier's own values.
 */
export function wanRecipeValidationError(form: GenerateForm): string | null {
  const caps = generationCapabilitiesForFamily(form.family, form.model);
  if (!caps.wanRecipe.supported) return null;
  return wanRecipeError(
    caps.wanRecipe.supportsDistillStrength
      ? form.wanRecipe
      : { ...form.wanRecipe, distillStrengthHigh: null, distillStrengthLow: null },
  );
}

export function advancedVideoValidationError(form: GenerateForm): string | null {
  if (!generationCapabilitiesForFamily(form.family).supportsAdvancedVideo) return null;
  const guidanceError = guidanceOverridesError(form.guidanceOverrides);
  if (guidanceError) return guidanceError;
  if (form.sourceVideo && !form.sourceVideo.base64) return "Source video cannot be empty.";
  if (form.keyframes.some((keyframe) => !keyframe.image.base64)) {
    return "Keyframe images cannot be empty.";
  }
  const keyframeError = keyframePositionValidationError(form);
  if (keyframeError) return keyframeError;
  if (form.icLoraControl) {
    if (!form.sourceVideo) return "Reference control requires a guide video.";
    if (form.loras.length + 1 > 4) {
      return "Reference control plus custom LoRAs exceeds the four-LoRA limit.";
    }
  }

  switch (form.pipeline) {
    case "a2-vid":
      if (!form.audioFile) return "Audio-to-video requires a conditioning audio file.";
      return form.audioFile.base64 ? null : "Conditioning audio cannot be empty.";
    case "retake":
      if (!form.sourceVideo) return "Retake requires a source video.";
      if (!form.retakeRange) return "Retake requires a start and end time.";
      if (
        !Number.isFinite(form.retakeRange.start_seconds) ||
        !Number.isFinite(form.retakeRange.end_seconds) ||
        form.retakeRange.start_seconds < 0
      ) {
        return "Retake times must be non-negative numbers.";
      }
      return form.retakeRange.end_seconds > form.retakeRange.start_seconds
        ? null
        : "Retake end time must be greater than its start time.";
    case "keyframe":
      return form.keyframes.length >= 2
        ? null
        : "Keyframe generation requires at least two keyframe images.";
    case "ic-lora":
      if (!form.sourceVideo && form.loras.length === 0 && !form.icLoraControl) {
        return "IC-LoRA requires a source video and at least one LoRA.";
      }
      if (!form.sourceVideo) return "IC-LoRA requires a source video.";
      return form.loras.length > 0 || form.icLoraControl
        ? null
        : "IC-LoRA requires at least one LoRA.";
    case "lip-dub":
      // Frames and fps come from the reference clip, so there is nothing to
      // check about duration here — only that a clip and the adapter exist.
      if (!form.sourceVideo) return "Lip dub requires a reference video to re-voice.";
      if (!form.icLoraControl && form.loras.length === 0) {
        return "Lip dub requires the lip-dub reference control.";
      }
      return form.width % 64 === 0 && form.height % 64 === 0
        ? null
        : "Lip dub renders in two stages, so width and height must be multiples of 64.";
    default:
      return null;
  }
}
