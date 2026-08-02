import { generationCapabilitiesForFamily } from "./capabilities";
import type { GenerateForm } from "./generateForm";
import { isCameraMotionPreset } from "@studio/lib/cameraMotion";
import { guidanceOverridesError } from "@studio/lib/guidanceOverrides";

export const MAX_GENERATION_PIXELS = 1_800_000;
export const MAX_INLINE_GENERATION_MEDIA_BYTES = 64 * 1024 * 1024;
// JSON base64 expands bytes by roughly 4/3 and the server body limit is 64
// MiB. Keep the whole iPhone conditioning payload below 45 MiB so the request
// has room for JSON/scalars without a late extractor rejection.
export const MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES = 45 * 1024 * 1024;
export const MOBILE_MEDIA_BUDGET_ERROR =
  "Combined generation media must be 45 MiB or smaller on iPhone.";

export type InlineGenerationMediaField =
  | "sourceImage"
  | "maskImage"
  | "controlImage"
  | "imageAttachments"
  | "sourceVideo"
  | "extendVideo"
  | "keyframes"
  | "audioFile";

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
  if (exclude !== "maskImage") total += decodedBase64Bytes(form.maskImage);
  if (exclude !== "controlImage") total += decodedBase64Bytes(form.controlImage);
  if (exclude !== "imageAttachments") {
    total += form.imageAttachments.reduce((sum, image) => sum + decodedBase64Bytes(image), 0);
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

export function guidanceValidationError(value: number): string | null {
  return Number.isFinite(value) && value >= 0 && value <= 100
    ? null
    : "Guidance must be from 0 to 100.";
}

export function fpsValidationError(value: number): string | null {
  return Number.isInteger(value) && value >= 1 && value <= 60
    ? null
    : "FPS must be a whole number from 1 to 60.";
}

export function resolutionValidationError(width: number, height: number): string | null {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 64 || height < 64) {
    return "Width and height must each be at least 64 pixels.";
  }
  if (width % 16 !== 0 || height % 16 !== 0) {
    return "Width and height must be multiples of 16.";
  }
  const megapixels = (width * height) / 1_000_000;
  return width * height <= MAX_GENERATION_PIXELS
    ? null
    : `${width} × ${height} is ${megapixels.toFixed(1)} MP. Choose a resolution at or below 1.8 MP.`;
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
    default:
      return null;
  }
}
