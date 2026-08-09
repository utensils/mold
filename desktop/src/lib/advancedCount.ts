/**
 * How many Advanced sections currently hold a non-default value. Drives
 * the inspector's "N on" pill and the section header's "N active" badge from one
 * pure function so the two never disagree. Only counts controls the current
 * family actually exposes (a stale negative prompt on a FLUX model isn't
 * "active" because FLUX has no negative-prompt control).
 */
import { generationCapabilitiesForFamily } from "./capabilities";
import type { GenerateForm } from "./generateForm";
import { guidanceOverridesAreEmpty } from "@studio/lib/guidanceOverrides";
import { negativePromptWireValue } from "@studio/lib/negativePrompt";
import { wanRecipeCount } from "@studio/lib/wanRecipe";

export function advancedActiveCount(form: GenerateForm): number {
  const caps = generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    form.guidanceCapabilities,
  );
  let count = 0;
  // Differs-from-default (#787): an untouched wan default is not "active",
  // while an explicit clear (the empty-uncond opt-out) is.
  if (
    caps.supportsNegativePrompt &&
    negativePromptWireValue(form.negativePrompt, form.negativePromptDefault) !== undefined
  )
    count += 1;
  if (caps.supportsScheduler && form.scheduler !== "default") count += 1;
  if (caps.supportsCfgPlus && form.cfgPlus) count += 1;
  if (caps.wanRecipe.supported) count += wanRecipeCount(form.wanRecipe);
  if (caps.supportsImg2img && (form.sourceImage || form.imageAttachments.length > 0)) count += 1;
  if (caps.supportsLora && form.loras.length > 0) count += 1;
  if (!caps.supportsVideo && form.upscaleModel) count += 1;
  if (caps.supportsVideo && form.cameraControl) count += 1;
  if (caps.supportsAdvancedVideo && form.icLoraControl) count += 1;
  if (caps.supportsAdvancedVideo && !guidanceOverridesAreEmpty(form.guidanceOverrides)) count += 1;
  if (
    caps.supportsAdvancedVideo &&
    (form.pipeline ||
      form.keyframes.length > 0 ||
      form.sourceVideo ||
      form.spatialUpscale ||
      form.temporalUpscale)
  ) {
    count += 1;
  }
  return count;
}
