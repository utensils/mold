import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import { defaultOutputFormat, outputFormatsForFamily } from "../lib/capabilities";
import { applyMetadataToForm, type GenerateForm } from "../lib/generateForm";

export interface MobileGalleryReuseResult {
  modelName: string;
  substitutedModel: boolean;
}

/**
 * Restore the gallery fields the iPhone composer actually exposes. Start from
 * the desktop's canonical metadata mapper so model defaults and legacy fields
 * stay consistent, then neutralize advanced controls that mobile cannot show
 * or clear yet. A reused prompt must never submit invisible settings.
 */
export function applyMobileGalleryMetadata(
  form: GenerateForm,
  metadata: OutputMetadata,
  models: ModelEntry[],
): MobileGalleryReuseResult {
  const originalModelInstalled = models.some((model) => model.name === metadata.model);
  const fallbackModel = models.find((model) => model.name === form.model) ?? models[0];
  const mobileMetadata =
    originalModelInstalled || !fallbackModel
      ? metadata
      : { ...metadata, model: fallbackModel.name };

  applyMetadataToForm(form, mobileMetadata, models);

  // A substituted model can belong to a different family. Keep the original
  // canvas/settings where they remain portable, but never leave an impossible
  // output format selected (for example MP4 on an image-only model).
  if (!outputFormatsForFamily(form.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(form.family);
  }

  form.originalPrompt = null;
  form.scheduler = "default";
  form.cfgPlus = false;
  form.batchSize = 1;
  form.upscaleModel = "";
  form.strength = 0.75;
  form.controlModel = "";
  form.controlScale = 1;
  form.loras = [];
  form.enableAudio = false;
  form.pipeline = null;
  form.retakeRange = null;
  form.spatialUpscale = null;
  form.temporalUpscale = null;
  form.cameraControl = null;

  return {
    modelName: form.model,
    substitutedModel: !originalModelInstalled && !!fallbackModel,
  };
}
