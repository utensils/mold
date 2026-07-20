import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import { defaultOutputFormat, outputFormatsForFamily } from "../lib/capabilities";
import { applyMetadataToForm, type GenerateForm } from "../lib/generateForm";

export interface MobileGalleryReuseResult {
  modelName: string;
  substitutedModel: boolean;
}

/**
 * Restore gallery settings through the desktop's canonical metadata mapper so
 * iPhone and desktop keep the same model defaults, legacy normalization, and
 * capability-aware request behavior. Binary media still clears in the shared
 * mapper because output metadata cannot carry those bytes.
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

  if (!originalModelInstalled && fallbackModel) {
    // Adapters and auxiliary models are host/model artifacts, not portable
    // print settings. Replaying them against a fallback can fail outright or,
    // worse, produce a result with unrelated weights.
    form.loras = [];
    form.upscaleModel = "";
    form.controlModel = "";
    form.cameraControl = null;
  }

  // A substituted model can belong to a different family. Keep the original
  // canvas/settings where they remain portable, but never leave an impossible
  // output format selected (for example MP4 on an image-only model).
  if (!outputFormatsForFamily(form.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(form.family);
  }

  return {
    modelName: form.model,
    substitutedModel: !originalModelInstalled && !!fallbackModel,
  };
}
