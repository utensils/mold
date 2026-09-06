import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import { coerceFormOutputFormat, generationCapabilitiesForFamily } from "../lib/capabilities";
import { applyMetadataToForm, type GenerateForm } from "../lib/generateForm";
import { emptyGuidanceOverrides } from "@studio/lib/guidanceOverrides";
import { emptyWanRecipe } from "@studio/lib/wanRecipe";
import { canonicalMinimaxH3ModelName, isMinimaxH3Identity } from "@studio/lib/minimaxH3Authoring";
import { reusedPrintTitle } from "./libraryOrganization";

export interface MobileGalleryReuseResult {
  modelName: string;
  substitutedModel: boolean;
  /** The print's saved title, restored into the Create title field (`""`
   * when the print was untitled — restoring clears a stale title too). */
  title: string;
}

/**
 * Restore gallery settings through the desktop's canonical metadata mapper so
 * iPhone and desktop keep the same model defaults, legacy normalization, and
 * capability-aware request behavior. Binary media still clears in the shared
 * mapper because output metadata cannot carry those bytes.
 *
 * A STITCHED print restores as an ordinary one-shot — model, canvas, fps, seed
 * and clip 1's prompt. There is no clip rail to reload on any surface any
 * more, and a stitched print is never refused: whatever the host recorded is
 * handed back as a single render the user can run again. The prompt rule
 * (`chain.stages[0].prompt` over the newline-joined `metadata.prompt`) lives
 * in `applyMetadataToForm`, so the phone and desktop cannot disagree about it.
 */
export function applyMobileGalleryMetadata(
  form: GenerateForm,
  metadata: OutputMetadata,
  models: ModelEntry[],
): MobileGalleryReuseResult {
  const canonicalRecordedModel = canonicalMinimaxH3ModelName(metadata.model);
  const recordedH3Identity = isMinimaxH3Identity(null, metadata.model);
  const installedRecordedModel = models.find(
    (model) =>
      model.name === metadata.model ||
      (canonicalRecordedModel !== null &&
        canonicalMinimaxH3ModelName(model.name) === canonicalRecordedModel),
  );
  const originalModelInstalled = installedRecordedModel !== undefined;
  // Every H3-shaped identity is a fail-closed family boundary. Released
  // aliases canonicalize to their exact task/layout; an unknown future
  // partition remains byte-for-byte unavailable instead of being guessed or
  // substituted into another family.
  const preserveMissingH3 = !originalModelInstalled && recordedH3Identity;

  const fallbackModel = models.find((model) => model.name === form.model) ?? models[0];
  const mobileMetadata = installedRecordedModel
    ? { ...metadata, model: installedRecordedModel.name }
    : preserveMissingH3
      ? { ...metadata, model: canonicalRecordedModel ?? metadata.model }
      : !fallbackModel
        ? metadata
        : { ...metadata, model: fallbackModel.name };

  applyMetadataToForm(form, mobileMetadata, models);

  if (preserveMissingH3) {
    form.model = mobileMetadata.model;
    form.family = "minimax-h3";
  }

  const substitutedModel = !originalModelInstalled && !preserveMissingH3 && !!fallbackModel;
  if (substitutedModel) {
    // Adapters and auxiliary models are host/model artifacts, not portable
    // print settings. Replaying them against a fallback can fail outright or,
    // worse, produce a result with unrelated weights.
    form.loras = [];
    form.upscaleModel = "";
    form.controlModel = "";
    form.cameraControl = null;
    form.icLoraControl = null;
    const fallbackCaps = generationCapabilitiesForFamily(fallbackModel.family, fallbackModel.name);
    if (!fallbackCaps.supportsAdvancedVideo) {
      form.guidanceOverrides = emptyGuidanceOverrides();
    }
    // The recorded solver and recipe belong to the print's own family and
    // tier; the server rejects either across that boundary.
    if (!fallbackCaps.schedulerOptions.includes(form.scheduler)) form.scheduler = "default";
    if (!fallbackCaps.wanRecipe.supported) {
      form.wanRecipe = emptyWanRecipe();
    } else if (!fallbackCaps.wanRecipe.supportsDistillStrength) {
      form.wanRecipe = {
        ...form.wanRecipe,
        distillStrengthHigh: null,
        distillStrengthLow: null,
      };
    }
  }

  // A substituted model can belong to a different family. Keep the original
  // canvas/settings where they remain portable, but never leave an impossible
  // output format selected (for example MP4 on an image-only model).
  //
  // The RECIPE answers, through the one shared coercion every restore site
  // uses; the pre-profile family list is only the fallback inside it. Asking
  // the family directly recognizes exactly one mesh family name, so a second
  // one would have had its advertised `glb` rewritten to `png` here — after
  // the shared mapper had already resolved it correctly.
  form.outputFormat =
    coerceFormOutputFormat(form.outputFormat, form.family, form.recipeCapabilities) ??
    form.outputFormat;

  if (!metadata.prompt?.trim() && !metadata.chain?.stages?.[0]?.prompt?.trim()) {
    // A promptless print has no prompt provenance. Clear any provenance left
    // by the previously edited print so typing a new prompt cannot revive it.
    form.originalPrompt = null;
  }

  return {
    modelName: form.model,
    substitutedModel,
    title: reusedPrintTitle(metadata),
  };
}
