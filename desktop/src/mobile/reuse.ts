import { modelsForOutput, sequenceMotionTailFrames } from "@studio/lib/sequence";
import type { SequenceClipForm } from "@studio/lib/sequenceForm";
import {
  clampClipsToMotionTail,
  planSequenceReuse,
  type SequenceReuseLossiness,
} from "@studio/lib/sequenceReuse";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import {
  defaultOutputFormat,
  generationCapabilitiesForFamily,
  outputFormatsForFamily,
} from "../lib/capabilities";
import { applyMetadataToForm, type GenerateForm } from "../lib/generateForm";
import { emptyGuidanceOverrides } from "@studio/lib/guidanceOverrides";
import { emptyWanRecipe } from "@studio/lib/wanRecipe";
import { canonicalMinimaxH3ModelName, isMinimaxH3Identity } from "@studio/lib/minimaxH3Authoring";

/** The clip rail a sequence print reloads, plus what it could not give back. */
export interface MobileSequenceReuse {
  clips: SequenceClipForm[];
  lossy: SequenceReuseLossiness;
  /** Clips raised because the resolved model's motion tail grew — the caller
   * must say so rather than silently resizing. */
  raised: number;
}

export interface MobileGalleryReuseResult {
  modelName: string;
  substitutedModel: boolean;
  /** Non-null only for a print stitched from a sequence (`metadata.chain`). */
  sequence: MobileSequenceReuse | null;
  /** Present when durable metadata claims a sequence for a model partition
   * that cannot safely be reinterpreted as a sequence-capable checkpoint. */
  sequenceUnsupportedReason?: string;
}

/**
 * Restore gallery settings through the desktop's canonical metadata mapper so
 * iPhone and desktop keep the same model defaults, legacy normalization, and
 * capability-aware request behavior. Binary media still clears in the shared
 * mapper because output metadata cannot carry those bytes.
 *
 * A sequence print additionally returns its recorded clips: iPhone gets
 * **Reuse only** in this pass — Edit sequence needs a chain-detail fetch on
 * the recovery route, which mobile does not have yet.
 */
export function applyMobileGalleryMetadata(
  form: GenerateForm,
  metadata: OutputMetadata,
  models: ModelEntry[],
): MobileGalleryReuseResult {
  const plan = planSequenceReuse(metadata);
  const oneShotPrompt = form.prompt;
  const oneShotOriginalPrompt = form.originalPrompt;
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

  // H3 is one-shot-only. A malformed or future durable snapshot must not turn
  // an H3 chain into an unrelated installed sequence model merely because the
  // original exact checkpoint is absent. Leave the form untouched and give
  // the caller a concrete recovery error instead.
  if (plan && recordedH3Identity) {
    return {
      modelName: canonicalRecordedModel ?? metadata.model,
      substitutedModel: false,
      sequence: null,
      sequenceUnsupportedReason:
        "This print records a MiniMax H3 checkpoint, which cannot render a clip sequence.",
    };
  }
  // A sequence must fall back to a SEQUENCE-capable model; the first installed
  // model could be an image model the clip rail can never render.
  const candidates = plan ? modelsForOutput(models, "sequence") : models;
  const fallbackModel =
    candidates.find((model) => model.name === form.model) ?? candidates[0] ?? models[0];
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
  if (!outputFormatsForFamily(form.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(form.family);
  }

  let sequence: MobileSequenceReuse | null = null;
  if (plan) {
    // The live tail belongs to the model resolved above, not the recorded one
    // — and for wan it belongs to that checkpoint's advertised `source_image`
    // contract rather than to the family, because only an image-conditioned
    // checkpoint carries anything across the seam (#783).
    const resolvedModel = models.find((entry) => entry.name === form.model) ?? null;
    const tail = sequenceMotionTailFrames({
      name: form.model,
      family: form.family,
      source_image: resolvedModel?.source_image ?? null,
    });
    const { clips, raised } = clampClipsToMotionTail(plan.clips, tail, 9);
    sequence = { clips, lossy: plan.lossy, raised };
    // Reusing a sequence must not overwrite the parked one-shot prompt.
    form.prompt = oneShotPrompt;
    form.originalPrompt = oneShotOriginalPrompt;
  } else if (!metadata.prompt?.trim()) {
    // A promptless print has no prompt provenance. Clear any provenance left
    // by the previously edited print so typing a new prompt cannot revive it.
    form.originalPrompt = null;
  }

  return {
    modelName: form.model,
    substitutedModel,
    sequence,
  };
}
