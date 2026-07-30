import { modelsForOutput, sequenceMotionTailFrames } from "@studio/lib/sequence";
import type { SequenceClipForm } from "@studio/lib/sequenceForm";
import {
  clampClipsToMotionTail,
  planSequenceReuse,
  type SequenceReuseLossiness,
} from "@studio/lib/sequenceReuse";
import type { ModelEntry, OutputMetadata } from "../lib/api/types";
import { defaultOutputFormat, outputFormatsForFamily } from "../lib/capabilities";
import { applyMetadataToForm, type GenerateForm } from "../lib/generateForm";

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
  const originalModelInstalled = models.some((model) => model.name === metadata.model);
  // A sequence must fall back to a SEQUENCE-capable model; the first installed
  // model could be an image model the clip rail can never render.
  const candidates = plan ? modelsForOutput(models, "sequence") : models;
  const fallbackModel =
    candidates.find((model) => model.name === form.model) ?? candidates[0] ?? models[0];
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
    form.icLoraControl = null;
  }

  // A substituted model can belong to a different family. Keep the original
  // canvas/settings where they remain portable, but never leave an impossible
  // output format selected (for example MP4 on an image-only model).
  if (!outputFormatsForFamily(form.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(form.family);
  }

  let sequence: MobileSequenceReuse | null = null;
  if (plan) {
    // The live tail belongs to the model resolved above, not the recorded one.
    const tail = sequenceMotionTailFrames({ name: form.model, family: form.family });
    const { clips, raised } = clampClipsToMotionTail(plan.clips, tail, 9);
    sequence = { clips, lossy: plan.lossy, raised };
    // `metadata.prompt` for a sequence is every clip newline-joined; clip 1's
    // prompt is the honest single-shot value to leave behind.
    form.prompt = clips[0]?.prompt ?? "";
  }

  return {
    modelName: form.model,
    substitutedModel: !originalModelInstalled && !!fallbackModel,
    sequence,
  };
}
