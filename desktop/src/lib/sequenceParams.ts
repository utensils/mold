/**
 * Project the desktop generate form onto the shared @studio
 * `SequenceSharedParams`. Read at submit/export time so the sequence always
 * sees the LIVE inspector values — the split that fixed the old composer's
 * stale private copies of width/steps/guidance.
 */
import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import type { GenerateForm } from "./generateForm";
import type { ModelEntry } from "./api/types";

export function sequenceParams(
  form: GenerateForm,
  selectedModel: ModelEntry | null = null,
): SequenceSharedParams {
  return {
    model: form.model,
    family: selectedModel?.family || form.family,
    width: form.width,
    height: form.height,
    fps: form.fps,
    steps: form.steps,
    guidance: form.guidance,
    strength: form.strength,
    sourceFitPolicy: form.sourceFit,
    upscalerModel: form.upscaleModel,
    seed: form.seed,
  };
}
