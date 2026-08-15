/**
 * Project the desktop generate form onto the shared @studio
 * `SequenceSharedParams`. Read at submit/export time so the sequence always
 * sees the LIVE inspector values — the split that fixed the old composer's
 * stale private copies of width/steps/guidance.
 */
import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import type { GenerateForm } from "./generateForm";
import type { ModelEntry } from "./api/types";
import { generationCapabilitiesForFamily } from "./capabilities";
import { effectiveGenerationGuidance } from "@studio/lib/generationCapabilities";

export function sequenceParams(
  form: GenerateForm,
  selectedModel: ModelEntry | null = null,
): SequenceSharedParams {
  const family = selectedModel?.family || form.family;
  const capabilities = generationCapabilitiesForFamily(
    family,
    form.model,
    form.pipeline,
    selectedModel?.guidance_capabilities ?? form.guidanceCapabilities,
  );
  return {
    model: form.model,
    family,
    width: form.width,
    height: form.height,
    fps: form.fps,
    steps: form.steps,
    guidance: effectiveGenerationGuidance(capabilities, form.guidance),
    strength: form.strength,
    sourceFitPolicy: form.sourceFit,
    upscalerModel: form.upscaleModel,
    seed: form.seed,
  };
}
