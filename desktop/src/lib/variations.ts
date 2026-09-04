import { conditioningForRequest } from "@studio/lib/sourceMediaPlan";
import { isAudioCompletion } from "@studio/lib/ltx2Pipeline";
import { isMeshArtifact } from "@studio/lib/meshCompletion";
import type { GenerationCapabilities } from "./capabilities";
import type { GenerateForm } from "./generateForm";
import type { CompleteEvent } from "./api/types";

/**
 * The recipe renders one at a time: an edit model, or a request that carries
 * references. New image resolves the capabilities against the recipe it has
 * loaded; a surface with only the form asks the family and model.
 */
export function batchLockedForForm(form: GenerateForm, caps: GenerationCapabilities): boolean {
  if (caps.forcesBatchSizeOne) return true;
  return (
    conditioningForRequest(caps.sourceImageMode, {
      hasSource: Boolean(form.sourceImage),
      referenceCount: form.imageAttachments.length,
      lastWrite: form.exclusiveWell ?? null,
    }) === "references"
  );
}

/**
 * Whether **Make 4 variations** means anything for what is on the canvas.
 * Only a finished still can be made again as a batch — a clip, a mesh and a
 * sound have no batch at all — and a batch-locked recipe coerces the count to
 * one, so the action would promise four and make exactly one.
 */
export function canRepeatPrint(
  job: { status: string; result: CompleteEvent | null } | null | undefined,
  batchLocked: boolean,
): boolean {
  if (batchLocked || job?.status !== "complete") return false;
  return !job.result?.video_frames && !isAudioCompletion(job.result) && !isMeshArtifact(job.result);
}
