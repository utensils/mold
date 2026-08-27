import { classifyMissingModelHold } from "@studio/api/generationPlacement";

/**
 * The phone's half of the durable missing-model offer.
 *
 * A print is admitted through `POST /api/generation-batches` BEFORE the
 * machine resolves its model, so "nobody has this model" arrives as a HELD
 * child carrying the machine's own typed code rather than as an infeasible
 * placement preview. `classifyMissingModelHold` stays the single authority for
 * WHICH codes mean that; this owns only the phone-shaped question of whether
 * a given job should raise the prompt right now, so `MobileApp.vue` stays an
 * orchestrator.
 *
 * Returns `null` for anything that must not raise the prompt: a job with no
 * durable identity, a hold that is not about the model, or one already
 * offered — a machine re-reports the same hold on every reconciliation wave,
 * and a second prompt for one parked print would be noise.
 */
export function planHeldMissingModelPull(input: {
  jobId: string | null | undefined;
  model: string;
  /** The held child's typed `error_code`, never its sentence. */
  heldCode: string | null;
  alreadyOffered: ReadonlySet<string>;
}): { model: string; jobId: string } | null {
  const jobId = input.jobId;
  if (!jobId || input.alreadyOffered.has(jobId)) return null;
  const missing = classifyMissingModelHold(input.heldCode, input.model);
  return missing ? { model: missing.model, jobId } : null;
}
