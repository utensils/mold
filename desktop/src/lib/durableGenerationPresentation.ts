/**
 * Thin desktop/iPhone adapter from the shared `GenerationChildPresentation`
 * onto `generationJob.ts`'s `Job`. No policy lives here: the arm and every
 * sentence come from `@studio/lib/generationPresentation`. The `complete`
 * arm is deliberately not mapped — building the result from the recovery
 * record's child summary is the store's own.
 */

import type { GenerationChildPresentation } from "@studio/lib/generationPresentation";
import { markJobSettled, type Job } from "./generationJob";

function clearHold(job: Job): void {
  job.holdError = null;
  job.holdCode = null;
  job.retryable = false;
  job.retrying = false;
}

function settle(job: Job, error: string, settledAtMs: number | null): void {
  job.status = "error";
  job.error = error;
  job.cancelling = false;
  clearHold(job);
  if (settledAtMs !== null) job.settledAtMs ??= settledAtMs;
  markJobSettled(job);
  if (job.previewUrl) {
    URL.revokeObjectURL(job.previewUrl);
    job.previewUrl = null;
  }
}

export function applyDurablePresentation(job: Job, p: GenerationChildPresentation): void {
  if (job.status === "complete" || job.status === "error") return;
  switch (p.kind) {
    case "waiting":
      job.status = "queued";
      // A plain queue wait keeps `stage` empty so `jobStatusCode` resolves
      // the live position; a resync is a label-only overlay over the hold.
      job.stage = p.reason === "queued" ? null : p.label;
      if (p.reason !== "resync") clearHold(job);
      return;
    case "held":
      job.status = "queued";
      job.stage = p.label;
      job.holdError = p.error;
      job.holdCode = p.code;
      job.retryable = p.retryable;
      return;
    case "cancelling":
      job.status = "queued";
      job.stage = p.label;
      job.cancelling = true;
      clearHold(job);
      return;
    case "running":
      // A retry can move a held child straight to running; the hold is over.
      const alreadyRunning = job.status === "loading";
      job.status = "loading";
      if (!alreadyRunning || !job.stage) job.stage = p.label;
      clearHold(job);
      return;
    case "complete":
      return;
    case "cancelled":
      settle(job, p.label, p.settledAtMs);
      return;
    case "failed":
    case "complete_without_file":
      settle(job, p.message, p.settledAtMs);
      return;
    case "rejected":
      settle(job, p.message, null);
      return;
    case "unknown":
      // Settled and advisory: the row stops moving and is never re-attached
      // to a stream-era recovery pass.
      settle(job, p.message, p.settledAtMs);
      job.stage = p.label;
      job.interrupted = false;
      job.outcomeUnknown = true;
      return;
  }
}
