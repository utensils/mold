/**
 * Thin web adapter from the shared `GenerationChildPresentation` onto the
 * rail's `Job`. No policy lives here: which arm a child is in, and every
 * sentence, come from `@studio/lib/generationPresentation`. The `complete`
 * arm is deliberately not mapped — media hydration is the composable's own.
 */

import type { GenerationChildPresentation } from "@studio/lib/generationPresentation";
import type { Job } from "../composables/useGenerateStream";

function clearHold(job: Job): void {
  job.holdError = null;
  job.holdCode = null;
  job.retryable = false;
  job.retrying = false;
}

function settle(job: Job, state: "error" | "canceled", settledAt: number) {
  job.state = state;
  job.settledAt = settledAt;
  job.cancelling = false;
  job.cancelRequested = false;
  job.previewUrl = null;
}

export function applyDurablePresentation(
  job: Job,
  p: GenerationChildPresentation,
  now: number,
): void {
  if (job.state !== "running") return;
  switch (p.kind) {
    case "waiting":
      job.progress.stage = p.label;
      job.workStarted = false;
      // A resync is a label-only overlay: the hold it covers is still the
      // host's word until the snapshot says otherwise.
      if (p.reason !== "resync") clearHold(job);
      return;
    case "held":
      job.progress.stage = p.label;
      job.workStarted = false;
      job.holdError = p.error;
      job.holdCode = p.code;
      job.retryable = p.retryable;
      return;
    case "cancelling":
      job.progress.stage = p.label;
      job.workStarted = false;
      job.cancelling = true;
      clearHold(job);
      return;
    case "running":
      // A retry can move a held child straight to running; the hold is over.
      if (!job.workStarted || !job.progress.stage) job.progress.stage = p.label;
      job.workStarted = true;
      job.progress.queuePosition = null;
      clearHold(job);
      return;
    case "complete":
      return;
    case "cancelled":
      settle(job, "canceled", p.settledAtMs);
      return;
    case "failed":
    case "complete_without_file":
      job.error = p.message;
      settle(job, "error", p.settledAtMs);
      return;
    case "rejected":
      job.error = p.message;
      settle(job, "error", now);
      return;
    case "unknown":
      // The rail has no "unknown" state; `detached` already carries the
      // semantics — advisory, retired rather than labelled "Failed".
      job.error = p.message;
      settle(job, "error", p.settledAtMs);
      job.detached = true;
      return;
  }
}
