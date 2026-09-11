/**
 * The app and the phone's `generationJob.ts` `Job`, written from a durable
 * child's presentation.
 *
 * No policy lives here: the arm and every sentence come from
 * `@studio/lib/generationPresentation`, and which arm reaches which handler
 * (plus what happens to the hold on the way) from
 * `@studio/lib/durableGenerationPresentation`. All that is left is this
 * surface's field names.
 */

import {
  applyDurablePresentationWith,
  clearHold,
  type DurableRowSurface,
} from "@studio/lib/durableGenerationPresentation";
import type { GenerationChildPresentation } from "@studio/lib/generationPresentation";
import { markJobSettled, type Job } from "./generationJob";

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

const ROW: DurableRowSurface<Job> = {
  isLive: (job) => job.status !== "complete" && job.status !== "error",
  waiting: (job, p) => {
    job.status = "queued";
    // A plain queue wait keeps `stage` empty so `jobStatusCode` resolves the
    // live position; a resync is a label-only overlay over the hold.
    job.stage = p.reason === "queued" ? null : p.label;
  },
  held: (job, p) => {
    job.status = "queued";
    job.stage = p.label;
  },
  cancelling: (job, p) => {
    job.status = "queued";
    job.stage = p.label;
    job.cancelling = true;
  },
  running: (job, p) => {
    const alreadyRunning = job.status === "loading";
    job.status = "loading";
    if (!alreadyRunning || !job.stage) job.stage = p.label;
  },
  cancelled: (job, p) => settle(job, p.label, p.settledAtMs),
  failed: (job, p) => settle(job, p.message, p.settledAtMs),
  rejected: (job, p) => settle(job, p.message, null),
  unknown: (job, p) => {
    settle(job, p.message, p.settledAtMs);
    job.stage = p.label;
    job.interrupted = false;
    job.outcomeUnknown = true;
  },
};

const apply = applyDurablePresentationWith(ROW);

export function applyDurablePresentation(job: Job, p: GenerationChildPresentation): void {
  apply(job, p);
}
