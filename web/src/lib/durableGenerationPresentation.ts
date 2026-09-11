/**
 * The browser's rail `Job`, written from a durable child's presentation.
 *
 * No policy lives here: which arm a child is in and every sentence come from
 * `@studio/lib/generationPresentation`, and which arm reaches which handler
 * (plus what happens to the hold on the way) from
 * `@studio/lib/durableGenerationPresentation`. All that is left is this
 * surface's field names.
 */

import {
  applyDurablePresentationWith,
  type DurableRowSurface,
} from "@studio/lib/durableGenerationPresentation";
import type { GenerationChildPresentation } from "@studio/lib/generationPresentation";
import type { Job } from "../composables/useGenerateStream";

function settle(job: Job, state: "error" | "canceled", settledAt: number) {
  job.state = state;
  job.settledAt = settledAt;
  job.cancelling = false;
  job.cancelRequested = false;
  job.previewUrl = null;
}

/* The clock is the reducer pass's, not this module's: a rejection never
 * reached the queue, so it has no settled time of its own. */
const railAt = (now: number): DurableRowSurface<Job> => ({
  isLive: (job) => job.state === "running",
  waiting: (job, p) => {
    job.progress.stage = p.label;
    job.workStarted = false;
  },
  held: (job, p) => {
    job.progress.stage = p.label;
    job.workStarted = false;
  },
  cancelling: (job, p) => {
    job.progress.stage = p.label;
    job.workStarted = false;
    job.cancelling = true;
  },
  running: (job, p) => {
    if (!job.workStarted || !job.progress.stage) job.progress.stage = p.label;
    job.workStarted = true;
    job.progress.queuePosition = null;
  },
  cancelled: (job, p) => settle(job, "canceled", p.settledAtMs),
  failed: (job, p) => {
    job.error = p.message;
    settle(job, "error", p.settledAtMs);
  },
  rejected: (job, p) => {
    job.error = p.message;
    settle(job, "error", now);
  },
  unknown: (job, p) => {
    // The rail has no "unknown" state; `detached` already carries the
    // semantics — advisory, retired rather than labelled "Failed".
    job.error = p.message;
    settle(job, "error", p.settledAtMs);
    job.detached = true;
  },
});

export function applyDurablePresentation(
  job: Job,
  p: GenerationChildPresentation,
  now: number,
): void {
  applyDurablePresentationWith(railAt(now))(job, p);
}
