/**
 * How a durable child's presentation reaches a surface's own row.
 *
 * The sentences and the arm a child is in are `generationPresentation.ts`'s;
 * this is the layer under that — which arm goes to which handler, and what
 * happens to the hold on the way. Every surface had its own copy of that
 * switch over `Job` fields with different names, so an arm added upstream
 * fell through silently in two places instead of failing to compile in one.
 *
 * The `complete` arm is deliberately unhandled here: hydrating the media from
 * what the host published belongs to whoever owns the row.
 */

import type { GenerationChildPresentation } from "./generationPresentation";

type Arm<K extends GenerationChildPresentation["kind"]> = Extract<
  GenerationChildPresentation,
  { kind: K }
>;

/**
 * The hold quartet, which every surface's row spells the same way — a held
 * child is the one piece of this that is not a label.
 */
export interface DurableRowHold {
  holdError?: string | null;
  holdCode?: string | null;
  retryable?: boolean;
  retrying?: boolean;
}

export function clearHold(row: DurableRowHold): void {
  row.holdError = null;
  row.holdCode = null;
  row.retryable = false;
  row.retrying = false;
}

/** What a surface does with each arm, once the hold has been dealt with. */
export interface DurableRowSurface<J extends DurableRowHold> {
  /** False for a row that has settled: nothing may move it again. */
  isLive: (job: J) => boolean;
  waiting: (job: J, p: Arm<"waiting">) => void;
  held: (job: J, p: Arm<"held">) => void;
  cancelling: (job: J, p: Arm<"cancelling">) => void;
  running: (job: J, p: Arm<"running">) => void;
  cancelled: (job: J, p: Arm<"cancelled">) => void;
  /** A completion that published no file is shown exactly like a failure. */
  failed: (job: J, p: Arm<"failed"> | Arm<"complete_without_file">) => void;
  /** Admission refused by name; nothing was queued, so nothing settled. */
  rejected: (job: J, p: Arm<"rejected">) => void;
  /** Settled and advisory: the row stops moving and is never re-attached. */
  unknown: (job: J, p: Arm<"unknown">) => void;
}

export function applyDurablePresentationWith<J extends DurableRowHold>(
  surface: DurableRowSurface<J>,
): (job: J, p: GenerationChildPresentation) => void {
  return (job, p) => {
    if (!surface.isLive(job)) return;
    switch (p.kind) {
      case "waiting":
        // A resync is a label-only overlay: the hold it covers is still the
        // machine's word until the snapshot says otherwise.
        if (p.reason !== "resync") clearHold(job);
        surface.waiting(job, p);
        return;
      case "held":
        job.holdError = p.error;
        job.holdCode = p.code;
        job.retryable = p.retryable;
        surface.held(job, p);
        return;
      case "cancelling":
        clearHold(job);
        surface.cancelling(job, p);
        return;
      case "running":
        // A retry can move a held child straight to running; the hold is over.
        clearHold(job);
        surface.running(job, p);
        return;
      case "complete":
        return;
      case "cancelled":
        surface.cancelled(job, p);
        return;
      case "failed":
      case "complete_without_file":
        surface.failed(job, p);
        return;
      case "rejected":
        surface.rejected(job, p);
        return;
      case "unknown":
        surface.unknown(job, p);
        return;
    }
  };
}
