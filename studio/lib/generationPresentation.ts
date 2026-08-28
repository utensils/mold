/**
 * One presentation policy for a durable generation child, shared by web,
 * desktop, and iPhone.
 *
 * The reducer in `generationLifecycle.ts` owns what is TRUE about a child;
 * this module owns what the user is TOLD about it. Every surface maps the
 * union below onto its own job shape through a thin adapter and keeps only
 * media hydration, effect ledgers, and previews to itself, so three
 * tracker-to-job ladders can no longer disagree on stage strings, the
 * terminal split, cancellation, failure prose, reconciliation reasons, or
 * what "settled" means.
 */

import { describeTransportError } from "./errors";
import {
  isTerminalGenerationPhase,
  type GenerationBatchTracker,
  type GenerationLifecycleJob,
  type GenerationReconciliationState,
} from "./generationLifecycle";
import { queueWaitLabel } from "./queuePosition";

/** Every stage sentence a durable child can show, in one table. */
export const GENERATION_STAGE_LABELS = {
  submitting: "Submitting",
  confirming: "Confirming with host",
  resync: "Re-syncing with host",
  queued: queueWaitLabel({ kind: "queued" }),
  paused: queueWaitLabel({ kind: "paused" }),
  running: "Developing",
  held: "Held by host — action required",
  cancelling: "Cancellation pending",
  cancelled: "Cancelled",
  unknown: "Outcome unknown",
} as const;

export type GenerationChildPresentation =
  | {
      kind: "waiting";
      reason: "submitting" | "confirming" | "queued" | "paused" | "resync";
      label: string;
    }
  | {
      kind: "held";
      label: string;
      error: string | null;
      code: string | null;
      retryable: boolean;
    }
  | { kind: "cancelling"; label: string }
  | { kind: "running"; label: string }
  | {
      kind: "complete";
      filename: string;
      originalFilename: string | null;
      settledAtMs: number;
      generationTimeMs: number;
    }
  /** The host settled this child on what reached its gallery, so a completion
   * that names no file is a contradiction the user must see — never a stub
   * that starves Photos auto-save or a hydration that retries forever. */
  | { kind: "complete_without_file"; message: string; settledAtMs: number }
  | { kind: "failed"; message: string; settledAtMs: number }
  | { kind: "cancelled"; label: string; settledAtMs: number }
  /** Admission refused by name; nothing was queued. */
  | { kind: "rejected"; message: string }
  /** The outcome is not knowable on this authority and nothing more can
   * arrive: advisory, settled, and never an effect. */
  | { kind: "unknown"; label: string; message: string; settledAtMs: number };

export type GenerationReconciliationPresentation =
  | { kind: "none" }
  /** `event_gap` | `incomplete_response` — the next snapshot repairs it. */
  | { kind: "resync" }
  /** `instance_mismatch` | `missing` | `batch_mismatch` — the reducer fences
   * every later event and `expectedInstanceId` never changes. */
  | { kind: "unknown"; message: string };

export interface PresentGenerationChildInput {
  tracker: GenerationBatchTracker;
  childIndex: number;
  hostLabel: string | null;
  /** Injected wall clock; the policy never reads `Date.now()`. */
  now: number;
}

function hostName(hostLabel: string | null): string {
  return hostLabel?.trim() || "The host";
}

export function reconciliationPresentation(
  state: GenerationReconciliationState,
  hostLabel: string | null,
): GenerationReconciliationPresentation {
  if (!state.required) return { kind: "none" };
  const host = hostName(hostLabel);
  switch (state.reason) {
    case "instance_mismatch":
      return {
        kind: "unknown",
        message: `${host} was replaced by a new server instance. The previous instance still owns this print's outcome, which is unknown here.`,
      };
    case "missing":
      return {
        kind: "unknown",
        message: `${host} no longer has a record of this print; its outcome is unknown.`,
      };
    case "batch_mismatch":
      return {
        kind: "unknown",
        message: `${host} reported a different batch identity for this print; its outcome is unknown.`,
      };
    case "event_gap":
    case "incomplete_response":
      return { kind: "resync" };
    case null:
      return { kind: "none" };
  }
}

function terminalErrorText(value: unknown): string | null {
  if (typeof value === "string") return value.trim() || null;
  if (value == null) return null;
  if (typeof value === "object") {
    for (const key of ["message", "error"] as const) {
      const field = (value as Record<string, unknown>)[key];
      if (typeof field === "string" && field.trim()) return field.trim();
    }
  }
  try {
    return JSON.stringify(value) || null;
  } catch {
    return String(value);
  }
}

/** The machine's own sentence, then its structured terminal error, run
 * through the shared describer so memory advice composes on every surface. */
export function generationFailureMessage(
  lifecycle: Pick<GenerationLifecycleJob, "error" | "terminalError">,
  hostLabel: string | null,
): string {
  const detail =
    lifecycle.error?.trim() ||
    terminalErrorText(lifecycle.terminalError) ||
    "Generation failed";
  return describeTransportError(new Error(detail), hostLabel);
}

function waiting(
  reason: "submitting" | "confirming" | "queued" | "paused" | "resync",
): GenerationChildPresentation {
  return { kind: "waiting", reason, label: GENERATION_STAGE_LABELS[reason] };
}

function presentTerminal(
  child: GenerationLifecycleJob,
  hostLabel: string | null,
): GenerationChildPresentation {
  const settledAtMs = child.completedAtMs ?? child.version.updatedAtMs;
  switch (child.phase) {
    case "complete":
      if (!child.result?.filename) {
        return {
          kind: "complete_without_file",
          message: `${hostName(hostLabel)} reported this print complete but published no file.`,
          settledAtMs,
        };
      }
      return {
        kind: "complete",
        filename: child.result.filename,
        originalFilename: child.result.originalFilename ?? null,
        settledAtMs,
        generationTimeMs: Math.max(0, settledAtMs - child.createdAtMs),
      };
    case "failed":
      return {
        kind: "failed",
        message: generationFailureMessage(child, hostLabel),
        settledAtMs,
      };
    default:
      return {
        kind: "cancelled",
        label: GENERATION_STAGE_LABELS.cancelled,
        settledAtMs,
      };
  }
}

export function presentGenerationChild({
  tracker,
  childIndex,
  hostLabel,
  now,
}: PresentGenerationChildInput): GenerationChildPresentation {
  if (tracker.admission.phase === "rejected") {
    return {
      kind: "rejected",
      message: tracker.admission.error ?? "Generation was not accepted",
    };
  }
  const child =
    Object.values(tracker.jobs).find((job) => job.childIndex === childIndex) ??
    null;
  // A terminal the reducer already froze is a KNOWN outcome, whatever the
  // batch's reconciliation later says about the authority.
  if (child && isTerminalGenerationPhase(child.phase)) {
    return presentTerminal(child, hostLabel);
  }
  const reconciliation = reconciliationPresentation(
    tracker.reconciliation,
    hostLabel,
  );
  if (reconciliation.kind === "unknown") {
    return {
      kind: "unknown",
      label: GENERATION_STAGE_LABELS.unknown,
      message: reconciliation.message,
      settledAtMs: now,
    };
  }
  if (reconciliation.kind === "resync") return waiting("resync");
  if (!child) {
    switch (tracker.admission.phase) {
      case "pending":
        return waiting("submitting");
      case "uncertain":
        return waiting("confirming");
      case "confirmed":
        // The snapshot omitted this child; the next read repairs it or the
        // reducer fences the batch.
        return waiting("resync");
    }
  }
  switch (child.phase) {
    case "paused":
      return waiting("paused");
    case "held":
      return {
        kind: "held",
        label: GENERATION_STAGE_LABELS.held,
        error: child.error,
        code: child.errorCode,
        retryable: child.retryable === true,
      };
    case "cancelling":
      return { kind: "cancelling", label: GENERATION_STAGE_LABELS.cancelling };
    case "running":
      return { kind: "running", label: GENERATION_STAGE_LABELS.running };
    default:
      return waiting("queued");
  }
}

export function presentationIsSettled(p: GenerationChildPresentation): boolean {
  switch (p.kind) {
    case "waiting":
    case "held":
    case "cancelling":
    case "running":
      return false;
    default:
      return true;
  }
}

export function presentationWorkStarted(
  p: GenerationChildPresentation,
): boolean {
  return p.kind === "running";
}

/** A batch is settled when it was refused, lost its authority, or every one
 * of its `childCount` children is present and terminal. */
export function generationTrackerSettled(
  tracker: GenerationBatchTracker,
  childCount: number,
): boolean {
  if (tracker.admission.phase === "rejected") return true;
  if (
    reconciliationPresentation(tracker.reconciliation, null).kind === "unknown"
  ) {
    return true;
  }
  const jobs = Object.values(tracker.jobs);
  return (
    childCount > 0 &&
    jobs.length === childCount &&
    jobs.every((job) => isTerminalGenerationPhase(job.phase))
  );
}
