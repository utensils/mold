/**
 * Unified activity view-model: one discriminated union rendering BOTH
 * single-generation ("print") jobs and durable sequence (chain) jobs in the
 * same jobs surface — desktop's ActivityStrip, web's Create strip, and the
 * iPhone queue list all consume this merge instead of keeping a separate
 * chain-jobs list (mockup 1c: "the chain Jobs list merges with the
 * activity strip").
 */

import type { ChainJobState, ChainJobSummary } from "./api/chainTypes";
import { friendlySequenceError } from "./sequence";

export type ActivityAction =
  | "cancel"
  | "watch"
  | "retake"
  | "edit"
  | "resume"
  | "delete";

export type PrintPhase = "queued" | "running" | "done" | "failed" | "cancelled";

export type ActivityJobVM =
  | {
      kind: "print";
      key: string;
      hostId: string;
      hostLabel: string;
      model: string;
      prompt: string;
      phase: PrintPhase;
      progress: { step: number; total: number } | null;
      /** Auto-chained long single videos already carry stage info. */
      chain: { stageIndex: number; stageCount: number } | null;
      actions: ActivityAction[];
      createdAtMs: number;
    }
  | {
      kind: "sequence";
      key: string;
      jobId: string;
      hostId: string;
      hostLabel: string;
      model: string;
      state: ChainJobState;
      stageCount: number;
      currentStage: number;
      progress: { step: number; total: number } | null;
      error: string | null;
      actions: ActivityAction[];
      createdAtMs: number;
    };

/** Actions available for a sequence job in a given state. Interrupted /
 * failed / cancelled jobs surface `resume` — resumability with cached
 * stages is a server feature the strip must not hide. */
export function sequenceActions(state: ChainJobState): ActivityAction[] {
  switch (state) {
    case "queued":
    case "running":
      return ["watch", "cancel"];
    case "completed":
      return ["watch", "edit", "delete"];
    case "interrupted":
    case "failed":
    case "cancelled":
      return ["resume", "edit", "delete"];
  }
}

export function sequenceToVM(
  summary: ChainJobSummary,
  host: { hostId: string; hostLabel: string },
  progress: { step: number; total: number } | null = null,
): ActivityJobVM {
  return {
    kind: "sequence",
    key: `seq:${host.hostId}:${summary.id}`,
    jobId: summary.id,
    hostId: host.hostId,
    hostLabel: host.hostLabel,
    model: summary.model,
    state: summary.state,
    stageCount: summary.stage_count,
    currentStage: summary.current_stage,
    progress,
    error: summary.error ? friendlySequenceError(summary.error) : null,
    actions: sequenceActions(summary.state),
    createdAtMs: summary.created_at_unix_ms,
  };
}

function isActive(vm: ActivityJobVM): boolean {
  return vm.kind === "print"
    ? vm.phase === "queued" || vm.phase === "running"
    : vm.state === "queued" || vm.state === "running";
}

function isRunning(vm: ActivityJobVM): boolean {
  return vm.kind === "print" ? vm.phase === "running" : vm.state === "running";
}

/** Merge prints and sequences into one list: active work first (running
 * before queued), then everything by recency. */
export function mergeActivity(
  prints: readonly ActivityJobVM[],
  sequences: readonly ActivityJobVM[],
): ActivityJobVM[] {
  return [...prints, ...sequences].sort((a, b) => {
    const activeDelta = Number(isActive(b)) - Number(isActive(a));
    if (activeDelta !== 0) return activeDelta;
    const runningDelta = Number(isRunning(b)) - Number(isRunning(a));
    if (runningDelta !== 0) return runningDelta;
    return b.createdAtMs - a.createdAtMs;
  });
}
