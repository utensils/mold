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
  "cancel" | "watch" | "retake" | "edit" | "resume" | "delete";

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
      /** Wall clock at submit — NOT a client counter; attention rows sort
       *  against sequence epoch stamps. */
      createdAtMs: number;
      /** Wall clock when the job settled; null while it is still in flight. */
      settledAtMs: number | null;
      error: string | null;
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
      /** Server truth (`updated_at_unix_ms`) once settled, else null. The age
       *  rule reads this, so an app restart never resurrects an old failure. */
      settledAtMs: number | null;
    };

/** How long a settled-but-wrong job keeps a row in the Create strip. */
export const SETTLED_VISIBLE_MS = 5 * 60_000;
/** Strip rows for wrong-but-settled work; the overflow becomes a digest count. */
export const MAX_ATTENTION_ROWS = 2;
/** Rows rendered in Library ▸ History ▸ Sequences before the count footnote. */
export const HISTORY_JOBS_RENDER_CAP = 200;

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

const ACTION_LABEL: Record<ActivityAction, string> = {
  watch: "Watch",
  cancel: "Cancel",
  resume: "Resume",
  edit: "Edit",
  delete: "Delete",
  retake: "Retake",
};

/** §11 voice: `watch` is present tense, so a settled job is *opened* instead.
 *  Same action id on the wire — only the label moves. */
export function sequenceActionLabel(
  action: ActivityAction,
  state: ChainJobState,
): string {
  if (action === "watch")
    return state === "queued" || state === "running" ? "Watch" : "Open";
  return ACTION_LABEL[action];
}

export function sequenceToVM(
  summary: ChainJobSummary,
  host: { hostId: string; hostLabel: string },
  progress: { step: number; total: number } | null = null,
): ActivityJobVM {
  const settled = summary.state !== "queued" && summary.state !== "running";
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
    settledAtMs: settled ? summary.updated_at_unix_ms : null,
  };
}

function isActive(vm: ActivityJobVM): boolean {
  return vm.kind === "print"
    ? vm.phase === "queued" || vm.phase === "running"
    : vm.state === "queued" || vm.state === "running";
}

/** Settled work that still wants a decision: a failed print, a failed or
 *  interrupted (resumable) sequence. `completed` is not news the composer
 *  needs to carry, and a `cancelled` job was the user's own call. */
export function needsAttention(vm: ActivityJobVM): boolean {
  return vm.kind === "print"
    ? vm.phase === "failed"
    : vm.state === "failed" || vm.state === "interrupted";
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

export interface ActivityPartition {
  /** Queued / running prints and sequences, in merge order. */
  active: ActivityJobVM[];
  /** Settled-but-wrong rows the strip still shows, newest first. */
  attention: ActivityJobVM[];
  /** Settled sequences the strip is deliberately not showing. */
  settledSequences: number;
  /** Attention-eligible rows dropped by the cap, counted rather than lost. */
  hiddenAttention: number;
}

export interface PartitionActivityOptions {
  nowMs?: number;
  /** Session-only, client-side dismissals keyed by `ActivityJobVM.key`. */
  dismissed?: ReadonlySet<string> | readonly string[];
  settledVisibleMs?: number;
  maxAttentionRows?: number;
}

/**
 * Split a merged activity list into what the Create strip renders and what it
 * only counts. "Activity is present tense": in-flight work, plus a capped and
 * expiring set of settled-but-wrong rows, plus a digest of everything else.
 *
 * Settled prints are never counted — every settled print is already a Library
 * row and the freshest ones are in the sidebar's Now developing window, so a
 * count would be double bookkeeping.
 */
export function partitionActivity(
  rows: readonly ActivityJobVM[],
  options: PartitionActivityOptions = {},
): ActivityPartition {
  const nowMs = options.nowMs ?? Date.now();
  const settledVisibleMs = options.settledVisibleMs ?? SETTLED_VISIBLE_MS;
  const maxAttentionRows = options.maxAttentionRows ?? MAX_ATTENTION_ROWS;
  const dismissed =
    options.dismissed instanceof Set
      ? options.dismissed
      : new Set(options.dismissed ?? []);

  const active: ActivityJobVM[] = [];
  const eligible: ActivityJobVM[] = [];
  let settledSequences = 0;

  for (const vm of rows) {
    if (isActive(vm)) {
      active.push(vm);
      continue;
    }
    // A missing stamp means "just settled" rather than "ancient": hiding a
    // failure we can't date would lose its only pointer.
    const fresh =
      vm.settledAtMs === null || nowMs - vm.settledAtMs < settledVisibleMs;
    if (needsAttention(vm) && fresh) {
      // A dismissal is a decision, not a deferral — the row leaves the strip
      // without reappearing as a count. The durable job is untouched and
      // still listed in Library ▸ History ▸ Sequences.
      if (!dismissed.has(vm.key)) eligible.push(vm);
      continue;
    }
    if (vm.kind === "sequence") settledSequences += 1;
  }

  eligible.sort(
    (a, b) =>
      (b.settledAtMs ?? b.createdAtMs) - (a.settledAtMs ?? a.createdAtMs),
  );
  const attention = eligible.slice(0, Math.max(0, maxAttentionRows));
  return {
    active,
    attention,
    settledSequences,
    hiddenAttention: eligible.length - attention.length,
  };
}

/** The one mono chip at the end of the strip header, or null for silence. */
export function activityDigestLabel(partition: {
  settledSequences: number;
  hiddenAttention: number;
}): string | null {
  const parts: string[] = [];
  if (partition.hiddenAttention > 0)
    parts.push(`${partition.hiddenAttention} failed`);
  if (partition.settledSequences > 0) {
    const n = partition.settledSequences;
    parts.push(`${n} settled sequence${n === 1 ? "" : "s"}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}
