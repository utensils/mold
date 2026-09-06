/**
 * Activity view-model for the jobs surfaces — desktop's ActivityStrip, web's
 * Create strip, and the iPhone queue list.
 *
 * This used to be a discriminated union carrying durable sequence (chain) jobs
 * beside prints, because the apps authored sequences. They no longer do, so
 * there is exactly one kind of row again. A long clip the host renders as
 * chained clips is still ONE print carrying `chain: { stageIndex, stageCount }`
 * — a progress detail, not a second kind of work — and it must stay that way:
 * a sequence arm here is what made a single render read as two rows.
 */

import {
  queueStatusFor,
  queueWaitLabel,
  resolveQueueWait,
  type QueuePreparation,
  type QueueStatusIndex,
} from "./queuePosition";
import { compareNewestSubmitted } from "./activityOrder";

export type ActivityAction =
  "cancel" | "watch" | "retake" | "edit" | "resume" | "delete";

export type PrintPhase =
  "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

export interface ActivityJobVM {
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
  /** Wall clock at submit — NOT a client counter. */
  createdAtMs: number;
  /** Wall clock when the job settled; null while it is still in flight. */
  settledAtMs: number | null;
  error: string | null;
  /** Live 0-based dispatch order from the host's `/api/queue` listing, not
   *  the one-shot SSE `Queued` frame. Absent when the host has not been
   *  read or is too old to list the job. */
  queuePosition?: number | null;
  /** Effective waiting lifecycle after projecting host-wide pause state. */
  queueState?: string | null;
  /** Raw scheduler `blocked_reason` for this job, when the plan named one. */
  blockedReason?: string | null;
  /** Live dependency preparation detail from the scheduler plan. */
  preparation?: QueuePreparation | null;
}

/** How long a settled-but-wrong job keeps a row in the Create strip. */
export const SETTLED_VISIBLE_MS = 5 * 60_000;
/** Strip rows for wrong-but-settled work; the overflow becomes a digest count. */
export const MAX_ATTENTION_ROWS = 2;
/** Rows a History lens renders before the count footnote. */
export const HISTORY_JOBS_RENDER_CAP = 200;

/** Retained name for the surfaces that still spell the row out. */
export type PrintActivityVM = ActivityJobVM;

/**
 * Attach the host's live dispatch order to a print row.
 *
 * Positions belong to work that is still waiting: the moment a job starts
 * denoising (or settles) its slot number is history, and leaving it on the row
 * would contradict the present-tense rule the strip is built on (G15). A host
 * that has not been read contributes nothing rather than a guess.
 */
export function withLiveQueueStatus(
  vm: PrintActivityVM,
  index: QueueStatusIndex | null | undefined,
  serverJobId: string | null | undefined,
): PrintActivityVM {
  if (vm.phase !== "queued") return vm;
  const status = queueStatusFor(index, vm.hostId, serverJobId);
  if (!status) return vm;
  const next: PrintActivityVM = { ...vm };
  if (status.state !== null) next.queueState = status.state;
  if (status.position !== null) next.queuePosition = status.position;
  if (status.blockedReason !== null) next.blockedReason = status.blockedReason;
  if (status.preparation !== null) next.preparation = status.preparation;
  return next;
}

/**
 * The one line a waiting print shows, in every shell. A job the scheduler
 * genuinely parked says why — that is the answer the user is after — the head
 * of the line says "Next up", everyone behind it counts down, and a host that
 * lists nothing still says "Queued". Null means the row is not waiting at all
 * (running or settled), so the surface renders its own chrome.
 */
export function queueStatusLabel(vm: ActivityJobVM): string | null {
  if (vm.phase !== "queued") return null;
  return queueWaitLabel(
    resolveQueueWait({
      state: vm.queueState,
      position: vm.queuePosition,
      blockedReason: vm.blockedReason,
      preparation: vm.preparation,
    }),
  );
}

/** Running vs waiting, counted separately because they are not the same news. */
export interface ActivityCounts {
  running: number;
  waiting: number;
}

/**
 * The count beside a queue header. Queued work is NOT active work — calling
 * five rows "5 ACTIVE" when one GPU is rendering one of them is the header
 * half of the same lie the row labels told.
 */
export function activityCountLabel(counts: ActivityCounts): string {
  const parts: string[] = [];
  if (counts.running > 0) parts.push(`${counts.running} active`);
  if (counts.waiting > 0) parts.push(`${counts.waiting} queued`);
  return parts.length > 0 ? parts.join(" · ") : "0 active";
}

/** The same truth as a spoken sentence, for the live region. */
export function activityAnnouncement(counts: ActivityCounts): string {
  const { running, waiting } = counts;
  if (running === 0 && waiting === 0) return "No active generations.";
  if (running === 0) {
    return `${waiting} queued generation${waiting === 1 ? "" : "s"}.`;
  }
  const active = `${running} active generation${running === 1 ? "" : "s"}`;
  return waiting === 0 ? `${active}.` : `${active}, ${waiting} queued.`;
}

function isActive(vm: ActivityJobVM): boolean {
  return (
    vm.phase === "queued" || vm.phase === "running" || vm.phase === "paused"
  );
}

/** Settled work that still wants a decision: a failed print. `done` is not
 *  news the composer needs to carry, and a `cancelled` job was the user's own
 *  call. */
export function needsAttention(vm: ActivityJobVM): boolean {
  return vm.phase === "failed";
}

/** Order the rows the strip renders. Active work stays newest-first,
 * regardless of whether a row is queued or running; settled work follows by
 * recency and is partitioned into attention/history below. */
export function mergeActivity(
  prints: readonly ActivityJobVM[],
): ActivityJobVM[] {
  return [...prints].sort((a, b) => {
    const activeDelta = Number(isActive(b)) - Number(isActive(a));
    if (activeDelta !== 0) return activeDelta;
    if (isActive(a)) return compareNewestSubmitted(a, b);
    return b.createdAtMs - a.createdAtMs;
  });
}

export interface ActivityPartition {
  /** Queued / running prints, in merge order. */
  active: ActivityJobVM[];
  /** Settled-but-wrong rows the strip still shows, newest first. */
  attention: ActivityJobVM[];
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
      // without reappearing as a count.
      if (!dismissed.has(vm.key)) eligible.push(vm);
    }
  }

  eligible.sort(
    (a, b) =>
      (b.settledAtMs ?? b.createdAtMs) - (a.settledAtMs ?? a.createdAtMs),
  );
  const attention = eligible.slice(0, Math.max(0, maxAttentionRows));
  return {
    active,
    attention,
    hiddenAttention: eligible.length - attention.length,
  };
}

/** The one mono chip at the end of the strip header, or null for silence. */
export function activityDigestLabel(partition: {
  hiddenAttention: number;
}): string | null {
  return partition.hiddenAttention > 0
    ? `${partition.hiddenAttention} failed`
    : null;
}
