/**
 * Live queue position and blocked-reason copy, shared by web, desktop, and
 * iPhone.
 *
 * The SSE `Queued { position }` frame is a one-shot: it says where a job
 * landed at submit time and never again. `GET /api/queue` is the live view —
 * `entries[].position` is the 0-based dispatch order and it shifts as jobs
 * finish and on a PATCH reorder. Every surface resolves through here so a
 * queued row counts down instead of freezing at the number it was born with.
 */

import type { QueueEntry, QueuePlan, QueueWorkItem } from "../api/queuePlan";

export interface QueueStatus {
  /** The row's lifecycle as the host listed it (`queued`, `running`, `held`), or null. */
  state: string | null;
  /** Live 0-based dispatch order, or null when the host did not list the job. */
  position: number | null;
  /** Raw scheduler reason this work is parked, or null when it is simply waiting. */
  blockedReason: string | null;
  /** What a preparing job is working through, when its host reports it. */
  preparation: QueuePreparation | null;
}

/** Progress of a job the host is still preparing (weights, references, admission). */
export interface QueuePreparation {
  component: string | null;
  /** 0..1, or null when the preparer reported no byte total. */
  fraction: number | null;
  elapsedMs: number | null;
}

/** The plan's preparation report for a job's work item, if it carries one. */
function planPreparation(
  plan: QueuePlan | null | undefined,
  jobId: string,
): QueuePreparation | null {
  if (!plan) return null;
  const items: readonly QueueWorkItem[] = plan.work_items ?? [];
  for (const item of items) {
    if (item.parent_id !== jobId && item.work_id !== jobId) continue;
    if (item.blocked_reason !== "preparing") continue;
    const progress = item.preparation_progress ?? null;
    const total = progress?.bytes_total ?? 0;
    return {
      component: progress?.component || null,
      fraction:
        progress && total > 0
          ? Math.min(1, Math.max(0, progress.bytes_done / total))
          : null,
      elapsedMs:
        typeof item.preparation_elapsed_ms === "number" &&
        Number.isFinite(item.preparation_elapsed_ms)
          ? item.preparation_elapsed_ms
          : null,
    };
  }
  return null;
}

/** "Preparing", or "Preparing · Verifying MiniMax H3 artifacts 41%". */
export function preparationLabel(
  preparation: QueuePreparation | null | undefined,
): string {
  if (!preparation?.component) return "Preparing";
  const pct =
    preparation.fraction === null
      ? ""
      : ` ${Math.round(preparation.fraction * 100)}%`;
  return `Preparing · ${preparation.component}${pct}`;
}

/** One host's live queue read, in whatever shape the shell already holds. */
export interface QueueStatusSource {
  hostId: string;
  entries?: readonly QueueEntry[] | null | undefined;
  plan?: QueuePlan | null | undefined;
}

export type QueueStatusIndex = ReadonlyMap<string, QueueStatus>;

/** Job ids live in per-host id spaces, so the key always carries the host. */
export function queueStatusKey(hostId: string, jobId: string): string {
  return `${hostId}\u0000${jobId}`;
}

function finitePosition(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.floor(value)
    : null;
}

/**
 * Every value `QueueBlockedReason::as_str()` can produce
 * (`crates/mold-core/src/types.rs`). The wire also carries the legacy
 * `assignment_reason` alias in `QueueWorkItem.reason`, handled separately
 * below. Keep this list in step with the Rust enum: the copy table is typed
 * against it, so a new reason will not compile until it is classified.
 */
export const QUEUE_BLOCKED_REASONS = [
  "device_disabled",
  "device_draining",
  "device_startup_excluded",
  "device_unavailable",
  "device_degraded",
  "hard_pin_unavailable",
  "backend_unsupported",
  "model_not_installed",
  "insufficient_vram",
  "insufficient_host_ram",
  "aggregate_host_ram_reserved",
  "execution_plan_incompatible",
  "dependency_wait",
  "warm_wait",
  "queue_paused",
  "maintenance_mode",
  "cancelling",
  "no_schedulable_device",
  "no_idle_device",
  "lower_priority_opening",
  "preparing",
] as const;

export type QueueBlockedReasonId = (typeof QUEUE_BLOCKED_REASONS)[number];

/**
 * What each reason means to a person waiting on a print.
 *
 * `null` is the load-bearing half: those reasons are ordinary queue
 * bookkeeping, not faults. A one-GPU host reports `no_idle_device` for every
 * job behind the running one, `warm_wait` while it holds a slot for a warm
 * device, and `lower_priority_opening` when higher-priority work took the
 * opening this pass (`mold-scheduler/src/planner.rs`). Rendering those is how
 * four ordinary queued rows came to read "no idle device" instead of their
 * place in line. A `null` row falls through to its position.
 */
const BLOCKED_REASON_COPY: Record<QueueBlockedReasonId, string | null> = {
  device_disabled: "Device turned off",
  device_draining: "Device draining",
  device_startup_excluded: "Device excluded at startup",
  device_unavailable: "Waiting for a device",
  device_degraded: "Device degraded",
  hard_pin_unavailable: "Pinned device unavailable",
  backend_unsupported: "Not supported on this machine",
  model_not_installed: "Model not installed",
  insufficient_vram: "Waiting for GPU memory",
  insufficient_host_ram: "Waiting for memory",
  aggregate_host_ram_reserved: "Waiting for memory",
  execution_plan_incompatible: "Cannot run as planned",
  dependency_wait: null,
  warm_wait: null,
  queue_paused: "Queue paused",
  maintenance_mode: "Host in maintenance",
  cancelling: "Cancelling",
  no_schedulable_device: "No usable device",
  no_idle_device: null,
  lower_priority_opening: null,
  preparing: "Preparing",
};

/**
 * `QueueWorkItem.reason` is a display alias that may carry an
 * `AssignmentReason` rather than a blocking one. Those describe why work WON
 * a device, so they are never a reason to say anything at all.
 */
const ASSIGNMENT_REASONS = new Set([
  "priority",
  "starvation_forced",
  "warm_resident",
]);

/** Copy the whole fleet says for a reason nobody has taught it yet. */
const UNKNOWN_REASON_LABEL = "Waiting on the host";

function knownCopy(reason: string): string | null | undefined {
  // Not `Object.hasOwn`: the desktop app ships `minimumSystemVersion: 12.0`
  // and WebKit only gained it in Safari 15.4 (macOS 12.3), so on 12.0-12.2 the
  // first non-empty reason in a queue plan would throw and take the queue and
  // device panels down with it. Vite does not polyfill it.
  return Object.prototype.hasOwnProperty.call(BLOCKED_REASON_COPY, reason)
    ? BLOCKED_REASON_COPY[reason as QueueBlockedReasonId]
    : undefined;
}

/**
 * True when a reason is ordinary bookkeeping rather than something worth
 * saying. Same predicate the device panel filters on, so the two surfaces
 * never disagree about what counts as blocked.
 */
export function isBenignQueueReason(
  reason: string | null | undefined,
): boolean {
  if (!reason) return true;
  if (ASSIGNMENT_REASONS.has(reason)) return true;
  return knownCopy(reason) === null;
}

/** Scheduler reason → display text, or null when it is not worth surfacing. */
export function normalizeBlockedReason(
  reason: string | null | undefined,
): string | null {
  if (!reason || isBenignQueueReason(reason)) return null;
  return knownCopy(reason) ?? reason.replaceAll("_", " ");
}

/**
 * Short, plain-language copy for a queued row, or null when the reason is
 * ordinary bookkeeping and the row should keep counting its place in line.
 * A reason this build has never heard of still says something a person can
 * read — never the raw scheduler identifier.
 */
export function blockedReasonLabel(
  reason: string | null | undefined,
): string | null {
  if (isBenignQueueReason(reason)) return null;
  return knownCopy(reason as string) ?? UNKNOWN_REASON_LABEL;
}

/** The parked reason for a job's work items, if the plan named one. */
function planBlockedReason(
  plan: QueuePlan | null | undefined,
  jobId: string,
): string | null {
  if (!plan) return null;
  const items: readonly QueueWorkItem[] = plan.work_items ?? [];
  for (const item of items) {
    if (item.parent_id !== jobId && item.work_id !== jobId) continue;
    const label = normalizeBlockedReason(item.blocked_reason ?? item.reason);
    if (label !== null) return item.blocked_reason ?? item.reason ?? null;
  }
  return null;
}

/**
 * Fold every host's live queue read into one lookup. Hosts that have not been
 * read contribute nothing, which is exactly right: absence of an entry is
 * absence of evidence, never "position 0".
 */
export function buildQueueStatusIndex(
  sources: Iterable<QueueStatusSource>,
): QueueStatusIndex {
  const index = new Map<string, QueueStatus>();
  for (const source of sources) {
    if (!source || !source.hostId) continue;
    for (const entry of source.entries ?? []) {
      if (!entry || typeof entry.id !== "string" || entry.id.length === 0)
        continue;
      index.set(queueStatusKey(source.hostId, entry.id), {
        state: typeof entry.state === "string" ? entry.state : null,
        position: finitePosition(entry.position),
        blockedReason: planBlockedReason(source.plan, entry.id),
        preparation: planPreparation(source.plan, entry.id),
      });
    }
  }
  return index;
}

/** Live status for one job, or null when its host has not been read. */
export function queueStatusFor(
  index: QueueStatusIndex | null | undefined,
  hostId: string | null | undefined,
  jobId: string | null | undefined,
): QueueStatus | null {
  if (!index || !hostId || !jobId) return null;
  return index.get(queueStatusKey(hostId, jobId)) ?? null;
}

/**
 * Pill copy for a queued print. Position is 0-based, so slot 0 is the job
 * about to run and has no line to be in — the same threshold the Machines
 * queue panel already uses for `QUEUED #N`.
 */
export function queuePositionLabel(
  position: number | null | undefined,
): string | null {
  const value = finitePosition(position);
  if (value === null || value < 1) return null;
  return `#${value} in line`;
}

/**
 * What one waiting row is actually doing, decided once for web, desktop, and
 * iPhone. Surfaces format it in their own casing idiom; none of them decides
 * the vocabulary, which is how the same host came to describe four identical
 * queued jobs three different ways.
 */
export type QueueWaitStatus =
  /** Parked by the host: never dispatched on its own, so never "in line". */
  | { kind: "held" }
  /** An actionable reason outranks the position: say what to fix. */
  | { kind: "blocked"; label: string }
  /** Head of the line — running next, with nobody in front. */
  | { kind: "next" }
  /** 0-based dispatch order, so `position` is how many jobs are ahead. */
  | { kind: "position"; position: number }
  /** No evidence at all (older server, or the host was never read). */
  | { kind: "queued" };

export interface QueueWaitInput {
  /**
   * The row's lifecycle. A `held` row still carries a listing position, and
   * reading that position as a place in line is how a parked job came to
   * render as "Next up" on the phone — telling the operator to wait for work
   * the host will never start on its own.
   */
  state?: string | null | undefined;
  position?: number | null | undefined;
  blockedReason?: string | null | undefined;
  preparation?: QueuePreparation | null | undefined;
}

/** Resolve one waiting row. Absent evidence degrades to a plain "Queued". */
export function resolveQueueWait(
  input: QueueWaitInput | null | undefined,
): QueueWaitStatus {
  if (input?.state === "held") return { kind: "held" };
  if (input?.blockedReason === "preparing") {
    return { kind: "blocked", label: preparationLabel(input.preparation) };
  }
  const label = blockedReasonLabel(input?.blockedReason);
  if (label !== null) return { kind: "blocked", label };
  const position = finitePosition(input?.position);
  if (position === null) return { kind: "queued" };
  return position < 1 ? { kind: "next" } : { kind: "position", position };
}

/** Sentence-case copy — web and desktop pills, and the iPhone status line. */
export function queueWaitLabel(wait: QueueWaitStatus): string {
  switch (wait.kind) {
    case "held":
      return "Held";
    case "blocked":
      return wait.label;
    case "next":
      return "Next up";
    case "position":
      return queuePositionLabel(wait.position) ?? "Queued";
    case "queued":
      return "Queued";
  }
}

/** Compact uppercase code — the iPhone queue list's existing idiom. */
export function queueWaitCode(wait: QueueWaitStatus): string {
  switch (wait.kind) {
    case "held":
      return "HELD";
    case "blocked":
      return wait.label.toUpperCase();
    case "next":
      return "NEXT UP";
    case "position":
      return `QUEUED #${wait.position}`;
    case "queued":
      return "QUEUED";
  }
}
