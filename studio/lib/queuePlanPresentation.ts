import type { QueuePlan, QueueWorkItem } from "../api/queuePlan";
import { normalizeBlockedReason } from "./queuePosition";

/** Scheduler work that is not already represented by the durable queue list. */
export function queuePlanOnlyWork(
  plan: QueuePlan | null | undefined,
  excludeIds: readonly string[],
): QueueWorkItem[] {
  const excluded = new Set(excludeIds);
  return [...(plan?.work_items ?? [])]
    .filter((item) => {
      const rawReason = item.blocked_reason ?? item.reason;
      const reason = normalizeBlockedReason(rawReason);
      const preparing = rawReason === "preparing";
      return (
        !excluded.has(item.work_id) &&
        !excluded.has(item.parent_id) &&
        (item.activity_phase !== "blocked" || preparing) &&
        (reason === null || preparing)
      );
    })
    .sort(
      (a, b) =>
        a.queue_rank - b.queue_rank ||
        (a.lane_order ?? Number.MAX_SAFE_INTEGER) -
          (b.lane_order ?? Number.MAX_SAFE_INTEGER),
    );
}

export function queueScopeLabel(
  loadedCount: number,
  totalCount: number | null | undefined,
  available = true,
): string {
  if (!available) return "Queue details unavailable";
  const loaded = `${loadedCount} ${loadedCount === 1 ? "job" : "jobs"}`;
  if (totalCount == null) return `${loaded} loaded`;
  const total = Math.max(loadedCount, totalCount);
  return loadedCount < total
    ? `Showing ${loadedCount} of ${total} jobs`
    : `All ${loaded} loaded`;
}

export function queuePlanUpdateLabel(
  deadlineUnixMs: number | null | undefined,
  nowUnixMs: number,
): string | null {
  if (deadlineUnixMs == null) return null;
  const seconds = Math.ceil((deadlineUnixMs - nowUnixMs) / 1_000);
  return seconds > 1 ? `Updating order in ${seconds}s` : "Updating order…";
}

function approximateDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.ceil(milliseconds / 1_000));
  if (seconds < 45) return "under a minute";
  const minutes = Math.max(1, Math.round(seconds / 60));
  if (minutes < 60)
    return `about ${minutes} ${minutes === 1 ? "minute" : "minutes"}`;
  const hours = Math.round((minutes / 60) * 2) / 2;
  return `about ${hours} ${hours === 1 ? "hour" : "hours"}`;
}

export function queueCompletionLabel(
  finishUnixMs: number | null | undefined,
  confidence: string | null | undefined,
  nowUnixMs: number,
): string {
  if (finishUnixMs == null) return "Completion time is still being estimated";
  if (finishUnixMs <= nowUnixMs) return "Finishing now";
  const base = `Done in ${approximateDuration(finishUnixMs - nowUnixMs)}`;
  return confidence === "high" ? base : `${base} · estimate may change`;
}

export function queueLanePositionLabel(index: number): string {
  if (index === 0) return "Next";
  if (index === 1) return "After that";
  const position = index + 1;
  const remainder100 = position % 100;
  const suffix =
    remainder100 >= 11 && remainder100 <= 13
      ? "th"
      : position % 10 === 1
        ? "st"
        : position % 10 === 2
          ? "nd"
          : position % 10 === 3
            ? "rd"
            : "th";
  return `${position}${suffix} in this lane`;
}
