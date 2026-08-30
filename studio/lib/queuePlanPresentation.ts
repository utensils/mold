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
