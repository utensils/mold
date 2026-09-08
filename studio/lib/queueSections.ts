import { isBenignQueueReason } from "./queuePosition";
/** Presentation only; lifecycle and mutation authority remain with the host. */
export type QueueSection = "making" | "waiting" | "attention";
export const QUEUE_SECTIONS: ReadonlyArray<{
  id: QueueSection;
  label: string;
}> = [
  { id: "making", label: "Being made" },
  { id: "waiting", label: "Waiting" },
  { id: "attention", label: "Needs attention" },
];
export function queueSection(
  phase: string | null,
  stale = false,
  blocked: boolean | string = false,
): QueueSection {
  const actionable =
    typeof blocked === "string" ? !isBenignQueueReason(blocked) : blocked;
  if (
    stale ||
    actionable ||
    ["held", "blocked", "failed", "error", "paused"].includes(phase ?? "")
  )
    return "attention";
  if (["queued", "accepted", "waiting", "pending"].includes(phase ?? ""))
    return "waiting";
  return "making";
}
