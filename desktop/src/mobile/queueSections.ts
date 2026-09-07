/** Presentation only; lifecycle and mutation authority remain with the host. */
export type MobileQueueSection = "making" | "waiting" | "attention";
export const MOBILE_QUEUE_SECTIONS: ReadonlyArray<{ id: MobileQueueSection; label: string }> = [
  { id: "making", label: "Being made" },
  { id: "waiting", label: "Waiting" },
  { id: "attention", label: "Needs attention" },
];
export function mobileQueueSection(
  phase: string | null,
  stale = false,
  blocked = false,
): MobileQueueSection {
  if (stale || blocked || ["held", "blocked", "failed", "error", "paused"].includes(phase ?? ""))
    return "attention";
  if (["queued", "accepted", "waiting", "pending"].includes(phase ?? "")) return "waiting";
  return "making";
}
