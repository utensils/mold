/**
 * Dock badge = THIS app's active jobs (queued + developing), not the whole
 * engine's queue depth. A shared or remote engine runs other clients' work —
 * badging those numbers reads as "my app is busy" when it isn't.
 * `null` clears the badge.
 */
export function dockBadgeValue(pendingJobs: number, enabled: boolean): number | null {
  if (!enabled || pendingJobs <= 0) return null;
  return pendingJobs;
}
