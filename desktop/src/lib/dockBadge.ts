/**
 * Dock badge = prints that landed while this app was in the background, on
 * any connected machine, made by any client. It answers "what arrived while
 * you were away" and clears the moment the window comes back, the way a
 * messages badge does — not "how deep is this app's queue", which reads as
 * "my app is busy" and never clears while a long clip renders.
 * `null` clears the badge.
 */
export function dockBadgeValue(landedCount: number, enabled: boolean): number | null {
  if (!enabled || landedCount <= 0) return null;
  return landedCount;
}
