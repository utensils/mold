/**
 * Compact relative timestamps for MRU lists — "just now", "5m ago", "3d ago".
 *
 * Shared by shell lists through the domain layer.
 */
export function timeAgo(thenMs: number, nowMs: number = Date.now()): string {
  const seconds = Math.max(0, Math.floor((nowMs - thenMs) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(thenMs).toLocaleDateString();
}
