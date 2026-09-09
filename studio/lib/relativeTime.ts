/**
 * Compact relative timestamps for MRU lists — "just now", "5m ago", "3d ago".
 *
 * It lives here rather than in a shell because both the desktop's own lists
 * and the shared 3-D Studio's Recent panel say the same thing, and `studio/`
 * is the layer both can read.
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
