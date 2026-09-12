/** Compact uptime for host telemetry: 42 → "42s", 300 → "5m", 5400 → "1h 30m", 200_000 → "2d 7h". */
export function formatUptime(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}
