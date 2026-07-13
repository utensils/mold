/**
 * Pure multi-host helpers. `hostIdFromUrl` and `normalizeHostUrl` are TS
 * twins of `connection.rs::host_id` / `normalize_host_url` — per-host secret
 * names (`remote-api-key.<id>`) are derived on both sides and must agree.
 */

export function hostIdFromUrl(url: string): string {
  const stripped = url.replace(/^https?:\/\//, "");
  return stripped
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Default scheme/port, strip trailing slashes. Throws on garbage input. */
export function normalizeHostUrl(input: string): string {
  const trimmed = input.trim().replace(/\/+$/, "");
  if (!trimmed) throw new Error("Enter a host, like hal9000");
  const hasScheme = trimmed.startsWith("http://") || trimmed.startsWith("https://");
  const withScheme = hasScheme ? trimmed : `http://${trimmed}`;
  const url = new URL(withScheme);
  if (!url.hostname) throw new Error("Enter a valid host.");
  if (!hasScheme && !url.port) url.port = "7680";
  return url.toString().replace(/\/+$/, "");
}

/** The slice of a host the Auto router needs. */
export interface RoutableHost {
  id: string;
  kind: "local" | "remote";
  status: "connecting" | "ready" | "error";
  /** Live queue depth; null while unknown. */
  queueDepth: number | null;
}

/**
 * Auto routing: the ready host with the shallowest queue wins; unknown depth
 * counts as busiest; the local host wins ties (no network hop). Null when
 * nothing is ready.
 */
export function pickAutoHost<T extends RoutableHost>(hosts: T[]): T | null {
  const ready = hosts.filter((h) => h.status === "ready");
  if (ready.length === 0) return null;
  return ready.reduce((best, h) => {
    const depth = (x: RoutableHost) => x.queueDepth ?? Number.MAX_SAFE_INTEGER;
    if (depth(h) < depth(best)) return h;
    if (depth(h) === depth(best) && h.kind === "local" && best.kind !== "local") return h;
    return best;
  });
}

/**
 * Which host the status bar should mirror: the host of the most recently
 * submitted still-live job (the bar follows the action), else the primary.
 * Jobs without a routed host (single-host submissions) run on the primary.
 */
export function pickDisplayHost(
  liveJobHostIds: ReadonlyArray<string | null>,
  primaryId: string,
): string {
  for (let i = liveJobHostIds.length - 1; i >= 0; i--) {
    const id = liveJobHostIds[i];
    if (id) return id;
  }
  return primaryId;
}
