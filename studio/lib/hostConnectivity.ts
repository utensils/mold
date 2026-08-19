/*
 * Host reachability transitions, shared by web and desktop (spec §08 G11).
 *
 * Both surfaces poll every listed host on a timer, so an unreachable machine
 * reconnects on its own the moment it answers again — nothing about that is a
 * user action. These helpers are the one policy for narrating it: a host that
 * drops raises a WARNING (it is retrying, the entry stays listed), and one that
 * comes back raises a SUCCESS. Both fire exactly once per edge.
 *
 * "connecting" is a probe, never evidence of reachability, so the snapshot
 * carries the last settled status forward — otherwise the manual Retry path
 * (error → connecting → ready) would silently swallow its own recovery.
 */

export interface HostStatusSnapshot {
  id: string;
  label: string;
  /** "ready" | "connecting" | "error" — surfaces widen this, so keep it loose. */
  status: string;
}

/** Sticky supporting copy for the offline notice: the retry is automatic. */
export const HOST_OFFLINE_DESCRIPTION =
  "Retrying automatically — it stays listed for reconnect.";

/** Inline status shown on an offline machine's card while the poll retries. */
export const HOST_RECONNECTING_LABEL = "reconnecting…";

export function hostOfflineTitle(label: string): string {
  return `Can't reach ${label}`;
}

export function hostReconnectedTitle(label: string): string {
  return `Reconnected to ${label}`;
}

/**
 * Snapshot the current statuses so the next poll can diff against them.
 * `previous` (when supplied) keeps a host's last settled status while it is
 * merely "connecting".
 */
export function snapshotHostStatuses(
  current: readonly HostStatusSnapshot[],
  previous: Readonly<Record<string, string>> = {},
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const host of current) {
    const settled = previous[host.id];
    out[host.id] =
      host.status === "connecting" && settled ? settled : host.status;
  }
  return out;
}

/** Hosts that just went reachable → offline (ready → error). */
export function detectOfflineTransitions(
  previous: Readonly<Record<string, string>>,
  current: readonly HostStatusSnapshot[],
): HostStatusSnapshot[] {
  return current.filter(
    (host) => host.status === "error" && previous[host.id] === "ready",
  );
}

/**
 * Hosts that just came back (error → ready). A first successful connection is
 * not a recovery — only a host we already reported unreachable qualifies.
 */
export function detectReconnectTransitions(
  previous: Readonly<Record<string, string>>,
  current: readonly HostStatusSnapshot[],
): HostStatusSnapshot[] {
  return current.filter(
    (host) => host.status === "ready" && previous[host.id] === "error",
  );
}
