/*
 * Host reachability, shared by web and desktop (spec §08 G11).
 *
 * Both surfaces poll every listed host on a timer, so an unreachable machine
 * reconnects on its own the moment it answers again — nothing about that is a
 * user action. `reconcileHostConnectivity` is the one policy for narrating it:
 * a host that drops raises a WARNING (it is retrying, the entry stays listed),
 * and one that comes back raises a SUCCESS.
 *
 * Two rules make it honest, and both are load-bearing:
 *
 * - A recovery is reported only for a host in the caller's `warned` set. The
 *   error → ready edge alone is not enough: desktop's boot probe errors
 *   quietly, so a machine that was merely asleep at launch would otherwise
 *   produce a green "Reconnected to …" for a drop nobody was ever told about.
 * - "connecting" is a probe, never evidence of reachability, so the snapshot
 *   carries the last settled status forward — otherwise the manual Retry path
 *   (error → connecting → ready) would silently swallow its own recovery.
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

export interface HostConnectivityInput {
  /** Last settled status per host id, from a previous `changes.next`. */
  previous: Readonly<Record<string, string>>;
  current: readonly HostStatusSnapshot[];
  /** Host ids currently carrying an offline notice that was never withdrawn. */
  warned: Iterable<string>;
  /**
   * Whether a host that has never been reachable counts as a drop. Web says
   * yes: a registered machine that does not answer on the first probe is news.
   * Desktop says no: its boot probe is deliberately quiet because the Machines
   * workspace already shows the errored row.
   */
  warnOnFirstContact?: boolean;
}

export interface HostConnectivityChanges {
  /** Raise an offline notice for these. */
  offline: HostStatusSnapshot[];
  /** Withdraw the notice and confirm recovery for these. */
  reconnected: HostStatusSnapshot[];
  /**
   * Warned ids that are gone from `current` — disconnected or forgotten while
   * offline. Nothing will ever poll them again, so their notice must be
   * retired with the entry rather than left to hang forever.
   */
  retired: string[];
  /** The snapshot to carry into the next pass. */
  next: Record<string, string>;
}

/**
 * The whole host-reachability policy in one place, for both surfaces.
 *
 * A recovery is reported only for a host we actually warned about — the edge
 * alone is not enough. Otherwise a machine that was already asleep at launch
 * (quietly errored, never announced) produces a green "Reconnected to …" for a
 * drop the user was never told about.
 */
export function reconcileHostConnectivity(
  input: HostConnectivityInput,
): HostConnectivityChanges {
  const warned = new Set(input.warned);
  const present = new Set(input.current.map((host) => host.id));
  const offline = input.current.filter((host) => {
    if (host.status !== "error" || warned.has(host.id)) return false;
    const before = input.previous[host.id];
    if (before === "ready") return true;
    return input.warnOnFirstContact === true && before === undefined;
  });
  return {
    offline,
    reconnected: input.current.filter(
      (host) => host.status === "ready" && warned.has(host.id),
    ),
    retired: [...warned].filter((id) => !present.has(id)),
    next: snapshotHostStatuses(input.current, input.previous),
  };
}
