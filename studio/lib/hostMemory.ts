/**
 * Host-RAM pressure, shared by web, desktop, and iPhone.
 *
 * Servers report `host_memory` on the queue plan (and mirror it on
 * `/api/status` for non-V2 hosts) as the ledger's own view: what physically
 * exists, what is spendable right now, and the reserve the scheduler refuses
 * to dip into. Clients turn that into exactly one level so a RAM meter reads
 * the same everywhere. It is additive — an older server omits it and every
 * consumer must keep today's uncolored look rather than invent a number.
 */

export interface HostMemorySnapshot {
  total_bytes: number;
  available_bytes: number;
  /** Spendable bytes after the scheduler's outstanding reservations. */
  headroom_bytes: number;
  /** Reserve the scheduler will not admit work into. */
  safety_floor_bytes: number;
  /**
   * Evictable ZFS ARC the server counted into `headroom_bytes` (#1439).
   * Present only on a ZFS host with the credit enabled; `0` is a cold cache
   * and absence is no ZFS, an older server, or `MOLD_HOST_RAM_ZFS_ARC=0`.
   * `available_bytes` stays `MemAvailable` — the credit rides beside it.
   */
  reclaimable_zfs_arc_bytes?: number;
}

/** Mirrors the VRAM meter's vocabulary; `null` means "the host did not say". */
export type HostMemoryLevel = "ok" | "warn" | "critical";

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Read an additive `host_memory` object. Anything malformed or partial reads
 * as absent — a half-populated meter is worse than no meter.
 */
export function parseHostMemory(value: unknown): HostMemorySnapshot | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const row = value as Record<string, unknown>;
  const total = finite(row.total_bytes);
  const available = finite(row.available_bytes);
  const headroom = finite(row.headroom_bytes);
  const floor = finite(row.safety_floor_bytes);
  if (
    total === null ||
    available === null ||
    headroom === null ||
    floor === null
  ) {
    return null;
  }
  const snapshot: HostMemorySnapshot = {
    total_bytes: total,
    available_bytes: available,
    headroom_bytes: headroom,
    safety_floor_bytes: floor,
  };
  // Additive: a malformed credit reads as absent rather than spoiling the
  // four fields every meter depends on.
  const arc = finite(row.reclaimable_zfs_arc_bytes);
  if (arc !== null) snapshot.reclaimable_zfs_arc_bytes = arc;
  return snapshot;
}

/**
 * The one sentence every surface prints for what the host can still
 * schedule — `"41.3 GB available to schedule"`, optionally `"… of 62.5 GB"` —
 * naming the evictable ZFS ARC that headroom already includes whenever the
 * credit is positive, so the figure a user reads says what it counts.
 */
export function hostMemoryScheduleLabel(
  hostMemory: HostMemorySnapshot,
  formatBytes: (bytes: number) => string,
  options: { withTotal?: boolean } = {},
): string {
  const base = options.withTotal
    ? `${formatBytes(hostMemory.headroom_bytes)} of ${formatBytes(hostMemory.total_bytes)} available to schedule`
    : `${formatBytes(hostMemory.headroom_bytes)} available to schedule`;
  const arc = finite(hostMemory.reclaimable_zfs_arc_bytes);
  if (arc === null || arc <= 0) return base;
  return `${base} (includes ${formatBytes(arc)} evictable ZFS ARC)`;
}

/**
 * Pressure level from the ledger's own headroom, NOT from used/total: a host
 * whose RAM is mostly committed to reservations that have not allocated yet
 * looks idle to `free`, and that gap is what stalls a queue.
 *
 * - no headroom left → `critical` (nothing more can be admitted)
 * - less than one safety floor of headroom → `warn`
 * - otherwise → `ok`
 */
export function hostMemoryLevel(
  hostMemory: HostMemorySnapshot | null | undefined,
): HostMemoryLevel | null {
  if (!hostMemory) return null;
  const headroom = finite(hostMemory.headroom_bytes);
  const floor = finite(hostMemory.safety_floor_bytes);
  if (headroom === null || floor === null) return null;
  if (headroom <= 0) return "critical";
  if (floor > 0 && headroom < floor) return "warn";
  return "ok";
}
