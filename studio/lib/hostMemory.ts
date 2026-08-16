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
  return {
    total_bytes: total,
    available_bytes: available,
    headroom_bytes: headroom,
    safety_floor_bytes: floor,
  };
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
