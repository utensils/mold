/**
 * The single authority for MEMORY readings — VRAM and RAM — across web,
 * desktop, and mobile.
 *
 * Memory is binary and storage is decimal, and mold needs both: a 24 GiB card
 * is sold, boxed, and reported by nvidia-smi as "24 GB", while a 22.2 GB
 * checkpoint is what Hugging Face bills and what a drive vendor's label
 * promises. `formatBytes` owns the decimal half for downloads and disk; this
 * module owns the binary half, and a memory figure must never be routed
 * through the other one.
 *
 * That split used to be implicit, and both halves spelled their unit "GB", so
 * one Machines screen showed the SAME 4090 as "1.8 / 25.8 GB" on the meter
 * (decimal) and "1.7 GB of 24.0 GB" on the compute-plan card below it
 * (binary). Neither number was wrong for its divisor; the screen was wrong for
 * having two.
 *
 * The label stays "GB" rather than the pedantically-correct "GiB" because
 * every place a person already reads these numbers — the card's box,
 * nvidia-smi, Task Manager, Activity Monitor — says GB for the binary value.
 * "GiB" would be the only surface in the user's life spelling it that way.
 */

const BYTES_PER_GIB = 1024 ** 3;

/** An unreadable metric renders as a dash, never as a fabricated zero. */
const DASH = "—";

function gib(bytes: number | null | undefined): string | null {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return null;
  return (bytes / BYTES_PER_GIB).toFixed(1);
}

/** A single memory reading: `24.0 GB`. */
export function formatMemoryGB(bytes: number | null | undefined): string {
  const value = gib(bytes);
  return value === null ? DASH : `${value} GB`;
}

/**
 * A used/total memory pair sharing one unit — `1.8 / 24.0 GB`. The unit is
 * named once, at the end, because the two halves are the same quantity; a
 * meter beside it already says what is being measured.
 */
export function formatMemoryGBPair(
  used: number | null | undefined,
  total: number | null | undefined,
): string {
  const left = gib(used);
  return `${left === null ? DASH : left} / ${formatMemoryGB(total)}`;
}
