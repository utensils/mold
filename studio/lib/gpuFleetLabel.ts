/**
 * What the cards in a box are called, in one line.
 *
 * `GpuSnapshot` here is STRUCTURAL and deliberately minimal: a name and,
 * optionally, the backend the host reported. Desktop's richer
 * `GpuSnapshot` (ordinals, VRAM bytes, utilization) satisfies it, and so does
 * anything web builds from `/api/status` — `studio/` is the lower layer and
 * never imports a shell's wire types.
 */
export interface GpuSnapshot {
  name: string;
  backend?: string | null;
}

/** "4× L40S" for a uniform box, "RTX 4090 + B200" for a mixed one, "" for none. */
export function gpuFleetLabel(gpus: readonly GpuSnapshot[]): string {
  if (!gpus.length) return "";
  const names = [...new Set(gpus.map((gpu) => gpu.name))];
  return names.length === 1 && gpus.length > 1
    ? `${gpus.length}× ${names[0]}`
    : names.join(" + ");
}
