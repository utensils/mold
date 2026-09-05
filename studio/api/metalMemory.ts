/** Optional telemetry: an older or malformed extension never hides its device. */
export interface MetalMemory {
  wired_limit:
    | { mode: "automatic" | "unsupported" | "unavailable" }
    | { mode: "explicit"; mib: number };
  physical_bytes: number | null;
  available_host_bytes: number | null;
  recommended_bytes: number | null;
  allocated_bytes: number | null;
  effective_capacity_bytes: number | null;
  allocation_headroom_bytes: number | null;
  error: string | null;
}

export function parseMetalMemory(value: unknown): MetalMemory | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return;
  const sample = value as Record<string, unknown>;
  const limit = sample.wired_limit as Record<string, unknown> | null;
  if (!limit || typeof limit !== "object" || Array.isArray(limit)) return;
  if (
    !["automatic", "explicit", "unsupported", "unavailable"].includes(
      String(limit.mode),
    )
  )
    return;
  if (
    limit.mode === "explicit" &&
    (typeof limit.mib !== "number" ||
      !Number.isInteger(limit.mib) ||
      limit.mib <= 0 ||
      limit.mib > 0xffffffff)
  )
    return;
  for (const key of [
    "physical_bytes",
    "available_host_bytes",
    "recommended_bytes",
    "allocated_bytes",
    "effective_capacity_bytes",
    "allocation_headroom_bytes",
  ]) {
    const bytes = sample[key];
    if (
      bytes !== null &&
      (typeof bytes !== "number" || !Number.isSafeInteger(bytes) || bytes < 0)
    )
      return;
  }
  if (sample.error !== null && typeof sample.error !== "string") return;
  return sample as unknown as MetalMemory;
}

export function metalBytes(value: number | null): string {
  return value === null
    ? "unavailable"
    : `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

export function metalLimitLabel(memory: MetalMemory): string {
  return memory.wired_limit.mode === "explicit"
    ? `${memory.wired_limit.mib} MiB`
    : memory.wired_limit.mode;
}
