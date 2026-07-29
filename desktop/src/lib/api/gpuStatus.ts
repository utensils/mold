import { inferBackendFromGpuName } from "../hosts";
import type { GpuInfo, GpuSnapshot, GpuWorkerStatus, ServerStatus } from "./types";

const BYTES_PER_DECIMAL_MB = 1_000_000;

export function bestGpuBackend(gpus: ReadonlyArray<{ backend?: string | null }>): string | null {
  let best: string | null = null;
  let bestRank = -1;
  for (const gpu of gpus) {
    const backend = gpu.backend ?? null;
    const rank = backend === "cuda" ? 2 : backend === "metal" ? 1 : 0;
    if (backend && rank > bestRank) {
      best = backend;
      bestRank = rank;
    }
  }
  return best;
}

/** Normalize additive `/api/status.gpus` rows for telemetry UIs. */
export function gpuSnapshotsFromWorkers(
  info: GpuInfo | null | undefined,
  workers: GpuWorkerStatus[] | null | undefined,
): GpuSnapshot[] {
  if (workers?.length) {
    const backend = info?.backend ?? inferBackendFromGpuName(info?.name ?? workers[0]!.name);
    return workers.map((gpu) => ({
      ordinal: gpu.ordinal,
      name: gpu.name,
      backend,
      vram_total: gpu.vram_total_bytes,
      vram_used: gpu.vram_used_bytes,
      gpu_utilization: null,
    }));
  }
  const gpu = info;
  if (!gpu) return [];
  return [
    {
      ordinal: 0,
      name: gpu.name,
      backend: gpu.backend ?? inferBackendFromGpuName(gpu.name),
      vram_total: gpu.vram_total_mb * BYTES_PER_DECIMAL_MB,
      vram_used: gpu.vram_used_mb * BYTES_PER_DECIMAL_MB,
      gpu_utilization: null,
    },
  ];
}

export function gpuSnapshotsFromStatus(status: ServerStatus | null | undefined): GpuSnapshot[] {
  return status ? gpuSnapshotsFromWorkers(status.gpu_info, status.gpus) : [];
}

export function gpuFleetLabel(gpus: readonly GpuSnapshot[]): string {
  if (!gpus.length) return "";
  const names = [...new Set(gpus.map((gpu) => gpu.name))];
  return names.length === 1 && gpus.length > 1 ? `${gpus.length}× ${names[0]}` : names.join(" + ");
}

/** Aggregate memory for compact cards while preserving per-device rows elsewhere. */
export function summarizeStatusGpuMemory(
  status: ServerStatus | null | undefined,
): { usedMb: number; totalMb: number } | null {
  const gpus = gpuSnapshotsFromStatus(status);
  if (!gpus.length) return null;
  return gpus.reduce(
    (total, gpu) => ({
      usedMb: total.usedMb + gpu.vram_used / BYTES_PER_DECIMAL_MB,
      totalMb: total.totalMb + gpu.vram_total / BYTES_PER_DECIMAL_MB,
    }),
    { usedMb: 0, totalMb: 0 },
  );
}
