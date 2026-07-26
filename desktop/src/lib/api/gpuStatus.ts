import { inferBackendFromGpuName } from "../hosts";
import type { GpuSnapshot, ServerStatus } from "./types";

const BYTES_PER_DECIMAL_MB = 1_000_000;

/** Normalize additive `/api/status.gpus` rows for telemetry UIs. */
export function gpuSnapshotsFromStatus(
  status: ServerStatus | null | undefined,
): GpuSnapshot[] {
  if (!status) return [];
  if (status.gpus?.length) {
    const backend =
      status.gpu_info?.backend ??
      inferBackendFromGpuName(status.gpu_info?.name ?? status.gpus[0]!.name);
    return status.gpus.map((gpu) => ({
      ordinal: gpu.ordinal,
      name: gpu.name,
      backend,
      vram_total: gpu.vram_total_bytes,
      vram_used: gpu.vram_used_bytes,
      gpu_utilization: null,
    }));
  }
  const gpu = status.gpu_info;
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
