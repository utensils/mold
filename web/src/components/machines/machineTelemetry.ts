/*
 * Pure telemetry mapping for the host detail view. Prefers the richer
 * `/api/resources` snapshot (backend + GPU utilization + byte-accurate VRAM)
 * and falls back to `/api/status` fields. Every metric the host does not
 * expose (notably GPU temperature, which the resource aggregator omits)
 * renders as an em dash per spec §08 G4 rather than a fabricated zero.
 */
import { unifiedMemoryHost } from "@studio/lib/telemetryMemory";
import type { ResourceSnapshot, ServerStatus } from "../../types";
import { formatGB } from "../../util/format";

export const DASH = "—";

/** GB, one decimal, from a byte count. */
export function formatGb(bytes: number): string {
  return formatGB(bytes).replace(" GB", "");
}

/** Compact uptime, e.g. "3d 4h", "4h 12m", "12m", "45s". */
export function formatUptime(secs: number): string {
  if (!Number.isFinite(secs) || secs < 0) return DASH;
  const d = Math.floor(secs / 86_400);
  const h = Math.floor((secs % 86_400) / 3_600);
  const m = Math.floor((secs % 3_600) / 60);
  const s = Math.floor(secs % 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${s}s`;
}

export interface TelemetryView {
  gpuLine: string;
  loadPct: number | null;
  loadLabel: string;
  memPct: number | null;
  memLabel: string;
  cpuPct: number | null;
  cpuLabel: string;
  ramPct: number | null;
  ramLabel: string;
  /** Apple Metal shares one physical pool — GPU memory IS system RAM, so
   *  the panel renders one Memory row instead of duplicating the numbers. */
  unifiedMemory: boolean;
  queue: string;
  uptime: string;
  storageLabel: string | null;
}

export interface HostCardGpuSummary {
  label: string;
  used: number;
  total: number;
}

export function deriveHostCardGpu(
  status: ServerStatus | null,
): HostCardGpuSummary | null {
  const gpus = status?.gpus ?? [];
  if (gpus.length > 0) {
    const counts = new Map<string, number>();
    for (const gpu of gpus) {
      counts.set(gpu.name, (counts.get(gpu.name) ?? 0) + 1);
    }
    const label = [...counts.entries()]
      .map(([name, count]) => (count > 1 ? `${count}× ${name}` : name))
      .join(" + ");
    return {
      label,
      used: gpus.reduce((sum, gpu) => sum + gpu.vram_used_bytes, 0),
      total: gpus.reduce((sum, gpu) => sum + gpu.vram_total_bytes, 0),
    };
  }

  const info = status?.gpu_info;
  return info
    ? {
        label: info.name,
        used: info.vram_used_mb * 1_000_000,
        total: info.vram_total_mb * 1_000_000,
      }
    : null;
}

function pct(used: number, total: number): number | null {
  if (!total || total <= 0) return null;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

export function deriveTelemetry(
  status: ServerStatus | null,
  resources: ResourceSnapshot | null,
): TelemetryView {
  const rGpus = resources?.gpus ?? [];
  const sGpus = status?.gpus ?? [];
  const info = status?.gpu_info ?? null;

  // GPU name + backend line.
  const gpuLine = rGpus.length
    ? rGpus
        .map((gpu) =>
          rGpus.length > 1
            ? `GPU ${gpu.ordinal}: ${gpu.name} · ${gpu.backend}`
            : `${gpu.name} · ${gpu.backend}`,
        )
        .join(" | ")
    : sGpus.length
      ? sGpus
          .map((gpu) =>
            sGpus.length > 1 ? `GPU ${gpu.ordinal}: ${gpu.name}` : gpu.name,
          )
          .join(" | ")
      : (info?.name ?? DASH);

  // GPU load (utilization is null on Metal / nvidia-smi fallback).
  const loads = rGpus
    .map((gpu) => gpu.gpu_utilization)
    .filter((value): value is number => value != null);
  const loadPct = loads.length
    ? Math.min(
        100,
        Math.max(
          0,
          loads.reduce((sum, value) => sum + value, 0) / loads.length,
        ),
      )
    : null;
  const loadLabel = loadPct != null ? `${Math.round(loadPct)}%` : DASH;

  // VRAM — resources (bytes) → status.gpus (bytes) → gpu_info (MB).
  let memUsed: number | null = null;
  let memTotal: number | null = null;
  if (rGpus.length) {
    memUsed = rGpus.reduce((sum, gpu) => sum + gpu.vram_used, 0);
    memTotal = rGpus.reduce((sum, gpu) => sum + gpu.vram_total, 0);
  } else if (sGpus.length) {
    memUsed = sGpus.reduce((sum, gpu) => sum + gpu.vram_used_bytes, 0);
    memTotal = sGpus.reduce((sum, gpu) => sum + gpu.vram_total_bytes, 0);
  } else if (info) {
    memUsed = info.vram_used_mb * 1_000_000;
    memTotal = info.vram_total_mb * 1_000_000;
  }
  const memPct =
    memUsed != null && memTotal != null ? pct(memUsed, memTotal) : null;
  const memLabel =
    memUsed != null && memTotal != null
      ? `${formatGb(memUsed)} / ${formatGb(memTotal)} GB`
      : DASH;

  const queueDepth = status?.queue_depth;
  const queue = queueDepth != null ? String(queueDepth) : DASH;

  const uptime =
    status?.uptime_secs != null ? formatUptime(status.uptime_secs) : DASH;

  const disk = status?.models_disk ?? null;
  const storageLabel = disk
    ? `${formatGb(disk.free_bytes)} GB free of ${formatGb(disk.total_bytes)} GB`
    : null;

  const unifiedMemory = unifiedMemoryHost(rGpus.length ? rGpus : sGpus);

  const cpuPct = resources?.cpu?.usage_percent ?? null;
  const cpuLabel = cpuPct != null ? `${Math.round(cpuPct)}%` : DASH;
  const ram = resources?.system_ram;
  const ramPct = ram ? pct(ram.used, ram.total) : null;
  const ramLabel = ram
    ? `${formatGb(ram.used)} / ${formatGb(ram.total)} GB`
    : DASH;

  return {
    gpuLine,
    loadPct,
    loadLabel,
    memPct,
    memLabel,
    cpuPct,
    cpuLabel,
    ramPct,
    ramLabel,
    unifiedMemory,
    queue,
    uptime,
    storageLabel,
  };
}
