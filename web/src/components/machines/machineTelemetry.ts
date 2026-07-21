/*
 * Pure telemetry mapping for the host detail view. Prefers the richer
 * `/api/resources` snapshot (backend + GPU utilization + byte-accurate VRAM)
 * and falls back to `/api/status` fields. Every metric the host does not
 * expose (notably GPU temperature, which the resource aggregator omits)
 * renders as an em dash per spec §08 G4 rather than a fabricated zero.
 */
import type { ResourceSnapshot } from "../../types";
import type { HostStatus } from "./hostClient";

export const DASH = "—";

/** GB, one decimal, from a byte count. */
export function formatGb(bytes: number): string {
  return (bytes / 1_000_000_000).toFixed(1);
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
  temp: string;
  queue: string;
  uptime: string;
  storageLabel: string | null;
}

function pct(used: number, total: number): number | null {
  if (!total || total <= 0) return null;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

export function deriveTelemetry(
  status: HostStatus | null,
  resources: ResourceSnapshot | null,
): TelemetryView {
  const rGpu = resources?.gpus?.[0] ?? null;
  const sGpu = status?.gpus?.[0] ?? null;
  const info = status?.gpu_info ?? null;

  // GPU name + backend line.
  const name = rGpu?.name ?? sGpu?.name ?? info?.name ?? null;
  const backend = rGpu?.backend ?? null;
  const gpuLine = name ? (backend ? `${name} · ${backend}` : name) : DASH;

  // GPU load (utilization is null on Metal / nvidia-smi fallback).
  const loadPct =
    rGpu?.gpu_utilization != null
      ? Math.min(100, Math.max(0, rGpu.gpu_utilization))
      : null;
  const loadLabel = loadPct != null ? `${Math.round(loadPct)}%` : DASH;

  // VRAM — resources (bytes) → status.gpus (bytes) → gpu_info (MB).
  let memUsed: number | null = null;
  let memTotal: number | null = null;
  if (rGpu) {
    memUsed = rGpu.vram_used;
    memTotal = rGpu.vram_total;
  } else if (sGpu) {
    memUsed = sGpu.vram_used_bytes;
    memTotal = sGpu.vram_total_bytes;
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

  return {
    gpuLine,
    loadPct,
    loadLabel,
    memPct,
    memLabel,
    temp: DASH,
    queue,
    uptime,
    storageLabel,
  };
}
