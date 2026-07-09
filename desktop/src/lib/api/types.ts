/**
 * Hand-mirrored subsets of mold-core wire types (serde snake_case).
 * Source of truth: /api/openapi.json — a drift snapshot test guards these
 * once the OpenAPI harness lands (see desktop/docs/architecture.md §5).
 */

export interface GpuSnapshot {
  ordinal: number;
  name: string;
  backend: "cuda" | "metal" | "cpu" | string;
  vram_total: number;
  vram_used: number;
  vram_used_by_mold?: number | null;
  gpu_utilization?: number | null;
}

export interface RamSnapshot {
  total: number;
  used: number;
  used_by_mold: number;
  used_by_other: number;
}

export interface ResourceSnapshot {
  hostname: string;
  timestamp: number;
  gpus: GpuSnapshot[];
  system_ram: RamSnapshot;
}

export interface ServerStatus {
  version: string;
  git_sha?: string | null;
  models_loaded: string[];
  busy?: boolean;
  uptime_secs: number;
  hostname?: string | null;
  queue_depth?: number | null;
  queue_capacity?: number | null;
}

export interface ServerCapabilities {
  gallery: { can_delete: boolean };
  catalog?: { available: boolean; families: string[] } | null;
}
