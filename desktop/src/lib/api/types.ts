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

// ── Models ───────────────────────────────────────────────────────────────

/** Flattened ModelInfoExtended (ModelInfo + ModelDefaults + extras). */
export interface ModelEntry {
  name: string;
  family: string;
  size_gb: number;
  is_loaded: boolean;
  last_used?: number | null;
  hf_repo: string;
  default_steps: number;
  default_guidance: number;
  default_width: number;
  default_height: number;
  description: string;
  downloaded: boolean;
  disk_usage_bytes?: number | null;
  remaining_download_bytes?: number | null;
}

// ── Generation ───────────────────────────────────────────────────────────

export type OutputFormat = "png" | "jpeg" | "webp" | "gif" | "apng" | "mp4";

/** Subset of mold-core GenerateRequest the desktop sends today (grows in M3+). */
export interface GenerateRequest {
  prompt: string;
  negative_prompt?: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance?: number;
  seed?: number;
  batch_size?: number;
  output_format?: OutputFormat;
}

/** serde tag = "type", snake_case — /api/generate/stream `progress` events. */
export type ProgressEvent =
  | { type: "queued"; position: number; id?: string }
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "info"; message: string }
  | { type: "cache_hit"; resource: string }
  | { type: "denoise_step"; step: number; total: number; elapsed_ms: number }
  | {
      type: "weight_load";
      bytes_loaded: number;
      bytes_total: number;
      component: string;
    }
  | {
      type: "download_progress";
      filename: string;
      file_index: number;
      total_files: number;
      bytes_downloaded: number;
      bytes_total: number;
      batch_bytes_downloaded: number;
      batch_bytes_total: number;
      batch_elapsed_ms: number;
    }
  | {
      type: "download_done";
      filename: string;
      file_index: number;
      total_files: number;
      batch_bytes_downloaded: number;
      batch_bytes_total: number;
      batch_elapsed_ms: number;
    }
  | { type: "pull_complete"; model: string };

export interface CompleteEvent {
  /** Base64 payload — image bytes for images, video bytes for video. */
  image: string;
  format: OutputFormat;
  width: number;
  height: number;
  seed_used: number;
  generation_time_ms: number;
  model: string;
  video_frames?: number | null;
  video_fps?: number | null;
}

// ── Gallery ──────────────────────────────────────────────────────────────

export interface OutputMetadata {
  prompt: string;
  negative_prompt?: string | null;
  original_prompt?: string | null;
  model: string;
  seed: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  strength?: number | null;
  scheduler?: string | null;
  output_format?: OutputFormat | null;
  lora?: string | null;
  lora_scale?: number | null;
  upscale_model?: string | null;
  enable_audio?: boolean | null;
  video_frames?: number | null;
  video_fps?: number | null;
}

export interface GalleryImage {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  metadata_synthetic?: boolean;
}
