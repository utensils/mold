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

/** Scheduler override for UNet families (SD1.5, SDXL). Mirrors mold-core's
 * kebab-case enum; only these string variants are surfaced in the desktop UI. */
export type Scheduler = "default" | "ddim" | "euler-ancestral" | "unipc";

/** One entry in a LoRA stack. `path` is the server-side safetensors path
 * (`LoraInfo.path`); `scale` is 0–2, 1 = full strength. Mirrors mold-core
 * `LoraWeight`. */
export interface LoraWeight {
  path: string;
  scale: number;
}

/** Installed LoRA adapter (`GET /api/loras`). Mirrors mold-core `LoraInfo`. */
export interface LoraInfo {
  id: string;
  name: string;
  family: string;
  author?: string | null;
  path: string;
  trained_words: string[];
  size_bytes?: number | null;
  thumbnail_url?: string | null;
  added_at: number;
}

/** `POST /api/expand` — mirrors mold-core `ExpandRequest`. */
export interface ExpandRequest {
  prompt: string;
  model_family?: string;
  variations?: number;
}

/** `POST /api/expand` — mirrors mold-core `ExpandResponse`. */
export interface ExpandResponse {
  original: string;
  expanded: string[];
}

/** `POST /api/generate/estimate` — mirrors mold-core `GenerationMemoryEstimate`. */
export interface GenerationMemoryEstimate {
  model: string;
  peak_memory_bytes: number;
  activation_memory_bytes: number;
  available_memory_bytes?: number | null;
  load_strategy: string;
  fits_available_memory?: boolean | null;
}

/**
 * Subset of mold-core GenerateRequest the desktop sends.
 *
 * Wire note: mold-core serializes every image field as a base64 STRING in
 * JSON (no `data:` prefix), so `source_image` / `mask_image` / `control_image`
 * are typed as `string` here, not bytes.
 */
export interface GenerateRequest {
  prompt: string;
  negative_prompt?: string | null;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance?: number;
  seed?: number;
  batch_size?: number;
  output_format?: OutputFormat;
  scheduler?: Scheduler;
  cfg_plus?: boolean;
  /** img2img source, base64 (no data-URI prefix). */
  source_image?: string | null;
  strength?: number;
  /** Inpaint mask, base64. */
  mask_image?: string;
  /** ControlNet conditioning image, base64. */
  control_image?: string;
  control_model?: string;
  control_scale?: number;
  loras?: LoraWeight[];
  lora?: LoraWeight;
  expand?: boolean;
  original_prompt?: string;
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

// ── Model components ──────────────────────────────────────────────────────

/** One component of an installed model (mirrors mold-core `ModelComponentStatus`). */
export interface ModelComponentStatus {
  kind: string;
  name: string;
  present: boolean;
  path?: string | null;
  repair_model?: string | null;
}

/** `GET /api/models/:model/components` — mirrors `ModelComponentsResponse`. */
export interface ModelComponentsResponse {
  model: string;
  components: ModelComponentStatus[];
}

// ── Downloads ─────────────────────────────────────────────────────────────

export type DownloadJobStatus = "queued" | "active" | "completed" | "failed" | "cancelled";

/** Download queue entry (mirrors mold-core `DownloadJob`). */
export interface DownloadJob {
  id: string;
  model: string;
  catalog_id?: string | null;
  status: DownloadJobStatus;
  files_done: number;
  files_total: number;
  bytes_done: number;
  bytes_total: number;
  current_file?: string | null;
  started_at?: number | null;
  completed_at?: number | null;
  error?: string | null;
}

/** `GET /api/downloads` — mirrors mold-core `DownloadsListing`. */
export interface DownloadsListing {
  active?: DownloadJob | null;
  queued: DownloadJob[];
  history: DownloadJob[];
}

/**
 * SSE `download` frames (internally tagged, `type` discriminant). The first
 * frame to a new subscriber is always `snapshot`. Mirrors mold-core
 * `DownloadEvent`.
 */
export type DownloadEvent =
  | { type: "snapshot"; listing: DownloadsListing }
  | { type: "enqueued"; id: string; model: string; position: number }
  | { type: "dequeued"; id: string }
  | { type: "started"; id: string; files_total: number; bytes_total: number }
  | {
      type: "progress";
      id: string;
      files_done: number;
      bytes_done: number;
      current_file?: string | null;
    }
  | { type: "file_done"; id: string; filename: string }
  | { type: "job_done"; id: string; model: string }
  | { type: "job_failed"; id: string; error: string }
  | { type: "job_cancelled"; id: string }
  | { type: "catalog_ready"; id: string; ok: boolean };

/** `POST /api/downloads` — mirrors `CreateDownloadResponse`. */
export interface CreateDownloadResponse {
  id: string;
  position: number;
}

// ── Catalog ───────────────────────────────────────────────────────────────

/** One companion (shared component) attached to a catalog entry. */
export interface CatalogCompanionDetail {
  name: string;
  kind?: string;
  repo?: string;
  size_bytes?: number | null;
}

/**
 * Minimal subset of a `GET /api/catalog/search` entry the desktop renders.
 * `size_bytes` is the primary weights; `companion_details[].size_bytes` are the
 * shared components. The endpoint does not report which companions are already
 * on disk, so "fetch" is the full weights-plus-companions download (see
 * `lib/catalog.ts`).
 */
export interface CatalogEntry {
  id: string;
  source: string;
  name: string;
  author?: string | null;
  family: string;
  kind: string;
  size_bytes?: number | null;
  nsfw: boolean;
  installed: boolean;
  thumbnail_url?: string | null;
  trained_words?: string[];
  companions?: string[];
  companion_details?: CatalogCompanionDetail[];
}

/** `GET /api/catalog/search` response envelope. */
export interface CatalogSearchResponse {
  entries: CatalogEntry[];
  page: number;
  page_size: number;
  total: number;
}

/** One family from `GET /api/catalog/families`. */
export interface CatalogFamily {
  family: string;
}

/** `POST /api/catalog/:id/download` response. */
export interface CatalogDownloadResponse {
  primary_job_id?: string | null;
  companion_jobs: { name: string; job_id: string }[];
}
