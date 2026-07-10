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

// ── LTX-2 advanced video (mold-core `Ltx2*`, kebab-case on the wire) ────────

/** Explicit LTX-2 pipeline mode. Mirrors mold-core `Ltx2PipelineMode`. */
export type Ltx2PipelineMode =
  | "one-stage"
  | "two-stage"
  | "two-stage-hq"
  | "distilled"
  | "ic-lora"
  | "keyframe"
  | "a2vid"
  | "retake";

/** Spatial latent upscale factor. Mirrors mold-core `Ltx2SpatialUpscale`
 * (`X1_5` → `"x1-5"`, `X2` → `"x2"`). */
export type Ltx2SpatialUpscale = "x1-5" | "x2";

/** Temporal latent upscale factor. Mirrors mold-core `Ltx2TemporalUpscale`. */
export type Ltx2TemporalUpscale = "x2";

/** Retake / partial-regeneration time window. Mirrors mold-core `TimeRange`. */
export interface TimeRange {
  start_seconds: number;
  end_seconds: number;
}

/** One keyframe conditioning image on the wire — mirrors mold-core
 * `KeyframeCondition`, whose `image` field serializes as a base64 STRING. */
export interface KeyframeConditionWire {
  frame: number;
  image: string;
}

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
  // Video families (ltx-video / ltx2). Frame count must be 8n+1.
  frames?: number;
  fps?: number;
  enable_audio?: boolean;
  // LTX-2 advanced video (ltx2 only). Omitted → engine auto-selects.
  /** Source video for video-to-video / retake, base64 (no data-URI prefix). */
  source_video?: string;
  keyframes?: KeyframeConditionWire[];
  pipeline?: Ltx2PipelineMode;
  retake_range?: TimeRange;
  spatial_upscale?: Ltx2SpatialUpscale;
  temporal_upscale?: Ltx2TemporalUpscale;
}

/** serde tag = "type", snake_case — /api/generate/stream `progress` events. */
export type ProgressEvent =
  | { type: "queued"; position: number; id?: string }
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "info"; message: string }
  | { type: "cache_hit"; resource: string }
  | { type: "denoise_step"; step: number; total: number; elapsed_ms: number }
  /** Live latent preview: base64 PNG at latent resolution (client upscales). */
  | { type: "preview"; image: string; step: number; total: number }
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

// ── Config (mold config surface) ───────────────────────────────────────────

/** Where a config value currently resolves from (highest precedence wins). */
export type ConfigSource = "db" | "file" | "env" | "default";

/** One row from `GET /api/config`. Value shape drifts; kept as a scalar union. */
export interface ConfigRow {
  key: string;
  value: string | number | boolean | null;
  source: ConfigSource;
  profile?: string | null;
  /** Name of the environment variable that wins when source is "env". */
  env_var?: string | null;
}

/** `GET /api/config/profiles`. */
export interface ConfigProfiles {
  profiles: string[];
  active: string;
}

// ── Chains (mold.chain.v1 + durable chain jobs) ────────────────────────────

/** Boundary style between the previous stage and this one. */
export type TransitionMode = "smooth" | "cut" | "fade";

/** One rendered clip in a chain (mirrors mold-core `ChainStage`). */
export interface ChainStage {
  prompt: string;
  frames: number;
  /** Starting image, base64 (no data-URI prefix); v1 only honors it on stage 0. */
  source_image?: string | null;
  negative_prompt?: string | null;
  seed_offset?: number | null;
  transition?: TransitionMode;
  /** Crossfade length when `transition === "fade"`. */
  fade_frames?: number | null;
}

/**
 * Canonical chain request posted to `POST /api/chain-jobs` (mirrors mold-core
 * `ChainRequest`, canonical form only — note the key is `stages`).
 */
export interface ChainRequest {
  model: string;
  stages: ChainStage[];
  motion_tail_frames?: number;
  width: number;
  height: number;
  fps?: number;
  seed?: number | null;
  steps: number;
  guidance: number;
  strength?: number;
  output_format?: OutputFormat;
  enable_audio?: boolean | null;
}

/** `[chain]` table of a `mold.chain.v1` script (mirrors `ChainScriptChain`). */
export interface ChainScriptChain {
  model: string;
  width: number;
  height: number;
  fps: number;
  seed?: number | null;
  steps: number;
  guidance: number;
  strength: number;
  motion_tail_frames: number;
  output_format: OutputFormat;
  enable_audio?: boolean | null;
}

/**
 * `mold.chain.v1` script (mirrors `ChainScript`). On the JSON wire the stages
 * array key is `stage` (serde rename), matching the TOML `[[stage]]` tables.
 */
export interface ChainScript {
  schema: string;
  chain: ChainScriptChain;
  stage: ChainStage[];
}

/** `GET /api/capabilities/chain-limits?model=` (mirrors `ChainLimits`). */
export interface ChainLimits {
  model: string;
  frames_per_clip_cap: number;
  frames_per_clip_recommended: number;
  max_stages: number;
  max_total_frames: number;
  fade_frames_max: number;
  transition_modes: string[];
  quantization_family: string;
  supports_audio: boolean;
}

export type ChainJobState =
  "queued" | "running" | "interrupted" | "failed" | "completed" | "cancelled";

export type StageState = "pending" | "running" | "completed" | "failed";

export type RetakeMode = "cascade" | "splice";

export interface ChainJobSummary {
  id: string;
  state: ChainJobState;
  model: string;
  stage_count: number;
  current_stage: number;
  created_at_unix_ms: number;
  updated_at_unix_ms: number;
  error?: string | null;
  ephemeral: boolean;
}

export interface ChainJobStageDetail {
  idx: number;
  state: StageState;
  /** u64 effective seed, serialized as a decimal string. */
  seed: string;
  frames_emitted?: number | null;
  generation_time_ms?: number | null;
  has_preview: boolean;
  error?: string | null;
}

/** `GET /api/chain-jobs/:id` — `ChainJobSummary` flattened + stages/script. */
export interface ChainJobDetail extends ChainJobSummary {
  stages: ChainJobStageDetail[];
  finalizes: { output: string; at_unix_ms: number; stage_seeds: string[] }[];
  retakes: unknown[];
  script: ChainScript;
}

/** `GET /api/chain-jobs`. */
export interface ChainJobListing {
  jobs: ChainJobSummary[];
}

/** `POST /api/chain-jobs` (202). */
export interface CreateChainJobResponse {
  job_id: string;
}

/** `POST /api/chain-jobs/:id/retake` body (mirrors `RetakeRequest`). */
export interface RetakeRequest {
  stage_idx: number;
  mode: RetakeMode;
  /** u64 seed offset as a decimal string. */
  seed_offset?: string;
  prompt?: string;
}

/** `POST /api/chain-jobs/gc`. */
export interface GcOutcome {
  swept_ephemeral_jobs: number;
  pruned_artifact_dirs: number;
}

/**
 * SSE `chain_job` frames from `GET /api/chain-jobs/:id/events` (internally
 * tagged snake_case; first frame is always `snapshot`). Mirrors mold-core
 * `ChainJobEvent`.
 */
export type ChainJobEvent =
  | { type: "snapshot"; job: ChainJobDetail }
  | { type: "stage_start"; stage_idx: number }
  | { type: "denoise_step"; stage_idx: number; step: number; total: number }
  | { type: "stage_done"; stage_idx: number; frames_emitted: number; has_preview: boolean }
  | { type: "yielded"; pending_small_jobs: number }
  | { type: "finalizing"; total_frames: number }
  | { type: "finalized"; output: string; take: number }
  | { type: "state_changed"; state: ChainJobState; error?: string | null };
