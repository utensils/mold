/**
 * Hand-mirrored subsets of mold-core wire types (serde snake_case).
 * Source of truth: /api/openapi.json — a drift snapshot test guards these
 * once the OpenAPI harness lands (see desktop/docs/architecture.md §5).
 */

import type { ChainOutputMetadata } from "@studio/lib/api/chainTypes";

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

/** `/api/status` per-GPU worker row; present only on multi-GPU-aware servers. */
export interface GpuWorkerStatus {
  ordinal: number;
  name: string;
  vram_total_bytes: number;
  vram_used_bytes: number;
  loaded_model?: string | null;
  state: string;
}

/** `/api/status` GPU summary (MB units, unlike GpuSnapshot's bytes). */
export interface GpuInfo {
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  /** "cuda" | "metal"; absent from servers ≤ 0.16 — infer from `name` then. */
  backend?: string | null;
}

export interface ServerStatus {
  version: string;
  git_sha?: string | null;
  models_loaded: string[];
  busy?: boolean;
  uptime_secs: number;
  hostname?: string | null;
  gpu_info?: GpuInfo | null;
  /** One row per GPU worker; absent/null on single-GPU or older servers. */
  gpus?: GpuWorkerStatus[] | null;
  queue_depth?: number | null;
  queue_capacity?: number | null;
  /** Stable server-installation UUID; absent on older servers. */
  instance_id?: string | null;
  /** Disk stats for the filesystem holding the models dir; absent on older servers. */
  models_disk?: { total_bytes: number; free_bytes: number } | null;
}

export type ExpandBackend = "local" | "api";

export interface ExpandCapabilities {
  configured: boolean;
  model_present: boolean | null;
  backend: ExpandBackend;
}

export interface ServerCapabilities {
  gallery: { can_delete: boolean };
  catalog?: { available: boolean; families: string[] } | null;
  /** Server-assisted DNS-SD browse support; absent on older servers. */
  discovery?: { can_browse: boolean } | null;
  /** Absent on servers that predate `GET /api/events`. */
  events?: { available: boolean } | null;
  /** Absent on older servers means unknown, not unavailable. */
  expand?: ExpandCapabilities | null;
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
  /** Human-readable title for catalog installs whose `name` is an opaque
   * `cv:<id>` / `hf:<repo>` identifier. Additive — absent on older servers
   * and on manifest models. Display only: every API call keeps `name`. */
  display_name?: string | null;
  /** Installed-catalog classification; absent on older servers and manifest rows. */
  kind?: string | null;
  /** Installed-catalog modality (`image` / `video`); absent when unknown. */
  modality?: string | null;
  /** Explicit mature-content classification; absent means unknown. */
  nsfw?: boolean | null;
  /** Model-specific LTX-2 audio output support; absent on older servers. */
  supports_audio?: boolean | null;
  /** Explicit durable sequence eligibility; absent on older servers. */
  supports_sequence?: boolean | null;
  /** Server-advertised per-clip frame default (LTX-2 ships 97, LTX-Video
   * 25); absent on older servers. Sizes new sequence clips. */
  default_frames?: number | null;
  /** Server-advertised frame rate (LTX-Video ships 30, LTX-2 24); absent on
   * older servers and image models. Applied like steps/guidance. */
  default_fps?: number | null;
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
  /** Natural-language visual-style directive the server weaves into the
   * expander's system message (additive; absent on the wire when unset). */
  style?: string;
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
  /** Provenance label for the source image (gallery filename or upload
   * name) — recorded into OutputMetadata for Reuse-settings restore. */
  source_image_name?: string;
  /** Qwen-Image-Edit multi-image inputs, base64 each (no data-URI prefix).
   * Order is load-bearing: first = primary edit target, rest = references. */
  edit_images?: string[];
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
  /** Durable prepared-batch provenance. Index is one-based. */
  batch_id?: string;
  batch_index?: number;
  batch_count?: number;
  /** Post-generate upscaler model (e.g. "real-esrgan-x4plus"); image-only. */
  upscale_model?: string;
  // Video families (ltx-video / ltx2). Frame count must be 8n+1.
  frames?: number;
  fps?: number;
  enable_audio?: boolean;
  // LTX-2 advanced video (ltx2 only). Omitted → engine auto-selects.
  /** Conditioning audio for the a2vid (audio-to-video) pipeline, base64 (no
   * data-URI prefix). mold-core `audio_file`. */
  audio_file?: string;
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
  /** Base64 payload — empty when the client requested a metadata-only completion. */
  image: string;
  format: OutputFormat;
  width: number;
  height: number;
  original_image?: string | null;
  original_width?: number | null;
  original_height?: number | null;
  seed_used: number;
  generation_time_ms: number;
  model: string;
  video_frames?: number | null;
  video_fps?: number | null;
  /** Large base64 video previews; metadata-only completions omit them. */
  video_thumbnail?: string | null;
  video_gif_preview?: string | null;
  video_has_audio?: boolean;
  video_duration_ms?: number | null;
  video_audio_sample_rate?: number | null;
  video_audio_channels?: number | null;
  gpu?: number | null;
  /** Gallery filename the server saved this payload under (additive; absent
   * on older servers). Mirrored saves keep it so the local copy and the
   * origin stay one logical print in the merged gallery. */
  filename?: string | null;
  /** Gallery filename of the pre-upscale original, when one was saved. */
  original_filename?: string | null;
  /** The exact metadata the server recorded for this payload — the local DB
   * row for a mirrored save uses it verbatim (videos embed nothing). */
  metadata?: OutputMetadata | null;
}

// ── Gallery ──────────────────────────────────────────────────────────────

/** Embedded `mold:parameters` metadata (mirrors mold-core `OutputMetadata`).
 * Everything beyond the required core is optional — the desktop talks to
 * arbitrary-version remote servers, older ones simply omit newer fields. */
export interface OutputMetadata {
  prompt: string;
  negative_prompt?: string | null;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  model: string;
  seed: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  /** Generation canvas before any post-generate upscaler resized the file. */
  generation_width?: number | null;
  generation_height?: number | null;
  strength?: number | null;
  /** Provenance label of the img2img source (additive; newer servers only). */
  source_image_name?: string | null;
  /** SHA-256 (hex) of the exact source bytes used — local stash lookup key
   * for Reuse-settings source restore (additive; newer servers only). */
  source_image_sha256?: string | null;
  /** Durable sequence job this print was stitched from. Present only for
   * chain jobs with a server-side record — ephemeral chain outputs and
   * pre-#564 rows carry nothing (additive; newer servers only). */
  chain_job_id?: string | null;
  /** Per-clip provenance for a stitched sequence — what the Library's
   * sequence-aware Reuse settings reloads into the clip rail. */
  chain?: ChainOutputMetadata | null;
  /** Plain kebab-case name, or a serde-tagged object for parameterized
   * variants (e.g. `{ "ddim": … }`). Normalize before feeding the form. */
  scheduler?: string | Record<string, unknown> | null;
  output_format?: OutputFormat | null;
  cfg_plus?: boolean | null;
  lora?: string | null;
  lora_scale?: number | null;
  loras?: LoraWeight[] | null;
  control_model?: string | null;
  control_scale?: number | null;
  upscale_model?: string | null;
  gif_preview?: boolean | null;
  enable_audio?: boolean | null;
  audio_file_path?: string | null;
  source_video_path?: string | null;
  pipeline?: Ltx2PipelineMode | null;
  retake_range?: TimeRange | null;
  spatial_upscale?: Ltx2SpatialUpscale | null;
  temporal_upscale?: Ltx2TemporalUpscale | null;
  frames?: number | null;
  fps?: number | null;
  /** mold version that produced the print. */
  version?: string | null;
  /** Legacy desktop-only aliases for `frames` / `fps`; never sent by current
   * servers but kept so older synthesized rows still display. */
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
  active_jobs?: DownloadJob[];
  /** Compatibility view from older servers and for older clients. */
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

// ── Server events ─────────────────────────────────────────────────────────

/**
 * `GET /api/events` frames (internally tagged, `type` discriminant) —
 * server-wide job lifecycle + gallery mutations over one SSE connection.
 * Mirrors mold-core `ServerEvent`; feature-detect via
 * `ServerCapabilities.events`. Deltas only — bootstrap from `GET /api/queue`
 * + `GET /api/gallery`.
 */
export type ServerEvent =
  | { type: "job_queued"; id: string; model: string }
  | { type: "job_started"; id: string; model: string; gpu?: number | null }
  | { type: "job_ended"; id: string }
  | { type: "gallery_added"; filename: string; image?: GalleryImage | null }
  | { type: "gallery_removed"; filename: string }
  | { type: "queue_paused" }
  | { type: "queue_resumed" };

// ── Catalog ───────────────────────────────────────────────────────────────

/** One companion (shared component) attached to a catalog entry. */
export interface CatalogCompanionDetail {
  name: string;
  kind?: string;
  repo?: string;
  size_bytes?: number | null;
}

/** One file of a catalog entry's download recipe (`mold-catalog RecipeFile`). */
export interface CatalogRecipeFile {
  url: string;
  dest: string;
  sha256?: string | null;
  size_bytes?: number | null;
  role?: string | null;
}

/** Primary-download plan for a catalog entry (`mold-catalog DownloadRecipe`). */
export interface CatalogDownloadRecipe {
  files: CatalogRecipeFile[];
  needs_token?: string | null;
}

/**
 * Subset of a `GET /api/catalog/search` / `GET /api/catalog/:id` entry the
 * desktop renders. `size_bytes` is the primary weights;
 * `companion_details[].size_bytes` are the shared components. The endpoint
 * does not report which companions are already on disk, so "fetch" is the
 * full weights-plus-companions download (see `lib/catalog.ts`).
 *
 * Every descriptive field past the search-summary core is optional: the
 * desktop connects to arbitrary-version remote hosts, and older servers omit
 * them from the wire.
 */
export interface CatalogEntry {
  id: string;
  source: string;
  /** Upstream id without the `hf:`/`cv:` prefix — the HF repo id or Civitai version id. */
  source_id?: string | null;
  name: string;
  /** Human-readable title when `name` is an opaque catalog id (installed
   * `cv:`/`hf:` rows). Client-side only today — render `display_name ??
   * name`, but keep every id/equality/API use on `name`. */
  display_name?: string | null;
  author?: string | null;
  family: string;
  kind: string;
  /** `"image"` / `"video"`. */
  modality?: string;
  /** `"safetensors"` / `"gguf"` / `"diffusers"`. */
  file_format?: string;
  /** Live HF layout; separated repos can contain several checkpoints. */
  bundling?: "single-file" | "separated" | string | null;
  size_bytes?: number | null;
  download_count?: number | null;
  rating?: number | null;
  likes?: number | null;
  /** `null` means an older installed sidecar did not classify the model. */
  nsfw: boolean | null;
  installed: boolean;
  /** Absolute weights path when installed — `GET /api/catalog/installed` fills
   * it from the sidecar walk; search results usually omit it. */
  primary_path?: string | null;
  thumbnail_url?: string | null;
  /** Human-facing model page. Additive wire field — absent on older servers. */
  page_url?: string | null;
  description?: string | null;
  license?: string | null;
  tags?: string[];
  trained_words?: string[];
  /** Upstream catalog timestamps; current servers emit Unix seconds. */
  created_at?: number | null;
  updated_at?: number | null;
  added_at?: number | null;
  companions?: string[];
  companion_details?: CatalogCompanionDetail[];
  /** Primary weights files a pull will fetch (detail drawer itemization). */
  download_recipe?: CatalogDownloadRecipe;
  /** `>= 6` marks catalog packages no shipped engine can run yet. */
  supported?: boolean;
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
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
}

/**
 * Auto-expand request accepted by `POST /api/generate/chain/stream`. The
 * server expands one prompt into the canonical `stages` array using the
 * requested total/clip frame budgets.
 */
export interface AutoChainRequest {
  model: string;
  prompt: string;
  total_frames: number;
  clip_frames: number;
  motion_tail_frames: number;
  width: number;
  height: number;
  fps?: number;
  seed?: number | null;
  steps: number;
  guidance: number;
  strength?: number;
  output_format?: OutputFormat;
  /** Starting image for the first generated clip, base64 without a data URI. */
  source_image?: string | null;
  enable_audio?: boolean | null;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
}

/**
 * SSE `progress` frames from `POST /api/generate/chain/stream`. Newer
 * servers add the durable shim job id to every variant so a connected client
 * can cancel the underlying chain job; older servers omit it.
 */
export type ChainProgressEvent =
  | {
      type: "chain_start";
      stage_count: number;
      estimated_total_frames: number;
      job_id?: string;
    }
  | { type: "stage_start"; stage_idx: number; job_id?: string }
  | {
      type: "denoise_step";
      stage_idx: number;
      step: number;
      total: number;
      job_id?: string;
    }
  | {
      type: "stage_done";
      stage_idx: number;
      frames_emitted: number;
      job_id?: string;
    }
  | { type: "stitching"; total_frames: number; job_id?: string };

/** Final `complete` frame from `POST /api/generate/chain/stream`. */
export interface SseChainCompleteEvent {
  /** Base64 stitched media; empty for metadata-only completions. */
  video: string;
  format: OutputFormat;
  width: number;
  height: number;
  frames: number;
  fps: number;
  thumbnail?: string | null;
  gif_preview?: string | null;
  has_audio?: boolean;
  duration_ms?: number | null;
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
  stage_count: number;
  gpu?: number | null;
  generation_time_ms?: number | null;
  script: ChainScript;
  vram_estimate?: { worst_case_bytes: number; fits: boolean } | null;
  /** Saved gallery filename; present on metadata-only completions. */
  filename?: string | null;
  /** Exact metadata persisted beside the stitched output. */
  metadata?: OutputMetadata | null;
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
