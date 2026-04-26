// Matches `mold_core::OutputFormat` on the wire (lowercase strings).
export type OutputFormat = "png" | "jpeg" | "gif" | "apng" | "webp" | "mp4";

export type Scheduler =
  | "default"
  | "ddim"
  | "euler-ancestral"
  | "unipc"
  | { ddim: unknown }
  | { "euler-ancestral": unknown }
  | { unipc: unknown };

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
  scheduler?: Scheduler | null;
  lora?: string | null;
  lora_scale?: number | null;
  frames?: number | null;
  fps?: number | null;
  version: string;
}

export interface GalleryImage {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  metadata_synthetic?: boolean;
}

export type MediaKind = "image" | "animated" | "video";

export const VIDEO_FORMATS: ReadonlyArray<OutputFormat> = ["mp4"];
export const ANIMATED_FORMATS: ReadonlyArray<OutputFormat> = [
  "gif",
  "apng",
  "webp",
];

export function mediaKind(
  fmt: OutputFormat | null | undefined,
  filename: string,
): MediaKind {
  const resolved = fmt ?? inferFormatFromName(filename);
  if (resolved && VIDEO_FORMATS.includes(resolved)) return "video";
  if (resolved && ANIMATED_FORMATS.includes(resolved)) return "animated";
  return "image";
}

// Mirror of `mold_core::GalleryCapabilities`.
export interface GalleryCapabilities {
  can_delete: boolean;
}

// Mirror of `mold_core::ServerCapabilities`.
export interface ServerCapabilities {
  gallery: GalleryCapabilities;
}

export function inferFormatFromName(filename: string): OutputFormat | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".mp4")) return "mp4";
  if (lower.endsWith(".gif")) return "gif";
  if (lower.endsWith(".apng")) return "apng";
  if (lower.endsWith(".webp")) return "webp";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "jpeg";
  if (lower.endsWith(".png")) return "png";
  return null;
}

// ──────────────────────────────────────────────────────────────────────────────
// Generation types (mirror of mold_core::GenerateRequest / GenerateResponse /
// SseProgressEvent / SseCompleteEvent / ModelInfoExtended / ServerStatus).
// Client-side uses camelCase; serialization to/from the wire happens in api.ts.
// ──────────────────────────────────────────────────────────────────────────────

export interface LoraWeight {
  path: string;
  scale: number;
}

// ── Device placement (Agent C: model-ui-overhaul §3) ──────────────────────
export type DeviceRef =
  | { kind: "auto" }
  | { kind: "cpu" }
  | { kind: "gpu"; ordinal: number };

export interface AdvancedPlacement {
  transformer: DeviceRef;
  vae: DeviceRef;
  clip_l?: DeviceRef | null;
  clip_g?: DeviceRef | null;
  t5?: DeviceRef | null;
  qwen?: DeviceRef | null;
}

export interface DevicePlacement {
  text_encoders: DeviceRef;
  advanced?: AdvancedPlacement | null;
}

// Wire shape — what we POST to /api/generate/stream. snake_case to match serde.
export interface GenerateRequestWire {
  prompt: string;
  negative_prompt?: string | null;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed?: number | null;
  batch_size?: number;
  output_format?: OutputFormat;
  scheduler?: Scheduler | null;
  source_image?: string | null; // base64 (no data-URI prefix)
  strength?: number;
  expand?: boolean;
  original_prompt?: string | null;
  frames?: number | null;
  fps?: number | null;
  placement?: DevicePlacement | null;
}

export interface ModelDefaults {
  default_steps: number;
  default_guidance: number;
  default_width: number;
  default_height: number;
  description: string;
}

export interface ModelInfoExtended extends ModelDefaults {
  name: string;
  family: string;
  size_gb: number;
  is_loaded: boolean;
  last_used: number | null;
  hf_repo: string;
  downloaded: boolean;
  disk_usage_bytes?: number | null;
  remaining_download_bytes?: number | null;
}

export interface GpuInfo {
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
}

export type GpuWorkerState = "idle" | "generating" | "loading" | "degraded";

export interface GpuWorkerStatus {
  ordinal: number;
  name: string;
  vram_total_bytes: number;
  vram_used_bytes: number;
  loaded_model?: string | null;
  state: GpuWorkerState;
}

export interface ServerStatus {
  version: string;
  git_sha?: string | null;
  build_date?: string | null;
  models_loaded: string[];
  busy: boolean;
  gpu_info?: GpuInfo | null;
  uptime_secs: number;
  hostname?: string | null;
  memory_status?: string | null;
  gpus?: GpuWorkerStatus[] | null;
  queue_depth?: number | null;
  queue_capacity?: number | null;
}

export type SseProgressEvent =
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "info"; message: string }
  | { type: "cache_hit"; resource: string }
  | { type: "denoise_step"; step: number; total: number; elapsed_ms: number }
  | { type: "queued"; position: number }
  | {
      type: "weight_load";
      bytes_loaded: number;
      bytes_total: number;
      component: string;
    };

export interface SseCompleteEvent {
  image: string; // base64
  format: OutputFormat;
  width: number;
  height: number;
  seed_used: number;
  generation_time_ms: number;
  model: string;
  video_frames?: number | null;
  video_fps?: number | null;
  video_thumbnail?: string | null; // base64
  video_gif_preview?: string | null; // base64
  video_has_audio?: boolean;
  video_duration_ms?: number | null;
  video_audio_sample_rate?: number | null;
  video_audio_channels?: number | null;
  gpu?: number | null;
}

// ── Chained video generation (POST /api/generate/chain/stream) ────────────
// Mirrors `mold_core::chain::{ChainRequest, ChainStage,
// ChainProgressEvent, SseChainCompleteEvent}`.
export interface ChainStageWire {
  prompt: string;
  frames: number;
  source_image?: string | null;
  negative_prompt?: string | null;
  seed_offset?: number | null;
  transition?: "smooth" | "cut" | "fade";
  fade_frames?: number | null;
}

export interface ChainRequestWire {
  model: string;
  stages?: ChainStageWire[];
  motion_tail_frames?: number;
  width: number;
  height: number;
  fps?: number;
  seed?: number | null;
  steps: number;
  guidance: number;
  strength?: number;
  output_format?: OutputFormat;
  placement?: DevicePlacement | null;
  prompt?: string;
  total_frames?: number;
  clip_frames?: number;
  source_image?: string | null;
}

export type ChainProgressEvent =
  | {
      type: "chain_start";
      stage_count: number;
      estimated_total_frames: number;
    }
  | { type: "stage_start"; stage_idx: number }
  | {
      type: "denoise_step";
      stage_idx: number;
      step: number;
      total: number;
    }
  | { type: "stage_done"; stage_idx: number; frames_emitted: number }
  | { type: "stitching"; total_frames: number };

export interface SseChainCompleteEvent {
  video: string; // base64
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
}

export interface ExpandRequestWire {
  prompt: string;
  model_family: string;
  variations: number;
}

export interface ExpandResponseWire {
  original: string;
  expanded: string[];
}

// ── Client-side form shape (persisted in localStorage) ─────────────────────
export interface SourceImageState {
  kind: "upload" | "gallery";
  filename: string;
  base64: string; // stripped before localStorage persist
}

export interface ExpandFormState {
  enabled: boolean;
  variations: 1 | 3 | 5;
  familyOverride: string | null;
}

export interface GenerateFormState {
  version: 1;
  prompt: string;
  negativePrompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null; // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  fps: number | null;
  scheduler: Scheduler | null;
  outputFormat: OutputFormat;
  expand: ExpandFormState;
  sourceImage: SourceImageState | null;
  placement: DevicePlacement | null;
}

// ── Video-family detection helper used by multiple components ──────────────
export const VIDEO_FAMILIES: ReadonlyArray<string> = [
  "ltx-video",
  "ltx2",
  "ltx-2",
];

// Families whose image pipeline ignores the negative prompt.
export const NO_CFG_FAMILIES: ReadonlyArray<string> = [
  "flux",
  "flux2",
  "flux.2",
  "z-image",
  "qwen-image",
  "qwen_image",
];

// Families whose UNet responds to scheduler overrides.
export const UNET_SCHEDULER_FAMILIES: ReadonlyArray<string> = [
  "sd15",
  "sd1.5",
  "stable-diffusion-1.5",
  "sdxl",
];

// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
// Mirror of `mold_core::types::{DownloadJob, JobStatus, DownloadEvent,
// DownloadsListing}`. Keep field names / string literals in sync with the
// server's serde output.

export type JobStatusWire =
  | "queued"
  | "active"
  | "completed"
  | "failed"
  | "cancelled";

export interface DownloadJobWire {
  id: string;
  model: string;
  status: JobStatusWire;
  files_done: number;
  files_total: number;
  bytes_done: number;
  bytes_total: number;
  current_file?: string | null;
  started_at?: number | null;
  completed_at?: number | null;
  error?: string | null;
}

export interface DownloadsListingWire {
  active?: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
}

export type DownloadEventWire =
  | { type: "enqueued"; id: string; model: string; position: number }
  | { type: "dequeued"; id: string }
  | {
      type: "started";
      id: string;
      files_total: number;
      bytes_total: number;
    }
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
  | { type: "job_cancelled"; id: string };
// ──────────────────────────────────────────────────────────────────────────────
// Resource telemetry (Agent B scope). Mirror of `mold_core::ResourceSnapshot`
// et al. `vram_used_by_mold` / `vram_used_by_other` are null on Metal hosts
// and on CUDA hosts that fell back to the `nvidia-smi` subprocess path.
// ──────────────────────────────────────────────────────────────────────────────

export type GpuBackend = "cuda" | "metal";

export interface GpuSnapshot {
  ordinal: number;
  name: string;
  backend: GpuBackend;
  vram_total: number;
  vram_used: number;
  vram_used_by_mold: number | null;
  vram_used_by_other: number | null;
  /** 0-100. `null` on Metal and on the `nvidia-smi` fallback path. */
  gpu_utilization?: number | null;
}

export interface RamSnapshot {
  total: number;
  used: number;
  used_by_mold: number;
  used_by_other: number;
}

export interface CpuSnapshot {
  cores: number;
  /** 0-100 averaged across all cores. */
  usage_percent: number;
}

export interface ResourceSnapshot {
  hostname: string;
  timestamp: number;
  gpus: GpuSnapshot[];
  system_ram: RamSnapshot;
  /** `null` on the first sample (sysinfo needs a prior refresh to compute deltas). */
  cpu?: CpuSnapshot | null;
}

// ─── Catalog (sub-project A) ──────────────────────────────────────────────

export interface CatalogEntryWire {
  id: string;
  source: "hf" | "civitai";
  source_id: string;
  name: string;
  author: string | null;
  family: string;
  family_role: "foundation" | "finetune";
  sub_family: string | null;
  modality: "image" | "video";
  kind: "checkpoint" | "lora" | "vae" | "text-encoder" | "control-net";
  file_format: "safetensors" | "gguf" | "diffusers";
  bundling: "separated" | "single-file";
  size_bytes: number | null;
  download_count: number;
  rating: number | null;
  likes: number;
  nsfw: boolean;
  thumbnail_url: string | null;
  description: string | null;
  license: string | null;
  license_flags: {
    commercial?: boolean | null;
    derivatives?: boolean | null;
    different_license?: boolean | null;
  } | null;
  tags: string[];
  companions: string[];
  download_recipe: {
    files: {
      url: string;
      dest: string;
      sha256: string | null;
      size_bytes: number | null;
    }[];
    needs_token: "hf" | "civitai" | null;
  };
  engine_phase: number;
  created_at: number | null;
  updated_at: number | null;
  added_at: number;
}

export interface CatalogListResponse {
  entries: CatalogEntryWire[];
  page: number;
  page_size: number;
}

export interface CatalogFamilyCount {
  family: string;
  foundation: number;
  finetune: number;
}

export interface CatalogFamiliesResponse {
  families: CatalogFamilyCount[];
}

export interface CatalogListParams {
  family?: string;
  family_role?: "foundation" | "finetune";
  modality?: "image" | "video";
  source?: "hf" | "civitai";
  sub_family?: string;
  q?: string;
  include_nsfw?: boolean;
  max_engine_phase?: number;
  sort?: "downloads" | "rating" | "recent" | "name";
  page?: number;
  page_size?: number;
}

export type CatalogRefreshStatus =
  | { state: "pending" }
  | {
      state: "running";
      // All four optional so older servers (pre live-progress) still parse.
      // The web UI guards with nullish-coalescing where needed.
      families_total?: number;
      families_done?: number;
      current_family?: string | null;
      current_stage?: string | null;
      started_at_ms?: number;
    }
  | { state: "done"; total_entries: number; per_family: Record<string, string> }
  | { state: "failed"; message: string };
