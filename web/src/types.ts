import type { SourceFitPolicy } from "@studio/lib/sourceFit";

export type { SourceFitPolicy } from "@studio/lib/sourceFit";

// Matches `mold_core::OutputFormat` on the wire (lowercase strings).
export type OutputFormat = "png" | "jpeg" | "gif" | "apng" | "webp" | "mp4";

export type SeedMode = "random" | "static" | "increment";

export type Ltx2PipelineMode =
  | "one-stage"
  | "two-stage"
  | "two-stage-hq"
  | "distilled"
  | "ic-lora"
  | "keyframe"
  | "a2vid"
  | "retake";

export type Ltx2SpatialUpscale = "x1-5" | "x2";
export type Ltx2TemporalUpscale = "x2";

export interface TimeRange {
  start_seconds: number;
  end_seconds: number;
}

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
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  model: string;
  seed: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  generation_width?: number | null;
  generation_height?: number | null;
  strength?: number | null;
  scheduler?: Scheduler | null;
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
  can_delete?: boolean;
}

// Mirror of `mold_core::ServerCapabilities`.
export interface ServerCapabilities {
  gallery?: GalleryCapabilities;
  discovery?: { can_browse: boolean };
  queue?: {
    can_pause?: boolean;
    can_cancel_all?: boolean;
    can_reorder?: boolean;
  };
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
  { kind: "auto" } | { kind: "cpu" } | { kind: "gpu"; ordinal: number };

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
  cfg_plus?: boolean | null;
  scheduler?: Scheduler | null;
  source_image?: string | null; // base64 (no data-URI prefix)
  /** Upload/gallery label recorded as provenance only when source_image exists. */
  source_image_name?: string | null;
  /** Qwen-Image-Edit attachments in order: first image is the target,
   * subsequent images are references. Mutually exclusive with
   * `source_image`. */
  edit_images?: string[] | null;
  strength?: number;
  mask_image?: string | null;
  control_image?: string | null;
  control_model?: string | null;
  control_scale?: number;
  expand?: boolean;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  frames?: number | null;
  fps?: number | null;
  upscale_model?: string | null;
  gif_preview?: boolean;
  placement?: DevicePlacement | null;
  lora?: { path: string; scale: number } | null;
  /** Multi-LoRA stack. Wins over the singular `lora` field when both are
   * set. The server merges deltas additively so order is significant
   * mostly for human reasoning (the math commutes). */
  loras?: { path: string; scale: number }[] | null;
  /** AV-family (LTX-2 / LTX-2.3) audio decode toggle. `true` enables the
   * audio VAE + vocoder tail and produces an AAC track in the MP4 mux;
   * `false` skips audio decode; omit for "no preference" (server defaults
   * to on for MP4 output). The server rejects `true` for non-AV families. */
  enable_audio?: boolean | null;
  audio_file?: string | null;
  audio_file_path?: string | null;
  source_video?: string | null;
  source_video_path?: string | null;
  keyframes?: KeyframeConditionWire[] | null;
  pipeline?: Ltx2PipelineMode | null;
  retake_range?: TimeRange | null;
  spatial_upscale?: Ltx2SpatialUpscale | null;
  temporal_upscale?: Ltx2TemporalUpscale | null;
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
  /** Model-specific sequence support; absent on older servers. */
  supports_sequence?: boolean | null;
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
  instance_id?: string | null;
  models_disk?: HostDiskUsage | null;
  queue_paused?: boolean | null;
}

export interface HostDiskUsage {
  total_bytes: number;
  free_bytes: number;
}

// ── /api/queue (L3 reconciliation) ─────────────────────────────────────────

export type QueueJobState = "queued" | "running";

export interface QueueEntry {
  id: string;
  model: string;
  state: QueueJobState;
  started_at_unix_ms: number;
  position: number;
  /** Present only when `state === "running"`. */
  gpu?: number;
  /** Preferred lane for queued jobs. Omitted/null means Auto. */
  target_gpu?: number | null;
}

export interface QueueListing {
  entries: QueueEntry[];
}

export type SseProgressEvent =
  | { type: "dependency_wait"; dependency: string; reason: string }
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "info"; message: string }
  | { type: "cache_hit"; resource: string }
  | { type: "denoise_step"; step: number; total: number; elapsed_ms: number }
  /** Live latent preview: base64 PNG at latent resolution (client upscales). */
  | { type: "preview"; image: string; step: number; total: number }
  | { type: "queued"; position: number; id: string }
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
  | { type: "pull_complete"; model: string }
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
  original_image?: string | null;
  original_width?: number | null;
  original_height?: number | null;
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

export interface UpscaleRequestWire {
  model: string;
  image: string;
  output_format?: OutputFormat;
  tile_size?: number | null;
  metadata?: OutputMetadata | null;
}

export interface SseUpscaleCompleteEvent {
  image: string;
  format: OutputFormat;
  model: string;
  scale_factor: number;
  original_width: number;
  original_height: number;
  upscale_time_ms: number;
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
  /** Mux per-stage audio into the stitched MP4 (LTX-2 / LTX-2.3 only).
   * Omit for the wire default of off — chains opt in to audio
   * explicitly so existing callers don't suddenly produce audio they
   * didn't ask for. The server returns 400 if `true` for non-AV
   * families (see `chain_limits::family_supports_audio`). */
  enable_audio?: boolean | null;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
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
  | {
      type: "stage_done";
      stage_idx: number;
      frames_emitted: number;
    }
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

export interface ChainJobSummary {
  id: string;
  state: ChainJobState;
  model: string;
  stage_count: number;
  current_stage: number;
  created_at_unix_ms: number;
  updated_at_unix_ms: number;
  error: string | null;
  ephemeral: boolean;
}

export type ChainJobState =
  "queued" | "running" | "interrupted" | "failed" | "completed" | "cancelled";

export type StageState = "pending" | "running" | "completed" | "failed";

export interface ChainJobStageDetail {
  idx: number;
  state: StageState;
  seed: string;
  frames_emitted: number | null;
  generation_time_ms: number | null;
  has_preview: boolean;
  error: string | null;
}

export interface FinalizeRecord {
  output: string;
  at_unix_ms: number;
  stage_seeds: string[];
}

export interface RetakeAmendment {
  stage_idx: number;
  mode: "cascade" | "splice";
  old_seed: string;
  new_seed: string;
  old_prompt: string | null;
  new_prompt: string | null;
  at_unix_ms: number;
}

// script is a NEW wire-exact mirror (NOT lib/chainToml.ts's ChainScriptToml):
// Rust ChainScript serializes stages under the key "stage"
// (#[serde(rename = "stage")], chain.rs:236) — mirror pins that name.
export interface ChainScriptWire {
  schema: string;
  chain: Record<string, unknown>;
  stage: ChainStageWire[];
}

export interface ChainJobDetail extends ChainJobSummary {
  stages: ChainJobStageDetail[];
  finalizes: FinalizeRecord[];
  retakes: RetakeAmendment[];
  script: ChainScriptWire;
}

export interface ChainJobListing {
  jobs: ChainJobSummary[];
}

export interface CreateChainJobResponse {
  job_id: string;
}

export type ChainJobEvent =
  | { type: "snapshot"; job: ChainJobDetail }
  | { type: "stage_start"; stage_idx: number }
  | { type: "denoise_step"; stage_idx: number; step: number; total: number }
  | {
      type: "stage_done";
      stage_idx: number;
      frames_emitted: number;
      has_preview: boolean;
    }
  | { type: "yielded"; pending_small_jobs: number }
  | { type: "finalizing"; total_frames: number }
  | { type: "finalized"; output: string; take: number }
  | { type: "state_changed"; state: ChainJobState; error: string | null };

export interface RetakeRequest {
  stage_idx: number;
  mode: "cascade" | "splice";
  seed_offset?: string;
  prompt?: string;
}

export interface ExpandRequestWire {
  prompt: string;
  model_family: string;
  variations: number;
  /**
   * Natural-language style directive (see `styleHint`) the server appends to
   * the expander's system message so the look is woven into the rewrite —
   * never the literal preset suffix, and never appended to the prompt text.
   */
  style?: string;
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
  draftId?: string;
  width?: number | null;
  height?: number | null;
  mime?: string | null;
}

export interface SourceMediaState {
  kind: "upload";
  filename: string;
  base64: string; // stripped before localStorage persist
  draftId?: string;
  mime?: string | null;
}

export interface KeyframeConditionState {
  frame: number;
  image: SourceImageState;
}

export interface KeyframeConditionWire {
  frame: number;
  image: string;
}

export interface ExpandFormState {
  enabled: boolean;
  variations: 1 | 3 | 5;
  familyOverride: string | null;
}

export interface LoraSelection {
  path: string;
  scale: number;
  /** Trigger phrases (Civitai `trainedWords`) for the chosen LoRA. Carried
   * on the form-state row so the picker can render click-to-insert chips
   * without a second catalog lookup. Optional — populated when the user
   * picked the LoRA via the catalog-backed dropdown; empty otherwise. */
  trainedWords?: string[];
}

/// Soft cap on stacked LoRAs in the web UI. The inference engine has no
/// hard limit (the merge is `W' = W + Σ deltas`) but each adapter adds
/// matmul work and disk I/O at build time, so 4 is a sane UX ceiling.
export const MAX_LORA_STACK = 4;

/// Families whose engines actually merge LoRA adapters today. Mirrors
/// `crates/mold-tui/src/model_info.rs::capabilities_for_family` and the
/// server-side gate in `mold-core/src/validation.rs`. Keep all three in
/// sync — divergence shows up as a UI that lets the user pick a LoRA the
/// server then rejects.
export const LORA_CAPABLE_FAMILIES = [
  "flux",
  "flux2",
  "ltx2",
  "sd15",
  "sd3",
  "sdxl",
  "qwen-image",
  "qwen-image-edit",
  "z-image",
] as const;

export interface GenerateFormState {
  version: 3;
  prompt: string;
  /** Active style preset id (see `lib/stylePresets`). `null` = no style. The
   * preset's extras are appended to the outgoing prompt at request time; the
   * textarea content (`prompt`) is never rewritten by the style row. */
  stylePreset: string | null;
  negativePrompt: string;
  model: string;
  /** Family for the selected model. Stored so request serialization can
   * choose family-specific wire fields without needing the model catalog. */
  modelFamily: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seedMode: SeedMode;
  seed: number | null; // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  fps: number | null;
  scheduler: Scheduler | null;
  cfgPlus: boolean;
  outputFormat: OutputFormat;
  expand: ExpandFormState;
  sourceFitPolicy?: SourceFitPolicy;
  imageAttachments: SourceImageState[];
  maskImage: SourceImageState | null;
  controlImage: SourceImageState | null;
  controlModel: string;
  controlScale: number;
  upscaleModel: string;
  gifPreview: boolean;
  audioFile: SourceMediaState | null;
  audioFilePath: string;
  sourceVideo: SourceMediaState | null;
  sourceVideoPath: string;
  keyframes: KeyframeConditionState[];
  pipeline: Ltx2PipelineMode | null;
  retakeRange: TimeRange | null;
  spatialUpscale: Ltx2SpatialUpscale | null;
  temporalUpscale: Ltx2TemporalUpscale | null;
  /** LTX-2 camera preset id or an explicit .safetensors LoRA path. */
  cameraControl: string | null;
  placement: DevicePlacement | null;
  /** LoRA stack. Stored as an array so the UI can hold multiple
   * selections; serialized as `loras` on the wire. Defaults to an empty
   * array; absence-of-LoRA is `loras.length === 0`, never `null`. */
  loras: LoraSelection[];
  /** Per-form audio toggle. `true`/`false` send the corresponding
   * `enable_audio` on the wire; `null` omits the field so the server's
   * MP4 default-on behavior takes over. Auto-set to `true` when the
   * selected model's family supports audio (LTX-2 / LTX-2.3); otherwise
   * forced to `null` so the wire stays clean. */
  enableAudio: boolean | null;
}

// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
// Mirror of `mold_core::types::{DownloadJob, JobStatus, DownloadEvent,
// DownloadsListing}`. Keep field names / string literals in sync with the
// server's serde output.

export type JobStatusWire =
  "queued" | "active" | "completed" | "failed" | "cancelled";

export interface DownloadJobWire {
  id: string;
  model: string;
  catalog_id?: string | null;
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
  active_jobs?: DownloadJobWire[];
  active?: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
}

export type DownloadEventWire =
  /// First frame of any new SSE subscription — full queue snapshot so a
  /// fresh client paints current state without waiting for the next
  /// delta. The reducer replaces all state with the listing payload.
  | {
      type: "snapshot";
      listing: {
        active_jobs?: DownloadJobWire[];
        active?: DownloadJobWire | null;
        queued: DownloadJobWire[];
        history: DownloadJobWire[];
      };
    }
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
  | { type: "job_cancelled"; id: string }
  /// All jobs (primary + companions) for a catalog entry have settled.
  /// Emitted exactly once per catalog download. Listen for this instead
  /// of `job_done` when refreshing the model list after a catalog pull —
  /// the primary's `job_done` fires before companions are necessarily on
  /// disk, which is the "model sometimes doesn't show up" race.
  | { type: "catalog_ready"; id: string; ok: boolean };
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
  kind: CatalogKind;
  file_format: "safetensors" | "gguf" | "diffusers";
  bundling: "separated" | "single-file";
  size_bytes: number | null;
  download_count: number;
  rating: number | null;
  likes: number;
  /** `null` means an older installed sidecar did not classify the model. */
  nsfw: boolean | null;
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
  companion_details?: {
    name: string;
    kind: CatalogKind;
    repo: string;
    size_bytes: number | null;
  }[];
  download_recipe: {
    files: {
      url: string;
      dest: string;
      sha256: string | null;
      size_bytes: number | null;
    }[];
    needs_token: "hf" | "civitai" | null;
  };
  supported: boolean;
  /**
   * True when every file the entry needs is already present under the
   * configured models_dir. Computed server-side per request from the
   * recipe's `dest` paths (Civitai) or the resolved manifest's expected
   * file set (HF). Drives the Download↔Repair button swap in the
   * CatalogDetailDrawer and the "installed" chip on CatalogCard.
   */
  installed: boolean;
  /**
   * Absolute filesystem path to the primary file when installed (Civitai
   * entries only). Null for HF entries or when not installed. Used by the
   * web generate UI to pass `lora.path` directly to the generate request.
   */
  primary_path: string | null;
  created_at: number | null;
  updated_at: number | null;
  added_at: number;
  /** Trigger phrases (Civitai `trainedWords`) for LoRA entries. Empty
   * for non-LoRA rows or when the upstream API didn't supply any. The
   * SPA renders these as click-to-insert chips inside the LoRA picker. */
  trained_words?: string[];
  /** Upstream model page (civitai.com / huggingface.co). Absent on older
   * servers; the model detail drawer links out to it when present. */
  page_url?: string | null;
}

export interface CatalogListResponse {
  entries: CatalogEntryWire[];
  page: number;
  page_size: number;
  /** Total rows matching the request's filters, ignoring pagination. */
  total: number;
}

export interface CatalogFamilyCount {
  family: string;
}

export interface CatalogFamiliesResponse {
  families: CatalogFamilyCount[];
}

export interface CatalogListParams {
  family?: string;
  kind?: CatalogKind;
  source?: "hf" | "civitai";
  q?: string;
  include_nsfw?: boolean;
  /** Server vocabulary (`catalog.sort` capability); "name" was retired —
   *  no upstream supports it and the server rejects it with a 422. */
  sort?: "downloads" | "rating" | "recent";
  page?: number;
  page_size?: number;
}

export type CatalogKind =
  | "checkpoint"
  | "lora"
  | "vae"
  | "text-encoder"
  | "tokenizer"
  | "clip"
  | "control-net";

export interface GenerationMemoryEstimate {
  model: string;
  peak_memory_bytes: number;
  activation_memory_bytes: number;
  available_memory_bytes?: number | null;
  load_strategy: string;
  fits_available_memory?: boolean | null;
}

export interface ModelComponentStatus {
  kind: string;
  name: string;
  present: boolean;
  path?: string | null;
  repair_model?: string | null;
  options?: ModelComponentOption[];
}

export interface ModelComponentOption {
  label: string;
  path: string;
  present: boolean;
}

export interface ModelComponentsResponse {
  model: string;
  components: ModelComponentStatus[];
}
