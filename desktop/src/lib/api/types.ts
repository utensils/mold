/**
 * Hand-mirrored subsets of mold-core wire types (serde snake_case).
 * Source of truth: /api/openapi.json — a drift snapshot test guards these
 * once the OpenAPI harness lands (see desktop/docs/architecture.md §5).
 */

import type { ChainOutputMetadata, ChainRequestWire } from "@studio/lib/api/chainTypes";
import type { HostMemorySnapshot } from "@studio/lib/hostMemory";
import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "@studio/lib/generationReferences";
import type { Ltx2GuidanceOverrides } from "@studio/lib/guidanceOverrides";
import type { GenerationScheduler } from "@studio/lib/generationCapabilities";
import type { OutputFormat as WireOutputFormat } from "@studio/lib/generated/generationProfileV1";
import type { MiniMaxH3Capability } from "@studio/lib/minimaxH3Inventory";
import type { GenerationProfileSet } from "@studio/lib/generationProfile";
import type { SourceFitPolicy } from "@studio/lib/sourceFit";
import type {
  GalleryCollectionsChangedEvent,
  GalleryOrganizationFields,
  GalleryRestoredEvent,
  GalleryTrashCapabilities,
  GalleryTrashedEvent,
  GalleryUpdatedEvent,
} from "@studio/lib/api/galleryOrganization";
import type { DurableMediaCapabilities } from "@studio/api/generationAdmission";

// Library organization wire shapes are shared across surfaces; re-export the
// pieces desktop consumers reach for so `lib/api/types` stays the single
// desktop import for wire types.
export type {
  Collection,
  GalleryOrganizationFields,
  GalleryTrashCapabilities,
  TagCount,
} from "@studio/lib/api/galleryOrganization";
// Durable generation admission and outcome types are singular across web,
// desktop and mobile. Surfaces may keep importing them from this desktop
// facade while the shared client/reducer remains the authority.
export type {
  GenerationBatchAdmissionRequest,
  GenerationBatchChild,
  GenerationBatchResult,
  GenerationBatchStatus,
  GenerationBatchStatusRequest,
  GenerationBatchStatusResponse,
  GenerationLifecyclePhase,
} from "@studio/api/generationAdmission";

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
  /** `MemAvailable`; additive on newer servers. */
  available?: number;
  /** Evictable ZFS ARC beside `available`, never inside it (#1439). */
  reclaimable_zfs_arc?: number;
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
  /** Whether this host is currently holding queued work from dispatch. */
  queue_paused?: boolean | null;
  /** Stable server-installation UUID; absent on older servers. */
  instance_id?: string | null;
  /** Disk stats for the filesystem holding the models dir; absent on older servers. */
  models_disk?: { total_bytes: number; free_bytes: number } | null;
  /** Host-RAM ledger snapshot; absent on older servers. This is the FRESHER
   * copy — the queue plan only republishes its mirror when a plan is emitted
   * for some other reason. Read it through `hostMemoryLevel`, which validates
   * before anything renders. */
  host_memory?: HostMemorySnapshot | null;
}

export type ExpandBackend = "local" | "api";

export interface ExpandCapabilities {
  configured: boolean;
  model_present: boolean | null;
  backend: ExpandBackend;
  /** The manifest model local expansion resolves. Additive: absent on API
   * backends and on servers that predate the field, where clients fall back
   * to `DEFAULT_EXPAND_MODEL`. */
  model?: string | null;
}

export interface ServerCapabilities {
  generation_profile_v1?: boolean;
  /** Restart-safe encrypted request-media queueing. Absent is unsupported. */
  durable_media?: DurableMediaCapabilities | null;
  video_upscale?: {
    available: boolean;
    contract_version: number;
    source_library: boolean;
    source_upload: boolean;
    input_containers: string[];
    output_container: string;
    preserves_primary_audio_when_compatible: boolean;
    supports_vfr: boolean;
    supports_hdr: boolean;
    disclosure: string;
  } | null;
  gallery: {
    can_delete: boolean;
    /** Trash support (soft delete + retention). Absent on older servers,
     * which hard-delete, and when the metadata DB is disabled. */
    trash?: GalleryTrashCapabilities | null;
    /** Titles / favorites / tags / collections can be edited here. Absent
     * on older servers ⇒ hide the organization UI. */
    organize?: boolean;
    /** Replay-safe bulk organization endpoint. */
    bulk_mutations?: boolean;
    media_version?: boolean;
    conditional_get?: boolean;
    row_events?: boolean;
  };
  /** Server-enforced model families that are not activated in this build. */
  model_access?: {
    restrictions: Array<{
      code: string;
      family: string;
      message: string;
      license_url: string;
      authorization_url: string;
    }>;
  } | null;
  /** Host-authored, presentation-only H3 inventory. Current servers omit it;
   * model_access and runtime_available remain independent hard gates. */
  minimax_h3?: MiniMaxH3Capability | null;
  /** Continuation support. Absent on older servers, which means the Create
   * surfaces must hide the extend controls rather than send a rejected
   * request. */
  video?: {
    can_extend?: boolean;
    extend_default_overlap_frames?: number | null;
  } | null;
  catalog?: { available: boolean; families: string[] } | null;
  /** Server-assisted DNS-SD browse support; absent on older servers. */
  discovery?: { can_browse: boolean } | null;
  /** Absent on servers that predate `GET /api/events`. */
  events?: { available: boolean } | null;
  /** Stable-URL, header-secret reference ingress. Model access is advertised
   * separately and may still keep H3 legally unavailable. */
  reference_uploads?: {
    available: boolean;
    protocol_version: number;
    requires_api_key: boolean;
    session_path: string;
    upload_path: string;
    session_handle_header: string;
    upload_handle_header: string;
    max_file_bytes: number;
    max_session_bytes: number;
    max_active_sessions: number;
    session_ttl_ms: number;
  } | null;
  /** Live lifecycle and restart-only recovery support. */
  devices?: {
    available?: boolean;
    lifecycle?: boolean;
    restart_enable?: boolean;
    stable_pins?: boolean;
    planned_lanes?: boolean;
    learned_eta?: boolean;
  } | null;
  dispatch?: {
    active_mode?: string | null;
    v2_authoritative?: boolean;
    observes_v2_decisions?: boolean;
    request_placement_preview?: boolean;
  } | null;
  queue?: {
    can_pause?: boolean;
    can_pause_job?: boolean;
    can_cancel_all?: boolean;
    can_reorder?: boolean;
    cooperative_cancellation?: boolean;
    /** The batch chunk limit for durable admission. Its presence IS the
     * durable-generation contract; there is no separate version probe. */
    heterogeneous_batch_max_outputs?: number | null;
  } | null;
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
  max_pixels?: number | null;
  /**
   * Per-axis ceiling, independent of `max_pixels` (additive). Per model,
   * not per family: a checkpoint that ships the spatial upsampler composes
   * stage 1 at half size plus a tiled stage-2 refinement and reaches twice
   * the trained RoPE span; one that does not is capped at the span.
   */
  max_axis_pixels?: number | null;
  recommended_dimensions?: { width: number; height: number }[];
  dimension_alignment?: number | null;
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
  supports_duration_prediction?: boolean | null;
  runtime_ready?: boolean | null;
  runtime_readiness_error?: string | null;
  /** Checkpoint accepts a face-identity (PuLID) photo. Additive: absent on a
   * server that predates identity conditioning and on a build without the
   * adapter, and absence reads as "no" — offering the control optimistically
   * would only queue work the host refuses. Read it through
   * `@studio/lib/identityConditioning`, never raw. */
  supports_identity?: boolean | null;
  /** Model can continue an existing video in one request. Absent on servers
   * that predate continuation — read absence as "no". */
  supports_extend?: boolean | null;
  extend_default_overlap_frames?: number | null;
  /** Explicit durable sequence eligibility; absent on older servers. */
  supports_sequence?: boolean | null;
  /** Per-model source-image conditioning contract — `"unsupported"`,
   * `"optional"`, or `"required"`. Absent on older servers and on entries the
   * server could not classify; read it through
   * `@studio/lib/sourceImageCapability`, never raw. */
  source_image?: string | null;
  guidance_capabilities?: {
    adjustable: boolean;
    supports_negative_prompt: boolean;
    fixed_scale?: number | null;
  } | null;
  /** Versioned server-authoritative generation controls and recipes. */
  generation_profile?: GenerationProfileSet | null;
  /** Tuned default negative prompt the engine applies when a request omits
   * `negative_prompt` entirely (additive; wan today). Absent on older
   * servers and on families without one. An explicit `""` in a request
   * remains the opt-out — see `@studio/lib/negativePrompt`. */
  default_negative_prompt?: string | null;
  /** Server-advertised per-clip frame default (LTX-2 ships 97, LTX-Video
   * 25); absent on older servers. Sizes new sequence clips. */
  default_frames?: number | null;
  /** Server-advertised frame rate (LTX-Video ships 30, LTX-2 24); absent on
   * older servers and image models. Applied like steps/guidance. */
  default_fps?: number | null;
  /** Minimum requestable frame count; omitted means the legacy floor of one. */
  min_frames?: number | null;
  /** Requestable single-shot frame ceiling at `default_fps`. */
  max_frames?: number | null;
  /** Duration-based ceiling; clients recompute max frames when FPS changes. */
  max_runtime_seconds?: number | null;
  /** FPS-independent resource guard paired with `max_runtime_seconds`. */
  max_frames_absolute?: number | null;
  /** Valid frame counts are `k * frame_step + frame_offset` (offset defaults to 1). */
  frame_step?: number | null;
  /** Frame-grid offset; omitted means 1. MiniMax H3 advertises 5. */
  frame_offset?: number | null;
  /** Explicit runnable-contract boundary for future gated families. */
  runtime_available?: boolean | null;
  /** One sentence naming why `runtime_available` is false — a missing engine
   * arm for the weight layout, a task with no qualified route, or a build
   * compiled without the engine. Present exactly when `runtime_available` is
   * false; absent on servers that predate it (#1276). Read it through
   * `@studio/lib/modelRuntimeAvailability`, never by matching the text. */
  runtime_unavailable_reason?: string | null;
}

// ── Generation ───────────────────────────────────────────────────────────

/// Re-exported from the GENERATED mirror of `mold_core::OutputFormat` rather
/// than restated here.
///
/// This file used to declare its own copy, and in #1495 that copy went stale:
/// `web/src/types.ts` gained `glb` and this one did not, so the desktop build
/// failed on a comparison TypeScript could prove was always false. That is the
/// lucky outcome — the same divergence in `generationProfile.ts`'s runtime
/// format gate silently discarded a whole server-authored profile instead.
///
/// An alias cannot drift. Adding a variant in Rust and regenerating reaches
/// every surface at once.
///
/// `obj` is never a STORED format — mold only produces one as a gallery
/// export — but it is in the union so a hand-placed `.obj` classifies as a
/// mesh rather than falling through to the image branch.
export type OutputFormat = WireOutputFormat;

/** Solver override, spelled exactly as mold-core's kebab-case enum plus a
 * `"default"` sentinel meaning "omit the field". `ddim` / `euler-ancestral`
 * are UNet schedulers (SD1.5, SDXL) and `euler` / `dpm-pp` are wan's flow
 * sample solvers; only `uni-pc` is accepted by both. */
export type Scheduler = GenerationScheduler;

// ── LTX-2 advanced video (mold-core `Ltx2*`, kebab-case on the wire) ────────

/** Explicit LTX-2 pipeline mode. Mirrors mold-core `Ltx2PipelineMode`. */
export type Ltx2PipelineMode =
  | "one-stage"
  | "two-stage"
  | "two-stage-hq"
  | "distilled"
  | "ic-lora"
  | "keyframe"
  | "a2-vid"
  | "retake"
  | "lip-dub"
  | "t2a";
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
  name?: string | null;
}

export interface KeyframeMetadata {
  frame: number;
  name?: string | null;
  sha256: string;
}

/** One entry in a LoRA stack. `path` is the server-side safetensors path
 * (`LoraInfo.path`); `scale` is 0–2, 1 = full strength. Mirrors mold-core
 * `LoraWeight`. */
export interface LoraWeight {
  path: string;
  scale: number;
}

export type DeviceRef = { kind: "auto" } | { kind: "cpu" } | { kind: "gpu"; ordinal: number };

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
export type ExpandTask =
  | "text-to-image"
  | "text-to-video"
  | "image-to-video"
  | "video-to-video"
  | "retake"
  | "keyframe-interpolation"
  | "audio-driven-video"
  | "reference-to-audio-video"
  | "text-to-audio";

export interface ExpandRequest {
  prompt: string;
  model_family?: string;
  variations?: number;
  /** Natural-language visual-style directive the server weaves into the
   * expander's system message (additive; absent on the wire when unset). */
  style?: string;
  /** Resolved generation/conditioning policy; no media bytes travel here. */
  task?: ExpandTask;
}

/** `POST /api/expand` — mirrors mold-core `ExpandResponse`. */
export interface ExpandResponse {
  original: string;
  expanded: string[];
}

export type RemixSourceKind = "original" | "current" | "direct";
export type RemixDimension =
  "composition" | "camera" | "lighting" | "setting" | "mood" | "movement" | "style";

export interface RemixRequest {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  model_family: string;
  variations?: number;
  task: ExpandTask;
  style?: string;
  dimensions: RemixDimension[];
}

export interface RemixResponse {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  variants: Array<{ prompt: string; dimensions: RemixDimension[] }>;
}

export type PromptTransformOperation = "expand" | "remix";
export interface PromptTransformProvenance {
  operation: PromptTransformOperation;
  root_prompt?: string;
  source_prompt: string;
  source_kind: RemixSourceKind;
  task: ExpandTask;
  dimensions?: RemixDimension[];
}

/** `POST /api/generate/estimate` — mirrors mold-core `GenerationMemoryEstimate`. */
export interface GenerationMemoryEstimate {
  model: string;
  peak_memory_bytes: number;
  activation_memory_bytes: number;
  available_memory_bytes?: number | null;
  load_strategy: string;
  fits_available_memory?: boolean | null;
  /** Stable requirement resolved against physical capacity, not current load. */
  capacity_peak_memory_bytes?: number | null;
  /** Total capacity of the GPU used for the stable requirement. */
  device_capacity_bytes?: number | null;
  /** Stable, family-specific fit verdict resolved against that capacity. */
  fits_device_capacity?: boolean | null;
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
  prompt_transform?: PromptTransformProvenance | null;
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
  /** Client-shaped source-fit policy provenance, echoed verbatim into
   * OutputMetadata so crop controls restore on reuse. Engine never reads it. */
  source_fit?: SourceFitPolicy;
  /** Face-identity (PuLID) reference photo, base64 (no data-URI prefix).
   * Deliberately NOT a composition input: it is never fitted, cropped, or
   * resized against the canvas and carries no `source_fit` provenance. */
  id_image?: string;
  /** Ordered multi-photo identity form, mutually exclusive with id_image. */
  id_images?: string[];
  id_image_names?: string[];
  /** Provenance label for `id_image` — recorded into OutputMetadata (with the
   * digest, never the bytes) so Reuse settings can look the photo back up. */
  id_image_name?: string;
  /** Identity strength, `0.0..=3.0`. Absent takes the server's own default. */
  id_weight?: number;
  /** First identity-conditioned denoise step; must be below `steps`. Absent
   * takes the server's own default. */
  id_start_step?: number;
  /** Qwen-Image-Edit multi-image inputs, base64 each (no data-URI prefix).
   * Order is load-bearing: first = primary edit target, rest = references. */
  edit_images?: string[];
  /** Ordered heterogeneous MiniMax H3 Ref2VA inputs. */
  references?: GenerationReference[];
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
  /** User-authored print title (Library organization, D5). Additive: the
   * server embeds it into `OutputMetadata.title`, seeds the gallery row, and
   * folds a lossy slug into the output filename. Absent = untitled. */
  title?: string;
  /** Creation-time filing ("File under"): tags applied to the print the
   * moment it lands. Additive and ABSENT when nothing is filed — never `[]`.
   * Normalized and capped server-side; `mold_core::MAX_REQUEST_TAGS`. */
  tags?: string[];
  /** Creation-time collection. Clients send `{ name }` and let the routed
   * host get-or-create it by slug, so one request files correctly on any
   * machine in the fleet; `id` is only ever a host-local `Collection.id`. */
  collection?: { id?: string; name?: string };
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
  /** LTX-2 video-only opt-in (#1037): true skips the audio branch. */
  video_only?: boolean;
  // LTX-2 advanced video (ltx2 only). Omitted → engine auto-selects.
  /** Conditioning audio for the a2vid (audio-to-video) pipeline, base64 (no
   * data-URI prefix). mold-core `audio_file`. */
  audio_file?: string;
  /** Source video for video-to-video / retake, base64 (no data-URI prefix). */
  source_video?: string;
  /** Existing video to continue, base64 (no data-URI prefix). Makes the
   * request a continuation: the delivered output is this clip followed by the
   * newly rendered frames. Mutually exclusive with `source_video`. */
  extend_video?: string;
  /** Server-local path of the video to continue. */
  extend_video_path?: string;
  /** Pixel frames of the source tail used as motion context. Must be 8k+1 and
   * strictly less than `frames`; omit to use the server default. */
  extend_overlap_frames?: number;
  keyframes?: KeyframeConditionWire[];
  pipeline?: Ltx2PipelineMode;
  ic_lora_control?: string;
  retake_range?: TimeRange;
  spatial_upscale?: Ltx2SpatialUpscale;
  temporal_upscale?: Ltx2TemporalUpscale;
  /** Optional LTX-2 guider overrides. Absent fields keep pipeline defaults. */
  guidance_overrides?: Ltx2GuidanceOverrides | null;
  /** Wan flow shift (upstream `--sample_shift`) and the per-expert Lightning
   * distill strengths. Absent keeps the resolved tier's own values; the server
   * rejects — never ignores — any of them off-family. */
  sample_shift?: number | null;
  distill_strength_high?: number | null;
  distill_strength_low?: number | null;
  placement?: DevicePlacement | null;
}

export interface Ltx2ControlAdapterInfo {
  id: string;
  label: string;
  guide: string;
  size_bytes: number;
  installed: boolean;
  download_model: string;
  download_repo: string;
  download_filename: string;
  download_sha256: string;
}

export interface Ltx2CameraControlInfo {
  id: string;
  label: string;
  size_bytes: number;
  installed: boolean;
  download_model: string;
  download_repo: string;
  download_filename: string;
  download_sha256: string;
}

/** serde tag = "type", snake_case — /api/generate/stream `progress` events. */
export type ProgressEvent =
  | { type: "queued"; position: number; id?: string }
  | { type: "dependency_wait"; dependency: string; reason: string }
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "stage_progress"; name: string; current: number; total: number }
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
  /** Advisories the server attached to this render: it succeeded, but
   * something was adjusted, dropped, or is worth knowing — the multi-face
   * identity note among them. Additive; absent on older servers and on every
   * render that carried no advisory. An SSE render has no response headers to
   * read, so this is the only delivery `x-mold-request-warning` has here. */
  request_warnings?: string[] | null;
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
  /** Audio-only completion (`pipeline: "t2a"`). `image` then carries the WAV
   * itself, not a raster — probe these before the `video_*` fields, since an
   * audio print has no frames and would otherwise read as a still. */
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
  audio_duration_ms?: number | null;
  /** Rendered waveform PNG, base64. The only image an audio print has. */
  audio_thumbnail?: string | null;
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
  /** User-facing authoring mode; independent of internal auto-chaining. */
  output_mode?: "one-shot" | "sequence" | null;
  prompt: string;
  /** Creation-time print title; the gallery row is the editable authority. */
  title?: string | null;
  /** Tags the print was filed under at creation, exactly as applied. The
   * gallery row's tag links are the editable authority once it exists. */
  tags?: string[] | null;
  /** Display name of the collection the print was filed into at creation —
   * never the requested id, and never a name the host did not resolve. */
  collection?: string | null;
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
  /** Client-shaped source-fit provenance echoed verbatim by the server
   * (additive; newer servers only). Parse defensively before restoring. */
  source_fit?: unknown;
  /** Provenance label of the identity photo (additive; newer servers only). */
  id_image_name?: string | null;
  /** SHA-256 (hex) of the exact identity-photo bytes that rendered — the local
   * stash key Reuse settings looks the face back up with. Metadata never
   * carries the photo itself. */
  id_image_sha256?: string | null;
  /** Effective identity strength / first conditioned step the render applied
   * (the server records what actually applied, not what the request asked). */
  id_weight?: number | null;
  id_start_step?: number | null;
  /** Ordered content keys for Qwen Image Edit inputs (newer servers only). */
  edit_image_sha256s?: string[] | null;
  /** Redacted ordered H3 reference provenance (newer servers only). */
  references?: GenerationReferenceMetadata[] | null;
  /** Ordered byte-free keyframe provenance (newer servers only). */
  keyframes?: KeyframeMetadata[] | null;
  /** Durable sequence job this print was stitched from. Present only for
   * chain jobs with a server-side record — ephemeral chain outputs and
   * pre-#564 rows carry nothing (additive; newer servers only). */
  chain_job_id?: string | null;
  /** The queue id of the generation that produced this print. The server's own
   * replay idempotence key, and the exact answer to "is this print mine?" —
   * absent on hosts that predate it. */
  job_id?: string | null;
  /** Per-clip execution provenance for a stitched output. `output_mode`
   * decides whether Reuse settings exposes it as an authored sequence. */
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
  /** LTX-2 video-only opt-in as recorded at creation (#1037). */
  video_only?: boolean | null;
  audio_file_path?: string | null;
  source_video_path?: string | null;
  extend_video_path?: string | null;
  extend_overlap_frames?: number | null;
  pipeline?: Ltx2PipelineMode | null;
  pipeline_requested?: boolean | null;
  duration_prediction_requested?: boolean | null;
  /** LTX-2 attention arithmetic the print was rendered with (newer servers):
   * `ltx2-bf16-math` | `ltx2-bf16-flash` | `ltx2-f32-chunked` |
   * `ltx2-metal-sdpa`. Output-changing, so it is recorded, never inferred. */
  attention_path?: string | null;
  /** LTX-2 INT8 ConvRot execution arm the print was rendered with (newer
   * servers): `native-w8a8` | `dequant-cuda` | `dequant-metal` |
   * `dequant-host`. Output-changing between arms, so it is recorded. */
  int8_arm?: string | null;
  ic_lora_control?: string | null;
  retake_range?: TimeRange | null;
  spatial_upscale?: Ltx2SpatialUpscale | null;
  temporal_upscale?: Ltx2TemporalUpscale | null;
  guidance_overrides?: Ltx2GuidanceOverrides | null;
  /** Wan flow shift and per-expert distill strengths, recorded so Reuse
   * settings restores exactly what the print was rendered with. */
  sample_shift?: number | null;
  distill_strength_high?: number | null;
  distill_strength_low?: number | null;
  frames?: number | null;
  fps?: number | null;
  /** mold version that produced the print. */
  version?: string | null;
  /** Legacy desktop-only aliases for `frames` / `fps`; never sent by current
   * servers but kept so older synthesized rows still display. */
  video_frames?: number | null;
  video_fps?: number | null;
}

export interface GalleryImage extends GalleryOrganizationFields {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  media_version?: string | null;
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
  | { type: "job_state_committed"; id: string }
  | { type: "generation_states_committed" }
  | { type: "gallery_added"; filename: string; image?: GalleryImage | null }
  | { type: "gallery_removed"; filename: string }
  | GalleryUpdatedEvent<GalleryImage>
  | GalleryTrashedEvent
  | GalleryRestoredEvent<GalleryImage>
  | GalleryCollectionsChangedEvent
  | { type: "queue_paused" }
  | { type: "queue_resumed" }
  | { type: "queue_plan_changed"; plan: import("@studio/api/queuePlan").QueuePlan }
  | {
      type: "device_state_changed";
      device_id: string;
      desired_enabled: boolean;
      admin_state: "startup_excluded" | "starting" | "enabled" | "draining" | "disabled";
    };

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
  /** A merged search can keep one provider's rows when the other is down. */
  provider_errors?: CatalogProviderError[];
}

export interface CatalogProviderError {
  source: "hf" | "civitai";
  message: string;
  code?: "overloaded" | "rate-limited" | string;
  retry_after_seconds?: number;
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
  /** The persisted value applies when the engine/coordinator restarts. */
  restart_required?: boolean;
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
  output_mode?: "one-shot" | "sequence" | null;
  placement?: DevicePlacement | null;
  /** Title for the STITCHED print (mold-core `ChainRequest.title`). An
   * intermediate clip is a working artifact inside the job dir and never
   * reaches the gallery, so filing applies to the finished timeline only. */
  title?: string;
  /** Creation-time tags for the stitched print. */
  tags?: string[];
  /** Creation-time collection for the stitched print, by name. */
  collection?: { id?: string; name?: string };
}

/**
 * The exact `POST /api/chain-jobs` body desktop submits: studio's canonical
 * clip wire plus the additive creation-time filing the stitched print
 * carries. Kept here rather than in `@studio` because the shared wire is
 * every surface's contract and this is the desktop composer's body type.
 */
export type ChainCreateRequest = ChainRequestWire & {
  title?: string;
  tags?: string[];
  collection?: { id?: string; name?: string };
};

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
  output_mode?: "one-shot" | "sequence" | null;
  /** A stitched long video is still ONE print, so it carries the same title
   * and creation-time filing an unstitched one-shot would. */
  title?: string;
  tags?: string[];
  collection?: { id?: string; name?: string };
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
  fps?: number | null;
  frames_per_clip_runtime_seconds?: number | null;
  frames_per_clip_recommended: number;
  max_stages: number;
  max_total_frames: number;
  fade_frames_max: number;
  transition_modes: string[];
  quantization_family: string;
  supports_audio: boolean;
}

export type ChainJobState =
  "queued" | "running" | "paused" | "interrupted" | "failed" | "completed" | "cancelled";

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
  execution_phase?: "queued" | "running" | "finalizing" | null;
}

export interface ChainJobStageDetail {
  idx: number;
  state: StageState;
  /** u64 effective seed, serialized as a decimal string. */
  seed: string;
  frames_emitted?: number | null;
  generation_time_ms?: number | null;
  has_preview: boolean;
  has_media?: boolean;
  cache_ready?: boolean;
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
  | {
      type: "stage_done";
      stage_idx: number;
      frames_emitted: number;
      has_preview: boolean;
      has_media?: boolean;
      cache_ready?: boolean;
    }
  | { type: "yielded"; pending_small_jobs: number }
  | { type: "finalizing"; total_frames: number }
  | { type: "finalized"; output: string; take: number }
  | { type: "state_changed"; state: ChainJobState; error?: string | null };
