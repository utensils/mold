/** Shared server response and request fragments.
 * Authority: mold-core/src/types.rs, chain.rs, chain_job.rs, ltx2_control.rs
 * and ltx2_camera.rs. Optional additive fields accept older server responses.
 * Desktop-only synthesized metadata/telemetry lives in its legacy adapter. */
import type { MeshWorkflowProvenance } from "../meshWorkflowProvenance";
import type { ChainOutputMetadata, ChainScript } from "./chainTypes";
import type { OutputFormat } from "../generated/generationProfileV1";
import type { GenerationReferenceMetadata } from "../generationReferences";
import type { MiniMaxH3Capability } from "../minimaxH3Inventory";
import type { Ltx2GuidanceOverrides } from "../guidanceOverrides";
import type { GenerationScheduler } from "../generationCapabilities";
import type { MeshRequestOptions } from "../meshControls";
import type { MeshExportGeometryCapabilities } from "../meshExport";
import type { GenerationAsset } from "../../api/generationAssets";
import type {
  GalleryOrganizationFields,
  GalleryTrashCapabilities,
} from "./galleryOrganization";
import type { DurableMediaCapabilities } from "../../api/generationAdmission";
import type { HostMemorySnapshot } from "../hostMemory";
export type {
  ChainJobSummary,
  ChainJobState,
  CreateChainJobResponse,
} from "./chainTypes";
export type { OutputFormat } from "../generated/generationProfileV1";

export type Scheduler =
  | GenerationScheduler
  | { ddim: unknown }
  | { "euler-ancestral": unknown }
  | { "uni-pc": unknown }
  | { "edm-dpm-pp-2m": unknown };

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

export type Ltx2SpatialUpscale = "x1-5" | "x2";

export type Ltx2TemporalUpscale = "x2";

export interface TimeRange {
  start_seconds: number;
  end_seconds: number;
}

export interface OutputMetadata {
  output_mode?: "one-shot" | "sequence" | null;
  chain_job_id?: string | null;
  prefix_cache?: "retained" | "recomputed" | null;
  /** The durable 3-D workflow that produced this print, and the role it plays
   * in it. Additive: absent on every print made outside a durable mesh workflow and on
   * every host that predates the field, which reads as an ordinary print. */
  mesh_workflow?: MeshWorkflowProvenance | null;
  /** User-authored print title as it was at creation (D5). Embedded so
   * mirrors carry it; the gallery row's editable title wins for display. */
  title?: string | null;
  /** How long the render took, wall clock, in ms. Additive; absent on an
   * older host or a print made before it was recorded, zero = not measured. */
  generation_time_ms?: number | null;
  /** Tags the print was filed under at creation ("File under"), exactly as
   * the host applied them. The gallery row's links are the editable
   * authority once the print exists. Additive. */
  tags?: string[] | null;
  /** Display name of the collection the print was filed into at creation —
   * never the requested id, and never a name the host did not resolve. */
  collection?: string | null;
  prompt: string;
  negative_prompt?: string | null;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  model: string;
  /** The family the host resolved this render to. Additive: absent on every
   * print made before it was recorded, which reads as "unknown family", never
   * as a refusal. It is the only answer a client has for a print whose model
   * is not installed anywhere. */
  family?: string | null;
  seed: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  generation_width?: number | null;
  generation_height?: number | null;
  strength?: number | null;
  /** Source-image provenance supplied by newer servers. */
  source_image_name?: string | null;
  source_image_sha256?: string | null;
  /** Face-identity (PuLID) provenance — names and digests only, never the
   * face payload (#1224). Additive: absent on servers that predate identity
   * conditioning AND on every print that carried no identity photo. Read it
   * through `@studio/lib/identityConditioning`'s `identityProvenance`. */
  id_image_name?: string | null;
  id_image_sha256?: string | null;
  /** Effective values the render actually applied, not the request's. */
  id_weight?: number | null;
  id_start_step?: number | null;
  /** The request asked for a transparent background (only ever `true`;
   * Reuse restores the toggle from it). Additive. */
  transparent_background?: boolean | null;
  /** The STORED file carries alpha (some pixel below full opacity) — also
   * true for an edit of a transparent reference with the toggle off. The
   * Library draws the checkerboard from it (`showsAlphaBed`). Additive. */
  has_alpha?: boolean | null;
  /** Client-shaped source-fit provenance echoed verbatim by newer servers.
   * Parse defensively before restoring. */
  source_fit?: unknown;
  /** Ordered content keys for Qwen Image Edit inputs (newer servers only). */
  edit_image_sha256s?: string[] | null;
  reference_weight?: number | null;
  /** Redacted ordered H3 reference provenance (newer servers only). */
  references?: GenerationReferenceMetadata[] | null;
  /** Ordered byte-free keyframe provenance (newer servers only). */
  keyframes?: KeyframeMetadata[] | null;
  /** Queue id of the generation that produced this print — the server's replay
   * idempotence key, and the exact answer to "did my job produce this?".
   * Absent on hosts that predate it. */
  job_id?: string | null;
  /** Per-clip execution provenance for a stitched output. Web renders such a
   * print like any other and reuses it as a plain one-shot. */
  chain?: ChainOutputMetadata | null;
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
  /** LTX-2 video-only opt-in as recorded at creation (#1037). */
  video_only?: boolean | null;
  audio_file_path?: string | null;
  source_video_path?: string | null;
  extend_video_path?: string | null;
  extend_overlap_frames?: number | null;
  pipeline?: Ltx2PipelineMode | null;
  pipeline_requested?: boolean | null;
  duration_prediction_requested?: boolean | null;
  /** LTX-2 source-image preprocessing actually applied (newer servers). */
  source_preprocessing?: Ltx2SourcePreprocessing | null;
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
  /** The 3-D controls that shaped a mesh print, RESOLVED (request values or
   * the engine defaults). Present only on a mesh print (additive). */
  mesh?: MeshRequestOptions | null;
  version: string;
}

/**
 * How a `/api/models` row is named (`mold_core::model_presentation`). All
 * additive: absent on older servers, and `title`/`variant`/`summary`/
 * `family_label` are absent on catalog installs, whose provider title is
 * `display_name`. Display only — every API call keeps `name`. Read through
 * `studio/lib/modelDisplay.ts`'s `modelPresentation`, never directly.
 */
export interface ModelPresentationFields {
  display_name?: string | null;
  title?: string | null;
  variant?: string | null;
  summary?: string | null;
  family_label?: string | null;
}

export interface GalleryImage extends GalleryOrganizationFields {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  media_version?: string | null;
  metadata_synthetic?: boolean;
  /** Independently downloadable files retained by this print. */
  assets?: GenerationAsset[];
}

export interface MeshServerCapabilities {
  generation: boolean;
  formats: string[];
  /** The export menu is built from THIS list, never a client constant: a
   * host that adds an animated turntable container advertises it here. */
  export_formats: string[];
  textures: boolean;
  /** The size / up-axis / origin contract for geometry exports. ADDITIVE:
   * absent on every host that predates it, and that absence is the only gate
   * a client has — an older server drops the three keys instead of refusing
   * them, so a client that sent them anyway would promise a resize the host
   * never performed. The shape is the studio policy's, not a second copy. */
  export_geometry?: MeshExportGeometryCapabilities | null;
}

export interface ServerCapabilities {
  catalog?: { available: boolean; families: string[] } | null;
  generation_profile_v1?: boolean;
  /** 3-D mesh support (additive; absent on servers that predate it):
   * whether a mesh family has a runnable engine here, the containers this
   * host stores (GLB only), the containers `POST /api/gallery/export/:filename`
   * can transcode a stored mesh into, and whether textures are available. */
  mesh?: MeshServerCapabilities | null;
  /** Restart-safe encrypted request-media queueing. Absent is unsupported. */
  durable_media?: DurableMediaCapabilities | null;
  video_upscale?: {
    available: boolean;
    gallery_image?: boolean;
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
  gallery?: GalleryCapabilities;
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
  /** Host-authored H3 inventory and exact request-envelope facts. Absent on
   * older, non-H3, or incomplete hosts; model_access and runtime_available
   * remain independent hard gates. */
  minimax_h3?: MiniMaxH3Capability | null;
  /** Continuation support. Absent on older servers, which means the Create
   * surfaces must hide the extend controls rather than send a rejected
   * request. */
  video?: {
    can_extend?: boolean;
    extend_default_overlap_frames?: number | null;
  } | null;
  discovery?: { can_browse: boolean } | null;
  /** Stable-URL, header-secret reference ingress. H3 activation remains a
   * separate model_access decision. */
  reference_uploads?: {
    video_normalization?:
      | import("../../api/referenceNormalizationTypes").VideoNormalizationCapabilities
      | null;
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
    pre_render_transfer?: boolean;
    /** Explicit chain restart export and enforced destination installation header. */
    chain_restart?: boolean;
    chain_restart_controls?: boolean;
    transfer_identity?: string;
    can_cancel_all?: boolean;
    can_reorder?: boolean;
    /** Running singleton generations accept cooperative cancellation. */
    cooperative_cancellation?: boolean;
    /** The batch chunk limit for durable admission. Its presence IS the
     * durable-generation contract; there is no separate version probe. */
    heterogeneous_batch_max_outputs?: number | null;
  } | null;
  /** One server-wide, authenticated lifecycle stream. */
  events?: { available?: boolean } | null;
  /** Prompt expansion. `model_present` is the routing input: a host that is
   * known to lack the expander is the one case expansion leaves the
   * generation route. `model` names what to pull; absent on older servers. */
  expand?: {
    configured?: boolean;
    model_present?: boolean | null;
    backend?: "local" | "api";
    remix?: boolean;
    model?: string | null;
  } | null;
}

export interface LoraWeight {
  path: string;
  scale: number;
}

export type DeviceRef =
  | { kind: "auto" }
  | { kind: "cpu" }
  | { kind: "gpu"; ordinal: number }
  | { kind: "device"; id: string };

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

export interface GpuInfo {
  backend?: string | null;
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
}

export interface GpuWorkerStatus {
  ordinal: number;
  name: string;
  vram_total_bytes: number;
  vram_used_bytes: number;
  loaded_model?: string | null;
  state: GpuWorkerState;
}

export interface ServerStatus {
  gallery_storage?: {
    prints: number;
    bytes: number;
    trash_prints: number;
    trash_bytes: number;
  } | null;
  host_memory?: HostMemorySnapshot | null;
  version: string;
  git_sha?: string | null;
  build_date?: string | null;
  models_loaded: string[];
  busy?: boolean;
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

export interface SseChainCompleteEvent {
  script?: ChainScript;
  vram_estimate?: { worst_case_bytes: number; fits: boolean } | null;
  filename?: string | null;
  metadata?: OutputMetadata | null;
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

export type RemixSourceKind = "original" | "current" | "direct";

export type RemixDimension =
  | "composition"
  | "camera"
  | "lighting"
  | "setting"
  | "mood"
  | "movement"
  | "style";

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
  /** `null` on the first sample (sysinfo needs a prior refresh to compute deltas). */
  cpu?: CpuSnapshot | null;
}

export interface CatalogProviderError {
  source: "hf" | "civitai";
  message: string;
  code?: "overloaded" | "rate-limited" | string;
  retry_after_seconds?: number;
}

export interface GenerationMemoryEstimate {
  model: string;
  peak_memory_bytes: number;
  activation_memory_bytes: number;
  available_memory_bytes?: number | null;
  load_strategy: string;
  fits_available_memory?: boolean | null;
  capacity_peak_memory_bytes?: number | null;
  device_capacity_bytes?: number | null;
  fits_device_capacity?: boolean | null;
}

export interface ModelComponentStatus {
  kind: string;
  name: string;
  present: boolean;
  path?: string | null;
  repair_model?: string | null;
  options?: ModelComponentOption[];
}

export interface ModelComponentsResponse {
  model: string;
  components: ModelComponentStatus[];
}

export interface Ltx2SourcePreprocessing {
  profile: { generation: "v2" | "v2_3"; image_crf: number };
  codec: string;
  fit_policy: string;
}

export interface GalleryCapabilities {
  can_delete?: boolean;
  trash?: GalleryTrashCapabilities | null;
  organize?: boolean;
  /** Replay-safe titles/tags/favorites/collection and permanent-delete batches. */
  bulk_mutations?: boolean;
  media_version?: boolean;
  conditional_get?: boolean;
  row_events?: boolean;
  /** A saved print reads back byte-for-byte from
   * `GET /api/gallery/image/{filename}`. Absent means an older server. */
  persists_outputs?: boolean;
  /** `GET /api/gallery/changes` is served (read it through
   * `galleryChangesFeedAdvertised`). Absent means an older server, a
   * DB-less one, or a disabled output directory: keep the full listing. */
  changes_feed?: { protocol_version: number } | null;
}

export type GpuWorkerState = "idle" | "generating" | "loading" | "degraded";

export interface HostDiskUsage {
  total_bytes: number;
  free_bytes: number;
}

export type GpuBackend = "cuda" | "metal";

export interface CpuSnapshot {
  cores: number;
  /** 0-100 averaged across all cores. */
  usage_percent: number;
}

export interface ModelComponentOption {
  label: string;
  path: string;
  present: boolean;
}
