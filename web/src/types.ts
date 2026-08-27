import type { ChainOutputMetadata } from "@studio/lib/api/chainTypes";
import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "@studio/lib/generationReferences";
import type { MinimaxH3AuthoringState } from "@studio/lib/minimaxH3Authoring";
import type { MiniMaxH3Capability } from "@studio/lib/minimaxH3Inventory";
import type { SourceFitPolicy } from "@studio/lib/sourceFit";
import type {
  Ltx2GuidanceOverrides,
  Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import type { GenerationScheduler } from "@studio/lib/generationCapabilities";
import type { WanRecipeState } from "@studio/lib/wanRecipe";
import type { GenerationProfileSet } from "@studio/lib/generationProfile";
import type {
  GalleryOrganizationFields,
  GalleryTrashCapabilities,
} from "@studio/lib/api/galleryOrganization";
import type { DurableMediaCapabilities } from "@studio/api/generationAdmission";

export type { SourceFitPolicy } from "@studio/lib/sourceFit";
export type {
  Collection,
  GalleryOrganizationFields,
  GalleryTrashCapabilities,
  TagCount,
} from "@studio/lib/api/galleryOrganization";

// Matches `mold_core::OutputFormat` on the wire (lowercase strings).
export type OutputFormat =
  "png" | "jpeg" | "gif" | "apng" | "webp" | "mp4" | "wav";

export type SeedMode = "random" | "static" | "increment";

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

/** LTX-2 source-image conditioning preprocessing as actually executed. */
export interface Ltx2SourcePreprocessing {
  profile: { generation: "v2" | "v2_3"; image_crf: number };
  codec: string;
  fit_policy: string;
}
export type Ltx2TemporalUpscale = "x2";

export interface TimeRange {
  start_seconds: number;
  end_seconds: number;
}

/** Mirrors `mold-core`'s kebab-case `Scheduler` plus a `"default"` sentinel
 * meaning "omit the field". `ddim` / `euler-ancestral` are UNet schedulers and
 * `euler` / `dpm-pp` are wan sample solvers; only `uni-pc` is valid for both.
 * The object forms are the serde-tagged shapes older metadata may carry. */
export type Scheduler =
  | GenerationScheduler
  | { ddim: unknown }
  | { "euler-ancestral": unknown }
  | { "uni-pc": unknown };

export interface OutputMetadata {
  /** User-facing authoring mode; independent of internal auto-chaining. */
  output_mode?: "one-shot" | "sequence" | null;
  /** User-authored print title as it was at creation (D5). Embedded so
   * mirrors carry it; the gallery row's editable title wins for display. */
  title?: string | null;
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
  /** Client-shaped source-fit provenance echoed verbatim by newer servers.
   * Parse defensively before restoring. */
  source_fit?: unknown;
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
  /** Queue id of the generation that produced this print — the server's replay
   * idempotence key, and the exact answer to "did my job produce this?".
   * Absent on hosts that predate it. */
  job_id?: string | null;
  /** Per-clip execution provenance for a stitched output. `output_mode`
   * decides whether Reuse settings exposes it as an authored sequence. */
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
  audio_file_path?: string | null;
  source_video_path?: string | null;
  extend_video_path?: string | null;
  extend_overlap_frames?: number | null;
  pipeline?: Ltx2PipelineMode | null;
  pipeline_requested?: boolean | null;
  duration_prediction_requested?: boolean | null;
  /** LTX-2 source-image preprocessing actually applied (newer servers). */
  source_preprocessing?: Ltx2SourcePreprocessing | null;
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
  version: string;
}

/** One `/api/gallery` row. The organization fields (`title`, `tags`,
 * `favorite`, `collections`, `trashed_at`, `purge_at`) are additive: older
 * hosts omit them and every reader treats absence as "not organized". */
export interface GalleryImage extends GalleryOrganizationFields {
  filename: string;
  metadata: OutputMetadata;
  timestamp: number;
  format?: OutputFormat | null;
  size_bytes?: number | null;
  media_version?: string | null;
  metadata_synthetic?: boolean;
}

export type MediaKind = "image" | "animated" | "video" | "audio";

export const VIDEO_FORMATS: ReadonlyArray<OutputFormat> = ["mp4"];
export const AUDIO_FORMATS: ReadonlyArray<OutputFormat> = ["wav"];
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
  if (resolved && AUDIO_FORMATS.includes(resolved)) return "audio";
  if (resolved && ANIMATED_FORMATS.includes(resolved)) return "animated";
  return "image";
}

// Mirror of `mold_core::GalleryCapabilities`. `trash` and `organize` are
// additive: absent means an older host whose DELETE is permanent and whose
// prints cannot be titled, tagged, favorited, or collected — the Library hides
// that UI and keeps the hard-delete wording for such a host.
export interface GalleryCapabilities {
  can_delete?: boolean;
  trash?: GalleryTrashCapabilities | null;
  organize?: boolean;
  /** Replay-safe titles/tags/favorites/collection and permanent-delete batches. */
  bulk_mutations?: boolean;
  media_version?: boolean;
  conditional_get?: boolean;
  row_events?: boolean;
}

// Mirror of `mold_core::ServerCapabilities`.
export interface ServerCapabilities {
  generation_profile_v1?: boolean;
  /** Restart-safe encrypted request-media queueing. Absent is unsupported. */
  durable_media?: DurableMediaCapabilities | null;
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
  };
  /** Host-authored, presentation-only H3 inventory. Current servers omit it;
   * model_access and runtime_available remain independent hard gates. */
  minimax_h3?: MiniMaxH3Capability | null;
  /** Continuation support. Absent on older servers, which means the Create
   * surfaces must hide the extend controls rather than send a rejected
   * request. */
  video?: {
    can_extend?: boolean;
    extend_default_overlap_frames?: number | null;
  };
  discovery?: { can_browse: boolean };
  /** Stable-URL, header-secret reference ingress. H3 activation remains a
   * separate model_access decision. */
  reference_uploads?: {
    available: boolean;
    authless_inline?: boolean;
    protocol_version: number;
    requires_api_key: boolean;
    session_path: string;
    upload_path: string;
    session_handle_header: string;
    upload_handle_header: string;
    max_file_bytes: number;
    max_session_bytes: number;
    session_ttl_ms: number;
  };
  devices?: {
    available?: boolean;
    lifecycle?: boolean;
    restart_enable?: boolean;
    stable_pins?: boolean;
    planned_lanes?: boolean;
    learned_eta?: boolean;
  };
  dispatch?: {
    active_mode?: string | null;
    v2_authoritative?: boolean;
    observes_v2_decisions?: boolean;
    request_placement_preview?: boolean;
  };
  queue?: {
    can_pause?: boolean;
    can_cancel_all?: boolean;
    can_reorder?: boolean;
    /** Running singleton generations accept cooperative cancellation. */
    cooperative_cancellation?: boolean;
    /** The batch chunk limit for durable admission. Its presence IS the
     * durable-generation contract; there is no separate version probe. */
    heterogeneous_batch_max_outputs?: number | null;
  };
  /** One server-wide, authenticated lifecycle stream. */
  events?: { available?: boolean };
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

export function inferFormatFromName(filename: string): OutputFormat | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".mp4")) return "mp4";
  if (lower.endsWith(".wav")) return "wav";
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
  /** User-authored print title (D5). Additive; absent = untitled. */
  title?: string | null;
  /** Creation-time filing ("File under"). Tags the host applies to the
   * print's gallery row as it lands. Additive; absent = file nothing. */
  tags?: string[];
  /** Creation-time collection. Clients send `{ name }` and let the host
   * get-or-create by slug, so one request files correctly on any machine
   * in the fleet. Additive. */
  collection?: { id?: string; name?: string };
  prompt_transform?: PromptTransformProvenanceWire | null;
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
  /** Client-shaped crop/fit policy provenance — the server echoes it into
   * OutputMetadata verbatim; the engine never reads it. */
  source_fit?: SourceFitPolicy | null;
  /** Qwen-Image-Edit attachments in order: first image is the target,
   * subsequent images are references. Mutually exclusive with
   * `source_image`. */
  edit_images?: string[] | null;
  /** Ordered heterogeneous MiniMax H3 Ref2VA inputs. */
  references?: GenerationReference[] | null;
  /** Face-identity (PuLID) reference, base64 PNG/JPEG with no data-URI
   * prefix (#1224). Never fitted or cropped against the canvas — it is a
   * face reference, not a composition input. Admission accepts it only on an
   * identity-qualified checkpoint, and never beside a LoRA or `source_image`;
   * every rule lives in `@studio/lib/identityConditioning`. */
  id_image?: string | null;
  /** Upload label recorded as provenance only when `id_image` exists. */
  id_image_name?: string | null;
  /** `0.0..=3.0`; omit to let the server's own default (1.0) apply. */
  id_weight?: number | null;
  /** First identity-conditioned step, `< steps`; omit for the default (0). */
  id_start_step?: number | null;
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
  /** Existing video to continue, base64 (no data-URI prefix). Makes the
   * request a continuation: the delivered output is this clip followed by the
   * newly rendered frames. Mutually exclusive with `source_video`. */
  extend_video?: string | null;
  /** Server-local path of the video to continue. */
  extend_video_path?: string | null;
  /** Pixel frames of the source tail used as motion context. Must be 8k+1 and
   * strictly less than `frames`; omit to use the server default. */
  extend_overlap_frames?: number | null;
  keyframes?: KeyframeConditionWire[] | null;
  pipeline?: Ltx2PipelineMode | null;
  ic_lora_control?: string | null;
  retake_range?: TimeRange | null;
  spatial_upscale?: Ltx2SpatialUpscale | null;
  temporal_upscale?: Ltx2TemporalUpscale | null;
  guidance_overrides?: Ltx2GuidanceOverrides | null;
  /** Wan flow shift (upstream `--sample_shift`) and the per-expert Lightning
   * distill strengths. Absent keeps the resolved tier's own values; the
   * server rejects — never ignores — any of them off-family. */
  sample_shift?: number | null;
  distill_strength_high?: number | null;
  distill_strength_low?: number | null;
}

export interface ModelDefaults {
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
  /** Omitting frames invokes this model's qualified prompt duration head. */
  supports_duration_prediction?: boolean | null;
  /** Complete split-pack readiness on this host; absent on older servers. */
  runtime_ready?: boolean | null;
  runtime_readiness_error?: string | null;
  /** Model accepts a face-identity (PuLID) photo. Absent on servers that
   * predate identity conditioning — read absence as "no", which is what
   * keeps the control hidden instead of queueing work the host refuses.
   * Always read it through `@studio/lib/identityConditioning`'s
   * `supportsIdentity`, which prefers the server-authored recipe. */
  supports_identity?: boolean | null;
  /** Model can continue an existing video in one request. Absent on servers
   * that predate continuation — read absence as "no". */
  supports_extend?: boolean | null;
  extend_default_overlap_frames?: number | null;
  /** Model-specific sequence support; absent on older servers. */
  supports_sequence?: boolean | null;
  /**
   * Per-model source-image conditioning contract (#772): `"unsupported"`,
   * `"optional"`, or `"required"`. Additive — absent on older servers AND on
   * entries the current server could not classify, in which case the family
   * heuristic answers. Always read it through
   * `generationCapabilitiesForFamily`'s fifth argument, never raw.
   */
  source_image?: string | null;
  guidance_capabilities?: {
    adjustable: boolean;
    supports_negative_prompt: boolean;
    fixed_scale?: number | null;
  } | null;
  /** Versioned server-authoritative generation controls and recipes. */
  generation_profile?: GenerationProfileSet | null;
  /** Tuned default negative prompt the engine applies when a request omits
   * `negative_prompt` entirely (`/api/models`, additive; wan today). Absent
   * on older servers and on families without one. An explicit `""` in a
   * request remains the opt-out — see `@studio/lib/negativePrompt`. */
  default_negative_prompt?: string | null;
  /** Model's own default frame count (`/api/models`, additive) — LTX-2
   * ships 97, LTX-Video 25; absent on older servers and image models. */
  default_frames?: number | null;
  /** Model's own default frame rate (`/api/models`, additive) — LTX-Video
   * ships 30, LTX-2 24; absent on older servers and image models. */
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

/** `held` is additive: a journalled job the host parked after it exhausted its
 * replay or dispatch budget. It exists only in the durable queue, never starts
 * on its own, and is listed so it is not invisible. Absent on older servers. */
export type QueueJobState = "queued" | "running" | "held";

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
  /** Why the host parked this job. Present only for `state: "held"`. */
  held_reason?: string | null;
  /** Terminal error text for a held or failed job. */
  error?: string | null;
  /** The generation settings the job was admitted with (`OutputMetadata` shape). */
  metadata?: unknown;
  seed_pinned?: boolean | null;
  /** Whether this row was resumed from the durable queue after a restart. */
  replayed?: boolean | null;
  dispatch_attempts?: number | null;
  /** Whether the host journalled THIS job and will run it across a restart.
   * Additive and deliberately per-job: a host that advertises
   * `queue.durable_queue` still reports `false` for a job it excluded at
   * admission. Absent on servers without the durable queue. */
  durable?: boolean;
}

export interface QueueListing {
  entries: QueueEntry[];
  plan?: import("@studio/api/queuePlan").QueuePlan | null;
  live_only_entries?: QueueEntry[];
  page?: import("@studio/api/queuePlan").QueuePage;
}

export type SseProgressEvent =
  | { type: "dependency_wait"; dependency: string; reason: string }
  | { type: "stage_start"; name: string }
  | { type: "stage_done"; name: string; elapsed_ms: number }
  | { type: "stage_progress"; name: string; current: number; total: number }
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
  /** Advisories the server attached to this render: it succeeded, but
   * something was adjusted, dropped, or is worth knowing — the multi-face
   * identity note among them. Additive; absent on older servers and on every
   * render that carried no advisory. An SSE render has no response headers to
   * read, so this is the only delivery `x-mold-request-warning` has here. */
  request_warnings?: string[] | null;
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
  /** Audio-only completion (`pipeline: "t2a"`). `image` then carries the WAV
   * itself, not a raster — probe these before the `video_*` fields, since an
   * audio print has no frames and would otherwise read as a still. */
  audio_sample_rate?: number | null;
  audio_channels?: number | null;
  audio_duration_ms?: number | null;
  /** Rendered waveform PNG, base64. The only image an audio print has. */
  audio_thumbnail?: string | null;
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
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

export interface ChainRequestWire {
  model: string;
  /** An auto-chained one-shot: the machine renders and stitches it, records
   * the print with stage seeds but no chain job id, and deletes the job's
   * artifacts afterwards. Absent for an authored sequence, which is durable
   * and belongs in History. */
  ephemeral?: boolean;
  /** Title for the STITCHED print — a sequence renders one print, so this
   * titles that print and never an intermediate clip. Additive. */
  title?: string | null;
  /** Creation-time filing for the stitched print, same normalization and
   * limits as `GenerateRequestWire.tags`. Additive. */
  tags?: string[];
  /** Creation-time collection for the stitched print. Additive. */
  collection?: { id?: string; name?: string };
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
  output_mode?: "one-shot" | "sequence" | null;
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
  execution_phase?: "queued" | "running" | "finalizing" | null;
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
  has_media?: boolean;
  cache_ready?: boolean;
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

// script is a NEW wire-exact mirror (NOT @studio/lib/chainToml's ChainScript):
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
      has_media?: boolean;
      cache_ready?: boolean;
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
  /** Resolved generation/conditioning policy; no media bytes travel here. */
  task?: ExpandTask;
}

export interface ExpandResponseWire {
  original: string;
  expanded: string[];
}

export type RemixSourceKind = "original" | "current" | "direct";
export type RemixDimension =
  | "composition"
  | "camera"
  | "lighting"
  | "setting"
  | "mood"
  | "movement"
  | "style";
export interface RemixRequestWire {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  model_family: string;
  variations?: number;
  task: ExpandTask;
  style?: string;
  dimensions: RemixDimension[];
}
export interface RemixResponseWire {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  variants: Array<{ prompt: string; dimensions: RemixDimension[] }>;
}
export interface PromptTransformProvenanceWire {
  operation: "expand" | "remix";
  root_prompt?: string;
  source_prompt: string;
  source_kind: RemixSourceKind;
  task: ExpandTask;
  dimensions: RemixDimension[];
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
  name?: string | null;
}

export interface KeyframeMetadata {
  frame: number;
  name?: string | null;
  sha256: string;
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
export { MAX_LORA_STACK } from "@studio/lib/generationCapabilities";

/// Families whose engines actually merge LoRA adapters today. Mirrors
/// `crates/mold-tui/src/model_info.rs::capabilities_for_family` and the
/// server-side gate in `mold-core/src/validation.rs`. Keep all three in
/// sync — divergence shows up as a UI that lets the user pick a LoRA the
/// server then rejects.
export { LORA_CAPABLE_FAMILIES } from "@studio/lib/generationCapabilities";

/** Advanced overrides for the LTX-2 multimodal guider, and their form-side
 * mirror. Both shapes and every parse/serialize rule live in `@studio` so
 * each surface reads the same contract. */
export type {
  Ltx2GuidanceOverrides,
  Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";

export interface GenerateFormState {
  version: 3;
  prompt: string;
  /** Print title typed in Create's title field; rides every request this
   * form builds as `GenerateRequestWire.title`. Optional so persisted
   * pre-title drafts keep loading; `null`/absent = untitled. */
  title?: string | null;
  /** Root user-authored prompt retained across Expand/Remix and Gallery reuse. */
  originalPrompt?: string | null;
  /** Active style preset id (see `lib/stylePresets`). `null` = no style. The
   * preset's extras are appended to the outgoing prompt at request time; the
   * textarea content (`prompt`) is never rewritten by the style row. */
  stylePreset: string | null;
  negativePrompt: string;
  /** The selected model's advertised default negative
   * (`default_negative_prompt`, wan today; "" when none). Optional so
   * persisted pre-#787 form snapshots keep loading; semantics live in
   * `@studio/lib/negativePrompt`. */
  negativePromptDefault?: string;
  /** Restore-time explicit-clear authority (#787 round 3): true when a reuse
   * carried the explicit `""` opt-out while the advertised default was still
   * unknown. Keeps the clear from decaying to "untouched" once the model row
   * resolves; semantics live in `@studio/lib/negativePrompt`. Optional so
   * persisted pre-round-3 snapshots keep loading. */
  negativeExplicitClear?: boolean;
  model: string;
  /** Family for the selected model. Stored so request serialization can
   * choose family-specific wire fields without needing the model catalog. */
  modelFamily: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  guidanceCapabilities?: ModelInfoExtended["guidance_capabilities"];
  /** The selected model's advertised source-image contract (#772),
   * snapshotted on model change exactly like `guidanceCapabilities`. `null`
   * means the host advertised nothing and the family heuristic answers. */
  sourceImageCapability: string | null;
  /** Wan first/last-frame conditioning (#779): the closing still. Offered
   * only on a checkpoint whose advertised contract accepts a source image,
   * and meaningless without one — the pair ships as the two-entry `keyframes`
   * layout, never a lone keyframe. */
  endFrame: SourceImageState | null;
  /** Face-identity (PuLID) photo staged in the primary form (#1224). It is
   * never fitted against the canvas, so it carries no fit policy. Optional so
   * persisted pre-identity drafts keep loading; `null`/absent = none. */
  identityImage?: SourceImageState | null;
  /** `null` = untouched, which keeps `id_weight` off the wire so the
   * server's own default stays authoritative. Optional for the same reason
   * as `identityImage`. */
  identityWeight?: number | null;
  /** `null` = untouched; see `identityWeight`. */
  identityStartStep?: number | null;
  /** The selected model's advertised identity support, snapshotted on model
   * change exactly like `guidanceCapabilities` / `sourceImageCapability`:
   * request assembly runs long after the catalog row went out of scope.
   * `null`/absent means nothing has been read yet, which reads as "no". */
  identitySupported?: boolean | null;
  seedMode: SeedMode;
  seed: number | null; // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  /** When true, omit frames and let a qualified LTX-2.5 duration head decide. */
  predictDuration?: boolean;
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
  /** Existing video to continue. Set makes the request a continuation. */
  extendVideo: SourceMediaState | null;
  extendVideoPath: string;
  /** Pixel-frame overlap; `null` uses the server's advertised default. */
  extendOverlapFrames: number | null;
  keyframes: KeyframeConditionState[];
  pipeline: Ltx2PipelineMode | null;
  /** Official host-provided IC-LoRA control adapter ID. */
  icLoraControl?: string | null;
  retakeRange: TimeRange | null;
  spatialUpscale: Ltx2SpatialUpscale | null;
  temporalUpscale: Ltx2TemporalUpscale | null;
  /** Advanced LTX-2 guider overrides. Every entry is null/empty until the
   * user touches it, which is what keeps the request free of an override
   * object and the render on its pipeline defaults. Optional because saved
   * templates predate the field and are restored verbatim. */
  guidanceOverrides?: Ltx2GuidanceOverridesState;
  /** Wan flow shift and per-expert distill strengths. Null until touched for
   * the same reason as `guidanceOverrides`: the field must stay off the wire
   * so the resolved tier keeps its own value. Optional because saved
   * templates predate it. */
  wanRecipe?: WanRecipeState;
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
  /** MiniMax H3 first/last endpoints or ordered heterogeneous references.
   * Kept separate from legacy edit/source fields so no surface can flatten
   * Ref2VA into image-only editing. */
  h3Authoring?: MinimaxH3AuthoringState;
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
  provider_errors?: CatalogProviderError[];
}

export interface CatalogProviderError {
  source: "hf" | "civitai";
  message: string;
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

export interface ModelComponentOption {
  label: string;
  path: string;
  present: boolean;
}

export interface ModelComponentsResponse {
  model: string;
  components: ModelComponentStatus[];
}
