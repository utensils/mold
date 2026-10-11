export type { ChainScript, ChainScriptChain } from "@studio/lib/api/chainTypes";
import type {
  CatalogProviderError,
  DevicePlacement,
  ExpandTask,
  KeyframeConditionWire,
  LoraWeight,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  RemixDimension,
  RemixSourceKind,
  TimeRange,
  ModelPresentationFields,
} from "@studio/lib/api/serverTypes";
export type {
  AdvancedPlacement,
  CatalogProviderError,
  ChainJobState,
  ChainJobSummary,
  ChainProgressEvent,
  CreateChainJobResponse,
  DevicePlacement,
  DeviceRef,
  ExpandTask,
  GenerationMemoryEstimate,
  GpuInfo,
  GpuWorkerStatus,
  KeyframeConditionWire,
  KeyframeMetadata,
  LoraWeight,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  MeshServerCapabilities,
  ModelComponentStatus,
  ModelComponentsResponse,
  RamSnapshot,
  RemixDimension,
  RemixSourceKind,
  ServerCapabilities,
  ServerStatus,
  SseChainCompleteEvent,
  TimeRange,
} from "@studio/lib/api/serverTypes";
import type { OutputMetadata, GalleryImage } from "./legacyTypes";
export type { OutputMetadata, GalleryImage, GpuSnapshot, ResourceSnapshot } from "./legacyTypes";
/**
 * Hand-mirrored subsets of mold-core wire types (serde snake_case).
 * Source of truth: /api/openapi.json — a drift snapshot test guards these
 * once the OpenAPI harness lands (see desktop/docs/architecture.md §5).
 */
import type { MeshWorkflowProvenance } from "@studio/lib/meshWorkflowProvenance";
import type { GenerationReference } from "@studio/lib/generationReferences";
import type { Ltx2GuidanceOverrides } from "@studio/lib/guidanceOverrides";
import type { GenerationScheduler } from "@studio/lib/generationCapabilities";
import type { OutputFormat as WireOutputFormat } from "@studio/lib/generated/generationProfileV1";
import type { GenerationProfileSet } from "@studio/lib/generationProfile";
import type { MeshRequestOptions } from "@studio/lib/meshControls";

export type { MeshRequestOptions } from "@studio/lib/meshControls";
import type { SourceFitPolicy } from "@studio/lib/sourceFit";
import type {
  GalleryCollectionsChangedEvent,
  GalleryRestoredEvent,
  GalleryTrashedEvent,
  GalleryUpdatedEvent,
} from "@studio/lib/api/galleryOrganization";

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

export type ExpandCapabilities = NonNullable<
  import("@studio/lib/api/serverTypes").ServerCapabilities["expand"]
>;

// ── Models ───────────────────────────────────────────────────────────────

/** Flattened ModelInfoExtended (ModelInfo + ModelDefaults + extras). */
export interface ModelEntry extends ModelPresentationFields {
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

export interface ExpandRequest {
  prompt: string;
  model_family?: string;
  variations?: number;
  /** Natural-language visual-style directive the server weaves into the
   * expander's system message (additive; absent on the wire when unset). */
  style?: string;
  /** Resolved generation/conditioning policy; no media bytes travel here. */
  task?: ExpandTask;
  /** Identity, canvas, clip, and ordered references (additive). */
  context?: ExpandContext;
}

/** Mirrors `mold_core::ExpandContext` (see `@studio/lib/expandTask`). */
export type ExpandContext = import("@studio/lib/expandTask").ExpandContext;

/** `POST /api/expand` — mirrors mold-core `ExpandResponse`. */
export interface ExpandResponse {
  original: string;
  expanded: string[];
}

export interface RemixRequest {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  model_family: string;
  variations?: number;
  task: ExpandTask;
  style?: string;
  dimensions: RemixDimension[];
  /** Identity, canvas, clip, and ordered references (additive). */
  context?: ExpandContext;
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

/** Re-exported so a wire type and the policy that reads it cannot drift. */
export type { MeshWorkflowProvenance } from "@studio/lib/meshWorkflowProvenance";

/**
 * Subset of mold-core GenerateRequest the desktop sends.
 *
 * Wire note: mold-core serializes every image field as a base64 STRING in
 * JSON (no `data:` prefix), so `source_image` / `mask_image` / `control_image`
 * are typed as `string` here, not bytes.
 */
export interface GenerateRequest {
  /** Server-minted: which durable 3-D workflow this request is a stage of. A
   * client never sends it — the generate doors refuse one that does. */
  mesh_workflow?: MeshWorkflowProvenance | null;
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
  /** 3-D controls; refused at admission on a recipe with no `mesh` block. */
  mesh?: MeshRequestOptions | null;
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
  /**
   * IP-Adapter injection strength for `edit_images` on an ADDITIVE recipe
   * (`capabilities.reference_images.weight`). Absent takes the server's own
   * default (1.0), so a default-valued render is byte-identical on the wire
   * to one that never named it — and an older host, which drops the unknown
   * field rather than refusing it, renders exactly what it always did.
   */
  reference_weight?: number;
  /**
   * Cut the subject out onto a transparent background
   * (`capabilities.transparency`). Only `true` ever travels — off is the
   * field's absence, so an ordinary render is byte-identical on the wire.
   */
  transparent_background?: boolean;
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
  /** "Save every result" off: the host publishes the print, then moves it
   * straight to the trash. Additive; absent means save. */
  save_to_gallery?: boolean;
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
  /** Post-generate upscaler; video output queues durable Framewise processing. */
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
  /** Mesh completion. `image` then carries the binary glTF and `format` is
   * `glb`; probe `mesh_vertices` BEFORE the audio and video fields, since a
   * mesh has neither samples nor frames. `width`/`height` describe the poster. */
  mesh_vertices?: number | null;
  mesh_faces?: number | null;
  mesh_textured?: boolean;
  /** Rendered poster PNG, base64 — the only raster a mesh print has. */
  mesh_poster?: string | null;
  /** Axis-aligned bounds `[x, y, z]`, carried so a viewer frames a camera. */
  mesh_bounds_min?: [number, number, number] | null;
  mesh_bounds_max?: [number, number, number] | null;
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
  | { type: "gallery_added"; filename: string; image?: GalleryImage | null; imported?: boolean }
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

export type {
  ChainJobDetail,
  ChainJobListing,
  ChainJobStageDetail,
} from "@studio/lib/api/chainTypes";
