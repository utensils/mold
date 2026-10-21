import type { JobStallDiagnostic } from "@studio/lib/jobStall";
import type {
  CatalogProviderError,
  DevicePlacement,
  ExpandTask,
  KeyframeConditionWire,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  OutputMetadata,
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
  CpuSnapshot,
  CreateChainJobResponse,
  DevicePlacement,
  DeviceRef,
  ExpandTask,
  GalleryCapabilities,
  GalleryImage,
  GenerationMemoryEstimate,
  GpuBackend,
  GpuInfo,
  GpuSnapshot,
  GpuWorkerState,
  GpuWorkerStatus,
  HostDiskUsage,
  KeyframeConditionWire,
  KeyframeMetadata,
  LoraWeight,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  Ltx2PipelineMode,
  Ltx2SourcePreprocessing,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  MeshServerCapabilities,
  ModelComponentOption,
  ModelComponentStatus,
  ModelComponentsResponse,
  OutputMetadata,
  RamSnapshot,
  RemixDimension,
  RemixSourceKind,
  ResourceSnapshot,
  ServerCapabilities,
  ServerStatus,
  SseChainCompleteEvent,
  TimeRange,
} from "@studio/lib/api/serverTypes";
import type { OutputFormat as WireOutputFormat } from "@studio/lib/generated/generationProfileV1";
import type { GenerationReference } from "@studio/lib/generationReferences";
import type { MinimaxH3AuthoringState } from "@studio/lib/minimaxH3Authoring";
import type { SourceFitPolicy } from "@studio/lib/sourceFit";
import type {
  Ltx2GuidanceOverrides,
  Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import type { GenerationScheduler } from "@studio/lib/generationCapabilities";
import type { WanRecipeState } from "@studio/lib/wanRecipe";
import type { GenerationProfileSet } from "@studio/lib/generationProfile";
import type {
  MeshFormState,
  MeshRequestOptions,
} from "@studio/lib/meshControls";
import type { NamedViewsState } from "@studio/lib/namedViews";

export type { MeshRequestOptions } from "@studio/lib/meshControls";

export type { SourceFitPolicy } from "@studio/lib/sourceFit";
export type {
  Collection,
  GalleryOrganizationFields,
  GalleryTrashCapabilities,
  TagCount,
} from "@studio/lib/api/galleryOrganization";

// Aliased to the GENERATED mirror of `mold_core::OutputFormat` rather than
// restated. A hand-kept copy is exactly what went stale in #1495 — see the
// note on the same alias in `desktop/src/lib/api/types.ts`.
export type OutputFormat = WireOutputFormat;

export type SeedMode = "random" | "static" | "increment";

/** Mirrors `mold-core`'s kebab-case `Scheduler` plus a `"default"` sentinel
 * meaning "omit the field". `ddim` / `euler-ancestral` are UNet schedulers and
 * `euler` / `dpm-pp` are wan sample solvers; only `uni-pc` is valid for both.
 * The object forms are the serde-tagged shapes older metadata may carry. */
export type Scheduler =
  | GenerationScheduler
  | { ddim: unknown }
  | { "euler-ancestral": unknown }
  | { "uni-pc": unknown }
  | { "edm-dpm-pp-2m": unknown };

/** Re-exported so a wire type and the policy that reads it cannot drift. */
export type { MeshWorkflowProvenance } from "@studio/lib/meshWorkflowProvenance";

export type MediaKind = "image" | "animated" | "video" | "audio" | "mesh";

export const VIDEO_FORMATS: ReadonlyArray<OutputFormat> = ["mp4"];
export const AUDIO_FORMATS: ReadonlyArray<OutputFormat> = ["wav"];
// `obj` is never a STORED format — mold only produces one as a gallery export
// — but it is listed so a hand-placed `.obj` is classified as a mesh rather
// than falling through to the image branch and being handed to an <img> tag.
export const MESH_FORMATS: ReadonlyArray<OutputFormat> = ["glb", "obj"];
export const ANIMATED_FORMATS: ReadonlyArray<OutputFormat> = [
  "gif",
  "apng",
  "webp",
];

/** The frame counts that tell an animated WebP from a still one. */
export interface MediaKindFrames {
  frames?: number | null;
  video_frames?: number | null;
}

/**
 * WebP is BOTH a still format (every still recipe advertises it, and
 * `webp_still` encodes a single non-animated frame) and an animation
 * container. The file decides — `OutputFormat::is_video_artifact` reads the
 * `ANIM`/`VP8X` flag — and a client sees that decision as frame metadata: an
 * animation records its frame count, a still records none.
 */
function webpIsAnimated(metadata: MediaKindFrames | null | undefined): boolean {
  return (metadata?.frames ?? 0) > 1 || (metadata?.video_frames ?? 0) > 1;
}

export function mediaKind(
  fmt: OutputFormat | null | undefined,
  filename: string,
  metadata?: MediaKindFrames | null,
): MediaKind {
  const resolved = fmt ?? inferFormatFromName(filename);
  // Narrowest kind first, matching the probe order the server and the Rust
  // client both use: a mesh has no frames and no samples, so a wider test
  // running first would classify it as an image.
  if (resolved && MESH_FORMATS.includes(resolved)) return "mesh";
  if (resolved && VIDEO_FORMATS.includes(resolved)) return "video";
  if (resolved && AUDIO_FORMATS.includes(resolved)) return "audio";
  if (resolved === "webp") {
    return webpIsAnimated(metadata) ? "animated" : "image";
  }
  if (resolved && ANIMATED_FORMATS.includes(resolved)) return "animated";
  return "image";
}

export function inferFormatFromName(filename: string): OutputFormat | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".mp4")) return "mp4";
  if (lower.endsWith(".wav")) return "wav";
  if (lower.endsWith(".glb")) return "glb";
  if (lower.endsWith(".obj")) return "obj";
  if (lower.endsWith(".gif")) return "gif";
  if (lower.endsWith(".apng")) return "apng";
  if (lower.endsWith(".webp")) return "webp";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "jpeg";
  if (lower.endsWith(".png")) return "png";
  return null;
}

// Wire shape — what we POST to /api/generate/stream. snake_case to match serde.
export interface GenerateRequestWire {
  prompt: string;
  /** User-authored print title (D5). Additive; absent = untitled. */
  title?: string | null;
  /** "Save every result" off: the host publishes, then moves the print
   * straight to the trash. Additive; absent = save. */
  save_to_gallery?: boolean | null;
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
  seed?: number | string | null;
  batch_size?: number;
  output_format?: OutputFormat;
  /** 3-D controls; refused at admission on a recipe with no `mesh` block. */
  mesh?: MeshRequestOptions | null;
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
  /**
   * IP-Adapter injection strength for `edit_images` on an ADDITIVE recipe
   * (`capabilities.reference_images.weight`). Absent takes the server's own
   * default (1.0), so a default-valued render is byte-identical on the wire
   * to one that never named it — and an older host, which drops the unknown
   * field rather than refusing it, renders exactly what it always did.
   */
  reference_weight?: number | null;
  /**
   * Cut the subject out onto a transparent background
   * (`capabilities.transparency`). Only `true` ever travels — off is the
   * field's absence, so an ordinary render is byte-identical on the wire.
   */
  transparent_background?: boolean | null;
  /** Ordered heterogeneous MiniMax H3 Ref2VA inputs. */
  references?: GenerationReference[] | null;
  /** Face-identity (PuLID) reference, base64 PNG/JPEG with no data-URI
   * prefix (#1224). Never fitted or cropped against the canvas — it is a
   * face reference, not a composition input. Admission accepts it only on an
   * identity-qualified checkpoint, and it rides beside a LoRA and `source_image`;
   * every rule lives in `@studio/lib/identityConditioning`. */
  id_image?: string | null;
  /** Ordered multi-photo identity form, mutually exclusive with id_image. */
  id_images?: string[] | null;
  id_image_names?: string[] | null;
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
  /** LTX-2 video-only opt-in (#1037): true skips the audio branch. */
  video_only?: boolean | null;
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

export interface ModelInfoExtended
  extends ModelDefaults, ModelPresentationFields {
  name: string;
  family: string;
  size_gb: number;
  is_loaded: boolean;
  last_used: number | null;
  hf_repo: string;
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

// ── /api/queue (L3 reconciliation) ─────────────────────────────────────────

/** `held` is additive: a journalled job the host parked after it exhausted its
 * replay or dispatch budget. It exists only in the durable queue, never starts
 * on its own, and is listed so it is not invisible. Absent on older servers. */
export type QueueJobState = "queued" | "running" | "paused" | "held";

export interface QueueEntry {
  stall?: JobStallDiagnostic | null;
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
  /** For a `paused` row: whether SOMEONE paused this one row, as opposed to
   * the restart sweep parking the whole queue. Both wear `state: "paused"`,
   * and absent means the host does not distinguish them (every server built
   * before per-job pause could only ever have parked at restart). */
  explicitly_paused?: boolean | null;
}

export interface QueueListing {
  entries: QueueEntry[];
  plan?: import("@studio/api/queuePlan").QueuePlan | null;
  live_only_entries?: QueueEntry[];
  page?: import("@studio/api/queuePlan").QueuePage;
}

export type SseProgressEvent =
  | { type: "stall"; stall: JobStallDiagnostic | null }
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
  seed_used: number | string;
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
  /** The name the host saved this payload under (`SseCompleteEvent.filename`
   * in `mold_core`). The print's EXACT identity, and the only one that stays
   * right when a seed is deliberately reused; absent on older servers, which
   * is why every reader treats it as a hint and not a guarantee. */
  filename?: string | null;
  /** Gallery filename of the pre-upscale original, when one was saved. */
  original_filename?: string | null;
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

// ── Auto-chained long video (POST /api/chain-jobs, `ephemeral: true`) ─────
// A render whose requested length exceeds the checkpoint's single-pass clip
// size is split into clips and stitched back into ONE print. There is no
// authoring surface: web builds this wire body itself in `useGenerateStream`.
// Mirrors `mold_core::chain::{ChainRequest, ChainStage, ChainProgressEvent,
// SseChainCompleteEvent}`.
export type {
  ChainStageWire,
  ChainRequestWire,
} from "@studio/lib/api/chainTypes";

export interface ExpandRequestWire {
  prompt: string;
  model_family: string;
  variations: number;
  /**
   * Natural-language style directive the server appends to the expander's
   * system message. The composer preset that fed it is retired, so no client
   * sends one any more; the field stays because MCP still does.
   */
  style?: string;
  /** Resolved generation/conditioning policy; no media bytes travel here. */
  task?: ExpandTask;
  /** Identity, canvas, clip, and ordered references (additive). */
  context?: ExpandContextWire;
}

/** Mirrors `mold_core::ExpandContext` (see `@studio/lib/expandTask`). */
export type ExpandContextWire = import("@studio/lib/expandTask").ExpandContext;

export interface ExpandResponseWire {
  original: string;
  expanded: string[];
}
export interface RemixRequestWire {
  source_prompt: string;
  root_prompt?: string;
  source_kind: RemixSourceKind;
  model_family: string;
  variations?: number;
  task: ExpandTask;
  style?: string;
  dimensions: RemixDimension[];
  /** Identity, canvas, clip, and ordered references (additive). */
  context?: ExpandContextWire;
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

/// Families whose engines apply LoRA adapters today — merged into the
/// weights or, as on Qwen Image 2.1, as an unmerged bypass branch. Mirrors
/// `mold_core::validation::LORA_CAPABLE_FAMILIES` and its
/// `require_lora_capable_family` gate. Keep both in sync — divergence shows
/// up as a UI that lets the user pick a LoRA the server then rejects.
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
  seed: number | string | null; // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  /** When true, omit frames and let a qualified LTX-2.5 duration head decide. */
  predictDuration?: boolean;
  fps: number | null;
  scheduler: Scheduler | null;
  cfgPlus: boolean;
  outputFormat: OutputFormat;
  sourceFitPolicy?: SourceFitPolicy;
  imageAttachments: SourceImageState[];
  /**
   * The ordered reference strip of a TWO-WELL recipe — exclusive (FLUX.2
   * [klein], one well ships) or additive (IP-Adapter, both ship) — whose
   * source image and references need two stores:
   * `imageAttachments[0]` stays the source well, these are the references.
   * Every other layout keeps its single list — a `replaces` strip (Qwen edit,
   * FLUX.2 [dev]) is `imageAttachments`.
   *
   * Optional because a draft persisted before this field existed has none,
   * and every read spells that `?? []` rather than assuming an array.
   */
  referenceImages?: SourceImageState[];
  /**
   * Which exclusive well was written last. `resolveExclusiveWells` reads it
   * to decide which one ships and which parks; `null` — an untouched form or
   * a draft saved before the field existed — reads as the source well.
   */
  exclusiveWell?: "source" | "references" | null;
  /**
   * IP-Adapter injection strength for the reference strip on an ADDITIVE
   * recipe. `null` (and absence, on a draft saved before the field existed)
   * means untouched, which keeps `reference_weight` off the wire so the
   * server's own default stays the authority — the `identityWeight` rule.
   */
  referenceWeight?: number | null;
  /**
   * The Transparent background toggle. Kept even while a model that does not
   * advertise it is selected (it parks, like a staged identity photo) and
   * sent only while the recipe's `capabilities.transparency` is adjustable.
   * Absent on a draft saved before the field existed, which reads as off.
   */
  transparentBackground?: boolean;
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
  /** LTX-2 video-only opt-in (#1037): skips the audio branch structurally.
   * Output-changing, never a default; sent as `video_only: true` only when
   * conflict-free (see `@studio/lib/videoOnly`). */
  videoOnly?: boolean;
  /** MiniMax H3 first/last endpoints or ordered heterogeneous references.
   * Kept separate from legacy edit/source fields so no surface can flatten
   * Ref2VA into image-only editing. */
  h3Authoring?: MinimaxH3AuthoringState;
  /** Fixed semantic camera slots for Hunyuan3D multiview recipes. */
  namedViews?: NamedViewsState;
  /** 3-D geometry controls for a mesh recipe. `null` on every field means
   * "use the profile default", which is what keeps an untouched control off
   * the wire; cleared whenever the selected recipe has no `mesh` block.
   * Optional so persisted pre-mesh drafts keep loading. */
  mesh?: MeshFormState;
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
