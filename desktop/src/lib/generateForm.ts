/**
 * Reactive form state for the Generate workspace and the pure builder that
 * turns it into a wire `GenerateRequest`. The capability matrix
 * (`capabilities.ts`) decides which fields survive; this module only holds the
 * editable shape and the model-default / request-assembly plumbing.
 */
import type {
  GenerateRequest,
  KeyframeConditionWire,
  LoraWeight,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
  OutputFormat,
  OutputMetadata,
  Scheduler,
  TimeRange,
} from "./api/types";
import {
  MAX_LORA_STACK,
  defaultOutputFormat,
  generationCapabilitiesForFamily,
  outputFormatsForFamily,
  pruneRequestForFamily,
} from "./capabilities";
import { coerceSourceFitForMaskless, type SourceFitPolicy } from "@studio/lib/sourceFit";
import { defaultVideoFps } from "@studio/lib/sequence";
import { findInstalledModel } from "./generateModels";
import {
  cameraMotionFromLoraPath,
  cameraMotionLoraLabel,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import {
  emptyGuidanceOverrides,
  guidanceOverridesFromWire,
  guidanceOverridesToWire,
  type Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import { pipelineForControlId } from "@studio/lib/ltx2Control";
import { stripAudioOnlyIncompatibleFields } from "@studio/lib/ltx2Pipeline";

/** A LoRA row in the stack: wire fields plus display metadata (name, triggers). */
export interface FormLora {
  path: string;
  name: string;
  scale: number;
  trainedWords: string[];
}

/** A picked file (base64, no data-URI prefix); `filename` is display metadata. */
export interface PickedFile {
  filename: string;
  base64: string;
}

/** An image picked via {@link ImagePickerModal} (upload or gallery). */
export type PickedImage = PickedFile;

/** One LTX-2 keyframe: a conditioning image pinned to a frame index. */
export interface FormKeyframe {
  frame: number;
  image: PickedImage;
}

/** Whether the current seed field means "roll fresh" or "locked". */
export function seedMode(seed: string): "random" | "fixed" {
  return seed.trim() === "" ? "random" : "fixed";
}

export interface GenerateForm {
  prompt: string;
  /** Original prompt before an expand; sent as `original_prompt` and used for undo. */
  originalPrompt: string | null;
  model: string;
  family: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  /** Empty string = random seed. */
  seed: string;
  negativePrompt: string;
  scheduler: Scheduler;
  cfgPlus: boolean;
  batchSize: number;
  outputFormat: OutputFormat;
  /** Post-generate upscaler model name; empty = off. */
  upscaleModel: string;
  strength: number;
  /** base64, no data-URI prefix. */
  sourceImage: string | null;
  /** Provenance label for `sourceImage` — the gallery filename or upload
   * name it came from. Ships as `source_image_name` so Reuse settings can
   * restore the input image later; always cleared with the image. */
  sourceImageName: string | null;
  /** Decoded dimensions of the effective primary conditioning image. For
   * single-source families this describes `sourceImage`; for Qwen edit it
   * describes attachment 0 (the Target). These are UI sizing metadata only
   * and never travel on the generation wire. */
  sourceImageWidth: number | null;
  sourceImageHeight: number | null;
  /** Ordered edit/reference strip, base64 each (no data-URI prefix). For Qwen,
   * index 0 is the edit Target and the rest are References. FLUX.2 Dev treats
   * every entry as an ordered Reference. Empty in single-source mode. */
  imageAttachments: string[];
  /** How a source image that doesn't match width×height maps onto the canvas.
   * Applied client-side on submit (`sourceFitPreprocess.ts`), never wired. */
  sourceFit: SourceFitPolicy;
  maskImage: string | null;
  controlImage: string | null;
  controlModel: string;
  controlScale: number;
  loras: FormLora[];
  // Video families (ltx-video / ltx2).
  frames: number;
  fps: number;
  enableAudio: boolean;
  // LTX-2 advanced video (ltx2 only). All optional-safe: null / [] defaults so
  // a partial stored form (template snapshot) still hydrates cleanly.
  sourceVideo: PickedImage | null;
  /** Existing video to continue; set makes the request a continuation. */
  extendVideo: PickedImage | null;
  /** Pixel-frame overlap; null takes the host's advertised default. */
  extendOverlapFrames: number | null;
  keyframes: FormKeyframe[];
  pipeline: Ltx2PipelineMode | null;
  /** Official host-provided IC-LoRA control adapter ID. */
  icLoraControl?: string | null;
  retakeRange: TimeRange | null;
  spatialUpscale: Ltx2SpatialUpscale | null;
  temporalUpscale: Ltx2TemporalUpscale | null;
  /** Optional LTX-2 guider overrides. Empty values preserve pipeline defaults. */
  guidanceOverrides: Ltx2GuidanceOverridesState;
  /** Conditioning audio for the a2-vid pipeline; base64 on the wire. */
  audioFile: PickedFile | null;
  /** LTX-2 camera-motion LoRA: a preset id (dolly-in, …, static) or an
   * explicit `.safetensors` path; null = off. Ships as a `loras[]` entry
   * (`camera-control:<preset>` or the raw path) at scale 1.0 — exactly what
   * the CLI's `--camera-control` sends; there is no dedicated wire field. */
  cameraControl: string | null;
  /** Composer style preset id (see `stylePresets.ts`); `""` = none. A
   * look-and-feel modifier composed into the outgoing prompt at submit — never
   * mutates the textarea and carries no dedicated wire field. */
  stylePreset: string;
}

export function newGenerateForm(): GenerateForm {
  return {
    prompt: "",
    originalPrompt: null,
    model: "",
    family: "",
    width: 1024,
    height: 1024,
    steps: 4,
    guidance: 3.5,
    seed: "",
    negativePrompt: "",
    scheduler: "default",
    cfgPlus: false,
    batchSize: 1,
    outputFormat: "png",
    upscaleModel: "",
    strength: 0.75,
    sourceImage: null,
    sourceImageName: null,
    sourceImageWidth: null,
    sourceImageHeight: null,
    imageAttachments: [],
    sourceFit: { mode: "pad-repaint" },
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1.0,
    loras: [],
    frames: 97,
    fps: 24,
    enableAudio: false,
    sourceVideo: null,
    extendVideo: null,
    extendOverlapFrames: null,
    keyframes: [],
    pipeline: null,
    icLoraControl: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    guidanceOverrides: emptyGuidanceOverrides(),
    audioFile: null,
    cameraControl: null,
    stylePreset: "",
  };
}

/**
 * Take a submission-safe snapshot of the mutable composer state. Source-fit
 * preprocessing is asynchronous and may run for minutes; callers must never
 * read the live reactive form again after a user taps Generate.
 */
export function cloneGenerateForm(form: GenerateForm): GenerateForm {
  const sourceFit: SourceFitPolicy =
    form.sourceFit.mode === "upscale-then-fit"
      ? { ...form.sourceFit, fit: { ...form.sourceFit.fit } }
      : { ...form.sourceFit };
  return {
    ...form,
    imageAttachments: [...form.imageAttachments],
    sourceFit,
    loras: form.loras.map((lora) => ({
      ...lora,
      trainedWords: [...lora.trainedWords],
    })),
    sourceVideo: form.sourceVideo ? { ...form.sourceVideo } : null,
    extendVideo: form.extendVideo ? { ...form.extendVideo } : null,
    keyframes: form.keyframes.map((keyframe) => ({
      ...keyframe,
      image: { ...keyframe.image },
    })),
    retakeRange: form.retakeRange ? { ...form.retakeRange } : null,
    guidanceOverrides: { ...form.guidanceOverrides },
    audioFile: form.audioFile ? { ...form.audioFile } : null,
  };
}

/**
 * Apply a model's defaults and prune anything the new family can't use. LoRAs
 * clear on every model change — even FLUX→FLUX — because an adapter may not
 * target the new variant's tensor layout.
 */
export function applyModelDefaults(form: GenerateForm, m: ModelEntry): void {
  const cameraRows = form.loras.filter(
    (lora) => lora.path.startsWith("camera-control:") || lora.path === form.cameraControl?.trim(),
  );
  form.width = m.default_width;
  form.height = m.default_height;
  form.steps = m.default_steps;
  form.guidance = m.default_guidance;
  // The model's advertised rate is applied like steps/guidance — it is only
  // absent-server/absent-field that leaves the current value in place.
  form.fps = defaultVideoFps(m, form.fps);
  form.loras = [];
  form.icLoraControl = null;
  reconcileModelCapabilities(form, m);
  if (form.cameraControl) {
    form.loras = syncCameraMotionLora(
      cameraRows,
      form.cameraControl,
      form.cameraControl,
      (path, scale) => ({
        path,
        name: cameraMotionLoraLabel(path),
        scale,
        trainedWords: [],
      }),
    );
  }
}

/**
 * Refresh family/capability metadata for the same named model on a different
 * host without discarding portable user parameters. Host manifests are the
 * authority; two remotes may advertise corrected or aliased family metadata.
 */
export function reconcileModelCapabilities(form: GenerateForm, m: ModelEntry): void {
  form.model = m.name;
  form.family = m.family;
  const caps = generationCapabilitiesForFamily(m.family, m.name);
  if (!outputFormatsForFamily(m.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(m.family);
  }
  if (!caps.supportsScheduler) form.scheduler = "default";
  if (!caps.supportsCfgPlus) form.cfgPlus = false;
  if (caps.forcesBatchSizeOne) form.batchSize = 1;
  if (!caps.supportsImg2img) {
    form.sourceImage = null;
    form.sourceImageName = null;
    form.sourceImageWidth = null;
    form.sourceImageHeight = null;
    form.maskImage = null;
    form.imageAttachments = [];
  } else if (caps.sourceImageMode !== "single") {
    // Entering qwen-edit: a single-mode source seeds the strip as the Target
    // (web parity — the Composer's attachment survives the model switch).
    if (form.imageAttachments.length === 0 && form.sourceImage) {
      form.imageAttachments = [form.sourceImage];
    }
    form.sourceImage = null;
    // The picture strip carries no per-image labels.
    form.sourceImageName = null;
    form.maskImage = null;
  } else if (form.imageAttachments.length > 0) {
    // Leaving qwen-edit: the Target becomes the single img2img source (web
    // parity — attachments truncate to one, which single mode reads).
    if (!form.sourceImage) {
      form.sourceImage = form.imageAttachments[0] ?? null;
      form.sourceImageName = null;
    }
    form.imageAttachments = [];
  }
  if (!caps.supportsMask) {
    form.maskImage = null;
    // Maskless img2img (LTX-2 image-to-video) can't repaint pad bands, so a
    // mask-dependent fit policy flips to crop-fill on entry.
    if (caps.supportsImg2img && caps.sourceImageMode === "single") {
      form.sourceFit = coerceSourceFitForMaskless(form.sourceFit);
    }
  }
  if (!caps.supportsControlNet) {
    form.controlImage = null;
    form.controlModel = "";
  }
  if (!caps.supportsAudio || m.supports_audio === false) form.enableAudio = false;
  if (!caps.supportsAdvancedVideo) {
    form.sourceVideo = null;
    form.extendVideo = null;
    form.extendOverlapFrames = null;
    form.keyframes = [];
    form.pipeline = null;
    form.icLoraControl = null;
    form.retakeRange = null;
    form.spatialUpscale = null;
    form.temporalUpscale = null;
    form.guidanceOverrides = emptyGuidanceOverrides();
    form.audioFile = null;
    form.loras = syncCameraMotionLora(form.loras, form.cameraControl, null, (path, scale) => ({
      path,
      name: path,
      scale,
      trainedWords: [],
    }));
    form.cameraControl = null;
  }
}

/**
 * Restore every generation knob to the selected model's defaults. The prompt
 * (with its expand provenance), the model/family, and the batch size survive:
 * the prompt is the user's authored work, and prepared batch siblings must
 * never be silently resized by an unrelated control.
 *
 * With no `ModelEntry` — an uninstalled or not-yet-resolved model — the named
 * model and family are kept and the form falls back to `newGenerateForm()`
 * scalars.
 */
export function resetFormToModelDefaults(
  form: GenerateForm,
  m: ModelEntry | null | undefined,
): void {
  const { prompt, originalPrompt, batchSize, model, family } = form;
  Object.assign(form, newGenerateForm());
  if (m) {
    applyModelDefaults(form, m);
  } else {
    form.model = model;
    form.family = family;
  }
  form.prompt = prompt;
  form.originalPrompt = originalPrompt;
  form.batchSize = generationCapabilitiesForFamily(form.family, form.model).forcesBatchSizeOne
    ? 1
    : batchSize;
}

/**
 * Assemble the wire request from the form, honoring the family's capabilities.
 * `pruneRequestForFamily` is the final guard so no unsupported field ever
 * ships even if the form retained a stale value.
 */
export function buildRequest(form: GenerateForm): GenerateRequest {
  const caps = generationCapabilitiesForFamily(form.family, form.model);
  const parsedSeed = form.seed.trim() === "" ? undefined : Number(form.seed);
  let loras: LoraWeight[] = form.loras.map((l) => ({ path: l.path, scale: l.scale }));

  // Camera motion rides the ordinary loras[] stack (mirrors the CLI's
  // --camera-control, run.rs): presets ship as the `camera-control:<preset>`
  // virtual alias the server resolves; explicit `.safetensors` paths pass
  // through raw. The host-provided capability list is the compatibility
  // authority; the serializer never guesses from a public model id.
  const cameraControl = form.cameraControl?.trim();
  if (caps.supportsAdvancedVideo && cameraControl) {
    loras = syncCameraMotionLora(
      loras,
      cameraControl,
      cameraControl,
      (path, scale) => ({
        path,
        scale,
      }),
      MAX_LORA_STACK,
    );
  }

  const req: GenerateRequest = {
    prompt: form.prompt.trim(),
    model: form.model,
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
    batch_size:
      caps.forcesBatchSizeOne ||
      (caps.sourceImageMode === "references" && form.imageAttachments.length > 0)
        ? 1
        : form.batchSize,
    output_format: form.outputFormat,
  };

  if (parsedSeed !== undefined && Number.isFinite(parsedSeed)) req.seed = parsedSeed;
  if (form.originalPrompt && form.originalPrompt !== req.prompt) {
    req.original_prompt = form.originalPrompt;
  }
  if (caps.supportsNegativePrompt && form.negativePrompt.trim()) {
    req.negative_prompt = form.negativePrompt.trim();
  }
  if (caps.supportsScheduler && form.scheduler !== "default") req.scheduler = form.scheduler;
  if (caps.supportsCfgPlus && form.cfgPlus) req.cfg_plus = true;

  // qwen-edit ships the ordered picture strip (first = Target, rest =
  // References) and never source_image/strength; batch is already locked to 1
  // by forcesBatchSizeOne + pruneRequestForFamily.
  if (
    caps.supportsImg2img &&
    caps.sourceImageMode !== "single" &&
    form.imageAttachments.length > 0
  ) {
    req.edit_images = [...form.imageAttachments];
  }

  if (caps.supportsImg2img && caps.sourceImageMode === "single" && form.sourceImage) {
    req.source_image = form.sourceImage;
    if (form.sourceImageName) req.source_image_name = form.sourceImageName;
    req.strength = form.strength;
    if (caps.supportsMask && form.maskImage) req.mask_image = form.maskImage;
  }

  // ControlNet is independent conditioning, not an img2img derivative. An
  // SD1.5 request may carry a control image without a source image; nesting it
  // under source_image silently discarded that valid text-to-image workflow.
  if (caps.supportsControlNet && form.controlImage) {
    req.control_image = form.controlImage;
    if (form.controlModel.trim()) {
      req.control_model = form.controlModel.trim();
      req.control_scale = form.controlScale;
    }
  }

  if (caps.supportsLora && loras.length) req.loras = loras;

  // Post-generate upscale is image-only (the server skips it for video).
  if (!caps.supportsVideo && form.upscaleModel) req.upscale_model = form.upscaleModel;

  if (caps.supportsVideo) {
    req.frames = form.frames;
    req.fps = form.fps;
    if (caps.supportsAudio) req.enable_audio = form.enableAudio;
  }

  if (caps.supportsAdvancedVideo) {
    if (form.sourceVideo) req.source_video = form.sourceVideo.base64;
    if (form.extendVideo) {
      req.extend_video = form.extendVideo.base64;
      // Only travels with a clip to continue; the server rejects a bare
      // overlap, and omitting it takes the server's own default.
      if (form.extendOverlapFrames) {
        req.extend_overlap_frames = form.extendOverlapFrames;
      }
    }
    if (form.keyframes.length) {
      req.keyframes = form.keyframes.map<KeyframeConditionWire>((k) => ({
        frame: k.frame,
        image: k.image.base64,
      }));
    }
    if (form.icLoraControl) {
      req.ic_lora_control = form.icLoraControl;
      // Lip dub is a pipeline of its own; every other adapter drives `ic-lora`.
      req.pipeline = pipelineForControlId(form.icLoraControl);
    } else if (form.pipeline) req.pipeline = form.pipeline;
    if (form.retakeRange) req.retake_range = form.retakeRange;
    if (form.spatialUpscale) req.spatial_upscale = form.spatialUpscale;
    if (form.temporalUpscale) req.temporal_upscale = form.temporalUpscale;
    const guidanceOverrides = guidanceOverridesToWire(form.guidanceOverrides);
    if (guidanceOverrides) req.guidance_overrides = guidanceOverrides;
    // a2-vid (audio-to-video) requires conditioning audio; other pipelines ignore it.
    if (form.pipeline === "a2-vid" && form.audioFile) req.audio_file = form.audioFile.base64;
  }

  // Last, and after family pruning: an audio-only pipeline renders no frames,
  // so every conditioning input, upscaler, and a `false` audio flag is
  // something the server refuses. Stripping here rather than on the pipeline
  // transition keeps the user's source media intact if they switch back.
  return stripAudioOnlyIncompatibleFields(pruneRequestForFamily(req, form.family, form.model));
}

const KNOWN_SCHEDULERS: readonly Scheduler[] = ["default", "ddim", "euler-ancestral", "unipc"];

/** Match separator-insensitively: the server's `Display for Scheduler` writes
 * UniPc as `"uni-pc"` while the form union spells it `"unipc"`, and legacy
 * rows carry `"uni_pc"` / `"euler_ancestral"`. Squash `-`/`_` to compare. */
const squash = (name: string): string => name.toLowerCase().replace(/[-_]/g, "");
const SCHEDULER_BY_SQUASHED = new Map<string, Scheduler>(
  KNOWN_SCHEDULERS.map((s) => [squash(s), s]),
);

/** Collapse a metadata scheduler value (`"ddim"` or serde-tagged
 * `{ ddim: … }`) onto the form's string union; anything unknown → default. */
function normalizeMetadataScheduler(s: OutputMetadata["scheduler"]): Scheduler {
  if (!s) return "default";
  const name = typeof s === "string" ? s : (Object.keys(s)[0] ?? "default");
  return SCHEDULER_BY_SQUASHED.get(squash(name)) ?? "default";
}

/** Display name for a LoRA restored from metadata — the path's basename. */
function loraNameFromPath(path: string): string {
  if (cameraMotionFromLoraPath(path)) return cameraMotionLoraLabel(path);
  const base = path.split("/").pop() ?? path;
  return base.replace(/\.safetensors$/i, "");
}

/**
 * Full-fidelity "Reuse settings": restore every serialized generation knob a
 * gallery item's embedded metadata carries (port of the web SPA's
 * `applyMetadataToForm`). Static-seed semantics recreate the print exact-ish;
 * binary media (source/mask/control/video/audio bytes) is cleared because
 * output metadata never contains it. When the model isn't installed anywhere,
 * the name is still set (family blank) and the existing missing-model UI takes
 * over — reuse never forces a host, so model-aware Auto routing still applies.
 */
export function applyMetadataToForm(
  form: GenerateForm,
  metadata: OutputMetadata,
  models: ModelEntry[] = [],
): void {
  form.cameraControl = null;
  const model = findInstalledModel(models, metadata.model);
  if (model) {
    applyModelDefaults(form, model);
  } else {
    form.model = metadata.model;
    form.family = "";
  }

  form.prompt = metadata.prompt ?? "";
  form.originalPrompt = metadata.original_prompt ?? null;
  form.negativePrompt = metadata.negative_prompt ?? "";
  // Prefer the pre-upscale generation canvas over the saved raster size.
  form.width = metadata.generation_width || metadata.width || form.width;
  form.height = metadata.generation_height || metadata.height || form.height;
  form.steps = metadata.steps || form.steps;
  form.guidance = metadata.guidance ?? form.guidance;
  form.seed = metadata.seed == null ? "" : String(metadata.seed);
  form.scheduler = normalizeMetadataScheduler(metadata.scheduler);
  form.cfgPlus = metadata.cfg_plus ?? false;
  if (metadata.strength != null) form.strength = metadata.strength;

  const loras =
    metadata.loras ??
    (metadata.lora ? [{ path: metadata.lora, scale: metadata.lora_scale ?? 1.0 }] : []);
  form.loras = loras.slice(0, MAX_LORA_STACK).map<FormLora>((l) => ({
    path: l.path,
    name: loraNameFromPath(l.path),
    scale: l.scale,
    trainedWords: [],
  }));
  form.cameraControl =
    form.loras
      .map((lora) => cameraMotionFromLoraPath(lora.path))
      .find((value): value is string => value !== null) ?? null;

  form.controlModel = metadata.control_model ?? "";
  if (metadata.control_scale != null) form.controlScale = metadata.control_scale;
  form.upscaleModel = metadata.upscale_model ?? "";
  if (metadata.output_format) form.outputFormat = metadata.output_format;

  // Video params (`video_frames`/`video_fps` are legacy desktop aliases).
  const frames = metadata.frames ?? metadata.video_frames;
  if (frames != null) form.frames = frames;
  const fps = metadata.fps ?? metadata.video_fps;
  if (fps != null) form.fps = fps;
  if (metadata.enable_audio != null) form.enableAudio = metadata.enable_audio;
  form.pipeline = metadata.pipeline ?? null;
  form.icLoraControl = metadata.ic_lora_control ?? null;
  form.retakeRange = metadata.retake_range ?? null;
  form.spatialUpscale = metadata.spatial_upscale ?? null;
  form.temporalUpscale = metadata.temporal_upscale ?? null;
  form.guidanceOverrides = guidanceOverridesFromWire(metadata.guidance_overrides);

  // Output metadata never carries source/mask/control/video/audio bytes —
  // clear any stale attachment instead of silently pairing it with the print.
  // (The async source restore may repopulate the pair afterwards.)
  form.sourceImage = null;
  form.sourceImageName = null;
  form.sourceImageWidth = null;
  form.sourceImageHeight = null;
  form.maskImage = null;
  form.controlImage = null;
  form.imageAttachments = [];
  form.sourceVideo = null;
  form.keyframes = [];
  form.audioFile = null;
}

/** Lossy scalar prefill used by non-gallery callers (palette, history, jobs). */
export interface ScalarPrefill {
  prompt: string;
  model: string;
  seed: number | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  upscaleModel?: string;
}

/** Full-fidelity prefill: the gallery item's embedded metadata, verbatim. */
export interface MetadataPrefill {
  metadata: OutputMetadata;
}

/** Exact queued request, including advanced and media inputs. */
export interface RequestPrefill {
  request: GenerateRequest;
}

export type GeneratePrefill = ScalarPrefill | MetadataPrefill | RequestPrefill;

export function applyRequestToForm(
  form: GenerateForm,
  request: GenerateRequest,
  models: ModelEntry[],
): void {
  Object.assign(form, newGenerateForm());
  const model = findInstalledModel(models, request.model);
  if (model) applyModelDefaults(form, model);
  form.prompt = request.prompt;
  form.originalPrompt = request.original_prompt ?? null;
  form.negativePrompt = request.negative_prompt ?? "";
  form.model = request.model;
  form.width = request.width;
  form.height = request.height;
  form.steps = request.steps;
  form.guidance = request.guidance ?? form.guidance;
  form.seed = request.seed == null ? "" : String(request.seed);
  form.scheduler = request.scheduler ?? "default";
  form.cfgPlus = request.cfg_plus ?? false;
  form.batchSize = request.batch_size ?? 1;
  form.outputFormat = request.output_format ?? form.outputFormat;
  form.upscaleModel = request.upscale_model ?? "";
  form.strength = request.strength ?? form.strength;
  form.sourceImage = request.source_image ?? null;
  form.sourceImageName = request.source_image_name ?? null;
  form.sourceImageWidth = null;
  form.sourceImageHeight = null;
  form.imageAttachments = [...(request.edit_images ?? [])];
  form.maskImage = request.mask_image ?? null;
  form.controlImage = request.control_image ?? null;
  form.controlModel = request.control_model ?? "";
  form.controlScale = request.control_scale ?? 1;
  const loras = request.loras ?? (request.lora ? [request.lora] : []);
  form.loras = loras.slice(0, MAX_LORA_STACK).map((lora) => ({
    path: lora.path,
    name: loraNameFromPath(lora.path),
    scale: lora.scale,
    trainedWords: [],
  }));
  form.cameraControl =
    form.loras
      .map((lora) => cameraMotionFromLoraPath(lora.path))
      .find((value): value is string => value !== null) ?? null;
  form.frames = request.frames ?? form.frames;
  form.fps = request.fps ?? form.fps;
  form.enableAudio = request.enable_audio ?? false;
  form.audioFile = request.audio_file
    ? { filename: "Audio input", base64: request.audio_file }
    : null;
  form.sourceVideo = request.source_video
    ? { filename: "Video input", base64: request.source_video }
    : null;
  form.keyframes = (request.keyframes ?? []).map((keyframe) => ({
    frame: keyframe.frame,
    image: { filename: `Keyframe ${keyframe.frame}`, base64: keyframe.image },
  }));
  form.pipeline = request.pipeline ?? null;
  form.icLoraControl = request.ic_lora_control ?? null;
  form.retakeRange = request.retake_range ?? null;
  form.spatialUpscale = request.spatial_upscale ?? null;
  form.temporalUpscale = request.temporal_upscale ?? null;
  form.guidanceOverrides = guidanceOverridesFromWire(request.guidance_overrides);
}

/**
 * Route a composer prefill into the form: gallery reuse ships full metadata
 * through {@link applyMetadataToForm}; everything else keeps the legacy
 * scalar copy exactly as before.
 */
export function applyPrefillToForm(
  form: GenerateForm,
  prefill: GeneratePrefill,
  models: ModelEntry[] = [],
): void {
  if ("metadata" in prefill) {
    applyMetadataToForm(form, prefill.metadata, models);
    return;
  }
  if ("request" in prefill) {
    applyRequestToForm(form, prefill.request, models);
    return;
  }
  form.prompt = prefill.prompt;
  form.model = prefill.model;
  form.seed = prefill.seed !== null ? String(prefill.seed) : "";
  form.width = prefill.width;
  form.height = prefill.height;
  form.steps = prefill.steps;
  form.guidance = prefill.guidance;
  form.upscaleModel = prefill.upscaleModel ?? "";
  const m = findInstalledModel(models, prefill.model);
  if (m) form.family = m.family;
}
