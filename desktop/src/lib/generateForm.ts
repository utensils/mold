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
import { coerceSourceFitForMaskless, type SourceFitPolicy } from "./sourceFit";
import { findInstalledModel } from "./generateModels";

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
  /** Qwen-Image-Edit picture strip, base64 each (no data-URI prefix). Order is
   * load-bearing: index 0 is the primary edit Target, the rest are References.
   * Only read in `sourceImageMode === "qwen-edit"`; empty everywhere else. */
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
  keyframes: FormKeyframe[];
  pipeline: Ltx2PipelineMode | null;
  retakeRange: TimeRange | null;
  spatialUpscale: Ltx2SpatialUpscale | null;
  temporalUpscale: Ltx2TemporalUpscale | null;
  /** Conditioning audio for the a2vid pipeline; base64 on the wire. */
  audioFile: PickedFile | null;
  /** LTX-2 camera-motion LoRA: a preset id (dolly-in, …, static) or an
   * explicit `.safetensors` path; null = off. Ships as a `loras[]` entry
   * (`camera-control:<preset>` or the raw path) at scale 1.0 — exactly what
   * the CLI's `--camera-control` sends; there is no dedicated wire field. */
  cameraControl: string | null;
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
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    audioFile: null,
    cameraControl: null,
  };
}

/**
 * Apply a model's defaults and prune anything the new family can't use. LoRAs
 * clear on every model change — even FLUX→FLUX — because an adapter may not
 * target the new variant's tensor layout.
 */
export function applyModelDefaults(form: GenerateForm, m: ModelEntry): void {
  form.model = m.name;
  form.family = m.family;
  form.width = m.default_width;
  form.height = m.default_height;
  form.steps = m.default_steps;
  form.guidance = m.default_guidance;
  form.loras = [];

  const caps = generationCapabilitiesForFamily(m.family);
  if (!outputFormatsForFamily(m.family).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(m.family);
  }
  if (!caps.supportsScheduler) form.scheduler = "default";
  if (!caps.supportsCfgPlus) form.cfgPlus = false;
  if (caps.forcesBatchSizeOne) form.batchSize = 1;
  if (!caps.supportsImg2img) {
    form.sourceImage = null;
    form.maskImage = null;
    form.imageAttachments = [];
  } else if (caps.sourceImageMode === "qwen-edit") {
    // Entering qwen-edit: a single-mode source seeds the strip as the Target
    // (web parity — the Composer's attachment survives the model switch).
    if (form.imageAttachments.length === 0 && form.sourceImage) {
      form.imageAttachments = [form.sourceImage];
    }
    form.sourceImage = null;
    form.maskImage = null;
  } else if (form.imageAttachments.length > 0) {
    // Leaving qwen-edit: the Target becomes the single img2img source (web
    // parity — attachments truncate to one, which single mode reads).
    if (!form.sourceImage) form.sourceImage = form.imageAttachments[0] ?? null;
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
  if (!caps.supportsAudio) form.enableAudio = false;
  if (!caps.supportsAdvancedVideo) {
    form.sourceVideo = null;
    form.keyframes = [];
    form.pipeline = null;
    form.retakeRange = null;
    form.spatialUpscale = null;
    form.temporalUpscale = null;
    form.audioFile = null;
    form.cameraControl = null;
  } else if (m.name.includes("ltx-2.3")) {
    // Camera-motion presets are published for 19B only; on LTX-2.3 buildRequest
    // silently drops a preset value. Switching within the ltx2 family keeps the
    // advanced-video block above from clearing it, so drop a stale preset here
    // (a custom `.safetensors` path stays valid on 2.3).
    const cam = form.cameraControl?.trim();
    if (cam && !cam.endsWith(".safetensors")) form.cameraControl = null;
  }
}

/**
 * Assemble the wire request from the form, honoring the family's capabilities.
 * `pruneRequestForFamily` is the final guard so no unsupported field ever
 * ships even if the form retained a stale value.
 */
export function buildRequest(form: GenerateForm): GenerateRequest {
  const caps = generationCapabilitiesForFamily(form.family);
  const parsedSeed = form.seed.trim() === "" ? undefined : Number(form.seed);
  const loras: LoraWeight[] = form.loras.map((l) => ({ path: l.path, scale: l.scale }));

  // Camera motion rides the ordinary loras[] stack (mirrors the CLI's
  // --camera-control, run.rs): presets ship as the `camera-control:<preset>`
  // virtual alias the server resolves; explicit `.safetensors` paths pass
  // through raw. LTX-2 only — and presets are published for 19B only, so
  // they're skipped for LTX-2.3 models (the CLI errors there; a custom path
  // remains valid on 2.3).
  const cameraControl = form.cameraControl?.trim();
  if (caps.supportsAdvancedVideo && cameraControl) {
    if (cameraControl.endsWith(".safetensors")) {
      loras.push({ path: cameraControl, scale: 1.0 });
    } else if (!form.model.includes("ltx-2.3")) {
      loras.push({ path: `camera-control:${cameraControl}`, scale: 1.0 });
    }
  }

  const req: GenerateRequest = {
    prompt: form.prompt.trim(),
    model: form.model,
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
    batch_size: form.batchSize,
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
    caps.sourceImageMode === "qwen-edit" &&
    form.imageAttachments.length > 0
  ) {
    req.edit_images = [...form.imageAttachments];
  }

  if (caps.supportsImg2img && caps.sourceImageMode === "single" && form.sourceImage) {
    req.source_image = form.sourceImage;
    req.strength = form.strength;
    if (caps.supportsMask && form.maskImage) req.mask_image = form.maskImage;
    if (caps.supportsControlNet && form.controlImage) {
      req.control_image = form.controlImage;
      if (form.controlModel.trim()) {
        req.control_model = form.controlModel.trim();
        req.control_scale = form.controlScale;
      }
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
    if (form.keyframes.length) {
      req.keyframes = form.keyframes.map<KeyframeConditionWire>((k) => ({
        frame: k.frame,
        image: k.image.base64,
      }));
    }
    if (form.pipeline) req.pipeline = form.pipeline;
    if (form.retakeRange) req.retake_range = form.retakeRange;
    if (form.spatialUpscale) req.spatial_upscale = form.spatialUpscale;
    if (form.temporalUpscale) req.temporal_upscale = form.temporalUpscale;
    // a2vid (audio-to-video) requires conditioning audio; other pipelines ignore it.
    if (form.pipeline === "a2vid" && form.audioFile) req.audio_file = form.audioFile.base64;
  }

  return pruneRequestForFamily(req, form.family);
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
  form.retakeRange = metadata.retake_range ?? null;
  form.spatialUpscale = metadata.spatial_upscale ?? null;
  form.temporalUpscale = metadata.temporal_upscale ?? null;

  // Output metadata never carries source/mask/control/video/audio bytes —
  // clear any stale attachment instead of silently pairing it with the print.
  form.sourceImage = null;
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

export type GeneratePrefill = ScalarPrefill | MetadataPrefill;

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
