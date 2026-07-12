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
  Scheduler,
  TimeRange,
} from "./api/types";
import {
  defaultOutputFormat,
  generationCapabilitiesForFamily,
  outputFormatsForFamily,
  pruneRequestForFamily,
} from "./capabilities";

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
  }
  if (!caps.supportsMask) form.maskImage = null;
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
