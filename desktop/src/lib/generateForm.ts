/**
 * Reactive form state for the Generate workspace and the pure builder that
 * turns it into a wire `GenerateRequest`. The capability matrix
 * (`capabilities.ts`) decides which fields survive; this module only holds the
 * editable shape and the model-default / request-assembly plumbing.
 */
import type { GenerateRequest, LoraWeight, ModelEntry, OutputFormat, Scheduler } from "./api/types";
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

  if (caps.supportsVideo) {
    req.frames = form.frames;
    req.fps = form.fps;
    if (caps.supportsAudio) req.enable_audio = form.enableAudio;
  }

  return pruneRequestForFamily(req, form.family);
}
