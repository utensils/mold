import { ref, watch, type Ref } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  LoraSelection,
  ModelInfoExtended,
  OutputMetadata,
  OutputFormat,
  Scheduler,
} from "../types";
import {
  MAX_LORA_STACK,
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
  familySupportsAudio,
  supportsLora,
} from "../types";

/** Output-format options for a given model family, ordered by preference.
 * The first entry is the default the UI auto-selects when a model is
 * chosen. */
export function outputFormatsForFamily(family: string): OutputFormat[] {
  return VIDEO_FAMILIES.includes(family)
    ? ["mp4", "gif", "apng", "webp"]
    : ["png", "jpeg", "webp"];
}

export function defaultOutputFormat(family: string): OutputFormat {
  return outputFormatsForFamily(family)[0];
}

const STORAGE_KEY = "mold.generate.form";
const FORM_VERSION = 2;
const QWEN_IMAGE_EDIT_FAMILY = "qwen-image-edit";

function isQwenImageEditFamily(family: string): boolean {
  return family === QWEN_IMAGE_EDIT_FAMILY;
}

function selectedFamily(s: GenerateFormState): string {
  if (s.modelFamily) return s.modelFamily;
  if (s.model.startsWith(`${QWEN_IMAGE_EDIT_FAMILY}:`)) {
    return QWEN_IMAGE_EDIT_FAMILY;
  }
  return "";
}

function defaultForm(): GenerateFormState {
  return {
    version: FORM_VERSION,
    prompt: "",
    negativePrompt: "",
    model: "",
    modelFamily: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    seedMode: "random",
    seed: null,
    batchSize: 1,
    strength: 0.75,
    frames: null,
    fps: null,
    scheduler: null,
    cfgPlus: false,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1.0,
    upscaleModel: "",
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    placement: null,
    loras: [],
    enableAudio: null,
  };
}

function cloneFormState(state: GenerateFormState): GenerateFormState {
  return JSON.parse(JSON.stringify(state)) as GenerateFormState;
}

/** Remove browser-only binary media from a form snapshot before writing it to
 * localStorage or a local generation template. Server-local path fields
 * (`audioFilePath`, `sourceVideoPath`) are preserved because they are stable
 * references; upload/gallery base64 payloads are intentionally not stored. */
export function sanitizePersistedForm(
  state: GenerateFormState,
): GenerateFormState {
  return {
    ...cloneFormState(state),
    version: FORM_VERSION,
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    audioFile: null,
    sourceVideo: null,
    keyframes: [],
  };
}

/** Clone the generation configuration that can be safely represented in a
 * web-local template. Binary source/mask/control/media bytes are stripped by
 * `sanitizePersistedForm`; callers can store separate filename references for
 * display, but loading the template will not silently pretend those bytes are
 * still available. */
export function cloneTemplateForm(state: GenerateFormState): GenerateFormState {
  return sanitizePersistedForm(state);
}

function modelDefaultsPatch(
  current: GenerateFormState,
  model: ModelInfoExtended,
): GenerateFormState {
  const next: GenerateFormState = {
    ...cloneFormState(current),
    model: model.name,
    modelFamily: model.family,
    width: model.default_width,
    height: model.default_height,
    steps: model.default_steps,
    guidance: model.default_guidance,
    loras: [],
  };
  if (VIDEO_FAMILIES.includes(model.family)) {
    next.frames ??= 25;
    next.fps ??= 24;
  } else {
    next.frames = null;
    next.fps = null;
    next.gifPreview = false;
  }
  const formats = outputFormatsForFamily(model.family);
  if (!formats.includes(next.outputFormat)) {
    next.outputFormat = formats[0];
  }
  next.enableAudio = familySupportsAudio(model.family) ? true : null;
  if (model.family !== "ltx2" && model.family !== "ltx-2") {
    next.audioFile = null;
    next.audioFilePath = "";
    next.sourceVideo = null;
    next.sourceVideoPath = "";
    next.keyframes = [];
    next.pipeline = null;
    next.retakeRange = null;
    next.spatialUpscale = null;
    next.temporalUpscale = null;
  }
  if (isQwenImageEditFamily(model.family)) {
    next.batchSize = 1;
    next.maskImage = null;
    next.controlImage = null;
    next.controlModel = "";
  } else if (next.imageAttachments.length > 1) {
    next.imageAttachments = next.imageAttachments.slice(0, 1);
  }
  return next;
}

export interface ApplyMetadataOptions {
  models?: ModelInfoExtended[];
  format?: OutputFormat | null;
}

/** Recreate-safe metadata application shared by Gallery Recreate and future
 * template/import paths. It restores serialized generation knobs, uses static
 * seed semantics for exact-ish recreation, and clears stale binary media
 * because output metadata does not contain source/mask/control bytes. */
export function applyMetadataToForm(
  current: GenerateFormState,
  metadata: OutputMetadata,
  options: ApplyMetadataOptions = {},
): GenerateFormState {
  const model = options.models?.find((m) => m.name === metadata.model);
  const next = model
    ? modelDefaultsPatch(current, model)
    : { ...cloneFormState(current), model: metadata.model, modelFamily: "" };
  const loras =
    metadata.loras?.map<LoraSelection>((l) => ({
      path: l.path,
      scale: l.scale,
    })) ??
    (metadata.lora
      ? [
          {
            path: metadata.lora,
            scale: metadata.lora_scale ?? 1.0,
          },
        ]
      : []);
  const outputFormat = metadata.output_format ?? options.format ?? null;
  return {
    ...next,
    prompt: metadata.prompt ?? "",
    negativePrompt: metadata.negative_prompt ?? "",
    width: metadata.width || next.width,
    height: metadata.height || next.height,
    steps: metadata.steps || next.steps,
    guidance: metadata.guidance ?? next.guidance,
    seedMode: metadata.seed == null ? "random" : "static",
    seed: metadata.seed ?? null,
    scheduler: metadata.scheduler ?? null,
    cfgPlus: metadata.cfg_plus ?? false,
    strength:
      metadata.strength !== undefined && metadata.strength !== null
        ? metadata.strength
        : next.strength,
    loras: loras.slice(0, MAX_LORA_STACK),
    controlModel: metadata.control_model ?? "",
    controlScale: metadata.control_scale ?? next.controlScale,
    upscaleModel: metadata.upscale_model ?? "",
    gifPreview: metadata.gif_preview ?? false,
    enableAudio: metadata.enable_audio ?? next.enableAudio,
    audioFilePath: metadata.audio_file_path ?? "",
    sourceVideoPath: metadata.source_video_path ?? "",
    pipeline: metadata.pipeline ?? null,
    retakeRange: metadata.retake_range ?? null,
    spatialUpscale: metadata.spatial_upscale ?? null,
    temporalUpscale: metadata.temporal_upscale ?? null,
    frames: metadata.frames ?? null,
    fps: metadata.fps ?? null,
    outputFormat: outputFormat ?? next.outputFormat,
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    audioFile: null,
    sourceVideo: null,
    keyframes: [],
  };
}

/// Drops users with pre-multi-LoRA persisted forms onto the new shape
/// without re-prompting them for everything else. The old `lora` field
/// (singular, nullable) becomes a 1- or 0-element `loras` array.
type LegacyFormState = Omit<Partial<GenerateFormState>, "version"> & {
  lora?: LoraSelection | null;
  version?: number;
  sourceImage?: GenerateFormState["imageAttachments"][number] | null;
};

function migrateLegacy(parsed: LegacyFormState): Partial<GenerateFormState> {
  const {
    lora,
    sourceImage: _sourceImage,
    imageAttachments: _imageAttachments,
    version: _version,
    ...rest
  } = parsed;
  const next: Partial<GenerateFormState> = {
    ...rest,
    version: FORM_VERSION,
    imageAttachments: [],
  };
  if (!Array.isArray(rest.loras)) {
    next.loras = lora ? [lora] : [];
  }
  return next;
}

function load(): GenerateFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultForm();
    const parsed = JSON.parse(raw) as LegacyFormState;
    if (parsed.version !== 1 && parsed.version !== FORM_VERSION) {
      return defaultForm();
    }
    return {
      ...defaultForm(),
      ...sanitizePersistedForm({
        ...defaultForm(),
        ...migrateLegacy(parsed),
      }),
    };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  try {
    // Drop base64 bytes from localStorage — they blow past the quota quickly
    // and the attachment is re-picked trivially on reload.
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(sanitizePersistedForm(state)),
    );
  } catch {
    /* ignore */
  }
}

export interface UseGenerateForm {
  state: Ref<GenerateFormState>;
  reset: () => void;
  applyModelDefaults: (model: ModelInfoExtended) => void;
  /** Replace the entire LoRA stack. Pass `[]` to clear. */
  setLoras: (loras: LoraSelection[]) => void;
  /** Append a LoRA to the stack, capped by `MAX_LORA_STACK`. No-op if
   * the cap is already reached. */
  addLora: (lora: LoraSelection) => void;
  /** Update a single LoRA's scale in place by index. */
  updateLoraScale: (index: number, scale: number) => void;
  /** Drop the LoRA at `index`. Out-of-range indices are no-ops. */
  removeLora: (index: number) => void;
  /** Append `phrase` to the active prompt with sensible whitespace. */
  appendPromptPhrase: (phrase: string) => void;
  toRequest: () => GenerateRequestWire;
  isVideoFamily: (family: string) => boolean;
  supportsNegativePrompt: (family: string) => boolean;
  supportsScheduler: (family: string) => boolean;
  /** Mirrors `mold-tui/src/model_info.rs::capabilities_for_family.supports_lora`
   * and the server-side `require_lora_capable_family` gate. Drives the
   * conditional render of `<LoraPicker>` in the SettingsModal. */
  supportsLora: (family: string) => boolean;
}

export function useGenerateForm(): UseGenerateForm {
  const state = ref<GenerateFormState>(load());

  let timer: ReturnType<typeof setTimeout> | null = null;
  watch(
    state,
    (v) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => persist(v), 300);
    },
    { deep: true },
  );

  return {
    state,
    reset: () => {
      state.value = defaultForm();
    },
    setLoras: (loras) => {
      state.value.loras = loras.slice(0, MAX_LORA_STACK);
    },
    addLora: (lora) => {
      if (state.value.loras.length >= MAX_LORA_STACK) return;
      state.value.loras = [...state.value.loras, lora];
    },
    updateLoraScale: (index, scale) => {
      const next = state.value.loras.slice();
      const row = next[index];
      if (!row) return;
      next[index] = { ...row, scale };
      state.value.loras = next;
    },
    removeLora: (index) => {
      if (index < 0 || index >= state.value.loras.length) return;
      const next = state.value.loras.slice();
      next.splice(index, 1);
      state.value.loras = next;
    },
    appendPromptPhrase: (phrase: string) => {
      const trimmed = phrase.trim();
      if (!trimmed) return;
      const current = state.value.prompt;
      // Reuse comma+space if the prompt is non-empty so trigger phrases
      // chain naturally ("a cat, cinematic, dramatic lighting") rather
      // than concatenating into a wall of text. Avoid the comma when the
      // prompt is empty to keep simple cases clean.
      state.value.prompt = current.trim()
        ? `${current.trimEnd()}, ${trimmed}`
        : trimmed;
    },
    applyModelDefaults: (m) => {
      // LoRA support is family-specific. Clear the stack on every model
      // change — even FLUX→FLUX swaps because the LoRA might not target
      // the new variant's tensor layout.
      state.value = modelDefaultsPatch(state.value, m);
    },
    toRequest: () => {
      const s = state.value;
      // The wire format strips per-row metadata (trigger phrases) — only
      // path + scale travel to the server. We send `loras` (plural) so
      // multi-LoRA stacks reach the FLUX engine; older single-LoRA
      // clients still set `lora`, which the server coalesces.
      const loras = s.loras.length
        ? s.loras.map((l) => ({ path: l.path, scale: l.scale }))
        : undefined;
      const qwenEdit = isQwenImageEditFamily(selectedFamily(s));
      const attachments = s.imageAttachments ?? [];
      const controlModel = s.controlModel.trim();
      const upscaleModel = s.upscaleModel.trim();
      const audioPath = s.audioFilePath.trim();
      const sourceVideoPath = s.sourceVideoPath.trim();
      const family = selectedFamily(s);
      const ltx2 = family === "ltx2" || family === "ltx-2";
      const sd3 = family === "sd3" || family === "sd3.5";
      return {
        prompt: s.prompt,
        negative_prompt: s.negativePrompt || null,
        model: s.model,
        width: s.width,
        height: s.height,
        steps: s.steps,
        guidance: s.guidance,
        seed: s.seedMode === "random" ? null : s.seed,
        batch_size: qwenEdit ? 1 : s.batchSize,
        output_format: s.outputFormat,
        cfg_plus: sd3 && s.cfgPlus ? true : undefined,
        scheduler: s.scheduler,
        ...(qwenEdit
          ? {
              edit_images: attachments.map((image) => image.base64),
            }
          : {
              source_image: attachments[0]?.base64 ?? null,
              strength: s.strength,
              mask_image: s.maskImage?.base64 ?? undefined,
              control_image: s.controlImage?.base64 ?? undefined,
              control_model:
                s.controlImage && controlModel ? controlModel : undefined,
              control_scale:
                s.controlImage && controlModel ? s.controlScale : undefined,
            }),
        expand: s.expand.enabled || undefined,
        frames: s.frames,
        fps: s.fps,
        upscale_model: upscaleModel || undefined,
        gif_preview: s.gifPreview || undefined,
        placement: s.placement ?? undefined,
        loras,
        enable_audio: s.enableAudio ?? undefined,
        ...(ltx2
          ? {
              audio_file: s.audioFile?.base64 ?? undefined,
              audio_file_path: s.audioFile ? undefined : audioPath || undefined,
              source_video: s.sourceVideo?.base64 ?? undefined,
              source_video_path: s.sourceVideo
                ? undefined
                : sourceVideoPath || undefined,
              keyframes: s.keyframes.length
                ? s.keyframes.map((k) => ({
                    frame: k.frame,
                    image: k.image.base64,
                  }))
                : undefined,
              pipeline: s.pipeline ?? undefined,
              retake_range: s.retakeRange ?? undefined,
              spatial_upscale: s.spatialUpscale ?? undefined,
              temporal_upscale: s.temporalUpscale ?? undefined,
            }
          : {}),
      };
    },
    isVideoFamily: (family: string) => VIDEO_FAMILIES.includes(family),
    supportsNegativePrompt: (family: string) =>
      !NO_CFG_FAMILIES.includes(family),
    supportsScheduler: (family: string) =>
      UNET_SCHEDULER_FAMILIES.includes(family),
    supportsLora,
  };
}

// Scheduler type is re-exported so callers can type-narrow without importing
// both modules.
export type { Scheduler };
export { isQwenImageEditFamily };
