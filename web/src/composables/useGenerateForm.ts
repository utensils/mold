import { ref, watch, type Ref, type WatchStopHandle } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  LoraSelection,
  ModelInfoExtended,
  OutputMetadata,
  OutputFormat,
  Scheduler,
} from "../types";
import { MAX_LORA_STACK } from "../types";
import {
  clearMemoryDraftsForTest,
  getDraftMedia,
  newDraftId,
  putDraftMedia,
} from "../lib/draftMediaStore";
import {
  generationCapabilitiesForFamily,
  isQwenImageEditFamily,
  isVideoFamily,
  supportsLora,
  supportsNegativePrompt,
  supportsScheduler,
} from "../lib/generateCapabilities";
import { composeStyle } from "../lib/stylePresets";
import { defaultVideoFps } from "@studio/lib/sequence";

/** The prompt actually sent to the server: the textarea content composed with
 * the active style preset (the shared kit substitutes a "{prompt}" template,
 * otherwise it comma-appends). Never mutates `state.prompt` — the style row is
 * a request-time modifier, not a rewrite of what the user typed. Shared by
 * `toRequest` and the estimate/summary display so both agree. */
export function promptWithStyle(state: GenerateFormState): string {
  return composeStyle(state.prompt, state.stylePreset ?? "", {
    supportsNegativePrompt: false,
  }).prompt;
}

/** Output-format options for a given model family, ordered by preference.
 * The first entry is the default the UI auto-selects when a model is
 * chosen. */
export function outputFormatsForFamily(family: string): OutputFormat[] {
  return isVideoFamily(family)
    ? ["mp4", "gif", "apng", "webp"]
    : ["png", "jpeg", "webp"];
}

const STORAGE_KEY = "mold.generate.form";
const FORM_VERSION = 3 as const;
const QWEN_IMAGE_EDIT_FAMILY = "qwen-image-edit";

function selectedFamily(s: GenerateFormState): string {
  if (s.modelFamily) return s.modelFamily;
  if (s.model.startsWith(`${QWEN_IMAGE_EDIT_FAMILY}:`)) {
    return QWEN_IMAGE_EDIT_FAMILY;
  }
  if (s.model.startsWith("sdxl")) return "sdxl";
  if (s.model.startsWith("sd15") || s.model.startsWith("sd1.5")) return "sd15";
  if (s.model.startsWith("sd3.5")) return "sd3.5";
  if (s.model.startsWith("sd3")) return "sd3";
  if (s.model.startsWith("flux2") || s.model.startsWith("flux.2"))
    return "flux2";
  if (s.model.startsWith("flux")) return "flux";
  if (s.model.startsWith("z-image")) return "z-image";
  if (s.model.startsWith("qwen-image-edit")) return "qwen-image-edit";
  if (s.model.startsWith("qwen-image")) return "qwen-image";
  if (s.model.startsWith("ltx-video")) return "ltx-video";
  if (s.model.startsWith("ltx2") || s.model.startsWith("ltx-2")) return "ltx2";
  return "";
}

function defaultForm(): GenerateFormState {
  return {
    version: FORM_VERSION,
    prompt: "",
    stylePreset: null,
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
    sourceFitPolicy: { mode: "pad-repaint" },
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
    icLoraControl: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    cameraControl: null,
    placement: null,
    loras: [],
    enableAudio: null,
  };
}

function cloneFormState<T>(state: T): T {
  return JSON.parse(JSON.stringify(state)) as T;
}

/** Remove browser-only binary media from a form snapshot before writing it to
 * localStorage or a local generation template. Server-local path fields
 * (`audioFilePath`, `sourceVideoPath`) are preserved because they are stable
 * references; upload/gallery base64 payloads are intentionally not stored. */
export function sanitizePersistedForm(
  state: GenerateFormState,
): GenerateFormState {
  const sanitized = {
    ...cloneFormState(state),
    version: FORM_VERSION,
    imageAttachments: state.imageAttachments.map(stripMediaBytes),
    maskImage: state.maskImage ? stripMediaBytes(state.maskImage) : null,
    controlImage: state.controlImage
      ? stripMediaBytes(state.controlImage)
      : null,
    audioFile: state.audioFile ? stripMediaBytes(state.audioFile) : null,
    sourceVideo: state.sourceVideo ? stripMediaBytes(state.sourceVideo) : null,
    keyframes: state.keyframes.map((k) => ({
      ...k,
      image: stripMediaBytes(k.image),
    })),
  };
  return sanitized;
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
    icLoraControl: null,
  };
  const capabilities = generationCapabilitiesForFamily(model.family);
  if (capabilities.supportsVideo) {
    next.frames ??= 25;
    // The model's advertised rate is applied like steps/guidance — it is only
    // absent-server/absent-field that leaves the current value in place.
    next.fps = defaultVideoFps(model, next.fps);
  } else {
    next.frames = null;
    next.fps = null;
    next.gifPreview = false;
  }
  const formats = outputFormatsForFamily(model.family);
  if (!formats.includes(next.outputFormat)) {
    next.outputFormat = formats[0];
  }
  next.enableAudio = capabilities.supportsAudio
    ? model.supports_audio !== false
    : null;
  if (!capabilities.supportsScheduler) {
    next.scheduler = null;
  }
  if (!capabilities.supportsCfgPlus) {
    next.cfgPlus = false;
  }
  if (model.family !== "ltx2" && model.family !== "ltx-2") {
    next.audioFile = null;
    next.audioFilePath = "";
    next.sourceVideo = null;
    next.sourceVideoPath = "";
    next.keyframes = [];
    next.pipeline = null;
    next.icLoraControl = null;
    next.retakeRange = null;
    next.spatialUpscale = null;
    next.temporalUpscale = null;
    next.cameraControl = null;
  } else if (model.name.includes("ltx-2.3")) {
    const camera = next.cameraControl?.trim();
    if (camera && !camera.endsWith(".safetensors")) next.cameraControl = null;
  }
  if (capabilities.sourceImageMode === "qwen-edit") {
    next.batchSize = 1;
    next.maskImage = null;
    next.controlImage = null;
    next.controlModel = "";
  } else if (next.imageAttachments.length > 1) {
    next.imageAttachments = next.imageAttachments.slice(0, 1);
  }
  if (!capabilities.supportsControlNet) {
    next.controlImage = null;
    next.controlModel = "";
  }
  return next;
}

/**
 * "Reset settings to defaults" (Create rail): every generation knob returns to
 * the factory default, then to the selected model's own defaults on top.
 *
 * What survives is what the reset is *for* — the user is re-tuning a print they
 * are still composing: the prompt, the style preset, the selected model/family,
 * and the batch size (prepared batch work is never silently resized, see
 * CLAUDE.md). Everything else — shape, resolution, detail, prompt strength,
 * seed, and every advanced field including source media, LoRAs and the video
 * suite — goes back to defaults.
 */
export function settingsResetPatch(
  current: GenerateFormState,
  model: ModelInfoExtended | null,
): GenerateFormState {
  const base: GenerateFormState = {
    ...defaultForm(),
    prompt: current.prompt,
    stylePreset: current.stylePreset,
    model: current.model,
    modelFamily: current.modelFamily,
    batchSize: current.batchSize,
  };
  // With a resolved catalog row the model's own defaults (and its capability
  // gates) win over the generic ones; without it the generic defaults stand and
  // the model name is left alone.
  return model ? modelDefaultsPatch(base, model) : base;
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
    // Saved metadata already carries the fully-composed prompt (style extras
    // included at generation time); re-applying a preset would double-append.
    stylePreset: null,
    negativePrompt: metadata.negative_prompt ?? "",
    width: metadata.generation_width || metadata.width || next.width,
    height: metadata.generation_height || metadata.height || next.height,
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
    icLoraControl: metadata.ic_lora_control ?? null,
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

function stripMediaBytes<T extends { base64: string }>(media: T): T {
  const { base64: _base64, ...rest } = cloneFormState(media) as T & {
    base64?: string;
  };
  void _base64;
  return rest as T;
}

function ensureDraftIds(state: GenerateFormState) {
  const ensure = <T extends { base64: string; draftId?: string } | null>(
    media: T,
  ): T => {
    if (media?.base64 && !media.draftId) media.draftId = newDraftId();
    return media;
  };
  state.imageAttachments = state.imageAttachments.map((m) => ensure(m)!);
  state.maskImage = ensure(state.maskImage);
  state.controlImage = ensure(state.controlImage);
  state.audioFile = ensure(state.audioFile);
  state.sourceVideo = ensure(state.sourceVideo);
  state.keyframes = state.keyframes.map((k) => ({
    ...k,
    image: ensure(k.image)!,
  }));
}

function mediaFromState(state: GenerateFormState) {
  return [
    ...state.imageAttachments,
    state.maskImage,
    state.controlImage,
    state.audioFile,
    state.sourceVideo,
    ...state.keyframes.map((k) => k.image),
  ].filter((m): m is NonNullable<typeof m> => Boolean(m?.base64));
}

let pendingDraftWrites: Promise<unknown>[] = [];
let pendingHydration: Promise<unknown> | null = null;

function persistDraftMedia(state: GenerateFormState) {
  ensureDraftIds(state);
  pendingDraftWrites = mediaFromState(state).map((m) => putDraftMedia(m));
  return Promise.allSettled(pendingDraftWrites);
}

async function hydrateDraftMedia(state: GenerateFormState) {
  const hydrate = async <
    T extends { draftId?: string; base64?: string } | null,
  >(
    media: T,
  ): Promise<T> => {
    if (!media?.draftId || media.base64) return media;
    const stored = await getDraftMedia(media.draftId);
    return stored ? ({ ...media, ...stored } as T) : media;
  };
  const attachmentIds = state.imageAttachments.map((m) => m.draftId ?? "");
  const attachments = await Promise.all(
    state.imageAttachments.map((m) => hydrate(m)),
  );
  if (
    state.imageAttachments.length === attachmentIds.length &&
    state.imageAttachments.every(
      (m, i) => (m.draftId ?? "") === attachmentIds[i],
    )
  ) {
    state.imageAttachments = attachments;
  }

  const maskId = state.maskImage?.draftId ?? "";
  const mask = await hydrate(state.maskImage);
  if ((state.maskImage?.draftId ?? "") === maskId) state.maskImage = mask;

  const controlId = state.controlImage?.draftId ?? "";
  const control = await hydrate(state.controlImage);
  if ((state.controlImage?.draftId ?? "") === controlId)
    state.controlImage = control;

  const audioId = state.audioFile?.draftId ?? "";
  const audio = await hydrate(state.audioFile);
  if ((state.audioFile?.draftId ?? "") === audioId) state.audioFile = audio;

  const sourceVideoId = state.sourceVideo?.draftId ?? "";
  const sourceVideo = await hydrate(state.sourceVideo);
  if ((state.sourceVideo?.draftId ?? "") === sourceVideoId) {
    state.sourceVideo = sourceVideo;
  }

  const keyframeIds = state.keyframes.map(
    (k) => `${k.frame}:${k.image.draftId ?? ""}`,
  );
  const keyframes = await Promise.all(
    state.keyframes.map(async (k) => ({ ...k, image: await hydrate(k.image) })),
  );
  if (
    state.keyframes.length === keyframeIds.length &&
    state.keyframes.every(
      (k, i) => `${k.frame}:${k.image.draftId ?? ""}` === keyframeIds[i],
    )
  ) {
    state.keyframes = keyframes;
  }
}

function load(): GenerateFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultForm();
    const parsed = JSON.parse(raw) as Partial<GenerateFormState>;
    if (parsed.version !== FORM_VERSION) {
      return defaultForm();
    }
    return {
      ...defaultForm(),
      ...sanitizePersistedForm({
        ...defaultForm(),
        ...parsed,
      }),
    };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  try {
    void persistDraftMedia(state);
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
  /** Restore every generation setting to the selected model's defaults while
   * keeping the prompt, style, model and batch size (`settingsResetPatch`). */
  resetSettings: (model: ModelInfoExtended | null) => void;
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
  /**
   * Build the wire request. Pass the resolved catalog row when available so
   * checkpoint-specific capabilities can override stale persisted form state.
   */
  toRequest: (model?: ModelInfoExtended | null) => GenerateRequestWire;
  isVideoFamily: (family: string) => boolean;
  supportsNegativePrompt: (family: string) => boolean;
  supportsScheduler: (family: string) => boolean;
  /** Mirrors `mold-tui/src/model_info.rs::capabilities_for_family.supports_lora`
   * and the server-side `require_lora_capable_family` gate. Drives the
   * conditional render of `<LoraPicker>` in the SettingsModal. */
  supportsLora: (family: string) => boolean;
}

let singleton: UseGenerateForm | null = null;
let stopPersistWatch: WatchStopHandle | null = null;
let persistTimer: ReturnType<typeof setTimeout> | null = null;

export function useGenerateForm(): UseGenerateForm {
  if (singleton) return singleton;
  const state = ref<GenerateFormState>(load());
  pendingHydration = hydrateDraftMedia(state.value);

  stopPersistWatch = watch(
    state,
    (v) => {
      if (persistTimer) clearTimeout(persistTimer);
      persistTimer = setTimeout(() => persist(v), 300);
    },
    { deep: true },
  );

  singleton = {
    state,
    reset: () => {
      state.value = defaultForm();
    },
    resetSettings: (model) => {
      state.value = settingsResetPatch(state.value, model);
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
    toRequest: (model = null) => {
      const s = state.value;
      // The wire format strips per-row metadata (trigger phrases) — only
      // path + scale travel to the server. We send `loras` (plural) so
      // multi-LoRA stacks reach the FLUX engine; older single-LoRA
      // clients still set `lora`, which the server coalesces.
      let loras = s.loras.map((l) => ({ path: l.path, scale: l.scale }));
      const capabilities = generationCapabilitiesForFamily(selectedFamily(s));
      const qwenEdit = capabilities.sourceImageMode === "qwen-edit";
      const attachments = s.imageAttachments ?? [];
      const controlModel = capabilities.supportsControlNet
        ? s.controlModel.trim()
        : "";
      const upscaleModel = s.upscaleModel.trim();
      const audioPath = s.audioFilePath.trim();
      const sourceVideoPath = s.sourceVideoPath.trim();
      const family = selectedFamily(s);
      const ltx2 = family === "ltx2" || family === "ltx-2";
      const cameraControl = s.cameraControl?.trim();
      if (ltx2 && cameraControl) {
        loras = loras.slice(0, MAX_LORA_STACK - 1);
        loras.push({
          path: cameraControl.endsWith(".safetensors")
            ? cameraControl
            : `camera-control:${cameraControl}`,
          scale: 1,
        });
      }
      // The style preset is baked into the OUTGOING request — prompt template
      // plus the preset's curated negative, merged after the user's own
      // fragments and only for families that accept a negative prompt. Neither
      // field on screen is rewritten.
      const styled = composeStyle(s.prompt, s.stylePreset ?? "", {
        supportsNegativePrompt: capabilities.supportsNegativePrompt,
        negative: s.negativePrompt,
      });
      return {
        prompt: styled.prompt,
        negative_prompt: capabilities.supportsNegativePrompt
          ? styled.negative || null
          : null,
        model: s.model,
        width: s.width,
        height: s.height,
        steps: s.steps,
        guidance: s.guidance,
        seed: s.seedMode === "random" ? null : s.seed,
        batch_size: capabilities.forcesBatchSizeOne ? 1 : s.batchSize,
        output_format: s.outputFormat,
        cfg_plus: capabilities.supportsCfgPlus && s.cfgPlus ? true : undefined,
        scheduler: capabilities.supportsScheduler ? s.scheduler : undefined,
        ...(qwenEdit
          ? {
              edit_images: attachments.map((image) => image.base64),
            }
          : {
              source_image: attachments[0]?.base64 ?? null,
              source_image_name: attachments[0]?.base64
                ? attachments[0].filename
                : undefined,
              strength: attachments[0]?.base64 ? s.strength : undefined,
              mask_image: s.maskImage?.base64 ?? undefined,
              control_image: capabilities.supportsControlNet
                ? (s.controlImage?.base64 ?? undefined)
                : undefined,
              control_model:
                s.controlImage && controlModel ? controlModel : undefined,
              control_scale:
                s.controlImage && controlModel ? s.controlScale : undefined,
            }),
        expand: s.expand.enabled || undefined,
        frames: capabilities.supportsVideo ? s.frames : undefined,
        fps: capabilities.supportsVideo ? s.fps : undefined,
        upscale_model: upscaleModel || undefined,
        gif_preview:
          capabilities.supportsVideo && s.gifPreview ? true : undefined,
        placement: s.placement ?? undefined,
        loras: loras.length ? loras : undefined,
        // A persisted draft can retain `true` from a previously-selected AV
        // checkpoint. The resolved model row is the final authority at submit
        // time: video-only LTX-2 exports must never reach the server with audio
        // enabled, even if initial model hydration did not reapply defaults.
        enable_audio:
          model?.supports_audio === false
            ? false
            : (s.enableAudio ?? undefined),
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
              pipeline: s.icLoraControl ? "ic-lora" : (s.pipeline ?? undefined),
              ic_lora_control: s.icLoraControl ?? undefined,
              retake_range: s.retakeRange ?? undefined,
              spatial_upscale: s.spatialUpscale ?? undefined,
              temporal_upscale: s.temporalUpscale ?? undefined,
            }
          : {}),
      };
    },
    isVideoFamily,
    supportsNegativePrompt,
    supportsScheduler,
    supportsLora,
  };
  return singleton;
}

// Scheduler type is re-exported so callers can type-narrow without importing
// both modules.
export type { Scheduler };
export { isQwenImageEditFamily };

export const __testing__ = {
  resetForTest() {
    stopPersistWatch?.();
    stopPersistWatch = null;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = null;
    singleton = null;
    pendingDraftWrites = [];
    pendingHydration = null;
  },
  async flushDraftWrites() {
    await Promise.allSettled(pendingDraftWrites);
  },
  async flushHydration() {
    await pendingHydration;
  },
  clearDraftsForTest() {
    clearMemoryDraftsForTest();
  },
};
