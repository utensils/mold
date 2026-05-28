import { ref, watch, type Ref } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  LoraSelection,
  ModelInfoExtended,
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
    return { ...defaultForm(), ...migrateLegacy(parsed) };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  try {
    // Drop base64 bytes from localStorage — they blow past the quota quickly
    // and the attachment is re-picked trivially on reload.
    const {
      imageAttachments: _attachments,
      maskImage: _mask,
      controlImage: _control,
      audioFile: _audio,
      sourceVideo: _video,
      keyframes: _keyframes,
      ...rest
    } = state;
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...rest, version: FORM_VERSION }),
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
      state.value.model = m.name;
      state.value.modelFamily = m.family;
      state.value.width = m.default_width;
      state.value.height = m.default_height;
      state.value.steps = m.default_steps;
      state.value.guidance = m.default_guidance;
      // LoRA support is family-specific. Clear the stack on every model
      // change — even FLUX→FLUX swaps because the LoRA might not target
      // the new variant's tensor layout.
      state.value.loras = [];
      // Video families need sensible frame/fps defaults.
      if (VIDEO_FAMILIES.includes(m.family)) {
        state.value.frames ??= 25; // 8n+1
        state.value.fps ??= 24;
      } else {
        state.value.frames = null;
        state.value.fps = null;
        state.value.gifPreview = false;
      }
      // Auto-pick a valid output format whenever the model family changes so
      // users never have to manually toggle this — switching from an image
      // to a video family and back would otherwise leave an invalid format
      // (e.g. `png` on LTX-2) stuck in the form.
      const formats = outputFormatsForFamily(m.family);
      if (!formats.includes(state.value.outputFormat)) {
        state.value.outputFormat = formats[0];
      }
      // Audio toggle defaults: ON for the only family with an audio path
      // today (LTX-2 / LTX-2.3), null for everyone else so the wire stays
      // clean and the server's MP4 default-on behaviour isn't fought.
      // Mirrors `chain_limits::family_supports_audio` on the server.
      state.value.enableAudio = familySupportsAudio(m.family) ? true : null;
      if (m.family !== "ltx2" && m.family !== "ltx-2") {
        state.value.audioFile = null;
        state.value.audioFilePath = "";
        state.value.sourceVideo = null;
        state.value.sourceVideoPath = "";
        state.value.keyframes = [];
        state.value.pipeline = null;
        state.value.retakeRange = null;
        state.value.spatialUpscale = null;
        state.value.temporalUpscale = null;
      }
      if (isQwenImageEditFamily(m.family)) {
        state.value.batchSize = 1;
        state.value.maskImage = null;
        state.value.controlImage = null;
        state.value.controlModel = "";
      } else if (state.value.imageAttachments.length > 1) {
        state.value.imageAttachments = state.value.imageAttachments.slice(0, 1);
      }
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
