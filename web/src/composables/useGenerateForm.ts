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
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
  familySupportsAudio,
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

function defaultForm(): GenerateFormState {
  return {
    version: 1,
    prompt: "",
    negativePrompt: "",
    model: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    seed: null,
    batchSize: 1,
    strength: 0.75,
    frames: null,
    fps: null,
    scheduler: null,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    sourceImage: null,
    placement: null,
    lora: null,
    enableAudio: null,
  };
}

function load(): GenerateFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultForm();
    const parsed = JSON.parse(raw) as Partial<GenerateFormState>;
    if (parsed.version !== 1) return defaultForm();
    return { ...defaultForm(), ...parsed, sourceImage: null };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  try {
    // Drop base64 bytes from localStorage — they blow past the quota quickly
    // and the attachment is re-picked trivially on reload.
    const { sourceImage: _omit, ...rest } = state;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rest));
  } catch {
    /* ignore */
  }
}

export interface UseGenerateForm {
  state: Ref<GenerateFormState>;
  reset: () => void;
  applyModelDefaults: (model: ModelInfoExtended) => void;
  setLora: (lora: LoraSelection | null) => void;
  toRequest: () => GenerateRequestWire;
  isVideoFamily: (family: string) => boolean;
  supportsNegativePrompt: (family: string) => boolean;
  supportsScheduler: (family: string) => boolean;
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
    setLora: (lora) => {
      state.value.lora = lora;
    },
    applyModelDefaults: (m) => {
      state.value.model = m.name;
      state.value.width = m.default_width;
      state.value.height = m.default_height;
      state.value.steps = m.default_steps;
      state.value.guidance = m.default_guidance;
      state.value.lora = null; // LoRA is family-specific; clear on model change
      // Video families need sensible frame/fps defaults.
      if (VIDEO_FAMILIES.includes(m.family)) {
        state.value.frames ??= 25; // 8n+1
        state.value.fps ??= 24;
      } else {
        state.value.frames = null;
        state.value.fps = null;
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
    },
    toRequest: () => {
      const s = state.value;
      return {
        prompt: s.prompt,
        negative_prompt: s.negativePrompt || null,
        model: s.model,
        width: s.width,
        height: s.height,
        steps: s.steps,
        guidance: s.guidance,
        seed: s.seed,
        batch_size: s.batchSize,
        output_format: s.outputFormat,
        scheduler: s.scheduler,
        source_image: s.sourceImage?.base64 ?? null,
        strength: s.strength,
        expand: s.expand.enabled || undefined,
        frames: s.frames,
        fps: s.fps,
        placement: s.placement ?? undefined,
        lora: s.lora ?? undefined,
        enable_audio: s.enableAudio ?? undefined,
      };
    },
    isVideoFamily: (family: string) => VIDEO_FAMILIES.includes(family),
    supportsNegativePrompt: (family: string) =>
      !NO_CFG_FAMILIES.includes(family),
    supportsScheduler: (family: string) =>
      UNET_SCHEDULER_FAMILIES.includes(family),
  };
}

// Scheduler type is re-exported so callers can type-narrow without importing
// both modules.
export type { Scheduler };
