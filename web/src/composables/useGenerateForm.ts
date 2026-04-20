import { ref, watch, type Ref } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  ModelInfoExtended,
  Scheduler,
} from "../types";
import {
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
} from "../types";

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
    applyModelDefaults: (m) => {
      state.value.model = m.name;
      state.value.width = m.default_width;
      state.value.height = m.default_height;
      state.value.steps = m.default_steps;
      state.value.guidance = m.default_guidance;
      // Video families need sensible frame/fps defaults.
      if (VIDEO_FAMILIES.includes(m.family)) {
        state.value.frames ??= 25; // 8n+1
        state.value.fps ??= 24;
      } else {
        state.value.frames = null;
        state.value.fps = null;
      }
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
