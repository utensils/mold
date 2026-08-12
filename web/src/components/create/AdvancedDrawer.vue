<script setup lang="ts">
/*
 * Advanced controls (Mold Studio Create) — capability-gated, always-open
 * icon sections. Each section maps onto EXISTING form fields; sections a
 * family doesn't support never render.
 *
 * Surface split (spec §06 v0.13): the web app is the power surface, so at
 * tablet width and above the applicable sections render INLINE as an
 * always-visible column in the Create controls region (`mobile` false → a plain <section>).
 * On phones (`mobile` true) the same content collapses into the full-screen
 * Advanced SheetPanel. Reset clears advanced fields only — the prompt, model,
 * shape, resolution, detail and seed survive.
 */
import { computed, ref, watch } from "vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import Chip from "@ui/components/Chip.vue";
import LoraPicker from "../LoraPicker.vue";
import PlacementPanel from "../PlacementPanel.vue";
import ExtendVideoControls from "./advanced/ExtendVideoControls.vue";
import Ltx2VideoControls from "./advanced/Ltx2VideoControls.vue";
import MinimaxH3AuthoringPanel from "@studio/components/MinimaxH3AuthoringPanel.vue";
import {
  DEFAULT_EXTEND_OVERLAP_FRAMES,
  submitsExtend,
} from "@studio/lib/extend";
import UpscaleSection from "./advanced/UpscaleSection.vue";
import type {
  DevicePlacement,
  GenerateFormState,
  Ltx2CameraControlInfo,
  LoraSelection,
  ModelInfoExtended,
  OutputFormat,
  Scheduler,
  SourceFitPolicy,
  SourceImageState,
} from "../../types";
import {
  generationCapabilitiesForFamily,
  isFlux2DevModel,
} from "../../lib/generateCapabilities";
import { schedulerLabel } from "@studio/lib/generationCapabilities";
import {
  MAX_WAN_DISTILL_STRENGTH,
  emptyWanRecipe,
  wanRecipeCount,
  wanRecipeError,
  type WanRecipeState,
} from "@studio/lib/wanRecipe";
import { outputFormatsForFamily } from "../../composables/useGenerateForm";
import { blobToBase64 } from "../../lib/base64";
import { useOverlayFocus } from "../../composables/useOverlayFocus";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { OutputMode } from "@studio/lib/sequence";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import {
  clampVideoFrames,
  fixedVideoFps,
  minVideoFrames,
  snapVideoFrames,
  videoFrameGridLabel,
  videoFrameStep,
} from "@studio/lib/videoDuration";
import {
  SOURCE_FIT_OPTIONS,
  coerceSourceFitForMaskless,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import {
  cameraMotionLoraPath,
  cameraMotionMode,
  parseCameraControlAvailability,
  isCameraMotionPreset,
} from "@studio/lib/cameraMotion";
import {
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  type MinimaxH3AuthoringState,
} from "@studio/lib/minimaxH3Authoring";
import { sourceImageValidationError } from "@studio/lib/sourceImageCapability";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";

const props = withDefaults(
  defineProps<{
    /** Sheet open state (phone only; ignored when inline). */
    open?: boolean;
    modelValue: GenerateFormState;
    family: string;
    advCount?: number;
    /** Phone surface → SheetPanel instead of DrawerPanel. */
    mobile?: boolean;
    /** GPUs for the placement section (empty → section hidden). */
    placementGpus?: { ordinal: number; name: string }[];
    /** Installed models on the selected generation route. */
    models?: ModelInfoExtended[];
    /** Selected model's resolved audio assets are available. */
    audioOutputSupported?: boolean;
    /** Host advertises `video.can_extend`; false hides the continuation UI. */
    canExtend?: boolean;
    extendDefaultOverlapFrames?: number;
    output?: OutputMode;
    sequenceSupportsAudio?: boolean;
  }>(),
  {
    open: false,
    advCount: 0,
    mobile: false,
    placementGpus: () => [],
    models: () => [],
    audioOutputSupported: true,
    canExtend: false,
    extendDefaultOverlapFrames: DEFAULT_EXTEND_OVERLAP_FRAMES,
    output: "single",
    sequenceSupportsAudio: false,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  close: [];
  "open-picker": [];
  "clear-source": [];
  /** Wan first/last-frame conditioning (#779). The closing still gets its own
   * picker so it can never overwrite the opening one. */
  "open-end-frame-picker": [];
  "clear-end-frame": [];
  "open-mask": [];
  "append-prompt": [phrase: string];
}>();
const host = ref<HTMLElement | { $el?: unknown } | null>(null);
const draft = useSequenceDraftStore();
const sequenceMode = computed(() => props.output === "sequence");
const selectedModel = computed(
  () =>
    props.models.find((model) => model.name === props.modelValue.model) ?? null,
);
const activeSequenceClip = computed(
  () =>
    draft.clips.find((clip) => clip.id === draft.activeClipId) ??
    draft.clips[0] ??
    null,
);
const activeSequenceIndex = computed(() =>
  activeSequenceClip.value
    ? draft.clips.findIndex((clip) => clip.id === activeSequenceClip.value?.id)
    : -1,
);
const sequenceCameraControls = ref<Ltx2CameraControlInfo[]>([]);
const sequenceCameraControlsLoaded = ref(false);
const sequenceCameraUnsupportedReason = ref<string | null>(null);
let cameraControlsEpoch = 0;
watch(
  [sequenceMode, () => props.family, () => props.modelValue.model],
  async () => {
    const epoch = ++cameraControlsEpoch;
    // Drop the previous model's reason immediately; keeping it while the
    // new request is in flight shows a stale explanation for the wrong model.
    sequenceCameraUnsupportedReason.value = null;
    sequenceCameraControls.value = [];
    sequenceCameraControlsLoaded.value = false;
    if (
      !sequenceMode.value ||
      props.family !== "ltx2" ||
      !props.modelValue.model
    )
      return;
    try {
      const response = await fetch(
        `/api/capabilities/ltx2-camera-controls?model=${encodeURIComponent(props.modelValue.model)}&detail=1`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const availability = parseCameraControlAvailability(
        await response.json(),
      );
      const options = availability.controls;
      if (epoch !== cameraControlsEpoch) return;
      sequenceCameraControls.value = options;
      sequenceCameraUnsupportedReason.value = availability.unsupportedReason;
      sequenceCameraControlsLoaded.value = true;
      for (const clip of draft.clips) {
        const camera = clip.cameraControl;
        if (
          camera &&
          isCameraMotionPreset(camera) &&
          !options.some((option) => option.id === camera)
        ) {
          clip.cameraControl = null;
        }
      }
    } catch {
      // The custom-path escape hatch remains usable while the host recovers.
      if (epoch === cameraControlsEpoch) {
        sequenceCameraUnsupportedReason.value = null;
        sequenceCameraControlsLoaded.value = false;
      }
    }
  },
  { immediate: true },
);
const overlayOpen = computed(() => props.mobile && props.open);
const { onKeydown } = useOverlayFocus(overlayOpen, host, () => emit("close"));

const NEG_CHIPS = [
  "blurry",
  "extra fingers",
  "watermark",
  "low quality",
  "oversaturated",
];

// Desktop/tablet web (spec §06 v0.12): render inline as an always-visible
// column. Phone: render inside the Advanced sheet.
const inline = computed(() => !props.mobile);

// The fifth argument is the selected checkpoint's own advertised
// source-image contract (#772): wan's checkpoints split T2V / I2V-optional /
// I2V-required and only the server can tell them apart. Never read
// `source_image` (or a family set) here — the shared kit owns the
// absent-field fallback that keeps older servers on today's behaviour.
const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.family,
    props.modelValue.model,
    props.modelValue.pipeline,
    selectedModel.value?.guidance_capabilities,
    selectedModel.value?.source_image ?? props.modelValue.sourceImageCapability,
  ),
);
const flux2Dev = computed(() => isFlux2DevModel(props.modelValue.model));
const h3Task = computed(() =>
  selectedModel.value ? minimaxH3TaskForModel(props.modelValue.model) : null,
);
const h3Family = computed(() =>
  isMinimaxH3Identity(props.family, props.modelValue.model),
);
const h3Authoring = computed(
  () => props.modelValue.h3Authoring ?? emptyMinimaxH3AuthoringState(),
);
function setH3Authoring(value: MinimaxH3AuthoringState) {
  patch({ h3Authoring: value });
}
const formats = computed(() => outputFormatsForFamily(props.family));

// Wan puts its solver in the recipe section below, next to the flow shift it
// belongs with, so the generic section would otherwise render a second picker
// onto the same field.
const showScheduler = computed(
  () =>
    (caps.value.supportsScheduler && !caps.value.wanRecipe.supported) ||
    caps.value.supportsCfgPlus,
);
const showPlacement = computed(() => props.placementGpus.length > 0);

// Post-generate upscale applies to stills only — the server skips the
// upscaler whenever the response is a video (`queue.rs`), so video families
// never see the section.
const showUpscale = computed(() => !caps.value.supportsVideo);

// The Scheduler type permits parameterized object variants; the drawer only
// surfaces the named string schedulers.
const schedulerChoices = computed<string[]>(() =>
  caps.value.schedulerOptions.flatMap((s) =>
    typeof s === "string" ? [s] : [],
  ),
);
const schedulerName = computed(() =>
  typeof props.modelValue.scheduler === "string"
    ? props.modelValue.scheduler
    : "default",
);
const schedulerSummary = computed(() =>
  caps.value.supportsScheduler ? schedulerLabel(schedulerName.value) : "CFG++",
);

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

// ── Scheduler & sampling ──────────────────────────────────────────────
function setScheduler(value: string) {
  // "default" is a UI-only sentinel meaning "omit the field" — the wire enum
  // has no such variant, so it must become null here, not travel (codex
  // review).
  patch({ scheduler: value === "default" ? null : (value as Scheduler) });
}

// ── Wan sampler recipe ────────────────────────────────────────────────
// Every control writes null when cleared: the engine keeps the resolved
// tier's shift and distill strengths only while the field is absent from the
// request, so a cleared box must not serialize as 0.
const wanRecipe = computed<WanRecipeState>(
  () => props.modelValue.wanRecipe ?? emptyWanRecipe(),
);
const wanRecipeActive = computed(() => wanRecipeCount(wanRecipe.value));
const wanRecipeMessage = computed(() => wanRecipeError(wanRecipe.value));
const wanRecipeSummary = computed(() =>
  wanRecipeActive.value
    ? `${wanRecipeActive.value} set · ${schedulerLabel(schedulerName.value)}`
    : schedulerLabel(schedulerName.value),
);
function setWanRecipe(next: Partial<WanRecipeState>) {
  patch({ wanRecipe: { ...wanRecipe.value, ...next } });
}
function resetWanRecipe() {
  patch({ wanRecipe: emptyWanRecipe() });
}
function numberOrNull(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

// ── Negative prompt ───────────────────────────────────────────────────
function addNegative(word: string) {
  const cur = props.modelValue.negativePrompt.trim();
  patch({ negativePrompt: cur ? `${cur}, ${word}` : word });
}

// ── Source image ──────────────────────────────────────────────────────
const hasSource = computed(() => props.modelValue.imageAttachments.length > 0);
const hasEndFrame = computed(
  () => caps.value.supportsEndFrame && props.modelValue.endFrame != null,
);
/** Why the attached conditioning would be refused, in the server's own order.
 * H3 keeps its own authoring validator, which names its boundaries precisely.
 * A continuation carries its own first frames (#783), so the well's notice has
 * to read it the way submit and admission do or the two disagree. */
const sourceConditioningError = computed(() =>
  h3Family.value
    ? null
    : sourceImageValidationError({
        capability: caps.value.sourceImageCapability,
        hasSourceImage: hasSource.value,
        isExtend: submitsExtend({
          family: props.family,
          extendVideo: props.modelValue.extendVideo,
          extendVideoPath: props.modelValue.extendVideoPath,
        }),
        hasEndFrame: hasEndFrame.value,
        frames: caps.value.supportsVideo ? props.modelValue.frames : null,
        model: props.modelValue.model,
      }),
);
const fitOptions = [
  { value: "pad-fit", label: "Contain" },
  { value: "crop-fill", label: "Cover" },
  { value: "pad-repaint", label: "Pad + repaint" },
  { value: "lanczos-resize", label: "Lanczos" },
  { value: "upscale-then-fit", label: "Upscale + fit" },
] as const;
const fitMode = computed(
  () => props.modelValue.sourceFitPolicy?.mode ?? "pad-repaint",
);
const sequenceFitOptions = SOURCE_FIT_OPTIONS.filter(
  (option) => option.value !== "pad-repaint",
);
const sequenceFitMode = computed(
  () =>
    coerceSourceFitForMaskless(
      props.modelValue.sourceFitPolicy ?? { mode: "crop-fill" },
    ).mode,
);
function setSequenceFit(mode: SourceFitMode) {
  patch({
    sourceFitPolicy: sourceFitPolicyForMode(mode, {
      supportsMask: false,
      upscalerModel: props.modelValue.upscaleModel || "real-esrgan-x4plus:fp16",
    }),
  });
}
function setFit(mode: string) {
  if (mode === "upscale-then-fit") {
    patch({
      sourceFitPolicy: {
        mode,
        upscalerModel:
          props.modelValue.upscaleModel || "real-esrgan-x4plus:fp16",
        fit: { mode: "crop-fill" },
      },
    });
    return;
  }
  patch({ sourceFitPolicy: { mode } as SourceFitPolicy });
}

// ── ControlNet (guidance image + model + scale) ───────────────────────
// Gated by capability (only sd15 today). The control image is an inline
// upload — unlike the primary source dropzone, which routes through the
// parent's picker modal — so the drawer stays self-contained. Control
// model + scale surface only once an image is attached, matching the
// server's request-time gate (`control_model`/`control_scale` are sent
// only when a control image is present).
const showControlNet = computed(() => caps.value.supportsControlNet);
const hasControl = computed(() => props.modelValue.controlImage != null);
const controlModels = computed(() =>
  props.models.filter(
    (model) => model.downloaded && model.family === "controlnet",
  ),
);
async function readControlImage(
  event: Event,
): Promise<SourceImageState | null> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = "";
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}
async function onControlImage(event: Event) {
  const image = await readControlImage(event);
  if (image) patch({ controlImage: image });
}
function clearControl() {
  patch({ controlImage: null });
}

// LTX-2 / LTX-2.3 own the full advanced video suite (pipeline, audio,
// keyframes, retake, spatial/temporal). `supportsAudio` is true for
// exactly those families, so it doubles as the suite gate; plain
// ltx-video keeps just frames/fps/GIF.
const showLtx2 = computed(() => caps.value.supportsAudio && !h3Family.value);

// ── Output & seed: exact size follows the active recipe grid ──────────
const resolutionAlignment = computed(
  () =>
    effectiveGenerationRecipe(selectedModel.value, props.modelValue.pipeline)
      ?.resolution.alignment ??
    selectedModel.value?.dimension_alignment ??
    16,
);
function snapDim(v: number): number {
  const alignment = resolutionAlignment.value;
  if (!Number.isFinite(v) || v <= 0) return Math.max(64, alignment);
  return Math.max(64, Math.round(v / alignment) * alignment);
}
function setWidth(raw: string) {
  patch({ width: snapDim(Number(raw)) });
}
function setHeight(raw: string) {
  patch({ height: snapDim(Number(raw)) });
}
function swapDims() {
  patch({ width: props.modelValue.height, height: props.modelValue.width });
}

const seedModes = [
  { value: "random", label: "Random" },
  { value: "static", label: "Fixed" },
  { value: "increment", label: "Increment" },
] as const;

// ── LoRA / placement passthrough ──────────────────────────────────────
function setLoras(loras: LoraSelection[]) {
  const cameraPath = cameraMotionLoraPath(props.modelValue.cameraControl);
  patch({
    loras,
    cameraControl:
      cameraPath && !loras.some((lora) => lora.path === cameraPath)
        ? null
        : props.modelValue.cameraControl,
  });
}
function setPlacement(placement: DevicePlacement | null) {
  patch({ placement });
}

const videoContract = computed(
  () => selectedModel.value ?? { family: props.family },
);
const frameGridLabel = computed(() => videoFrameGridLabel(videoContract.value));
const frameStep = computed(() => videoFrameStep(videoContract.value));
const frameMinimum = computed(() => minVideoFrames(videoContract.value));
const fixedFps = computed(() => fixedVideoFps(videoContract.value));
watch(
  [fixedFps, () => props.modelValue.fps] as const,
  ([fixed, current]) => {
    if (fixed !== null && current !== fixed) patch({ fps: fixed });
  },
  { immediate: true },
);
function clampFrames(n: number): number {
  const value = Number.isFinite(n)
    ? n
    : (selectedModel.value?.default_frames ?? 25);
  return fixedFps.value !== null
    ? clampVideoFrames(value, fixedFps.value, videoContract.value)
    : Math.max(frameMinimum.value, snapVideoFrames(value, videoContract.value));
}

// ── Reset (advanced fields only — prompt/model/shape/seed survive) ────
function resetAdvanced() {
  if (sequenceMode.value) {
    draft.openingImage = null;
    draft.enableAudio = false;
    for (const clip of draft.clips) {
      clip.negativePrompt = "";
      clip.cameraControl = null;
    }
    patch({ strength: 0.75, sourceFitPolicy: { mode: "crop-fill" } });
    return;
  }
  patch({
    // Reset restores the model's advertised default negative (wan), not the
    // explicit empty opt-out — matching the iPhone reset.
    negativePrompt: props.modelValue.negativePromptDefault,
    scheduler: null,
    cfgPlus: false,
    imageAttachments: [],
    endFrame: null,
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1.0,
    strength: 0.75,
    sourceFitPolicy: { mode: "pad-repaint" },
    loras: [],
    upscaleModel: "",
    // Video suite (frames/fps survive as core video params, like resolution).
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
    wanRecipe: emptyWanRecipe(),
  });
}

function setSequenceCameraMode(mode: string) {
  const clip = activeSequenceClip.value;
  if (!clip) return;
  if (mode === "custom") {
    if (cameraMotionMode(clip.cameraControl) !== "custom")
      clip.cameraControl = "";
  } else {
    clip.cameraControl = mode || null;
  }
}
</script>

<template>
  <component
    ref="host"
    :is="mobile ? SheetPanel : 'section'"
    :class="inline ? 'adv adv--inline' : undefined"
    data-test="advanced-root"
    :open="mobile ? open : undefined"
    :variant="mobile ? 'full' : undefined"
    :title="mobile ? 'Advanced' : undefined"
    @close="emit('close')"
    @keydown="onKeydown"
  >
    <template v-if="mobile" #header>
      <div class="adv__head" data-test="advanced-header">
        <div class="adv__title">Advanced</div>
        <div class="adv__subtitle">
          Fine controls, tucked away until you need them
        </div>
      </div>
      <BadgePill v-if="advCount > 0" data-test="advanced-active"
        >{{ advCount }} active</BadgePill
      >
    </template>

    <!-- Same head on both surfaces. SheetPanel's full variant has no header
         slot, so on phones this row leads the sheet body — a Reset parked
         under every section would need a full scroll to reach. -->
    <div class="adv__inline-head">
      <span v-if="inline" class="adv__kicker">Advanced</span>
      <BadgePill v-if="advCount > 0" data-test="advanced-active"
        >{{ advCount }} on</BadgePill
      >
      <span class="adv__inline-spacer" />
      <button
        type="button"
        class="adv__inline-reset"
        data-test="advanced-reset"
        aria-label="Reset advanced settings"
        @click="resetAdvanced"
      >
        ↺ Reset
      </button>
    </div>

    <div class="adv__sections">
      <template v-if="sequenceMode">
        <AccordionSection
          icon="image"
          title="Opening sequence image"
          :summary="
            draft.openingImage?.filename ?? 'Optional original starting frame'
          "
          :open="true"
          :header-interactive="false"
          data-test="sequence-section-opening-image"
        >
          <button
            type="button"
            class="adv__dropzone"
            data-test="sequence-opening-image-pick"
            @click="emit('open-picker')"
          >
            {{
              draft.openingImage
                ? `Replace ${draft.openingImage.filename}`
                : "Drop an image or browse for the original starting frame"
            }}
          </button>
          <button
            v-if="draft.openingImage"
            type="button"
            class="adv__remove"
            data-test="sequence-opening-image-clear"
            @click="draft.openingImage = null"
          >
            Remove opening image
          </button>
          <template v-if="draft.openingImage">
            <SliderRow
              label="Source strength"
              :model-value="modelValue.strength"
              :min="0"
              :max="1"
              :step="0.01"
              :value-label="modelValue.strength.toFixed(2)"
              data-test="sequence-source-strength"
              @update:model-value="patch({ strength: $event })"
            />
            <div class="adv__field">
              <label class="adv__label" for="sequence-source-fit"
                >Fit to video frame</label
              >
              <select
                id="sequence-source-fit"
                class="adv__input"
                data-test="sequence-source-fit"
                :value="sequenceFitMode"
                @change="
                  setSequenceFit(
                    ($event.target as HTMLSelectElement).value as SourceFitMode,
                  )
                "
              >
                <option
                  v-for="option in sequenceFitOptions"
                  :key="option.value"
                  :value="option.value"
                >
                  {{ option.label }}
                </option>
              </select>
              <p class="adv__hint">
                Applied to the opening image before the first clip renders.
              </p>
            </div>
          </template>
        </AccordionSection>

        <AccordionSection
          v-if="activeSequenceClip && family === 'ltx2'"
          icon="video"
          :title="`Clip ${activeSequenceIndex + 1} camera motion`"
          :summary="
            sequenceCameraControls.find(
              (control) => control.id === activeSequenceClip?.cameraControl,
            )?.label ??
            activeSequenceClip.cameraControl ??
            'None'
          "
          :open="true"
          :header-interactive="false"
          data-test="sequence-section-camera"
        >
          <select
            class="adv__input"
            data-test="sequence-camera-motion"
            aria-label="Active clip camera motion"
            :value="cameraMotionMode(activeSequenceClip.cameraControl)"
            @change="
              setSequenceCameraMode(($event.target as HTMLSelectElement).value)
            "
          >
            <option value="">None</option>
            <option
              v-for="control in sequenceCameraControls"
              :key="control.id"
              :value="control.id"
            >
              {{ control.label
              }}{{ control.installed ? "" : " · downloads on first use" }}
            </option>
            <option value="custom">Custom LoRA path…</option>
          </select>
          <input
            v-if="
              cameraMotionMode(activeSequenceClip.cameraControl) === 'custom'
            "
            v-model="activeSequenceClip.cameraControl"
            class="adv__input adv__camera-path"
            data-test="sequence-camera-motion-custom"
            aria-label="Active clip camera motion LoRA path"
            placeholder="/path/to/lora.safetensors"
          />
          <p
            v-if="
              sequenceCameraControlsLoaded &&
              sequenceCameraControls.length === 0
            "
            class="adv__hint"
            data-test="sequence-camera-motion-19b-hint"
          >
            {{
              sequenceCameraUnsupportedReason ??
              "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
            }}
          </p>
        </AccordionSection>

        <AccordionSection
          v-if="activeSequenceClip"
          icon="negative"
          :title="`Clip ${activeSequenceIndex + 1} negative prompt`"
          summary="What to steer away from in this clip"
          :open="true"
          :header-interactive="false"
          data-test="sequence-section-negative"
        >
          <textarea
            v-model="activeSequenceClip.negativePrompt"
            :disabled="!caps.supportsNegativePrompt"
            class="adv__textarea"
            data-test="sequence-negative-input"
            placeholder="blurry, low quality, deformed…"
          />
          <p
            v-if="!caps.supportsNegativePrompt"
            class="adv__hint"
            data-test="sequence-negative-unavailable-hint"
          >
            Saved for reuse, but this distilled recipe does not use
            negative-prompt guidance.
          </p>
          <div v-if="caps.supportsNegativePrompt" class="adv__chips">
            <Chip
              v-for="word in NEG_CHIPS"
              :key="word"
              @click="
                activeSequenceClip.negativePrompt =
                  activeSequenceClip.negativePrompt.trim()
                    ? `${activeSequenceClip.negativePrompt.trim()}, ${word}`
                    : word
              "
              >+ {{ word }}</Chip
            >
          </div>
        </AccordionSection>

        <AccordionSection
          v-if="sequenceSupportsAudio"
          icon="video"
          title="Sequence audio"
          summary="Generate and mux audio for this timeline"
          :open="true"
          :header-interactive="false"
          data-test="sequence-section-audio"
        >
          <div class="adv__row">
            <span class="adv__label">Generate audio</span>
            <SwitchToggle
              :model-value="draft.enableAudio"
              label="Generate sequence audio"
              @update:model-value="draft.enableAudio = $event"
            />
          </div>
        </AccordionSection>
      </template>
      <template v-else>
        <AccordionSection
          v-if="h3Task"
          icon="video"
          :title="h3Task === 'fl2va' ? 'Frame endpoints' : 'Ordered references'"
          :summary="
            h3Task === 'fl2va'
              ? 'First, last, both, or text only'
              : `${h3Authoring.references.length} in semantic order`
          "
          :open="true"
          :header-interactive="false"
          data-test="section-h3-authoring"
        >
          <MinimaxH3AuthoringPanel
            :model-value="h3Authoring"
            :task="h3Task"
            @update:model-value="setH3Authoring"
          />
        </AccordionSection>

        <AccordionSection
          v-if="showScheduler"
          icon="scheduler"
          title="Scheduler & sampling"
          :summary="schedulerSummary"
          :open="true"
          :header-interactive="false"
          data-test="section-scheduler"
        >
          <div v-if="caps.supportsScheduler" class="adv__field">
            <label class="adv__label">Scheduler</label>
            <select
              class="adv__select"
              data-test="scheduler-select"
              :value="schedulerName"
              @change="setScheduler(($event.target as HTMLSelectElement).value)"
            >
              <option v-for="s in schedulerChoices" :key="s" :value="s">
                {{ schedulerLabel(s) }}
              </option>
            </select>
          </div>
          <div v-if="caps.supportsCfgPlus" class="adv__row">
            <span class="adv__label">CFG++</span>
            <SwitchToggle
              :model-value="modelValue.cfgPlus"
              label="CFG++"
              data-test="cfg-plus"
              @update:model-value="patch({ cfgPlus: $event })"
            />
          </div>
        </AccordionSection>

        <AccordionSection
          v-if="caps.wanRecipe.supported"
          icon="scheduler"
          title="Sampler recipe"
          :summary="wanRecipeSummary"
          :open="true"
          :header-interactive="false"
          data-test="section-wan-recipe"
        >
          <div class="adv__field">
            <label class="adv__label">Sample solver</label>
            <select
              class="adv__select"
              data-test="wan-solver-select"
              :value="schedulerName"
              @change="setScheduler(($event.target as HTMLSelectElement).value)"
            >
              <option v-for="s in schedulerChoices" :key="s" :value="s">
                {{ schedulerLabel(s) }}
              </option>
            </select>
          </div>
          <div class="adv__field">
            <label class="adv__label">Flow shift</label>
            <input
              class="adv__input"
              type="number"
              inputmode="decimal"
              step="0.5"
              min="0"
              placeholder="Model default"
              data-test="wan-sample-shift"
              :value="wanRecipe.sampleShift ?? ''"
              @input="
                setWanRecipe({
                  sampleShift: numberOrNull(
                    ($event.target as HTMLInputElement).value,
                  ),
                })
              "
            />
            <p class="adv__hint">
              Higher shift spends more steps on structure. Empty keeps this
              model's own value.
            </p>
          </div>
          <div
            v-if="caps.wanRecipe.supportsDistillStrength"
            class="adv__field adv__pair"
          >
            <label class="adv__label">
              High-noise distill
              <input
                class="adv__input"
                type="number"
                inputmode="decimal"
                step="0.1"
                min="0"
                :max="MAX_WAN_DISTILL_STRENGTH"
                placeholder="1.0"
                data-test="wan-distill-high"
                :value="wanRecipe.distillStrengthHigh ?? ''"
                @input="
                  setWanRecipe({
                    distillStrengthHigh: numberOrNull(
                      ($event.target as HTMLInputElement).value,
                    ),
                  })
                "
              />
            </label>
            <label class="adv__label">
              Low-noise distill
              <input
                class="adv__input"
                type="number"
                inputmode="decimal"
                step="0.1"
                min="0"
                :max="MAX_WAN_DISTILL_STRENGTH"
                placeholder="1.0"
                data-test="wan-distill-low"
                :value="wanRecipe.distillStrengthLow ?? ''"
                @input="
                  setWanRecipe({
                    distillStrengthLow: numberOrNull(
                      ($event.target as HTMLInputElement).value,
                    ),
                  })
                "
              />
            </label>
          </div>
          <p
            v-if="wanRecipeMessage"
            class="adv__error"
            role="alert"
            data-test="wan-recipe-error"
          >
            {{ wanRecipeMessage }}
          </p>
          <button
            v-if="wanRecipeActive"
            type="button"
            class="adv__reset"
            data-test="wan-recipe-reset"
            @click="resetWanRecipe"
          >
            Reset recipe
          </button>
        </AccordionSection>

        <AccordionSection
          v-if="caps.supportsNegativePrompt || modelValue.negativePrompt.trim()"
          icon="negative"
          title="Negative prompt"
          summary="What to steer away from"
          :open="true"
          :header-interactive="false"
          data-test="section-negative"
        >
          <textarea
            class="adv__textarea"
            data-test="negative-input"
            placeholder="blurry, low quality, deformed…"
            :value="modelValue.negativePrompt"
            :disabled="!caps.supportsNegativePrompt"
            @input="
              patch({
                negativePrompt: ($event.target as HTMLTextAreaElement).value,
              })
            "
          />
          <p
            v-if="!caps.supportsNegativePrompt"
            class="adv__hint"
            data-test="negative-unavailable-hint"
          >
            Saved for reuse, but this distilled recipe fixes CFG and does not
            use negative-prompt guidance. Choose a Dev checkpoint with Auto or a
            guided pipeline to enable it.
          </p>
          <div class="adv__chips">
            <Chip
              v-for="word in NEG_CHIPS"
              :key="word"
              :data-test="`neg-chip-${word.replace(/\s+/g, '-')}`"
              @click="addNegative(word)"
              >+ {{ word }}</Chip
            >
          </div>
        </AccordionSection>

        <!-- Hidden outright for a checkpoint the server says takes no source
             image: admission rejects one, so offering the well can only
             produce a refused request. -->
        <AccordionSection
          v-if="!h3Family && caps.supportsSourceImage"
          icon="image"
          :title="
            caps.sourceImageMode !== 'single'
              ? flux2Dev
                ? 'Reference images'
                : 'Edit images'
              : 'Source image'
          "
          :summary="
            hasSource
              ? caps.sourceImageMode !== 'single'
                ? `${modelValue.imageAttachments.length} attached`
                : `1 image · denoise ${modelValue.strength.toFixed(2)}`
              : caps.sourceImageMode !== 'single'
                ? flux2Dev
                  ? 'Optional · up to 4 ordered references'
                  : 'Target and reference images'
                : 'Image-to-image & inpainting'
          "
          :open="true"
          :header-interactive="false"
          data-test="section-source"
        >
          <!-- Requiredness is never inferred: only an explicit advertised
               `required` marks the well, because a family heuristic can say a
               checkpoint reads a source image, not that it cannot render
               without one. -->
          <p
            v-if="caps.requiresSourceImage"
            class="adv__required"
            data-test="source-required-badge"
          >
            Required — this checkpoint renders from an image.
          </p>
          <button
            v-if="!hasSource"
            type="button"
            class="adv__dropzone"
            data-test="source-attach"
            :aria-required="caps.requiresSourceImage || undefined"
            @click="emit('open-picker')"
          >
            {{
              caps.sourceImageMode !== "single"
                ? flux2Dev
                  ? "Attach references or "
                  : "Attach images or "
                : "Drop an image or "
            }}<span class="adv__accent">browse</span>
          </button>
          <div v-else>
            <div class="adv__source-row">
              <span class="adv__source-name">{{
                modelValue.imageAttachments[0]?.filename
              }}</span>
              <button
                type="button"
                class="adv__remove"
                data-test="source-remove"
                @click="emit('clear-source')"
              >
                Remove
              </button>
            </div>
            <!-- Wan pins the first frame exactly and never reads strength. -->
            <SliderRow
              v-if="caps.sourceImageMode === 'single' && caps.supportsStrength"
              label="Denoise strength"
              :model-value="modelValue.strength"
              :min="0"
              :max="1"
              :step="0.01"
              :value-label="modelValue.strength.toFixed(2)"
              @update:model-value="patch({ strength: $event })"
            />
            <div v-if="caps.sourceImageMode === 'single'" class="adv__field">
              <label class="adv__label">Fit to canvas</label>
              <SegmentedControl
                :model-value="fitMode"
                :options="fitOptions"
                label="Fit to canvas"
                @update:model-value="setFit"
              />
            </div>
            <button
              v-if="caps.supportsMask"
              type="button"
              class="adv__mask"
              data-test="source-mask"
              @click="emit('open-mask')"
            >
              {{
                modelValue.maskImage
                  ? "Mask applied · edit"
                  : "Edit inpaint mask"
              }}
            </button>
          </div>

          <!-- Why the attached conditioning would be refused. One message at a
               time, in the same order admission checks them. -->
          <p
            v-if="sourceConditioningError"
            class="adv__error"
            role="alert"
            data-test="source-conditioning-error"
          >
            {{ sourceConditioningError }}
          </p>

          <!-- End frame (wan first/last-frame conditioning, #779). Offered
               only when the server advertised a source-image contract for this
               checkpoint: a host old enough to omit the field rejects wan
               keyframes outright. -->
          <div
            v-if="caps.supportsEndFrame"
            class="adv__end-frame"
            data-test="end-frame-well"
          >
            <div class="adv__subhead">End frame <span>· Optional</span></div>
            <button
              v-if="!modelValue.endFrame"
              type="button"
              class="adv__dropzone"
              data-test="end-frame-attach"
              @click="emit('open-end-frame-picker')"
            >
              Drop the closing image or
              <span class="adv__accent">browse</span>
            </button>
            <div v-else class="adv__source-row">
              <span class="adv__source-name">{{
                modelValue.endFrame.filename
              }}</span>
              <button
                type="button"
                class="adv__remove"
                data-test="end-frame-remove"
                @click="emit('clear-end-frame')"
              >
                Remove
              </button>
            </div>
            <p class="adv__hint">
              Renders a first/last-frame clip: the source image opens it, this
              one closes it.
            </p>
          </div>

          <div
            v-if="showControlNet"
            class="adv__controlnet"
            data-test="controlnet-block"
          >
            <div class="adv__subhead">ControlNet</div>
            <label
              v-if="!hasControl"
              class="adv__filezone"
              data-test="control-attach"
            >
              <span
                >Attach a control image or
                <span class="adv__accent">browse</span></span
              >
              <input
                type="file"
                accept="image/png,image/jpeg"
                class="adv__file-input"
                @change="onControlImage"
              />
            </label>
            <template v-else>
              <div class="adv__source-row">
                <span class="adv__source-name">{{
                  modelValue.controlImage?.filename
                }}</span>
                <button
                  type="button"
                  class="adv__remove"
                  data-test="control-remove"
                  @click="clearControl"
                >
                  Remove
                </button>
              </div>
              <div class="adv__field">
                <label class="adv__label">Control model</label>
                <input
                  class="adv__input"
                  data-test="control-model"
                  list="installed-controlnet-models"
                  placeholder="e.g. control_v11p_sd15_canny"
                  :value="modelValue.controlModel"
                  @input="
                    patch({
                      controlModel: ($event.target as HTMLInputElement).value,
                    })
                  "
                />
                <datalist id="installed-controlnet-models">
                  <option
                    v-for="model in controlModels"
                    :key="model.name"
                    :value="model.name"
                  />
                </datalist>
              </div>
              <SliderRow
                label="Control scale"
                data-test="control-scale"
                :model-value="modelValue.controlScale"
                :min="0"
                :max="2"
                :step="0.05"
                :value-label="modelValue.controlScale.toFixed(2)"
                @update:model-value="patch({ controlScale: $event })"
              />
            </template>
          </div>
        </AccordionSection>

        <AccordionSection
          v-if="caps.supportsLora"
          icon="layers"
          title="LoRA stack"
          :summary="`${modelValue.loras.length} active · style adapters`"
          :open="true"
          :header-interactive="false"
          data-test="section-lora"
        >
          <LoraPicker
            :family="family"
            :model-value="modelValue.loras"
            @update:model-value="setLoras"
            @append-prompt="emit('append-prompt', $event)"
          />
        </AccordionSection>

        <UpscaleSection
          v-if="showUpscale"
          :model-value="modelValue.upscaleModel"
          @update:model-value="patch({ upscaleModel: $event })"
        />

        <AccordionSection
          icon="output"
          title="Output & seed"
          summary="Format and reproducibility"
          :open="true"
          :header-interactive="false"
          data-test="section-output"
        >
          <div class="adv__field">
            <label class="adv__label">File format</label>
            <SegmentedControl
              :model-value="modelValue.outputFormat"
              :options="
                formats.map((f) => ({ value: f, label: f.toUpperCase() }))
              "
              label="File format"
              @update:model-value="
                patch({ outputFormat: $event as OutputFormat })
              "
            />
          </div>
          <div class="adv__field">
            <label class="adv__label">Exact size</label>
            <div class="adv__size">
              <input
                class="adv__input"
                type="number"
                min="64"
                :step="resolutionAlignment"
                data-test="exact-width"
                aria-label="Width in pixels"
                :value="modelValue.width"
                @change="setWidth(($event.target as HTMLInputElement).value)"
              />
              <button
                type="button"
                class="adv__swap"
                data-test="exact-swap"
                aria-label="Swap width and height"
                title="Swap width and height"
                @click="swapDims"
              >
                <Icon name="swap" :size="15" />
              </button>
              <input
                class="adv__input"
                type="number"
                min="64"
                :step="resolutionAlignment"
                data-test="exact-height"
                aria-label="Height in pixels"
                :value="modelValue.height"
                @change="setHeight(($event.target as HTMLInputElement).value)"
              />
            </div>
            <p class="adv__hint">
              snaps to the nearest {{ resolutionAlignment }}px.
            </p>
          </div>
          <div class="adv__field">
            <label class="adv__label">Seed</label>
            <SegmentedControl
              :model-value="modelValue.seedMode"
              :options="seedModes"
              label="Seed mode"
              @update:model-value="patch({ seedMode: $event })"
            />
          </div>
          <input
            v-if="modelValue.seedMode !== 'random'"
            class="adv__input"
            data-test="output-seed"
            type="number"
            min="0"
            placeholder="Seed"
            :value="modelValue.seed ?? ''"
            @input="
              patch({
                seed: Number(($event.target as HTMLInputElement).value) || null,
              })
            "
          />
        </AccordionSection>

        <AccordionSection
          v-if="caps.supportsVideo"
          icon="video"
          title="Video"
          :summary="`${modelValue.frames ?? 25} frames · ${modelValue.fps ?? 24} fps`"
          :open="true"
          :header-interactive="false"
          data-test="section-video"
        >
          <div class="adv__field">
            <VideoDurationSlider
              :frames="modelValue.frames ?? selectedModel?.default_frames ?? 25"
              :fps="modelValue.fps ?? selectedModel?.default_fps ?? 24"
              :model="selectedModel"
              label="Duration"
              @update:frames="patch({ frames: $event })"
            />
          </div>
          <div class="adv__field">
            <label class="adv__label">Frames ({{ frameGridLabel }})</label>
            <input
              class="adv__input"
              data-test="video-frames"
              type="number"
              :min="frameMinimum"
              :step="frameStep"
              :value="modelValue.frames ?? 25"
              @change="
                patch({
                  frames: clampFrames(
                    Number(($event.target as HTMLInputElement).value),
                  ),
                })
              "
            />
            <p class="adv__hint">
              Frames must follow {{ frameGridLabel
              }}<template v-if="h3Family">, from 124 through 362</template>.
            </p>
          </div>
          <div class="adv__field">
            <label class="adv__label">Frames per second</label>
            <input
              class="adv__input"
              data-test="video-fps"
              type="number"
              min="1"
              :disabled="fixedFps !== null"
              :value="modelValue.fps ?? 24"
              @change="
                patch({
                  fps: Number(($event.target as HTMLInputElement).value) || 24,
                })
              "
            />
          </div>
          <div v-if="!h3Family" class="adv__row">
            <span class="adv__label">GIF preview</span>
            <SwitchToggle
              :model-value="modelValue.gifPreview"
              label="Generate GIF preview"
              data-test="video-gif-preview"
              @update:model-value="patch({ gifPreview: $event })"
            />
          </div>
          <!-- Continuation is per model, not per family (#783): wan reaches
               it without the LTX-2 suite behind `showLtx2`. -->
          <ExtendVideoControls
            v-if="canExtend"
            :model-value="modelValue"
            :family="family"
            :default-overlap-frames="extendDefaultOverlapFrames"
            @update:model-value="emit('update:modelValue', $event)"
          />
          <Ltx2VideoControls
            v-if="showLtx2"
            :model-value="modelValue"
            :audio-output-supported="audioOutputSupported"
            @update:model-value="emit('update:modelValue', $event)"
          />
        </AccordionSection>

        <AccordionSection
          v-if="showPlacement"
          icon="machines"
          title="GPU placement"
          summary="Pin this job to a device"
          :open="true"
          :header-interactive="false"
          data-test="section-placement"
        >
          <PlacementPanel
            :model-value="modelValue.placement"
            :family="family"
            :model="modelValue.model"
            :gpus="placementGpus"
            @update:model-value="setPlacement"
          />
        </AccordionSection>
      </template>
    </div>

    <div v-if="mobile" class="adv__footer">
      <div class="adv__spacer" />
      <button
        type="button"
        class="adv__done"
        data-test="advanced-done"
        @click="emit('close')"
      >
        Done
      </button>
    </div>
  </component>
</template>

<style scoped>
/* Inline (tablet+ web) container — a card in the controls region. */
.adv--inline {
  display: block;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}
.adv__inline-head {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 14px;
}
.adv__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.adv__inline-spacer {
  flex: 1;
}
.adv__inline-reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.adv__inline-reset:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.adv__head {
  flex: 1;
}
.adv__title {
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 700;
}
.adv__subtitle {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  margin-top: 1px;
}
.adv__sections {
  display: flex;
  flex-direction: column;
  gap: 11px;
}
.adv__field {
  margin-bottom: 14px;
}
.adv__camera-path {
  margin-top: 9px;
}
.adv__field:last-child {
  margin-bottom: 0;
}
.adv__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.adv__label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.adv__select,
.adv__input {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
  font-family: var(--f-mono);
}
.adv__textarea {
  width: 100%;
  box-sizing: border-box;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13.5px;
  line-height: 1.45;
  min-height: 64px;
  resize: none;
  outline: none;
  padding: 11px 13px;
}
.adv__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
}
.adv__size {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 8px;
}
.adv__swap {
  height: 40px;
  width: 40px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.adv__swap:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}
.adv__dropzone {
  width: 100%;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-card);
  padding: 26px;
  font-size: 13px;
  cursor: pointer;
}
.adv__accent {
  color: var(--safelight);
  font-weight: 600;
}
.adv__source-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.adv__source-name {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.adv__remove {
  border: 0;
  background: transparent;
  color: var(--stop);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.adv__mask {
  margin-top: 12px;
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control-lg);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.adv__hint {
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
}
.adv__pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.adv__label .adv__input {
  margin-top: 6px;
}
/* Textual, never colour alone: the requirement has to survive a monochrome
 * or high-contrast rendering. */
.adv__required {
  font-size: 11px;
  font-weight: 600;
  color: var(--safelight);
  margin: 0 0 10px;
}
/* One error style serves the recipe controls and the end-frame well. */
.adv__error {
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--stop);
  margin: 10px 0 0;
}
.adv__reset {
  margin-top: 10px;
  background: transparent;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--ink-2);
  cursor: pointer;
  font-size: 11px;
  height: 28px;
  padding: 0 10px;
}
.adv__end-frame,
.adv__controlnet {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--edge);
}
.adv__subhead {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-2);
  margin-bottom: 10px;
}
.adv__filezone {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  box-sizing: border-box;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-card);
  padding: 20px;
  font-size: 13px;
  text-align: center;
  cursor: pointer;
}
.adv__file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}
.adv__footer {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--edge);
}
.adv__spacer {
  flex: 1;
}
.adv__done {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 11px 26px;
  border-radius: var(--radius-control-lg);
  font-size: 13.5px;
  font-weight: 700;
  cursor: pointer;
}
</style>
