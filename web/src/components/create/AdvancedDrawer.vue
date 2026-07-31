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
import Ltx2VideoControls from "./advanced/Ltx2VideoControls.vue";
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
import { generationCapabilitiesForFamily } from "../../lib/generateCapabilities";
import { outputFormatsForFamily } from "../../composables/useGenerateForm";
import { blobToBase64 } from "../../lib/base64";
import { useOverlayFocus } from "../../composables/useOverlayFocus";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { OutputMode } from "@studio/lib/sequence";
import {
  cameraMotionMode,
  isCameraMotionPreset,
} from "@studio/lib/cameraMotion";

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
    output: "single",
    sequenceSupportsAudio: false,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  close: [];
  "open-picker": [];
  "clear-source": [];
  "open-mask": [];
  "append-prompt": [phrase: string];
}>();
const host = ref<HTMLElement | { $el?: unknown } | null>(null);
const draft = useSequenceDraftStore();
const sequenceMode = computed(() => props.output === "sequence");
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
let cameraControlsEpoch = 0;
watch(
  [sequenceMode, () => props.family, () => props.modelValue.model],
  async () => {
    const epoch = ++cameraControlsEpoch;
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
        `/api/capabilities/ltx2-camera-controls?model=${encodeURIComponent(props.modelValue.model)}`,
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const options = (await response.json()) as Ltx2CameraControlInfo[];
      if (epoch !== cameraControlsEpoch) return;
      sequenceCameraControls.value = options;
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

const caps = computed(() => generationCapabilitiesForFamily(props.family));
const formats = computed(() => outputFormatsForFamily(props.family));

const showScheduler = computed(
  () => caps.value.supportsScheduler || caps.value.supportsCfgPlus,
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
  caps.value.supportsScheduler ? schedulerName.value : "CFG++",
);

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

// ── Scheduler & sampling ──────────────────────────────────────────────
function setScheduler(value: string) {
  patch({ scheduler: value as Scheduler });
}

// ── Negative prompt ───────────────────────────────────────────────────
function addNegative(word: string) {
  const cur = props.modelValue.negativePrompt.trim();
  patch({ negativePrompt: cur ? `${cur}, ${word}` : word });
}

// ── Source image ──────────────────────────────────────────────────────
const hasSource = computed(() => props.modelValue.imageAttachments.length > 0);
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
const showLtx2 = computed(() => caps.value.supportsAudio);

// ── Output & seed: exact size (snap to the 16px grid, like desktop) ───
function snapDim(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 16;
  return Math.max(16, Math.round(v / 16) * 16);
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
  patch({ loras });
}
function setPlacement(placement: DevicePlacement | null) {
  patch({ placement });
}

// frames must be 8n+1 (9, 17, 25, 33, …) — ported from GenerateParamsPanel.
function clampFrames(n: number): number {
  if (!Number.isFinite(n)) return 25;
  return Math.max(9, Math.round((n - 1) / 8) * 8 + 1);
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
    return;
  }
  patch({
    negativePrompt: "",
    scheduler: null,
    cfgPlus: false,
    imageAttachments: [],
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
            Built-in camera motions are available for LTX-2 19B only. This model
            accepts a custom LoRA path.
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
            class="adv__textarea"
            data-test="sequence-negative-input"
            placeholder="blurry, low quality, deformed…"
          />
          <div class="adv__chips">
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
                {{ s }}
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
          v-if="caps.supportsNegativePrompt"
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
            @input="
              patch({
                negativePrompt: ($event.target as HTMLTextAreaElement).value,
              })
            "
          />
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

        <AccordionSection
          icon="image"
          :title="
            caps.sourceImageMode === 'qwen-edit'
              ? 'Edit images'
              : 'Source image'
          "
          :summary="
            hasSource
              ? caps.sourceImageMode === 'qwen-edit'
                ? `${modelValue.imageAttachments.length} attached`
                : `1 image · denoise ${modelValue.strength.toFixed(2)}`
              : caps.sourceImageMode === 'qwen-edit'
                ? 'Target and reference images'
                : 'Image-to-image & inpainting'
          "
          :open="true"
          :header-interactive="false"
          data-test="section-source"
        >
          <button
            v-if="!hasSource"
            type="button"
            class="adv__dropzone"
            data-test="source-attach"
            @click="emit('open-picker')"
          >
            {{
              caps.sourceImageMode === "qwen-edit"
                ? "Attach images or "
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
            <SliderRow
              v-if="caps.sourceImageMode === 'single'"
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
                min="16"
                step="16"
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
                min="16"
                step="16"
                data-test="exact-height"
                aria-label="Height in pixels"
                :value="modelValue.height"
                @change="setHeight(($event.target as HTMLInputElement).value)"
              />
            </div>
            <p class="adv__hint">snaps to the nearest 16px.</p>
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
            <label class="adv__label">Frames (8n+1)</label>
            <input
              class="adv__input"
              data-test="video-frames"
              type="number"
              :value="modelValue.frames ?? 25"
              @change="
                patch({
                  frames: clampFrames(
                    Number(($event.target as HTMLInputElement).value),
                  ),
                })
              "
            />
            <p class="adv__hint">Frames must be 8n+1 — try 97.</p>
          </div>
          <div class="adv__field">
            <label class="adv__label">Frames per second</label>
            <input
              class="adv__input"
              data-test="video-fps"
              type="number"
              min="1"
              :value="modelValue.fps ?? 24"
              @change="
                patch({
                  fps: Number(($event.target as HTMLInputElement).value) || 24,
                })
              "
            />
          </div>
          <div class="adv__row">
            <span class="adv__label">GIF preview</span>
            <SwitchToggle
              :model-value="modelValue.gifPreview"
              label="Generate GIF preview"
              data-test="video-gif-preview"
              @update:model-value="patch({ gifPreview: $event })"
            />
          </div>
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
