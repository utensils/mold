<script setup lang="ts">
import { computed, ref, watch } from "vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import SegmentedControl, { type SegmentOption } from "@ui/components/SegmentedControl.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import Chip from "@ui/components/Chip.vue";
import {
  buildRequest,
  resetFormToModelDefaults,
  seedMode,
  type GenerateForm,
  type PickedImage,
} from "../../lib/generateForm";
import type {
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  OutputFormat,
} from "../../lib/api/types";
import {
  generationCapabilitiesForFamily,
  MAX_LORA_STACK,
  defaultOutputFormat,
  outputFormatsForFamily,
} from "../../lib/capabilities";
import { schedulerLabel } from "@studio/lib/generationCapabilities";
import {
  MAX_WAN_DISTILL_STRENGTH,
  emptyWanRecipe,
  wanRecipeCount,
  type WanRecipeState,
} from "@studio/lib/wanRecipe";
import {
  clampVideoFrames,
  fixedVideoFps,
  minVideoFrames,
  videoFrameGridLabel,
  videoFrameGridError,
  videoFramesError,
  videoFrameStep,
  snapVideoFrames,
} from "@studio/lib/videoDuration";
import {
  autoChainFieldList,
  decideGenerateRequestRouting,
  type ChainRoutingDecision,
} from "../../lib/chainRouting";
import { fileToBase64 } from "../../lib/image";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  wanRecipeValidationError,
} from "../../lib/generateValidation";
import { advancedActiveCount } from "../../lib/advancedCount";
import {
  cameraMotionLoraPath,
  cameraMotionLoraSlotAvailable,
  cameraMotionLoraLabel,
  cameraMotionMode,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import {
  canOfferExtend,
  extendNewFrames,
  extendOverlapOptions,
  extendValidationError,
  serverExtendOverlapDefault,
} from "@studio/lib/extend";
import {
  emptyGuidanceOverrides,
  guidanceOverrideCount,
  skipStepError,
  stgBlocksError,
  MAX_GUIDANCE_SCALE,
  MAX_GUIDANCE_SKIP_STEP,
  type Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import {
  LIP_DUB_TIMING_HINT,
  isControlAdapterPipeline,
  pipelineForControlId,
} from "@studio/lib/ltx2Control";
import { AUDIO_ONLY_PIPELINE, isAudioOnlyPipeline } from "@studio/lib/ltx2Pipeline";
import SourceImageWell from "../generate/SourceImageWell.vue";
import LoraStack from "../generate/LoraStack.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import MinimaxH3AuthoringPanel from "@studio/components/MinimaxH3AuthoringPanel.vue";
import {
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  type MinimaxH3AuthoringState,
} from "@studio/lib/minimaxH3Authoring";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** The picked model, used by Reset to restore its defaults. */
    selectedModel?: ModelEntry | null;
    upscalers?: ModelEntry[];
    controlAdapters?: Ltx2ControlAdapterInfo[];
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    selectedModel: null,
    upscalers: () => [],
    controlAdapters: () => [],
    cameraControls: () => [],
    cameraControlsLoaded: false,
    cameraUnsupportedReason: null,
  },
);

const emit = defineEmits<{ "append-word": [word: string] }>();

const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.selectedModel?.guidance_capabilities,
  ),
);
const audioOutputSupported = computed(() => props.selectedModel?.supports_audio !== false);
const formats = computed(() => outputFormatsForFamily(props.form.family));
const advancedCount = computed(() => advancedActiveCount(props.form));
const h3Task = computed(() =>
  props.selectedModel ? minimaxH3TaskForModel(props.form.model) : null,
);
const h3Family = computed(() => isMinimaxH3Identity(props.form.family, props.form.model));
const h3Authoring = computed(() => props.form.h3Authoring ?? emptyMinimaxH3AuthoringState());
function setH3Authoring(value: MinimaxH3AuthoringState): void {
  props.form.h3Authoring = value;
}

// ── Scheduler & sampling ─────────────────────────────────────────────────────
const schedulerSummary = computed(() => schedulerLabel(props.form.scheduler));

// ── Wan sampler recipe ───────────────────────────────────────────────────────
// Wan's solver belongs beside the flow shift it is tuned with, so the generic
// scheduler section stands down for the family rather than rendering a second
// picker onto the same field.
const showScheduler = computed(
  () =>
    (caps.value.supportsScheduler && !caps.value.wanRecipe.supported) || caps.value.supportsCfgPlus,
);
const wanRecipe = computed<WanRecipeState>(() => props.form.wanRecipe);
const wanRecipeActive = computed(() => wanRecipeCount(wanRecipe.value));
const wanRecipeMessage = computed(() => wanRecipeValidationError(props.form));
function setWanRecipe(next: Partial<WanRecipeState>) {
  props.form.wanRecipe = { ...wanRecipe.value, ...next };
}
function resetWanRecipe() {
  props.form.wanRecipe = emptyWanRecipe();
}

// ── Negative prompt quick-adds ───────────────────────────────────────────────
const NEGATIVE_QUICK_ADDS = [
  "blurry",
  "extra fingers",
  "watermark",
  "low quality",
  "oversaturated",
];
function addNegative(word: string) {
  const current = props.form.negativePrompt.trim();
  props.form.negativePrompt = current ? `${current}, ${word}` : word;
}

// ── Output & seed ────────────────────────────────────────────────────────────
const formatOptions = computed<SegmentOption<OutputFormat>[]>(() =>
  formats.value.map((f) => ({ value: f, label: f })),
);
const snap16 = (v: number) => Math.max(64, Math.round(v / 16) * 16);
function snapWidth() {
  props.form.width = snap16(props.form.width);
}
function snapHeight() {
  props.form.height = snap16(props.form.height);
}
function swapSize() {
  [props.form.width, props.form.height] = [props.form.height, props.form.width];
}
const seedFixed = computed(() => seedMode(props.form.seed) === "fixed");

// ── Video ────────────────────────────────────────────────────────────────────
const videoContract = computed(() => props.selectedModel ?? { family: props.form.family });
const frameGridLabel = computed(() => videoFrameGridLabel(videoContract.value));
const frameStep = computed(() => videoFrameStep(videoContract.value));
const frameMinimum = computed(() => minVideoFrames(videoContract.value));
const fixedFps = computed(() => fixedVideoFps(videoContract.value));
const framesError = computed(() =>
  caps.value.supportsVideo
    ? fixedFps.value !== null
      ? videoFramesError(props.form.frames, videoContract.value)
      : videoFrameGridError(props.form.frames, videoContract.value)
    : null,
);
watch(
  [fixedFps, () => props.form.fps] as const,
  ([fixed, current]) => {
    if (fixed !== null && current !== fixed) props.form.fps = fixed;
  },
  { immediate: true },
);
const generationRequest = computed(() => buildRequest(props.form));
const chainDecision = computed<ChainRoutingDecision>(() =>
  caps.value.supportsVideo
    ? decideGenerateRequestRouting(generationRequest.value, props.form.family)
    : { kind: "single" },
);
const singleShotPreservationNote = computed(() => {
  const decision = chainDecision.value;
  if (decision.kind !== "single" || !decision.preservedAutoChainFields?.length) return null;
  return `Will render as one ${props.form.frames}-frame clip to preserve ${autoChainFieldList(decision.preservedAutoChainFields)}. This may use more GPU memory than automatic chaining.`;
});
const fpsError = computed(() =>
  caps.value.supportsVideo
    ? fixedFps.value !== null && props.form.fps !== fixedFps.value
      ? `FPS is fixed at ${fixedFps.value} for this model.`
      : fpsValidationError(props.form.fps)
    : null,
);
const cameraError = computed(() =>
  cameraControlValidationError(
    props.form,
    props.cameraControlsLoaded ? props.cameraControls.map((c) => c.id) : undefined,
  ),
);
const audioFormatError = computed(() => audioOutputValidationError(props.form));
const advancedVideoError = computed(() => advancedVideoValidationError(props.form));

const uiCameraMode = ref(cameraMotionMode(props.form.cameraControl));
watch(
  () => props.form.cameraControl,
  (value) => {
    if (uiCameraMode.value === "custom" && (value === "" || cameraMotionMode(value) === "custom")) {
      return;
    }
    uiCameraMode.value = cameraMotionMode(value);
  },
);
function setCameraMode(mode: string) {
  uiCameraMode.value = mode;
  if (mode === "custom") {
    if (cameraMotionMode(props.form.cameraControl) !== "custom") setCameraControl("");
  } else {
    setCameraControl(mode === "" ? null : mode);
  }
}

function setCameraControl(next: string | null) {
  if (
    cameraMotionLoraPath(next) &&
    !cameraMotionLoraSlotAvailable(props.form.loras, props.form.cameraControl, MAX_LORA_STACK)
  ) {
    return;
  }
  props.form.loras = syncCameraMotionLora(
    props.form.loras,
    props.form.cameraControl,
    next,
    (path, scale) => ({
      path,
      name: cameraMotionLoraLabel(path),
      scale,
      trainedWords: [],
    }),
    MAX_LORA_STACK,
  );
  props.form.cameraControl = next;
}

const cameraSlotAvailable = computed(() =>
  cameraMotionLoraSlotAvailable(props.form.loras, props.form.cameraControl, MAX_LORA_STACK),
);

const keyframePickerOpen = ref(false);
const pipelineOptions: Ltx2PipelineMode[] = [
  "one-stage",
  "two-stage",
  "two-stage-hq",
  "distilled",
  "ic-lora",
  "keyframe",
  "a2-vid",
  "retake",
  "lip-dub",
  AUDIO_ONLY_PIPELINE,
];
const spatialOptions: Ltx2SpatialUpscale[] = ["x1-5", "x2"];
const temporalOptions: Ltx2TemporalUpscale[] = ["x2"];
function setPipeline(v: string) {
  props.form.pipeline = (v || null) as Ltx2PipelineMode | null;
  if (!isControlAdapterPipeline(props.form.pipeline)) props.form.icLoraControl = null;
  if (props.form.pipeline !== "retake") props.form.retakeRange = null;
  // `t2a` renders no frames, so the server rejects every video container.
  // Move the format with the mode rather than letting the user discover the
  // mismatch as a 422; leaving `t2a` restores the family default.
  if (isAudioOnlyPipeline(props.form.pipeline)) {
    props.form.outputFormat = "wav";
  } else if (props.form.outputFormat === "wav") {
    props.form.outputFormat = defaultOutputFormat(props.form.family);
  }
}
const audioOnlyPipeline = computed(() => isAudioOnlyPipeline(props.form.pipeline));
function setControlAdapter(value: string) {
  props.form.icLoraControl = value || null;
  // Lip dub is a pipeline of its own; every other adapter drives `ic-lora`.
  if (value) props.form.pipeline = pipelineForControlId(value);
}
function setSpatial(v: string) {
  props.form.spatialUpscale = (v || null) as Ltx2SpatialUpscale | null;
}
function setTemporal(v: string) {
  props.form.temporalUpscale = (v || null) as Ltx2TemporalUpscale | null;
}
const guidance = computed<Ltx2GuidanceOverridesState>(() => props.form.guidanceOverrides);
const guidanceCount = computed(() => guidanceOverrideCount(guidance.value));
const stgBlocksMessage = computed(() => stgBlocksError(guidance.value.stgBlocks));
const skipStepMessage = computed(() => skipStepError(guidance.value.skipStep));
function setGuidance(next: Partial<Ltx2GuidanceOverridesState>) {
  props.form.guidanceOverrides = { ...guidance.value, ...next };
}
function numberOrNull(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}
function resetGuidance() {
  props.form.guidanceOverrides = emptyGuidanceOverrides();
}
function setRetake(edge: "start" | "end", raw: string) {
  const value = Number(raw) || 0;
  const current = props.form.retakeRange ?? { start_seconds: 0, end_seconds: 1 };
  props.form.retakeRange =
    edge === "start" ? { ...current, start_seconds: value } : { ...current, end_seconds: value };
}
function suggestKeyframeFrame(): number {
  const last = props.form.keyframes.at(-1);
  return last ? last.frame + 24 : 0;
}
function onKeyframePick(picked: PickedImage[]) {
  const first = picked[0];
  if (!first) return;
  props.form.keyframes = [...props.form.keyframes, { frame: suggestKeyframeFrame(), image: first }];
}
function updateKeyframeFrame(index: number, raw: string) {
  const frame = Math.max(0, Math.round(Number(raw) || 0));
  const next = props.form.keyframes.slice();
  const item = next[index];
  if (!item) return;
  next[index] = { ...item, frame };
  props.form.keyframes = next;
}
function removeKeyframe(index: number) {
  props.form.keyframes = props.form.keyframes.filter((_, i) => i !== index);
}
async function setSourceVideo(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  props.form.sourceVideo = { filename: file.name, base64: await fileToBase64(file) };
}
function clearSourceVideo() {
  props.form.sourceVideo = null;
}
async function setExtendVideo(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  props.form.extendVideo = { filename: file.name, base64: await fileToBase64(file) };
}
function clearExtendVideo() {
  // Drop the overlap too: a value valid for this clip may be invalid for the
  // next one, and a stale number would silently ride along.
  props.form.extendVideo = null;
  props.form.extendOverlapFrames = null;
}
const canExtend = computed(() => canOfferExtend(props.selectedModel));
const extendOverlap = computed(
  () => props.form.extendOverlapFrames ?? serverExtendOverlapDefault(props.selectedModel),
);
const extendOverlapChoices = computed(() => extendOverlapOptions(props.form.frames));
const extendError = computed(() =>
  props.form.extendVideo
    ? extendValidationError({
        overlapFrames: extendOverlap.value,
        frames: props.form.frames,
        hasSourceImage: props.form.sourceImage !== null || props.form.imageAttachments.length > 0,
        hasSourceVideo: props.form.sourceVideo !== null,
        hasKeyframes: props.form.keyframes.length > 0,
      })
    : null,
);
const extendSummary = computed(() => {
  const added = extendNewFrames(props.form.frames, extendOverlap.value);
  return added === null ? "" : `Appends ${added} new frames after the source clip.`;
});
async function setAudioFile(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  props.form.audioFile = { filename: file.name, base64: await fileToBase64(file) };
}
function clearAudioFile() {
  props.form.audioFile = null;
}
function snapFramesField() {
  props.form.frames =
    fixedFps.value !== null
      ? clampVideoFrames(props.form.frames, fixedFps.value, videoContract.value)
      : Math.max(frameMinimum.value, snapVideoFrames(props.form.frames, videoContract.value));
}

// ── Reset — restore the model's defaults, preserve prompt + prepared state ────
function reset() {
  resetFormToModelDefaults(props.form, props.selectedModel);
}
</script>

<template>
  <section class="ms-adv" data-test="inline-advanced">
    <div class="ms-adv__toolbar">
      <span class="ms-adv__summary">
        {{ advancedCount > 0 ? `${advancedCount} active` : "Fine controls" }}
      </span>
      <button type="button" class="ms-adv__reset" data-test="advanced-reset" @click="reset">
        Reset
      </button>
    </div>

    <div class="ms-adv__list">
      <!-- 1 · Scheduler & sampling -->
      <AccordionSection
        v-if="showScheduler"
        icon="scheduler"
        title="Scheduler &amp; sampling"
        :summary="schedulerSummary"
        :open="true"
        :header-interactive="false"
        data-test="section-scheduler"
      >
        <template v-if="caps.supportsScheduler">
          <label class="ms-label">Scheduler</label>
          <select v-model="form.scheduler" class="ms-select" aria-label="Scheduler">
            <option v-for="opt in caps.schedulerOptions" :key="opt" :value="opt">
              {{ schedulerLabel(opt) }}
            </option>
          </select>
        </template>
        <div v-if="caps.supportsCfgPlus" class="ms-switch-row">
          <div>
            <div class="ms-switch-row__title">CFG++</div>
            <div class="ms-switch-row__hint">Lower guidance to 1.5–2.5</div>
          </div>
          <SwitchToggle
            :model-value="form.cfgPlus"
            label="CFG++"
            @update:model-value="form.cfgPlus = $event"
          />
        </div>
      </AccordionSection>

      <!-- 1b · Wan sampler recipe -->
      <AccordionSection
        v-if="caps.wanRecipe.supported"
        icon="scheduler"
        title="Sampler recipe"
        :summary="
          wanRecipeActive
            ? `${wanRecipeActive} set · ${schedulerLabel(form.scheduler)}`
            : schedulerLabel(form.scheduler)
        "
        :open="true"
        :header-interactive="false"
        data-test="section-wan-recipe"
      >
        <div class="ms-guidance__head">
          <label class="ms-label">
            Sample solver
            <span v-if="wanRecipeActive" class="ms-guidance__count" data-test="wan-recipe-count">{{
              wanRecipeActive
            }}</span>
          </label>
          <button
            v-if="wanRecipeActive"
            type="button"
            class="ms-guidance__reset"
            data-test="wan-recipe-reset"
            @click="resetWanRecipe"
          >
            Reset
          </button>
        </div>
        <select
          v-model="form.scheduler"
          class="ms-select ms-label--mt"
          aria-label="Sample solver"
          data-test="wan-solver-select"
        >
          <option v-for="opt in caps.schedulerOptions" :key="opt" :value="opt">
            {{ schedulerLabel(opt) }}
          </option>
        </select>
        <label class="ms-guidance__label ms-label--mt">
          Flow shift
          <input
            type="number"
            inputmode="decimal"
            step="0.5"
            min="0"
            placeholder="Model default"
            class="ms-input data-mono"
            data-test="wan-sample-shift"
            :value="wanRecipe.sampleShift ?? ''"
            @input="
              setWanRecipe({ sampleShift: numberOrNull(($event.target as HTMLInputElement).value) })
            "
          />
        </label>
        <p class="ms-hint">
          Higher shift spends more steps on structure. Empty keeps this model's own value.
        </p>
        <div v-if="caps.wanRecipe.supportsDistillStrength" class="ms-grid2 ms-label--mt">
          <label class="ms-guidance__label">
            High-noise distill
            <input
              type="number"
              inputmode="decimal"
              step="0.1"
              min="0"
              :max="MAX_WAN_DISTILL_STRENGTH"
              placeholder="1.0"
              class="ms-input data-mono"
              data-test="wan-distill-high"
              :value="wanRecipe.distillStrengthHigh ?? ''"
              @input="
                setWanRecipe({
                  distillStrengthHigh: numberOrNull(($event.target as HTMLInputElement).value),
                })
              "
            />
          </label>
          <label class="ms-guidance__label">
            Low-noise distill
            <input
              type="number"
              inputmode="decimal"
              step="0.1"
              min="0"
              :max="MAX_WAN_DISTILL_STRENGTH"
              placeholder="1.0"
              class="ms-input data-mono"
              data-test="wan-distill-low"
              :value="wanRecipe.distillStrengthLow ?? ''"
              @input="
                setWanRecipe({
                  distillStrengthLow: numberOrNull(($event.target as HTMLInputElement).value),
                })
              "
            />
          </label>
        </div>
        <p
          v-if="wanRecipeMessage"
          class="ms-error ms-error--mt"
          role="alert"
          data-test="wan-recipe-error"
        >
          {{ wanRecipeMessage }}
        </p>
      </AccordionSection>

      <!-- 2 · Negative prompt -->
      <AccordionSection
        v-if="caps.supportsNegativePrompt || form.negativePrompt.trim()"
        icon="negative"
        title="Negative prompt"
        summary="What to steer away from"
        :open="true"
        :header-interactive="false"
        data-test="section-negative"
      >
        <textarea
          v-model="form.negativePrompt"
          :disabled="!caps.supportsNegativePrompt"
          data-selectable
          rows="2"
          placeholder="blurry, low quality, deformed…"
          class="ms-textarea"
          aria-label="Negative prompt"
        />
        <p
          v-if="!caps.supportsNegativePrompt"
          class="ms-hint"
          data-test="negative-unavailable-hint"
        >
          Saved for reuse, but this distilled recipe fixes CFG and does not use negative-prompt
          guidance. Choose a Dev checkpoint with Auto or a guided pipeline to enable it.
        </p>
        <div class="ms-chips">
          <Chip
            v-for="word in NEGATIVE_QUICK_ADDS"
            :key="word"
            :data-test="`neg-add-${word.replace(/\s+/g, '-')}`"
            @click="addNegative(word)"
            >+ {{ word }}</Chip
          >
        </div>
      </AccordionSection>

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

      <!-- 3 · Source image -->
      <AccordionSection
        v-if="caps.supportsImg2img && !h3Family"
        icon="image"
        title="Source image"
        summary="Image-to-image &amp; inpainting"
        :open="true"
        :header-interactive="false"
        data-test="section-source"
      >
        <SourceImageWell :form="form" />
      </AccordionSection>

      <!-- 4 · LoRA stack -->
      <AccordionSection
        v-if="caps.supportsLora"
        icon="layers"
        title="LoRA stack"
        summary="Style adapters"
        :open="true"
        :header-interactive="false"
        data-test="section-lora"
      >
        <LoraStack :form="form" :model="form.model" @append-word="emit('append-word', $event)" />
      </AccordionSection>

      <!-- 5 · Upscale after generate -->
      <AccordionSection
        v-if="!caps.supportsVideo && upscalers.length"
        icon="upscale"
        title="Upscale after generate"
        :summary="form.upscaleModel || 'Off'"
        :open="true"
        :header-interactive="false"
        data-test="section-upscale"
      >
        <label class="ms-label">Upscaler</label>
        <select v-model="form.upscaleModel" data-test="upscale-select" class="ms-select">
          <option value="">Off</option>
          <option v-for="u in upscalers" :key="u.name" :value="u.name">
            {{ u.name }}{{ u.downloaded ? "" : " (downloads on first use)" }}
          </option>
        </select>
      </AccordionSection>

      <!-- 6 · Output & seed -->
      <AccordionSection
        icon="output"
        title="Output &amp; seed"
        summary="Format, exact size, reproducibility"
        :open="true"
        :header-interactive="false"
        data-test="section-output"
      >
        <label class="ms-label">File format</label>
        <SegmentedControl
          :model-value="form.outputFormat"
          :options="formatOptions"
          label="File format"
          @update:model-value="form.outputFormat = $event"
        />

        <label class="ms-label ms-label--mt">Exact size</label>
        <div class="ms-size">
          <input
            v-model.number="form.width"
            type="number"
            step="16"
            min="64"
            aria-label="Width"
            class="ms-input data-mono"
            @change="snapWidth"
          />
          <button
            type="button"
            class="ms-size__swap"
            title="Swap width and height"
            aria-label="Swap width and height"
            @click="swapSize"
          >
            ⇄
          </button>
          <input
            v-model.number="form.height"
            type="number"
            step="16"
            min="64"
            aria-label="Height"
            class="ms-input data-mono"
            @change="snapHeight"
          />
        </div>

        <template v-if="seedFixed">
          <label class="ms-label ms-label--mt">Fixed seed</label>
          <input
            v-model="form.seed"
            data-selectable
            data-test="advanced-seed-value"
            type="text"
            inputmode="numeric"
            aria-label="Fixed seed value"
            class="ms-input data-mono"
          />
        </template>
      </AccordionSection>

      <!-- 7 · Video (ltx families) -->
      <AccordionSection
        v-if="caps.supportsVideo"
        icon="video"
        title="Video"
        summary="Frames, motion &amp; pipeline"
        :open="true"
        :header-interactive="false"
        data-test="section-video"
      >
        <VideoDurationSlider
          :frames="form.frames"
          :fps="form.fps"
          :model="selectedModel"
          @update:frames="form.frames = $event"
        />

        <label class="ms-label">Frames ({{ frameGridLabel }})</label>
        <input
          v-model.number="form.frames"
          type="number"
          :step="frameStep"
          :min="frameMinimum"
          aria-label="Frames"
          :aria-invalid="framesError ? 'true' : undefined"
          class="ms-input data-mono"
          :class="framesError ? 'ms-input--error' : ''"
          @change="snapFramesField"
        />
        <p v-if="framesError" class="ms-error">{{ framesError }}</p>

        <div v-if="chainDecision.kind === 'chain'" data-test="chain-cue" class="ms-cue">
          Will render as
          <span class="data-mono font-semibold text-ink">{{ chainDecision.stageCount }}</span>
          chained clips of {{ chainDecision.clipFrames }} frames (motion-tail
          {{ chainDecision.motionTail }}) — expect this to take substantially longer than a single
          clip.
        </div>
        <div
          v-else-if="singleShotPreservationNote"
          data-test="single-shot-preservation-cue"
          class="ms-cue"
        >
          {{ singleShotPreservationNote }}
        </div>
        <p
          v-else-if="chainDecision.kind === 'reject'"
          data-test="chain-reject"
          class="ms-error ms-error--mt"
        >
          {{ chainDecision.reason }}
        </p>

        <label class="ms-label ms-label--mt">FPS</label>
        <input
          v-model.number="form.fps"
          type="number"
          min="1"
          max="60"
          :disabled="fixedFps !== null"
          aria-label="Frames per second"
          class="ms-input data-mono"
        />
        <p v-if="fpsError" class="ms-error" role="alert">{{ fpsError }}</p>

        <template v-if="caps.supportsAdvancedVideo">
          <label class="ms-label ms-label--mt">Camera motion</label>
          <select
            data-test="camera-motion"
            aria-label="Camera motion"
            class="ms-select"
            :value="uiCameraMode"
            @change="setCameraMode(($event.target as HTMLSelectElement).value)"
          >
            <option value="">None</option>
            <option
              v-for="p in cameraControls"
              :key="p.id"
              :value="p.id"
              :disabled="!cameraSlotAvailable"
            >
              {{ p.label }}{{ p.installed ? "" : " · downloads on first use" }}
            </option>
            <option value="custom" :disabled="!cameraSlotAvailable">Custom LoRA path…</option>
          </select>
          <input
            v-if="uiCameraMode === 'custom'"
            data-test="camera-motion-custom"
            data-selectable
            type="text"
            placeholder="/path/to/lora.safetensors"
            aria-label="Camera motion LoRA path"
            class="ms-input data-mono ms-input--mt"
            :value="form.cameraControl ?? ''"
            @input="setCameraControl(($event.target as HTMLInputElement).value)"
          />
          <p
            v-if="cameraControlsLoaded && cameraControls.length === 0"
            data-test="camera-motion-19b-hint"
            class="ms-hint"
          >
            {{
              cameraUnsupportedReason ??
              "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
            }}
          </p>
          <p v-if="cameraError" data-test="camera-motion-error" class="ms-error" role="alert">
            {{ cameraError }}
          </p>
        </template>

        <div v-if="caps.supportsAudio && !h3Family" class="ms-switch-row ms-switch-row--mt">
          <div class="ms-switch-row__title">Generate audio</div>
          <SwitchToggle
            :model-value="form.enableAudio"
            :disabled="!audioOutputSupported"
            label="Generate audio"
            @update:model-value="form.enableAudio = $event"
          />
        </div>
        <p v-if="caps.supportsAudio && !h3Family && !audioOutputSupported" class="ms-hint">
          Audio assets are not included with this checkpoint. Video generation remains available.
        </p>
        <p v-if="audioFormatError" class="ms-error" role="alert">{{ audioFormatError }}</p>

        <template v-if="caps.supportsAdvancedVideo">
          <p
            v-if="advancedVideoError"
            data-test="ltx2-validation-error"
            class="ms-error"
            role="alert"
          >
            {{ advancedVideoError }}
          </p>
          <div data-test="ltx2-controls">
            <label class="ms-label ms-label--mt">Pipeline</label>
            <select
              :value="form.pipeline ?? ''"
              aria-label="LTX-2 pipeline mode"
              class="ms-select"
              data-test="ltx2-pipeline"
              @change="setPipeline(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Auto</option>
              <option v-for="opt in pipelineOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
            <p v-if="form.pipeline === 'lip-dub'" class="ms-hint" data-test="ltx2-lip-dub-hint">
              {{ LIP_DUB_TIMING_HINT }}
            </p>
            <p v-if="audioOnlyPipeline" class="ms-hint" data-test="ltx2-t2a-hint">
              Text-to-audio renders sound only — no video. Duration comes from frames ÷ fps and the
              output is a WAV. Source media, keyframes, retake, and the upscalers do not apply.
            </p>

            <label class="ms-label ms-label--mt">Reference control</label>
            <select
              :value="form.icLoraControl ?? ''"
              aria-label="Reference control"
              class="ms-select"
              data-test="ltx2-reference-control"
              @change="setControlAdapter(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Custom / none</option>
              <option v-for="adapter in controlAdapters" :key="adapter.id" :value="adapter.id">
                {{ adapter.label }}{{ adapter.installed ? "" : " · download" }}
              </option>
            </select>
            <p v-if="form.icLoraControl" class="ms-hint" data-test="ltx2-reference-guide">
              {{
                controlAdapters.find((adapter) => adapter.id === form.icLoraControl)?.guide ??
                "Choose the frame-aligned guide video this control expects."
              }}
            </p>

            <div class="ms-grid2 ms-label--mt">
              <div>
                <label class="ms-label">Spatial</label>
                <select
                  :value="form.spatialUpscale ?? ''"
                  aria-label="Spatial upscale"
                  class="ms-select"
                  data-test="ltx2-spatial"
                  @change="setSpatial(($event.target as HTMLSelectElement).value)"
                >
                  <option value="">Native</option>
                  <option v-for="v in spatialOptions" :key="v" :value="v">{{ v }}</option>
                </select>
              </div>
              <div>
                <label class="ms-label">Temporal</label>
                <select
                  :value="form.temporalUpscale ?? ''"
                  aria-label="Temporal upscale"
                  class="ms-select"
                  data-test="ltx2-temporal"
                  @change="setTemporal(($event.target as HTMLSelectElement).value)"
                >
                  <option value="">Native</option>
                  <option v-for="v in temporalOptions" :key="v" :value="v">{{ v }}</option>
                </select>
              </div>
            </div>

            <div class="ms-guidance ms-label--mt">
              <div class="ms-guidance__head">
                <label class="ms-label">
                  Guidance overrides
                  <span
                    v-if="guidanceCount"
                    class="ms-guidance__count"
                    data-test="ltx2-guidance-count"
                    >{{ guidanceCount }}</span
                  >
                </label>
                <button
                  v-if="guidanceCount"
                  type="button"
                  class="ms-guidance__reset"
                  data-test="ltx2-guidance-reset"
                  @click="resetGuidance"
                >
                  Reset
                </button>
              </div>
              <p class="ms-hint">
                Empty keeps this checkpoint’s pipeline defaults. Used by two-stage, two-stage HQ,
                keyframe, and audio-to-video renders.
              </p>
              <div class="ms-grid2 ms-label--mt">
                <label class="ms-guidance__label">
                  STG scale
                  <input
                    type="number"
                    inputmode="decimal"
                    step="0.1"
                    min="0"
                    :max="MAX_GUIDANCE_SCALE"
                    placeholder="Default"
                    class="ms-input data-mono"
                    data-test="ltx2-stg-scale"
                    :value="guidance.stgScale ?? ''"
                    @input="
                      setGuidance({
                        stgScale: numberOrNull(($event.target as HTMLInputElement).value),
                      })
                    "
                  />
                </label>
                <label class="ms-guidance__label">
                  STG blocks
                  <input
                    type="text"
                    inputmode="numeric"
                    placeholder="28, 29"
                    class="ms-input data-mono"
                    data-test="ltx2-stg-blocks"
                    :aria-invalid="stgBlocksMessage ? 'true' : undefined"
                    :value="guidance.stgBlocks"
                    @input="setGuidance({ stgBlocks: ($event.target as HTMLInputElement).value })"
                  />
                </label>
              </div>
              <p
                v-if="stgBlocksMessage"
                class="ms-error"
                role="alert"
                data-test="ltx2-stg-blocks-error"
              >
                {{ stgBlocksMessage }}
              </p>
              <div class="ms-grid2 ms-label--mt">
                <label class="ms-guidance__label">
                  CFG rescale
                  <input
                    type="number"
                    inputmode="decimal"
                    step="0.05"
                    min="0"
                    max="1"
                    placeholder="Default"
                    class="ms-input data-mono"
                    data-test="ltx2-rescale-scale"
                    :value="guidance.rescaleScale ?? ''"
                    @input="
                      setGuidance({
                        rescaleScale: numberOrNull(($event.target as HTMLInputElement).value),
                      })
                    "
                  />
                </label>
                <label class="ms-guidance__label">
                  Modality scale
                  <input
                    type="number"
                    inputmode="decimal"
                    step="0.1"
                    min="0"
                    :max="MAX_GUIDANCE_SCALE"
                    placeholder="Default"
                    class="ms-input data-mono"
                    data-test="ltx2-modality-scale"
                    :value="guidance.modalityScale ?? ''"
                    @input="
                      setGuidance({
                        modalityScale: numberOrNull(($event.target as HTMLInputElement).value),
                      })
                    "
                  />
                </label>
              </div>
              <label class="ms-guidance__label ms-label--mt">
                Guidance skip stride
                <input
                  type="number"
                  inputmode="numeric"
                  step="1"
                  min="0"
                  :max="MAX_GUIDANCE_SKIP_STEP"
                  placeholder="Every step"
                  class="ms-input data-mono"
                  data-test="ltx2-guidance-skip-step"
                  :aria-invalid="skipStepMessage ? 'true' : undefined"
                  :value="guidance.skipStep ?? ''"
                  @input="
                    setGuidance({
                      skipStep: numberOrNull(($event.target as HTMLInputElement).value),
                    })
                  "
                />
              </label>
              <p
                v-if="skipStepMessage"
                class="ms-error"
                role="alert"
                data-test="ltx2-guidance-skip-step-error"
              >
                {{ skipStepMessage }}
              </p>
            </div>

            <template v-if="form.pipeline === 'retake'">
              <label class="ms-label ms-label--mt">Retake range (seconds)</label>
              <div class="ms-grid2">
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  :value="form.retakeRange?.start_seconds ?? ''"
                  placeholder="start"
                  aria-label="Retake start (seconds)"
                  class="ms-input data-mono"
                  data-test="ltx2-retake-start"
                  @input="setRetake('start', ($event.target as HTMLInputElement).value)"
                />
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  :value="form.retakeRange?.end_seconds ?? ''"
                  placeholder="end"
                  aria-label="Retake end (seconds)"
                  class="ms-input data-mono"
                  data-test="ltx2-retake-end"
                  @input="setRetake('end', ($event.target as HTMLInputElement).value)"
                />
              </div>
            </template>

            <template v-if="form.pipeline === 'a2-vid'">
              <label class="ms-label ms-label--mt">Conditioning audio</label>
              <div v-if="form.audioFile" class="ms-file-row">
                <span class="data-mono ms-file-row__name" :title="form.audioFile.filename">{{
                  form.audioFile.filename
                }}</span>
                <button type="button" class="ms-file-row__clear" @click="clearAudioFile">
                  clear
                </button>
              </div>
              <input
                v-else
                type="file"
                accept="audio/*"
                aria-label="Conditioning audio"
                class="ms-file"
                data-test="ltx2-audio-file"
                @change="setAudioFile"
              />
              <p class="ms-hint">a2-vid needs an audio track to drive the video.</p>
            </template>

            <label class="ms-label ms-label--mt">Source video</label>
            <div v-if="form.sourceVideo" class="ms-file-row">
              <span class="data-mono ms-file-row__name" :title="form.sourceVideo.filename">{{
                form.sourceVideo.filename
              }}</span>
              <button type="button" class="ms-file-row__clear" @click="clearSourceVideo">
                clear
              </button>
            </div>
            <input
              v-else
              type="file"
              accept="video/*"
              aria-label="Source video"
              class="ms-file"
              data-test="ltx2-source-video"
              @change="setSourceVideo"
            />

            <template v-if="canExtend">
              <label class="ms-label ms-label--mt">Continue a video</label>
              <div v-if="form.extendVideo" class="ms-file-row">
                <span class="data-mono ms-file-row__name" :title="form.extendVideo.filename">{{
                  form.extendVideo.filename
                }}</span>
                <button
                  type="button"
                  class="ms-file-row__clear"
                  data-test="ltx2-extend-clear"
                  @click="clearExtendVideo"
                >
                  clear
                </button>
              </div>
              <input
                v-else
                type="file"
                accept="video/*"
                aria-label="Video to continue"
                class="ms-file"
                data-test="ltx2-extend-video"
                @change="setExtendVideo"
              />
              <template v-if="form.extendVideo">
                <label class="ms-label ms-label--mt">Overlap (frames of motion context)</label>
                <select
                  class="ms-select"
                  data-test="ltx2-extend-overlap"
                  :value="String(extendOverlap)"
                  @change="
                    form.extendOverlapFrames = Number(($event.target as HTMLSelectElement).value)
                  "
                >
                  <option
                    v-for="option in extendOverlapChoices"
                    :key="option"
                    :value="String(option)"
                  >
                    {{ option }}
                  </option>
                </select>
                <p v-if="extendError" class="ms-error" data-test="ltx2-extend-error">
                  {{ extendError }}
                </p>
                <p v-else class="ms-hint" data-test="ltx2-extend-summary">{{ extendSummary }}</p>
              </template>
            </template>

            <div class="ms-kf-head ms-label--mt">
              <label class="ms-label">Keyframes</label>
              <button
                type="button"
                class="ms-kf-add"
                data-test="ltx2-add-keyframe"
                @click="keyframePickerOpen = true"
              >
                Add…
              </button>
            </div>
            <div
              v-for="(keyframe, index) in form.keyframes"
              :key="`${keyframe.image.filename}-${index}`"
              class="ms-kf-row"
            >
              <input
                type="number"
                min="0"
                :value="keyframe.frame"
                :aria-label="`Keyframe ${index + 1} frame`"
                class="ms-input data-mono ms-kf-row__frame"
                :data-test="`ltx2-keyframe-frame-${index}`"
                @input="updateKeyframeFrame(index, ($event.target as HTMLInputElement).value)"
              />
              <span class="ms-kf-row__name">{{ keyframe.image.filename }}</span>
              <button
                type="button"
                class="ms-kf-row__remove"
                :aria-label="`Remove keyframe ${index + 1}`"
                @click="removeKeyframe(index)"
              >
                ✕
              </button>
            </div>
          </div>

          <ImagePickerModal
            :open="keyframePickerOpen"
            title="Keyframe image"
            :multiple="false"
            @pick="onKeyframePick"
            @close="keyframePickerOpen = false"
          />
        </template>
      </AccordionSection>
    </div>
  </section>
</template>

<style scoped>
.ms-adv {
  padding-top: 10px;
}
.ms-adv__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.ms-adv__summary {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  letter-spacing: 0.04em;
}
.ms-adv__list {
  display: flex;
  flex-direction: column;
  gap: 11px;
}
.ms-label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ms-label--mt {
  margin-top: 16px;
}
.ms-select,
.ms-input {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: 9px;
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
}
.ms-input--mt {
  margin-top: 6px;
}
.ms-input--error {
  border-color: var(--stop);
}
.ms-textarea {
  width: 100%;
  box-sizing: border-box;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: 10px;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13.5px;
  resize: none;
  outline: none;
  min-height: 64px;
  line-height: 1.45;
  padding: 11px 13px;
}
.ms-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
}
.ms-switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 16px;
}
.ms-switch-row--mt {
  margin-top: 16px;
}
.ms-switch-row__title {
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
}
.ms-switch-row__hint {
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 2px;
}
.ms-size {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ms-size__swap {
  flex-shrink: 0;
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-grid2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.ms-error {
  font-size: 11px;
  color: var(--stop);
  margin-top: 6px;
}
.ms-error--mt {
  margin-top: 6px;
}
.ms-hint {
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
  line-height: 1.45;
}
.ms-cue {
  border: 1px solid var(--ce);
  margin-top: 6px;
  border-radius: 9px;
  background: color-mix(in srgb, var(--safelight) 10%, transparent);
  padding: 8px 10px;
  font-size: 11px;
  color: var(--ink-2);
}
.ms-file {
  display: block;
  width: 100%;
  font-size: 11px;
  color: var(--ink-3);
}
.ms-file-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ms-file-row__name {
  font-size: 11px;
  color: var(--rebate);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-file-row__clear {
  font-size: 11px;
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-kf-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ms-kf-add {
  font-size: 11px;
  color: var(--safelight);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-kf-row {
  display: grid;
  grid-template-columns: 4rem 1fr auto;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}
.ms-kf-row__frame {
  height: 32px;
}
.ms-kf-row__name {
  font-size: 11px;
  color: var(--ink-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-kf-row__remove {
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-guidance {
  border-top: 1px solid var(--ce);
  padding-top: 16px;
}
.ms-guidance__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ms-guidance__head .ms-label {
  margin-bottom: 0;
}
.ms-guidance__count {
  display: inline-block;
  margin-left: 5px;
  padding: 1px 6px;
  border: 1px solid var(--ce);
  border-radius: 999px;
  font-family: var(--f-mono);
  font-size: 10px;
}
.ms-guidance__reset {
  border: 0;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  font-size: 11px;
  font-weight: 600;
}
.ms-guidance__label {
  display: block;
  color: var(--ink-2);
  font-size: 11px;
  font-weight: 600;
}
.ms-guidance__label .ms-input {
  margin-top: 6px;
}
.ms-adv__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 9px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.ms-adv__reset:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
  color: var(--rebate);
}
</style>
