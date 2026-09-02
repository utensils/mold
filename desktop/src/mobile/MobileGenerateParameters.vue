<script setup lang="ts">
import { computed, ref, useId, watch } from "vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import type {
  Ltx2PipelineMode,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
} from "../lib/api/types";
import type { ApiTarget } from "../lib/api/client";
import { generationCapabilitiesForFamily, MAX_LORA_STACK } from "../lib/capabilities";
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
} from "../lib/chainRouting";
import {
  applyRecipeDefaults,
  buildRequest,
  formExtendOverlapFrames,
  type GenerateForm,
  type PickedImage,
} from "../lib/generateForm";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  inlineGenerationMediaBytes,
  wanRecipeValidationError,
  MAX_INLINE_GENERATION_MEDIA_BYTES,
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  type InlineGenerationMediaField,
} from "../lib/generateValidation";
import { fileToBase64, isStillImageFile } from "../lib/image";
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
} from "@studio/lib/extend";
import {
  guidanceOverrideCount,
  skipStepError,
  stgBlocksError,
  MAX_GUIDANCE_SCALE,
  MAX_GUIDANCE_SKIP_STEP,
  type Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import { schedulerLabel } from "@studio/lib/generationCapabilities";
import {
  IDENTITY_SECTION_LABEL,
  IDENTITY_START_STEP_HINT,
  IDENTITY_START_STEP_LABEL,
  IDENTITY_WEIGHT_HINT,
  IDENTITY_WEIGHT_LABEL,
  ID_START_STEP_DEFAULT,
  ID_WEIGHT_DEFAULT,
  ID_WEIGHT_MAX,
  ID_WEIGHT_MIN,
  ID_WEIGHT_STEP,
} from "@studio/lib/identityConditioning";
import { mobileIdentityStartStepMax } from "./identity";
import { identityConditioningValidationError } from "../lib/generateValidation";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import {
  MAX_WAN_DISTILL_STRENGTH,
  wanRecipeCount,
  type WanRecipeState,
} from "@studio/lib/wanRecipe";
import {
  LIP_DUB_TIMING_HINT,
  isControlAdapterPipeline,
  pipelineForControlId,
} from "@studio/lib/ltx2Control";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** Selected model row; carries the continuation capability. */
    selectedModel?: ModelEntry | null;
    target?: ApiTarget | null;
    upscalers?: ModelEntry[];
    controlAdapters?: Ltx2ControlAdapterInfo[];
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    selectedModel: null,
    target: null,
    upscalers: () => [],
    controlAdapters: () => [],
    cameraControls: () => [],
    cameraControlsLoaded: false,
    cameraUnsupportedReason: null,
  },
);
const emit = defineEmits<{
  "validity-change": [valid: boolean];
}>();

const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.selectedModel?.guidance_capabilities,
    props.selectedModel?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(props.selectedModel, props.form.pipeline),
  ),
);
const videoContract = computed(() => props.selectedModel ?? { family: props.form.family });
const frameGridLabel = computed(() => videoFrameGridLabel(videoContract.value));
const frameStep = computed(() => videoFrameStep(videoContract.value));
const frameMinimum = computed(() => minVideoFrames(videoContract.value));
const fixedFps = computed(() => fixedVideoFps(videoContract.value));
const generationRequest = computed(() => buildRequest(props.form));
const frameError = computed(() =>
  caps.value.supportsVideo
    ? fixedFps.value !== null
      ? videoFramesError(props.form.frames, videoContract.value)
      : videoFrameGridError(props.form.frames, videoContract.value)
    : null,
);
const fpsError = computed(() =>
  caps.value.supportsVideo
    ? fixedFps.value !== null && props.form.fps !== fixedFps.value
      ? `FPS is fixed at ${fixedFps.value} for this model.`
      : fpsValidationError(props.form.fps)
    : null,
);
const chainDecision = computed<ChainRoutingDecision>(() =>
  caps.value.supportsVideo
    ? decideGenerateRequestRouting(generationRequest.value, props.form.family, props.selectedModel)
    : { kind: "single" },
);

const singleShotPreservationNote = computed(() => {
  const decision = chainDecision.value;
  if (decision.kind !== "single" || !decision.preservedAutoChainFields?.length) return null;
  return `Rendering one ${props.form.frames}-frame clip to preserve ${autoChainFieldList(decision.preservedAutoChainFields)}. This may use more GPU memory than automatic chaining.`;
});

const audioFormatError = computed(() => audioOutputValidationError(props.form));
const advancedVideoError = computed(() => advancedVideoValidationError(props.form));
const cameraError = computed(() =>
  cameraControlValidationError(
    props.form,
    props.cameraControlsLoaded ? props.cameraControls.map((c) => c.id) : undefined,
  ),
);
const mediaReadError = ref("");

// Declared before `valid`: its immediate validity watch evaluates during
// setup, so everything the gate reads must already exist.
const wanRecipe = computed<WanRecipeState>(() => props.form.wanRecipe);
const wanRecipeActive = computed(() => wanRecipeCount(wanRecipe.value));
const wanRecipeMessage = computed(() => wanRecipeValidationError(props.form));
function setWanRecipe(next: Partial<WanRecipeState>): void {
  props.form.wanRecipe = { ...wanRecipe.value, ...next };
}

const valid = computed(
  () =>
    !frameError.value &&
    !fpsError.value &&
    chainDecision.value.kind !== "reject" &&
    !audioFormatError.value &&
    !advancedVideoError.value &&
    // An out-of-band wan recipe value must hold the Develop button, not be
    // silently dropped from the wire (codex review).
    !wanRecipeMessage.value &&
    !cameraError.value,
);

watch(valid, (next) => emit("validity-change", next), { immediate: true });
watch(
  [fixedFps, () => props.form.fps] as const,
  ([fixed, current]) => {
    if (fixed !== null && current !== fixed) props.form.fps = fixed;
  },
  { immediate: true },
);
function snapFramesField(): void {
  props.form.frames =
    fixedFps.value !== null
      ? clampVideoFrames(props.form.frames, fixedFps.value, videoContract.value)
      : Math.max(frameMinimum.value, snapVideoFrames(props.form.frames, videoContract.value));
}

const cameraMode = ref(cameraMotionMode(props.form.cameraControl));
watch(
  () => props.form.cameraControl,
  (value) => {
    if (cameraMode.value === "custom" && (value === "" || cameraMotionMode(value) === "custom")) {
      return;
    }
    cameraMode.value = cameraMotionMode(value);
  },
);

function setCameraMode(mode: string): void {
  cameraMode.value = mode;
  if (mode === "custom") {
    if (cameraMotionMode(props.form.cameraControl) !== "custom") setCameraControl("");
  } else {
    setCameraControl(mode || null);
  }
}

function setCameraControl(next: string | null): void {
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

function hasAdvancedVideoValue(): boolean {
  return !!(
    props.form.pipeline ||
    props.form.spatialUpscale ||
    props.form.temporalUpscale ||
    props.form.retakeRange ||
    props.form.audioFile ||
    props.form.sourceVideo ||
    props.form.extendVideo ||
    props.form.keyframes.length ||
    guidanceOverrideCount(props.form.guidanceOverrides) > 0
  );
}

const advancedOpen = ref(hasAdvancedVideoValue());
watch(
  () => [
    props.form.pipeline,
    props.form.spatialUpscale,
    props.form.temporalUpscale,
    props.form.retakeRange,
    props.form.audioFile,
    props.form.sourceVideo,
    props.form.extendVideo,
    props.form.keyframes.length,
    guidanceOverrideCount(props.form.guidanceOverrides),
  ],
  () => {
    if (hasAdvancedVideoValue()) advancedOpen.value = true;
  },
);
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
];
const spatialOptions: Ltx2SpatialUpscale[] = ["x1-5", "x2"];
const temporalOptions: Ltx2TemporalUpscale[] = ["x2"];

function setPipeline(value: string): void {
  const pipeline = (value || null) as Ltx2PipelineMode | null;
  applyRecipeDefaults(props.form, props.selectedModel, pipeline);
  if (!isControlAdapterPipeline(pipeline)) props.form.icLoraControl = null;
  if (pipeline !== "retake") props.form.retakeRange = null;
}
function setControlAdapter(value: string): void {
  // Lip dub is a pipeline of its own; every other adapter drives `ic-lora`.
  if (value) {
    applyRecipeDefaults(props.form, props.selectedModel, pipelineForControlId(value));
    props.form.icLoraControl = value;
  } else {
    props.form.icLoraControl = null;
  }
}

function setSpatial(value: string): void {
  props.form.spatialUpscale = (value || null) as Ltx2SpatialUpscale | null;
}

function setTemporal(value: string): void {
  props.form.temporalUpscale = (value || null) as Ltx2TemporalUpscale | null;
}

const guidance = computed<Ltx2GuidanceOverridesState>(() => props.form.guidanceOverrides);
const guidanceCount = computed(() => guidanceOverrideCount(guidance.value));
const stgBlocksMessage = computed(() => stgBlocksError(guidance.value.stgBlocks));
const skipStepMessage = computed(() => skipStepError(guidance.value.skipStep));

function setGuidance(next: Partial<Ltx2GuidanceOverridesState>): void {
  props.form.guidanceOverrides = { ...guidance.value, ...next };
}

function numberOrNull(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

// ── Face identity (PuLID, #1224) ────────────────────────────────────────────
// Only the two KNOBS live in Advanced; the photo well is primary-form media
// beside the source wells. Both stay absent from the request until touched, so
// the server's own defaults remain authoritative — an empty field is untouched,
// not zero.
const identitySupported = computed(() => props.form.identitySupported === true);
const identityStartStepMax = computed(() => mobileIdentityStartStepMax(props.form.steps));
const identityError = computed(() => identityConditioningValidationError(props.form));

function setIdentityWeight(raw: string): void {
  props.form.identityWeight = numberOrNull(raw);
}

function setIdentityStartStep(raw: string): void {
  // Deliberately NOT truncated: a fractional step is a mistake the shared
  // policy names inline ("must be a whole number from 0 to N"), and silently
  // rounding it would submit a step the user never typed. `identityRequestFields`
  // keeps a non-integer off the wire either way.
  props.form.identityStartStep = numberOrNull(raw);
}

function setRetake(edge: "start" | "end", raw: string): void {
  const value = Math.max(0, Number(raw) || 0);
  const current = props.form.retakeRange ?? { start_seconds: 0, end_seconds: 1 };
  props.form.retakeRange =
    edge === "start" ? { ...current, start_seconds: value } : { ...current, end_seconds: value };
}

function exceedsMobileRequestBudget(
  incomingBytes: number,
  exclude: InlineGenerationMediaField | null,
): boolean {
  return (
    inlineGenerationMediaBytes(props.form, exclude) + incomingBytes >
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES
  );
}

async function setSourceVideo(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  input.value = "";
  if (file.size === 0) {
    mediaReadError.value = "Source video cannot be empty.";
    return;
  }
  if (file.size > MAX_INLINE_GENERATION_MEDIA_BYTES) {
    mediaReadError.value = "Source videos must be 64 MiB or smaller.";
    return;
  }
  if (exceedsMobileRequestBudget(file.size, "sourceVideo")) {
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on this phone.";
    return;
  }
  try {
    props.form.sourceVideo = { filename: file.name, base64: await fileToBase64(file) };
    mediaReadError.value = "";
  } catch {
    mediaReadError.value = "Couldn’t read that source video. Try choosing it again.";
  }
}

async function setExtendVideo(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  input.value = "";
  if (file.size === 0) {
    mediaReadError.value = "The video to continue cannot be empty.";
    return;
  }
  if (file.size > MAX_INLINE_GENERATION_MEDIA_BYTES) {
    mediaReadError.value = "Videos to continue must be 64 MiB or smaller.";
    return;
  }
  if (exceedsMobileRequestBudget(file.size, "extendVideo")) {
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on this phone.";
    return;
  }
  try {
    props.form.extendVideo = { filename: file.name, base64: await fileToBase64(file) };
    mediaReadError.value = "";
  } catch {
    mediaReadError.value = "Couldn’t read that video. Try choosing it again.";
  }
}

function clearExtendVideo(): void {
  // Drop the overlap too: a value valid for this clip may be invalid for the
  // next one, and a stale number would silently ride along.
  props.form.extendVideo = null;
  props.form.extendOverlapFrames = null;
}

const canExtend = computed(() => canOfferExtend(props.selectedModel));
// Exactly what `buildRequest` will submit, so the number on screen is the
// number on the wire — wan offers one option, so `@change` never fires and
// the form field stays null.
const extendOverlap = computed(() => formExtendOverlapFrames(props.form));
// The overlap grid belongs to the family: LTX-2 re-encodes an 8k+1 tail,
// while wan carries the single frame it was seeded with (#783).
const extendFamily = computed(() => props.selectedModel?.family ?? props.form.family);
const extendOverlapChoices = computed(() =>
  extendOverlapOptions(props.form.frames, extendFamily.value),
);
const extendError = computed(() =>
  props.form.extendVideo
    ? extendValidationError({
        overlapFrames: extendOverlap.value,
        frames: props.form.frames,
        family: extendFamily.value,
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

async function setAudioFile(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  input.value = "";
  if (file.size === 0) {
    mediaReadError.value = "Conditioning audio cannot be empty.";
    return;
  }
  if (file.size > MAX_INLINE_GENERATION_MEDIA_BYTES) {
    mediaReadError.value = "Conditioning audio must be 64 MiB or smaller.";
    return;
  }
  if (exceedsMobileRequestBudget(file.size, "audioFile")) {
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on this phone.";
    return;
  }
  try {
    props.form.audioFile = { filename: file.name, base64: await fileToBase64(file) };
    mediaReadError.value = "";
  } catch {
    mediaReadError.value = "Couldn’t read that audio file. Try choosing it again.";
  }
}

function suggestKeyframeFrame(offset: number): number {
  const last = props.form.keyframes.at(-1);
  return (last ? last.frame + 24 : 0) + offset * 24;
}

async function addKeyframes(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const files = Array.from(input.files ?? []);
  if (!files.length) return;
  input.value = "";
  if (
    files.some(
      (file) =>
        file.size === 0 ||
        !(
          file.type === "image/png" ||
          file.type === "image/jpeg" ||
          (!file.type && isStillImageFile(file.name))
        ),
    )
  ) {
    mediaReadError.value = "Keyframes must be non-empty PNG or JPEG images.";
    return;
  }
  const incomingBytes = files.reduce((sum, file) => sum + file.size, 0);
  if (exceedsMobileRequestBudget(incomingBytes, null)) {
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on this phone.";
    return;
  }
  try {
    const picked = await Promise.all(
      files.map<Promise<PickedImage>>(async (file) => ({
        filename: file.name,
        base64: await fileToBase64(file),
      })),
    );
    props.form.keyframes = [
      ...props.form.keyframes,
      ...picked.map((image, index) => ({ frame: suggestKeyframeFrame(index), image })),
    ];
    mediaReadError.value = "";
  } catch {
    mediaReadError.value = "Couldn’t read those keyframes. Try choosing them again.";
  }
}

function updateKeyframeFrame(index: number, raw: string): void {
  const item = props.form.keyframes[index];
  if (!item) return;
  const next = props.form.keyframes.slice();
  next[index] = { ...item, frame: Math.max(0, Math.round(Number(raw) || 0)) };
  props.form.keyframes = next;
}

function removeKeyframe(index: number): void {
  props.form.keyframes = props.form.keyframes.filter((_, candidate) => candidate !== index);
}

const frameErrorId = `mobile-frame-error-${useId()}`;
const fpsErrorId = `mobile-fps-error-${useId()}`;
</script>

<template>
  <section class="mobile-generate-parameters" data-test="mobile-generate-parameters">
    <fieldset class="mobile-generate-section mobile-generate-print-options">
      <legend class="mobile-generate-legend">Print options</legend>

      <label
        v-if="caps.supportsScheduler && !caps.wanRecipe.supported"
        class="field mobile-generate-field"
      >
        <span>Scheduler</span>
        <select v-model="form.scheduler" class="control" data-test="mobile-scheduler">
          <option v-for="option in caps.schedulerOptions" :key="option" :value="option">
            {{ schedulerLabel(option) }}
          </option>
        </select>
      </label>

      <!--
        Wan's solver writes the same `scheduler` field the generic row above
        owns, so only one of the two ever renders; it sits with the flow shift
        and distill strengths it is tuned alongside.
      -->
      <template v-if="caps.wanRecipe.supported">
        <label class="field mobile-generate-field">
          <span>
            Sample solver
            <span
              v-if="wanRecipeActive"
              class="mobile-generate-inline-count"
              data-test="mobile-wan-recipe-count"
              >{{ wanRecipeActive }}</span
            >
          </span>
          <select v-model="form.scheduler" class="control" data-test="mobile-wan-solver">
            <option v-for="option in caps.schedulerOptions" :key="option" :value="option">
              {{ schedulerLabel(option) }}
            </option>
          </select>
        </label>
        <label class="field mobile-generate-field">
          <span>Flow shift</span>
          <input
            class="control"
            type="number"
            inputmode="decimal"
            step="0.5"
            min="0"
            placeholder="Model default"
            data-test="mobile-wan-sample-shift"
            :value="wanRecipe.sampleShift ?? ''"
            @input="
              setWanRecipe({ sampleShift: numberOrNull(($event.target as HTMLInputElement).value) })
            "
          />
        </label>
        <p class="mobile-generate-note">
          Empty keeps this model's own flow shift and distill strengths.
        </p>
        <div v-if="caps.wanRecipe.supportsDistillStrength" class="mobile-generate-field-grid">
          <label class="field mobile-generate-field">
            <span>High-noise distill</span>
            <input
              class="control"
              type="number"
              inputmode="decimal"
              step="0.1"
              min="0"
              :max="MAX_WAN_DISTILL_STRENGTH"
              placeholder="1.0"
              data-test="mobile-wan-distill-high"
              :value="wanRecipe.distillStrengthHigh ?? ''"
              @input="
                setWanRecipe({
                  distillStrengthHigh: numberOrNull(($event.target as HTMLInputElement).value),
                })
              "
            />
          </label>
          <label class="field mobile-generate-field">
            <span>Low-noise distill</span>
            <input
              class="control"
              type="number"
              inputmode="decimal"
              step="0.1"
              min="0"
              :max="MAX_WAN_DISTILL_STRENGTH"
              placeholder="1.0"
              data-test="mobile-wan-distill-low"
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
          class="mobile-generate-validation"
          role="alert"
          data-test="mobile-wan-recipe-error"
        >
          {{ wanRecipeMessage }}
        </p>
      </template>

      <label
        v-if="caps.supportsCfgPlus"
        class="mobile-generate-toggle-row"
        data-test="mobile-cfg-plus-row"
      >
        <span>
          <strong>CFG++</strong>
          <small>Lower guidance to 1.5–2.5</small>
        </span>
        <input v-model="form.cfgPlus" type="checkbox" data-test="mobile-cfg-plus" />
      </label>

      <label
        v-if="(!caps.supportsAudio || caps.supportsVideo) && upscalers.length"
        class="field mobile-generate-field"
      >
        <span>{{ caps.supportsVideo ? "Framewise upscale" : "Upscale" }}</span>
        <select v-model="form.upscaleModel" class="control" data-test="mobile-upscale">
          <option value="">Off</option>
          <option v-for="upscaler in upscalers" :key="upscaler.name" :value="upscaler.name">
            {{ upscaler.name }}{{ upscaler.downloaded ? "" : " (downloads on first use)" }}
          </option>
        </select>
      </label>
    </fieldset>

    <!--
      Identity knobs. The photo itself is picked in the primary Create stack;
      these two only ever ride the wire alongside it, and only on a checkpoint
      that advertises identity support (a parked partition shows nothing).
    -->
    <fieldset
      v-if="identitySupported"
      class="mobile-generate-section"
      data-test="mobile-identity-section"
    >
      <legend class="mobile-generate-legend">{{ IDENTITY_SECTION_LABEL }}</legend>

      <label class="field mobile-generate-field">
        <span>{{ IDENTITY_WEIGHT_LABEL }}</span>
        <input
          class="control"
          type="number"
          inputmode="decimal"
          :step="ID_WEIGHT_STEP"
          :min="ID_WEIGHT_MIN"
          :max="ID_WEIGHT_MAX"
          :placeholder="String(ID_WEIGHT_DEFAULT)"
          data-test="mobile-identity-weight"
          :value="form.identityWeight ?? ''"
          @input="setIdentityWeight(($event.target as HTMLInputElement).value)"
        />
        <small>{{ IDENTITY_WEIGHT_HINT }}</small>
      </label>

      <label class="field mobile-generate-field">
        <span>{{ IDENTITY_START_STEP_LABEL }}</span>
        <input
          class="control"
          type="number"
          inputmode="numeric"
          step="1"
          min="0"
          :max="identityStartStepMax"
          :placeholder="String(ID_START_STEP_DEFAULT)"
          data-test="mobile-identity-start-step"
          :value="form.identityStartStep ?? ''"
          @input="setIdentityStartStep(($event.target as HTMLInputElement).value)"
        />
        <small>{{ IDENTITY_START_STEP_HINT }}</small>
      </label>

      <p
        v-if="identityError"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-identity-error"
      >
        {{ identityError }}
      </p>
    </fieldset>

    <fieldset
      v-if="caps.supportsVideo"
      class="mobile-generate-section mobile-generate-video-options"
    >
      <legend class="mobile-generate-legend">Video</legend>

      <VideoDurationSlider
        :frames="form.frames"
        :fps="form.fps"
        :model="selectedModel"
        :family="form.family"
        :model-name="form.model"
        :source-image-capability="selectedModel?.source_image ?? form.sourceImageCapability"
        :routing-request="buildRequest(form)"
        touch-friendly
        data-test="mobile-advanced-duration"
        @update:frames="form.frames = $event"
      />

      <div class="mobile-generate-field-grid">
        <label class="field mobile-generate-field">
          <span>Frames</span>
          <small>{{ frameGridLabel }} grid</small>
          <input
            v-model.number="form.frames"
            class="control"
            data-test="mobile-frames"
            type="number"
            inputmode="numeric"
            :min="frameMinimum"
            :step="frameStep"
            :aria-invalid="frameError ? 'true' : undefined"
            :aria-describedby="frameError ? frameErrorId : undefined"
            @change="snapFramesField"
          />
        </label>

        <label v-if="caps.supportsAdvancedVideo" class="field mobile-generate-field">
          <span>Reference control</span>
          <select
            class="control"
            data-test="mobile-ltx2-reference-control"
            :value="form.icLoraControl ?? ''"
            @change="setControlAdapter(($event.target as HTMLSelectElement).value)"
          >
            <option value="">Custom / none</option>
            <option v-for="adapter in controlAdapters" :key="adapter.id" :value="adapter.id">
              {{ adapter.label }}{{ adapter.installed ? "" : " · download" }}
            </option>
          </select>
          <small v-if="form.icLoraControl" data-test="mobile-ltx2-reference-guide">
            {{
              controlAdapters.find((adapter) => adapter.id === form.icLoraControl)?.guide ??
              "Choose the frame-aligned guide video this control expects."
            }}
          </small>
        </label>
        <label class="field mobile-generate-field">
          <span>FPS</span>
          <input
            v-model.number="form.fps"
            class="control"
            data-test="mobile-fps"
            type="number"
            inputmode="numeric"
            min="1"
            max="60"
            :disabled="fixedFps !== null"
            :aria-invalid="fpsError ? 'true' : undefined"
            :aria-describedby="fpsError ? fpsErrorId : undefined"
          />
        </label>
      </div>

      <p
        v-if="frameError"
        :id="frameErrorId"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-frames-error"
      >
        {{ frameError }}
      </p>
      <p
        v-if="fpsError"
        :id="fpsErrorId"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-fps-error"
      >
        {{ fpsError }}
      </p>
      <p
        v-if="chainDecision.kind === 'chain'"
        class="mobile-generate-callout"
        data-test="mobile-chain-cue"
      >
        Will render as {{ chainDecision.stageCount }} chained clips of up to
        {{ chainDecision.clipFrames }} frames with a {{ chainDecision.motionTail }}-frame motion
        tail.
      </p>
      <p
        v-else-if="singleShotPreservationNote"
        class="mobile-generate-callout"
        data-test="mobile-single-shot-preservation-cue"
      >
        {{ singleShotPreservationNote }}
      </p>
      <p
        v-else-if="chainDecision.kind === 'reject'"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-chain-reject"
      >
        {{ chainDecision.reason }}
      </p>

      <p
        v-if="audioFormatError"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-audio-format-error"
      >
        {{ audioFormatError }}
      </p>

      <!-- Continuation is per model, not part of the LTX-2 suite (#783):
           wan continues by seeding the render with the source clip's final
           frame and never renders the LTX-2 pipeline disclosure. -->
      <div v-if="canExtend" class="mobile-generate-file-field">
        <span class="mobile-generate-label">Continue a video</span>
        <div v-if="form.extendVideo" class="mobile-generate-picked-file">
          <span>{{ form.extendVideo.filename }}</span>
          <button
            type="button"
            class="mobile-generate-clear"
            data-test="mobile-ltx2-extend-clear"
            @click="clearExtendVideo"
          >
            Remove
          </button>
        </div>
        <label v-else class="mobile-generate-file-button">
          <span>Choose video to continue</span>
          <input
            type="file"
            accept="video/*"
            data-test="mobile-ltx2-extend-video"
            @change="setExtendVideo"
          />
        </label>
        <template v-if="form.extendVideo">
          <span class="mobile-generate-label">Overlap (frames of motion context)</span>
          <select
            class="mobile-generate-select"
            data-test="mobile-ltx2-extend-overlap"
            :value="String(extendOverlap)"
            @change="form.extendOverlapFrames = Number(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="option in extendOverlapChoices" :key="option" :value="String(option)">
              {{ option }}
            </option>
          </select>
          <p v-if="extendError" class="mobile-generate-error" data-test="mobile-ltx2-extend-error">
            {{ extendError }}
          </p>
          <p v-else class="mobile-generate-note" data-test="mobile-ltx2-extend-summary">
            {{ extendSummary }}
          </p>
        </template>
      </div>

      <label v-if="caps.supportsAdvancedVideo" class="field mobile-generate-field">
        <span>Camera motion</span>
        <select
          class="control"
          data-test="mobile-camera-motion"
          aria-label="Camera motion"
          :value="cameraMode"
          @change="setCameraMode(($event.target as HTMLSelectElement).value)"
        >
          <option value="">None</option>
          <option
            v-for="preset in cameraControls"
            :key="preset.id"
            :value="preset.id"
            :disabled="!cameraSlotAvailable"
          >
            {{ preset.label }}{{ preset.installed ? "" : " · downloads on first use" }}
          </option>
          <option value="custom" :disabled="!cameraSlotAvailable">Custom LoRA path…</option>
        </select>
      </label>
      <label
        v-if="caps.supportsAdvancedVideo && cameraMode === 'custom'"
        class="field mobile-generate-field"
      >
        <span>Camera LoRA path</span>
        <input
          class="control"
          data-test="mobile-camera-motion-custom"
          data-selectable
          type="text"
          autocomplete="off"
          autocapitalize="none"
          placeholder="/path/to/lora.safetensors"
          :value="form.cameraControl ?? ''"
          @input="setCameraControl(($event.target as HTMLInputElement).value)"
        />
      </label>
      <p
        v-if="caps.supportsAdvancedVideo && cameraControlsLoaded && cameraControls.length === 0"
        class="mobile-generate-note"
        data-test="mobile-camera-motion-19b-hint"
      >
        {{
          cameraUnsupportedReason ??
          "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
        }}
      </p>
      <p
        v-if="cameraError"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-camera-motion-error"
      >
        {{ cameraError }}
      </p>
    </fieldset>

    <section
      v-if="caps.supportsAdvancedVideo"
      class="mobile-generate-section mobile-generate-advanced-video"
    >
      <button
        type="button"
        class="mobile-generate-disclosure"
        data-test="mobile-ltx2-disclosure"
        :aria-expanded="advancedOpen"
        aria-controls="mobile-ltx2-advanced-fields"
        @click="advancedOpen = !advancedOpen"
      >
        <span>LTX-2 pipeline</span>
        <span aria-hidden="true">{{ advancedOpen ? "−" : "+" }}</span>
      </button>

      <div
        v-if="advancedOpen"
        id="mobile-ltx2-advanced-fields"
        class="mobile-generate-disclosure-body"
      >
        <label class="field mobile-generate-field">
          <span>Pipeline</span>
          <select
            class="control"
            data-test="mobile-ltx2-pipeline"
            :value="form.pipeline ?? ''"
            @change="setPipeline(($event.target as HTMLSelectElement).value)"
          >
            <option value="">Auto</option>
            <option v-for="option in pipelineOptions" :key="option" :value="option">
              {{ option }}
            </option>
          </select>
        </label>
        <p
          v-if="form.pipeline === 'lip-dub'"
          class="mobile-generate-note"
          data-test="mobile-ltx2-lip-dub-hint"
        >
          {{ LIP_DUB_TIMING_HINT }}
        </p>

        <div class="mobile-generate-field-grid">
          <label class="field mobile-generate-field">
            <span>Spatial upscale</span>
            <select
              class="control"
              data-test="mobile-ltx2-spatial"
              :value="form.spatialUpscale ?? ''"
              @change="setSpatial(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Native</option>
              <option v-for="option in spatialOptions" :key="option" :value="option">
                {{ option }}
              </option>
            </select>
          </label>
          <label class="field mobile-generate-field">
            <span>Temporal upscale</span>
            <select
              class="control"
              data-test="mobile-ltx2-temporal"
              :value="form.temporalUpscale ?? ''"
              @change="setTemporal(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Native</option>
              <option v-for="option in temporalOptions" :key="option" :value="option">
                {{ option }}
              </option>
            </select>
          </label>
        </div>

        <div v-if="form.pipeline === 'retake'" class="mobile-generate-field">
          <span class="mobile-generate-label">Retake range</span>
          <div class="mobile-generate-field-grid">
            <label class="field mobile-generate-field">
              <span>Start seconds</span>
              <input
                class="control"
                data-test="mobile-ltx2-retake-start"
                type="number"
                inputmode="decimal"
                min="0"
                step="0.1"
                :value="form.retakeRange?.start_seconds ?? ''"
                @input="setRetake('start', ($event.target as HTMLInputElement).value)"
              />
            </label>
            <label class="field mobile-generate-field">
              <span>End seconds</span>
              <input
                class="control"
                data-test="mobile-ltx2-retake-end"
                type="number"
                inputmode="decimal"
                min="0"
                step="0.1"
                :value="form.retakeRange?.end_seconds ?? ''"
                @input="setRetake('end', ($event.target as HTMLInputElement).value)"
              />
            </label>
          </div>
        </div>

        <div class="mobile-generate-guidance" data-test="mobile-ltx2-guidance">
          <div class="mobile-generate-label-row">
            <span class="mobile-generate-label">
              Guidance overrides
              <span
                v-if="guidanceCount"
                class="mobile-generate-inline-count"
                data-test="mobile-ltx2-guidance-count"
                >{{ guidanceCount }}</span
              >
            </span>
          </div>
          <p class="mobile-generate-note">
            Empty keeps this checkpoint’s pipeline defaults. Used by two-stage, two-stage HQ,
            keyframe, and audio-to-video renders.
          </p>
          <div class="mobile-generate-field-grid">
            <label class="field mobile-generate-field">
              <span>STG scale</span>
              <input
                class="control"
                type="number"
                inputmode="decimal"
                step="0.1"
                min="0"
                :max="MAX_GUIDANCE_SCALE"
                placeholder="Default"
                data-test="mobile-ltx2-stg-scale"
                :value="guidance.stgScale ?? ''"
                @input="
                  setGuidance({ stgScale: numberOrNull(($event.target as HTMLInputElement).value) })
                "
              />
            </label>
            <label class="field mobile-generate-field">
              <span>STG blocks</span>
              <input
                class="control"
                type="text"
                inputmode="numeric"
                placeholder="28, 29"
                data-test="mobile-ltx2-stg-blocks"
                :aria-invalid="stgBlocksMessage ? 'true' : undefined"
                :value="guidance.stgBlocks"
                @input="setGuidance({ stgBlocks: ($event.target as HTMLInputElement).value })"
              />
            </label>
          </div>
          <p
            v-if="stgBlocksMessage"
            class="mobile-generate-validation"
            role="alert"
            data-test="mobile-ltx2-stg-blocks-error"
          >
            {{ stgBlocksMessage }}
          </p>
          <div class="mobile-generate-field-grid">
            <label class="field mobile-generate-field">
              <span>CFG rescale</span>
              <input
                class="control"
                type="number"
                inputmode="decimal"
                step="0.05"
                min="0"
                max="1"
                placeholder="Default"
                data-test="mobile-ltx2-rescale-scale"
                :value="guidance.rescaleScale ?? ''"
                @input="
                  setGuidance({
                    rescaleScale: numberOrNull(($event.target as HTMLInputElement).value),
                  })
                "
              />
            </label>
            <label class="field mobile-generate-field">
              <span>Modality scale</span>
              <input
                class="control"
                type="number"
                inputmode="decimal"
                step="0.1"
                min="0"
                :max="MAX_GUIDANCE_SCALE"
                placeholder="Default"
                data-test="mobile-ltx2-modality-scale"
                :value="guidance.modalityScale ?? ''"
                @input="
                  setGuidance({
                    modalityScale: numberOrNull(($event.target as HTMLInputElement).value),
                  })
                "
              />
            </label>
          </div>
          <label class="field mobile-generate-field">
            <span>Guidance skip stride</span>
            <input
              class="control"
              type="number"
              inputmode="numeric"
              step="1"
              min="0"
              :max="MAX_GUIDANCE_SKIP_STEP"
              placeholder="Every step"
              data-test="mobile-ltx2-guidance-skip-step"
              :aria-invalid="skipStepMessage ? 'true' : undefined"
              :value="guidance.skipStep ?? ''"
              @input="
                setGuidance({ skipStep: numberOrNull(($event.target as HTMLInputElement).value) })
              "
            />
          </label>
          <p
            v-if="skipStepMessage"
            class="mobile-generate-validation"
            role="alert"
            data-test="mobile-ltx2-guidance-skip-step-error"
          >
            {{ skipStepMessage }}
          </p>
        </div>

        <div v-if="form.pipeline === 'a2-vid'" class="mobile-generate-file-field">
          <span class="mobile-generate-label">Conditioning audio</span>
          <div v-if="form.audioFile" class="mobile-generate-picked-file">
            <span>{{ form.audioFile.filename }}</span>
            <button type="button" class="mobile-generate-clear" @click="form.audioFile = null">
              Remove
            </button>
          </div>
          <label v-else class="mobile-generate-file-button">
            <span>Choose audio</span>
            <input
              type="file"
              accept="audio/*"
              data-test="mobile-ltx2-audio-file"
              @change="setAudioFile"
            />
          </label>
          <p class="mobile-generate-note">Audio-to-video uses this track to drive the result.</p>
        </div>

        <div class="mobile-generate-file-field">
          <span class="mobile-generate-label">Source video</span>
          <div v-if="form.sourceVideo" class="mobile-generate-picked-file">
            <span>{{ form.sourceVideo.filename }}</span>
            <button type="button" class="mobile-generate-clear" @click="form.sourceVideo = null">
              Remove
            </button>
          </div>
          <label v-else class="mobile-generate-file-button">
            <span>Choose video</span>
            <input
              type="file"
              accept="video/*"
              data-test="mobile-ltx2-source-video"
              @change="setSourceVideo"
            />
          </label>
        </div>

        <div class="mobile-generate-file-field">
          <span class="mobile-generate-label">Keyframes</span>
          <label class="mobile-generate-file-button">
            <span>Add keyframe images</span>
            <input
              type="file"
              accept="image/png,image/jpeg"
              multiple
              data-test="mobile-ltx2-keyframe-file"
              @change="addKeyframes"
            />
          </label>
          <ol v-if="form.keyframes.length" class="mobile-generate-keyframes">
            <li
              v-for="(keyframe, index) in form.keyframes"
              :key="`${keyframe.image.filename}-${index}`"
              class="mobile-generate-keyframe"
              data-test="mobile-ltx2-keyframe-row"
            >
              <label>
                <span>Frame</span>
                <input
                  class="control"
                  type="number"
                  inputmode="numeric"
                  min="0"
                  :value="keyframe.frame"
                  :aria-label="`Keyframe ${index + 1} frame`"
                  :data-test="`mobile-ltx2-keyframe-frame-${index}`"
                  @input="updateKeyframeFrame(index, ($event.target as HTMLInputElement).value)"
                />
              </label>
              <span class="mobile-generate-keyframe-name">{{ keyframe.image.filename }}</span>
              <button
                type="button"
                class="mobile-generate-clear"
                :aria-label="`Remove keyframe ${index + 1}`"
                @click="removeKeyframe(index)"
              >
                Remove
              </button>
            </li>
          </ol>
        </div>

        <p
          v-if="advancedVideoError"
          class="mobile-generate-validation"
          role="alert"
          data-test="mobile-ltx2-validation-error"
        >
          {{ advancedVideoError }}
        </p>
        <p
          v-if="mediaReadError"
          class="mobile-generate-validation"
          role="alert"
          data-test="mobile-ltx2-media-error"
        >
          {{ mediaReadError }}
        </p>
      </div>
    </section>
  </section>
</template>
