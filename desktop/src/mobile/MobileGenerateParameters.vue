<script setup lang="ts">
import { computed, ref, useId, watch } from "vue";
import type {
  Ltx2PipelineMode,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
} from "../lib/api/types";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { frames8n1Error, snapFrames } from "../lib/chain";
import {
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
  type AutoChainUnsupportedField,
  type ChainRoutingDecision,
} from "../lib/chainRouting";
import { buildRequest, type GenerateForm, type PickedImage } from "../lib/generateForm";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
  inlineGenerationMediaBytes,
  MAX_INLINE_GENERATION_MEDIA_BYTES,
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  type InlineGenerationMediaField,
} from "../lib/generateValidation";
import { fileToBase64, isStillImageFile } from "../lib/image";
import { cameraMotionMode } from "@studio/lib/cameraMotion";
import {
  canOfferExtend,
  extendNewFrames,
  extendOverlapOptions,
  extendValidationError,
  serverExtendOverlapDefault,
} from "@studio/lib/extend";
import {
  guidanceOverrideCount,
  skipStepError,
  stgBlocksError,
  MAX_GUIDANCE_SCALE,
  MAX_GUIDANCE_SKIP_STEP,
  type Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** Selected model row; carries the continuation capability. */
    selectedModel?: ModelEntry | null;
    upscalers?: ModelEntry[];
    audioOutputSupported?: boolean;
    controlAdapters?: Ltx2ControlAdapterInfo[];
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
  }>(),
  {
    selectedModel: null,
    upscalers: () => [],
    audioOutputSupported: true,
    controlAdapters: () => [],
    cameraControls: () => [],
    cameraControlsLoaded: false,
  },
);
const MAX_BATCH_SIZE = 10_000;

const emit = defineEmits<{
  "validity-change": [valid: boolean];
}>();

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const generationRequest = computed(() => buildRequest(props.form));
const frameError = computed(() =>
  caps.value.supportsVideo ? frames8n1Error(props.form.frames) : null,
);
const fpsError = computed(() =>
  caps.value.supportsVideo ? fpsValidationError(props.form.fps) : null,
);
const chainDecision = computed<ChainRoutingDecision>(() =>
  caps.value.supportsVideo
    ? decideGenerateRequestRouting(generationRequest.value, props.form.family)
    : { kind: "single" },
);

const AUTO_CHAIN_FIELD_LABELS: Record<AutoChainUnsupportedField, string> = {
  negative_prompt: "negative prompt",
  loras: "LoRAs or camera motion",
  audio_file: "conditioning audio",
  source_video: "source video",
  keyframes: "keyframes",
  pipeline: "pipeline",
  ic_lora_control: "reference control",
  retake_range: "retake range",
  spatial_upscale: "spatial upscale",
  temporal_upscale: "temporal upscale",
  guidance_overrides: "guidance overrides",
};

const chainCompatibilityError = computed(() => {
  if (chainDecision.value.kind !== "chain") return null;
  const unsupported = unsupportedAutoChainFields(generationRequest.value);
  if (unsupported.length === 0) return null;
  const labels = unsupported.map((field) => AUTO_CHAIN_FIELD_LABELS[field]);
  return `Long-video chaining can’t preserve ${labels.join(", ")}. Remove those options or reduce Frames to 97 or fewer.`;
});

const audioFormatError = computed(() => audioOutputValidationError(props.form));
const advancedVideoError = computed(() => advancedVideoValidationError(props.form));
const cameraError = computed(() => cameraControlValidationError(props.form));
const mediaReadError = ref("");

const valid = computed(
  () =>
    !frameError.value &&
    !fpsError.value &&
    chainDecision.value.kind !== "reject" &&
    !chainCompatibilityError.value &&
    !audioFormatError.value &&
    !advancedVideoError.value &&
    !cameraError.value,
);

watch(valid, (next) => emit("validity-change", next), { immediate: true });
watch(
  () => [caps.value.forcesBatchSizeOne, props.form.batchSize] as const,
  ([forced, batchSize]) => {
    const normalized = forced
      ? 1
      : Math.min(MAX_BATCH_SIZE, Math.max(1, Math.round(batchSize) || 1));
    if (batchSize !== normalized) props.form.batchSize = normalized;
  },
  { immediate: true },
);

function stepBatch(delta: -1 | 1): void {
  if (caps.value.forcesBatchSizeOne) return;
  props.form.batchSize = Math.min(MAX_BATCH_SIZE, Math.max(1, props.form.batchSize + delta));
}

function setBatch(raw: string): void {
  if (caps.value.forcesBatchSizeOne) return;
  const value = Number(raw);
  props.form.batchSize = Number.isFinite(value)
    ? Math.min(MAX_BATCH_SIZE, Math.max(1, Math.round(value)))
    : 1;
}

function snapFramesField(): void {
  props.form.frames = snapFrames(props.form.frames);
}

const schedulerLabels: Record<string, string> = {
  default: "Default",
  ddim: "DDIM",
  "euler-ancestral": "Euler ancestral",
  unipc: "UniPC",
};

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
    if (cameraMotionMode(props.form.cameraControl) !== "custom") props.form.cameraControl = "";
  } else {
    props.form.cameraControl = mode || null;
  }
}

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
  "a2vid",
  "retake",
];
const spatialOptions: Ltx2SpatialUpscale[] = ["x1-5", "x2"];
const temporalOptions: Ltx2TemporalUpscale[] = ["x2"];

function setPipeline(value: string): void {
  props.form.pipeline = (value || null) as Ltx2PipelineMode | null;
  if (props.form.pipeline !== "ic-lora") props.form.icLoraControl = null;
  if (props.form.pipeline !== "retake") props.form.retakeRange = null;
}
function setControlAdapter(value: string): void {
  props.form.icLoraControl = value || null;
  if (value) props.form.pipeline = "ic-lora";
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
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on iPhone.";
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
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on iPhone.";
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
const extendOverlap = computed(
  () => props.form.extendOverlapFrames ?? serverExtendOverlapDefault(props.selectedModel),
);
const extendOverlapChoices = computed(() => extendOverlapOptions(props.form.frames));
const extendError = computed(() =>
  props.form.extendVideo
    ? extendValidationError({
        overlapFrames: extendOverlap.value,
        frames: props.form.frames,
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
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on iPhone.";
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
    mediaReadError.value = "Combined generation media must be 45 MiB or smaller on iPhone.";
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

      <div class="mobile-generate-field">
        <div class="mobile-generate-label-row">
          <span class="mobile-generate-label">Batch</span>
          <span
            v-if="caps.forcesBatchSizeOne"
            class="mobile-generate-note"
            data-test="mobile-batch-locked"
          >
            Edit models render one at a time
          </span>
        </div>
        <div class="mobile-generate-stepper" role="group" aria-label="Batch size">
          <button
            type="button"
            class="mobile-generate-stepper-button"
            data-test="mobile-batch-decrement"
            aria-label="Decrease batch size"
            :disabled="caps.forcesBatchSizeOne || form.batchSize <= 1"
            @click="stepBatch(-1)"
          >
            −
          </button>
          <input
            class="mobile-generate-stepper-value"
            data-test="mobile-batch-value"
            type="number"
            inputmode="numeric"
            min="1"
            step="1"
            aria-label="Batch size"
            :value="form.batchSize"
            :disabled="caps.forcesBatchSizeOne"
            @change="setBatch(($event.target as HTMLInputElement).value)"
          />
          <button
            type="button"
            class="mobile-generate-stepper-button"
            data-test="mobile-batch-increment"
            aria-label="Increase batch size"
            :disabled="caps.forcesBatchSizeOne || form.batchSize >= MAX_BATCH_SIZE"
            @click="stepBatch(1)"
          >
            +
          </button>
        </div>
      </div>

      <label v-if="caps.supportsScheduler" class="field mobile-generate-field">
        <span>Scheduler</span>
        <select v-model="form.scheduler" class="control" data-test="mobile-scheduler">
          <option v-for="option in caps.schedulerOptions" :key="option" :value="option">
            {{ schedulerLabels[option] ?? option }}
          </option>
        </select>
      </label>

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

      <label v-if="!caps.supportsVideo && upscalers.length" class="field mobile-generate-field">
        <span>Upscale</span>
        <select v-model="form.upscaleModel" class="control" data-test="mobile-upscale">
          <option value="">Off</option>
          <option v-for="upscaler in upscalers" :key="upscaler.name" :value="upscaler.name">
            {{ upscaler.name }}{{ upscaler.downloaded ? "" : " (downloads on first use)" }}
          </option>
        </select>
      </label>
    </fieldset>

    <fieldset
      v-if="caps.supportsVideo"
      class="mobile-generate-section mobile-generate-video-options"
    >
      <legend class="mobile-generate-legend">Video</legend>

      <div class="mobile-generate-field-grid">
        <label class="field mobile-generate-field">
          <span>Frames</span>
          <input
            v-model.number="form.frames"
            class="control"
            data-test="mobile-frames"
            type="number"
            inputmode="numeric"
            min="1"
            step="8"
            :aria-invalid="frameError ? 'true' : undefined"
            :aria-describedby="frameError ? frameErrorId : undefined"
            @change="snapFramesField"
          />
        </label>

        <label class="field mobile-generate-field">
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
        v-if="chainDecision.kind === 'chain' && !chainCompatibilityError"
        class="mobile-generate-callout"
        data-test="mobile-chain-cue"
      >
        Will render as {{ chainDecision.stageCount }} chained clips of
        {{ chainDecision.clipFrames }} frames with a {{ chainDecision.motionTail }}-frame motion
        tail.
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
        v-else-if="chainCompatibilityError"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-chain-compatibility-error"
      >
        {{ chainCompatibilityError }}
      </p>

      <label v-if="caps.supportsAudio" class="mobile-generate-toggle-row">
        <span>
          <strong>Generate audio</strong>
          <small>Include a synchronized soundtrack when the model supports it.</small>
        </span>
        <input
          v-model="form.enableAudio"
          type="checkbox"
          :disabled="!audioOutputSupported"
          data-test="mobile-enable-audio"
        />
      </label>
      <p v-if="caps.supportsAudio && !audioOutputSupported" class="mobile-generate-validation">
        Audio assets are not included with this checkpoint. Video generation remains available.
      </p>
      <p
        v-if="audioFormatError"
        class="mobile-generate-validation"
        role="alert"
        data-test="mobile-audio-format-error"
      >
        {{ audioFormatError }}
      </p>

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
          <option v-for="preset in cameraControls" :key="preset.id" :value="preset.id">
            {{ preset.label }}{{ preset.installed ? "" : " · downloads on first use" }}
          </option>
          <option value="custom">Custom LoRA path…</option>
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
          @input="form.cameraControl = ($event.target as HTMLInputElement).value"
        />
      </label>
      <p
        v-if="caps.supportsAdvancedVideo && cameraControlsLoaded && cameraControls.length === 0"
        class="mobile-generate-note"
        data-test="mobile-camera-motion-19b-hint"
      >
        Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA
        path.
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

        <div v-if="form.pipeline === 'a2vid'" class="mobile-generate-file-field">
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
              @change="
                form.extendOverlapFrames = Number(($event.target as HTMLSelectElement).value)
              "
            >
              <option v-for="option in extendOverlapChoices" :key="option" :value="String(option)">
                {{ option }}
              </option>
            </select>
            <p
              v-if="extendError"
              class="mobile-generate-error"
              data-test="mobile-ltx2-extend-error"
            >
              {{ extendError }}
            </p>
            <p v-else class="mobile-generate-note" data-test="mobile-ltx2-extend-summary">
              {{ extendSummary }}
            </p>
          </template>
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
