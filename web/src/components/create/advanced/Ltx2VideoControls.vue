<script setup lang="ts">
/*
 * LTX-2 advanced video suite — the family-specific controls the Advanced
 * drawer nests inside its "Video" section for LTX-2 / LTX-2.3 (the audio
 * families). Restores the full set the deleted GenerateParamsPanel offered:
 * pipeline mode, audio conditioning file, source video,
 * retake range, spatial/temporal upscale, and the keyframe editor. Every
 * control maps onto an existing GenerateFormState field (no renames); binary
 * uploads become SourceImage/SourceMediaState just like the source dropzone.
 * The component holds no local copy — it patches the parent form via v-model.
 */
import { computed, ref, watch } from "vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import type {
  GenerateFormState,
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  Ltx2GuidanceOverridesState,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  SourceImageState,
  SourceMediaState,
} from "../../../types";
import { MAX_LORA_STACK } from "../../../types";
import { blobToBase64 } from "../../../lib/base64";
import {
  cameraMotionLoraPath,
  cameraMotionLoraSlotAvailable,
  cameraMotionMode,
  isCameraMotionPreset,
  parseCameraControlAvailability,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import {
  LIP_DUB_TIMING_HINT,
  isControlAdapterPipeline,
  pipelineForControlId,
} from "@studio/lib/ltx2Control";
import {
  AUDIO_ONLY_PIPELINE,
  isAudioOnlyPipeline,
} from "@studio/lib/ltx2Pipeline";
import {
  emptyGuidanceOverrides,
  guidanceOverrideCount,
  stgBlocksError,
  skipStepError,
  MAX_GUIDANCE_SCALE,
  MAX_GUIDANCE_SKIP_STEP,
} from "@studio/lib/guidanceOverrides";

const props = defineProps<{ modelValue: GenerateFormState }>();
const emit = defineEmits<{ "update:modelValue": [value: GenerateFormState] }>();
const controlAdapters = ref<Ltx2ControlAdapterInfo[]>([]);
const cameraControls = ref<Ltx2CameraControlInfo[]>([]);
const cameraControlsLoaded = ref(false);
const cameraUnsupportedReason = ref<string | null>(null);
let controlEpoch = 0;
watch(
  () => props.modelValue.model,
  async (model) => {
    const epoch = ++controlEpoch;
    // Drop the previous model's reason immediately; keeping it while the
    // new request is in flight shows a stale explanation for the wrong model.
    cameraUnsupportedReason.value = null;
    controlAdapters.value = [];
    cameraControls.value = [];
    cameraUnsupportedReason.value = null;
    cameraControlsLoaded.value = false;
    if (!model) return;
    const read = async <T,>(path: string): Promise<T> => {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return (await response.json()) as T;
    };
    const controlsRequest = read<Ltx2ControlAdapterInfo[]>(
      `/api/capabilities/ltx2-control-adapters?model=${encodeURIComponent(model)}`,
    )
      .then((options) => {
        if (epoch !== controlEpoch) return;
        controlAdapters.value = options;
        if (
          props.modelValue.icLoraControl &&
          !options.some(
            (adapter) => adapter.id === props.modelValue.icLoraControl,
          )
        ) {
          patch({ icLoraControl: null });
        }
      })
      .catch(() => {
        if (epoch !== controlEpoch) return;
        controlAdapters.value = [];
        patch({ icLoraControl: null });
      });
    const camerasRequest = read<unknown>(
      `/api/capabilities/ltx2-camera-controls?model=${encodeURIComponent(model)}&detail=1`,
    )
      .then((body) => {
        if (epoch !== controlEpoch) return;
        const availability = parseCameraControlAvailability(body);
        const cameras = availability.controls;
        cameraControls.value = cameras;
        cameraUnsupportedReason.value = availability.unsupportedReason;
        cameraControlsLoaded.value = true;
        if (
          props.modelValue.cameraControl &&
          isCameraMotionPreset(props.modelValue.cameraControl) &&
          !cameras.some(
            (camera) => camera.id === props.modelValue.cameraControl,
          )
        ) {
          setCameraControl(null);
        }
      })
      .catch(() => {
        if (epoch !== controlEpoch) return;
        cameraControls.value = [];
        cameraUnsupportedReason.value = null;
        cameraControlsLoaded.value = false;
      });
    await Promise.allSettled([controlsRequest, camerasRequest]);
  },
  { immediate: true },
);

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

function setCameraControl(next: string | null) {
  if (
    cameraMotionLoraPath(next) &&
    !cameraMotionLoraSlotAvailable(
      props.modelValue.loras,
      props.modelValue.cameraControl,
      MAX_LORA_STACK,
    )
  ) {
    return;
  }
  patch({
    cameraControl: next,
    loras: syncCameraMotionLora(
      props.modelValue.loras,
      props.modelValue.cameraControl,
      next,
      (path, scale) => ({ path, scale, trainedWords: [] }),
      MAX_LORA_STACK,
    ),
  });
}

const cameraSlotAvailable = computed(() =>
  cameraMotionLoraSlotAvailable(
    props.modelValue.loras,
    props.modelValue.cameraControl,
    MAX_LORA_STACK,
  ),
);

// ── Pipeline mode ─────────────────────────────────────────────────────
const PIPELINE_OPTIONS: Ltx2PipelineMode[] = [
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
const pipelineValue = computed(() => props.modelValue.pipeline ?? "");
const audioOnlyPipeline = computed(() =>
  isAudioOnlyPipeline(props.modelValue.pipeline),
);
function setPipeline(raw: string) {
  const pipeline = (raw || null) as Ltx2PipelineMode | null;
  const audioOnly = isAudioOnlyPipeline(pipeline);
  patch({
    pipeline,
    icLoraControl: isControlAdapterPipeline(pipeline)
      ? props.modelValue.icLoraControl
      : null,
    // `t2a` renders no frames, so the server rejects every video container.
    // Move the format with the mode rather than leaving the user to discover
    // the mismatch as a 422; leaving `t2a` restores the family default.
    outputFormat: audioOnly
      ? "wav"
      : props.modelValue.outputFormat === "wav"
        ? "mp4"
        : props.modelValue.outputFormat,
  });
}
function setControlAdapter(raw: string) {
  patch({
    icLoraControl: raw || null,
    // Lip dub is a pipeline of its own; every other adapter drives `ic-lora`.
    pipeline: raw ? pipelineForControlId(raw) : props.modelValue.pipeline,
  });
}
const showLipDubTimingHint = computed(
  () => props.modelValue.pipeline === "lip-dub",
);

const cameraMode = ref(cameraMotionMode(props.modelValue.cameraControl));
watch(
  () => props.modelValue.cameraControl,
  (value) => {
    if (cameraMode.value === "custom" && cameraMotionMode(value) === "custom")
      return;
    cameraMode.value = cameraMotionMode(value);
  },
);
function setCameraMode(raw: string) {
  cameraMode.value = raw;
  if (raw === "custom") {
    if (cameraMotionMode(props.modelValue.cameraControl) !== "custom")
      setCameraControl("");
    return;
  }
  setCameraControl(raw || null);
}

// ── Spatial / temporal upscale (segmented; "" = native) ───────────────
const spatialOptions = [
  { value: "", label: "Native" },
  { value: "x1-5", label: "1.5×" },
  { value: "x2", label: "2×" },
] as const;
const temporalOptions = [
  { value: "", label: "Native" },
  { value: "x2", label: "2×" },
] as const;
const spatialValue = computed(() => props.modelValue.spatialUpscale ?? "");
const temporalValue = computed(() => props.modelValue.temporalUpscale ?? "");
function setSpatial(raw: string) {
  patch({ spatialUpscale: (raw || null) as Ltx2SpatialUpscale | null });
}
function setTemporal(raw: string) {
  patch({ temporalUpscale: (raw || null) as Ltx2TemporalUpscale | null });
}

// ── Media file readers (upload → base64, matching the source dropzone) ─
async function readFile(event: Event): Promise<File | null> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = "";
  return file;
}
async function readImage(event: Event): Promise<SourceImageState | null> {
  const file = await readFile(event);
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}
async function readMedia(event: Event): Promise<SourceMediaState | null> {
  const file = await readFile(event);
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}

async function onAudioFile(event: Event) {
  const audioFile = await readMedia(event);
  if (!audioFile) return;
  patch({ audioFile, audioFilePath: "" });
}
function clearAudioFile() {
  patch({ audioFile: null });
}
async function onSourceVideo(event: Event) {
  const sourceVideo = await readMedia(event);
  if (!sourceVideo) return;
  patch({ sourceVideo, sourceVideoPath: "" });
}
function clearSourceVideo() {
  patch({ sourceVideo: null });
}
// ── Retake range ──────────────────────────────────────────────────────
function setRetakeStart(raw: string) {
  patch({
    retakeRange: {
      start_seconds: Number(raw) || 0,
      end_seconds: props.modelValue.retakeRange?.end_seconds ?? 1,
    },
  });
}
function setRetakeEnd(raw: string) {
  patch({
    retakeRange: {
      start_seconds: props.modelValue.retakeRange?.start_seconds ?? 0,
      end_seconds: Number(raw) || 1,
    },
  });
}

// ── Guidance overrides ────────────────────────────────────────────────
// Each control writes null when cleared, because the engine keeps its
// per-pipeline constant only while the field is absent from the request.
// A template saved before this control existed restores without the field,
// so every read goes through one defaulted view of it.
const guidance = computed<Ltx2GuidanceOverridesState>(
  () => props.modelValue.guidanceOverrides ?? emptyGuidanceOverrides(),
);
function setGuidance(next: Partial<Ltx2GuidanceOverridesState>) {
  patch({ guidanceOverrides: { ...guidance.value, ...next } });
}
function numberOrNull(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}
const stgBlocksMessage = computed(() =>
  stgBlocksError(guidance.value.stgBlocks),
);
const skipStepMessage = computed(() => skipStepError(guidance.value.skipStep));
const guidanceCount = computed(() => guidanceOverrideCount(guidance.value));
function resetGuidance() {
  patch({ guidanceOverrides: emptyGuidanceOverrides() });
}

// ── Keyframe editor ───────────────────────────────────────────────────
async function addKeyframe(event: Event) {
  const image = await readImage(event);
  if (!image) return;
  const last = props.modelValue.keyframes.at(-1);
  const frame = last ? last.frame + 24 : 0;
  patch({ keyframes: [...props.modelValue.keyframes, { frame, image }] });
}
function updateKeyframeFrame(index: number, raw: string) {
  const frame = Math.max(0, Math.round(Number(raw) || 0));
  const keyframes = props.modelValue.keyframes.slice();
  const item = keyframes[index];
  if (!item) return;
  keyframes[index] = { ...item, frame };
  patch({ keyframes });
}
function removeKeyframe(index: number) {
  const keyframes = props.modelValue.keyframes.slice();
  keyframes.splice(index, 1);
  patch({ keyframes });
}
</script>

<template>
  <div class="ltx2" data-test="ltx2-suite">
    <div class="ltx2__field">
      <label class="ltx2__label">Reference control</label>
      <select
        class="ltx2__select"
        data-test="ltx2-reference-control"
        :value="modelValue.icLoraControl ?? ''"
        @change="setControlAdapter(($event.target as HTMLSelectElement).value)"
      >
        <option value="">Custom / none</option>
        <option
          v-for="adapter in controlAdapters"
          :key="adapter.id"
          :value="adapter.id"
        >
          {{ adapter.label }}{{ adapter.installed ? "" : " · download" }}
        </option>
      </select>
      <p
        v-if="modelValue.icLoraControl"
        class="ltx2__hint"
        data-test="ltx2-reference-guide"
      >
        {{
          controlAdapters.find(
            (adapter) => adapter.id === modelValue.icLoraControl,
          )?.guide ??
          "Choose the frame-aligned guide video this control expects."
        }}
      </p>
    </div>
    <div class="ltx2__field">
      <label class="ltx2__label">Pipeline</label>
      <select
        class="ltx2__select"
        data-test="ltx2-pipeline"
        :value="pipelineValue"
        @change="setPipeline(($event.target as HTMLSelectElement).value)"
      >
        <option value="">Auto</option>
        <option v-for="p in PIPELINE_OPTIONS" :key="p" :value="p">
          {{ p }}
        </option>
      </select>
      <p
        v-if="showLipDubTimingHint"
        class="ltx2__hint"
        data-test="ltx2-lip-dub-hint"
      >
        {{ LIP_DUB_TIMING_HINT }}
      </p>
      <p v-if="audioOnlyPipeline" class="ltx2__hint" data-test="ltx2-t2a-hint">
        Text-to-audio renders sound only — no video. Duration comes from frames
        &divide; fps, and the output is a WAV. Source media, keyframes, retake,
        and the upscalers do not apply.
      </p>
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Camera motion</label>
      <select
        class="ltx2__select"
        data-test="ltx2-camera-motion"
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
          {{ preset.label
          }}{{ preset.installed ? "" : " · downloads on first use" }}
        </option>
        <option value="custom" :disabled="!cameraSlotAvailable">
          Custom LoRA path…
        </option>
      </select>
      <input
        v-if="cameraMode === 'custom'"
        class="ltx2__input"
        data-test="ltx2-camera-motion-custom"
        placeholder="/path/to/lora.safetensors"
        :value="modelValue.cameraControl ?? ''"
        @input="setCameraControl(($event.target as HTMLInputElement).value)"
      />
      <p
        v-if="cameraControlsLoaded && cameraControls.length === 0"
        class="ltx2__hint"
        data-test="ltx2-camera-motion-19b-hint"
      >
        {{
          cameraUnsupportedReason ??
          "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
        }}
      </p>
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Audio conditioning</label>
      <label v-if="!modelValue.audioFile" class="ltx2__file">
        <span>Attach audio</span>
        <input
          type="file"
          accept="audio/*"
          class="ltx2__file-input"
          data-test="ltx2-audio-attach"
          @change="onAudioFile"
        />
      </label>
      <div v-else class="ltx2__source-row">
        <span class="ltx2__source-name">{{
          modelValue.audioFile.filename
        }}</span>
        <button
          type="button"
          class="ltx2__remove"
          data-test="ltx2-audio-clear"
          @click="clearAudioFile"
        >
          Remove
        </button>
      </div>
      <input
        class="ltx2__input"
        data-test="ltx2-audio-path"
        placeholder="or server audio path"
        :value="modelValue.audioFilePath"
        :disabled="modelValue.audioFile !== null"
        @input="
          patch({ audioFilePath: ($event.target as HTMLInputElement).value })
        "
      />
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Source video</label>
      <label v-if="!modelValue.sourceVideo" class="ltx2__file">
        <span>Attach video</span>
        <input
          type="file"
          accept="video/*"
          class="ltx2__file-input"
          data-test="ltx2-video-attach"
          @change="onSourceVideo"
        />
      </label>
      <div v-else class="ltx2__source-row">
        <span class="ltx2__source-name">{{
          modelValue.sourceVideo.filename
        }}</span>
        <button
          type="button"
          class="ltx2__remove"
          data-test="ltx2-video-clear"
          @click="clearSourceVideo"
        >
          Remove
        </button>
      </div>
      <input
        class="ltx2__input"
        data-test="ltx2-video-path"
        placeholder="or server video path"
        :value="modelValue.sourceVideoPath"
        :disabled="modelValue.sourceVideo !== null"
        @input="
          patch({ sourceVideoPath: ($event.target as HTMLInputElement).value })
        "
      />
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Retake range (seconds)</label>
      <div class="ltx2__pair">
        <input
          class="ltx2__input"
          type="number"
          step="0.1"
          min="0"
          placeholder="Start"
          data-test="ltx2-retake-start"
          :value="modelValue.retakeRange?.start_seconds ?? ''"
          @input="setRetakeStart(($event.target as HTMLInputElement).value)"
        />
        <input
          class="ltx2__input"
          type="number"
          step="0.1"
          min="0"
          placeholder="End"
          data-test="ltx2-retake-end"
          :value="modelValue.retakeRange?.end_seconds ?? ''"
          @input="setRetakeEnd(($event.target as HTMLInputElement).value)"
        />
      </div>
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Spatial upscale</label>
      <SegmentedControl
        :model-value="spatialValue"
        :options="spatialOptions"
        label="Spatial upscale"
        data-test="ltx2-spatial"
        @update:model-value="setSpatial"
      />
    </div>

    <div class="ltx2__field">
      <label class="ltx2__label">Temporal upscale</label>
      <SegmentedControl
        :model-value="temporalValue"
        :options="temporalOptions"
        label="Temporal upscale"
        data-test="ltx2-temporal"
        @update:model-value="setTemporal"
      />
    </div>

    <div class="ltx2__field">
      <div class="ltx2__keyhead">
        <label class="ltx2__label">
          Guidance overrides
          <span
            v-if="guidanceCount"
            class="ltx2__count"
            data-test="ltx2-guidance-count"
            >{{ guidanceCount }}</span
          >
        </label>
        <button
          v-if="guidanceCount"
          type="button"
          class="ltx2__reset"
          data-test="ltx2-guidance-reset"
          @click="resetGuidance"
        >
          Reset
        </button>
      </div>
      <p class="ltx2__hint">
        Empty keeps this checkpoint's pipeline defaults. Only two-stage,
        two-stage HQ, keyframe, and a2-vid renders use these.
      </p>
      <div class="ltx2__pair">
        <label class="ltx2__sublabel">
          STG scale
          <input
            class="ltx2__input"
            type="number"
            step="0.1"
            min="0"
            :max="MAX_GUIDANCE_SCALE"
            placeholder="Default"
            data-test="ltx2-stg-scale"
            :value="guidance.stgScale ?? ''"
            @input="
              setGuidance({
                stgScale: numberOrNull(
                  ($event.target as HTMLInputElement).value,
                ),
              })
            "
          />
        </label>
        <label class="ltx2__sublabel">
          STG blocks
          <input
            class="ltx2__input"
            type="text"
            inputmode="numeric"
            placeholder="e.g. 28, 29"
            data-test="ltx2-stg-blocks"
            :value="guidance.stgBlocks"
            @input="
              setGuidance({
                stgBlocks: ($event.target as HTMLInputElement).value,
              })
            "
          />
        </label>
      </div>
      <p
        v-if="stgBlocksMessage"
        class="ltx2__error"
        data-test="ltx2-stg-blocks-error"
      >
        {{ stgBlocksMessage }}
      </p>
      <div class="ltx2__pair ltx2__pair--spaced">
        <label class="ltx2__sublabel">
          CFG rescale
          <input
            class="ltx2__input"
            type="number"
            step="0.05"
            min="0"
            max="1"
            placeholder="Default"
            data-test="ltx2-rescale-scale"
            :value="guidance.rescaleScale ?? ''"
            @input="
              setGuidance({
                rescaleScale: numberOrNull(
                  ($event.target as HTMLInputElement).value,
                ),
              })
            "
          />
        </label>
        <label class="ltx2__sublabel">
          Modality scale
          <input
            class="ltx2__input"
            type="number"
            step="0.1"
            min="0"
            :max="MAX_GUIDANCE_SCALE"
            placeholder="Default"
            data-test="ltx2-modality-scale"
            :value="guidance.modalityScale ?? ''"
            @input="
              setGuidance({
                modalityScale: numberOrNull(
                  ($event.target as HTMLInputElement).value,
                ),
              })
            "
          />
        </label>
      </div>
      <label class="ltx2__sublabel ltx2__sublabel--spaced">
        Guidance skip stride
        <input
          class="ltx2__input"
          type="number"
          step="1"
          min="0"
          :max="MAX_GUIDANCE_SKIP_STEP"
          placeholder="Every step"
          data-test="ltx2-guidance-skip-step"
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
        class="ltx2__error"
        data-test="ltx2-guidance-skip-step-error"
      >
        {{ skipStepMessage }}
      </p>
    </div>

    <div class="ltx2__field ltx2__field--last">
      <div class="ltx2__keyhead">
        <label class="ltx2__label">Keyframes</label>
        <label class="ltx2__addkey" data-test="ltx2-keyframe-add-label">
          Add
          <input
            type="file"
            accept="image/png,image/jpeg"
            class="ltx2__file-input"
            data-test="ltx2-keyframe-add"
            @change="addKeyframe"
          />
        </label>
      </div>
      <div
        v-for="(keyframe, index) in modelValue.keyframes"
        :key="`${keyframe.image.filename}-${index}`"
        class="ltx2__keyrow"
        data-test="ltx2-keyframe-row"
      >
        <input
          class="ltx2__input ltx2__keyframe-input"
          type="number"
          min="0"
          data-test="ltx2-keyframe-frame"
          :value="keyframe.frame"
          @input="
            updateKeyframeFrame(
              index,
              ($event.target as HTMLInputElement).value,
            )
          "
        />
        <span class="ltx2__source-name">{{ keyframe.image.filename }}</span>
        <button
          type="button"
          class="ltx2__remove"
          data-test="ltx2-keyframe-remove"
          @click="removeKeyframe(index)"
        >
          Remove
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ltx2 {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--edge);
}
.ltx2__field {
  margin-bottom: 14px;
}
.ltx2__field--last {
  margin-bottom: 0;
}
.ltx2__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.ltx2__label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ltx2__row .ltx2__label {
  margin-bottom: 0;
}
.ltx2__select,
.ltx2__input {
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
.ltx2__input:disabled {
  opacity: 0.5;
}
.ltx2__pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.ltx2__file {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  padding: 14px;
  font-size: 12.5px;
  cursor: pointer;
  margin-bottom: 8px;
}
.ltx2__file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}
.ltx2__source-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.ltx2__source-name {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ltx2__remove {
  border: 0;
  background: transparent;
  color: var(--stop);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.ltx2__keyhead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.ltx2__keyhead .ltx2__label {
  margin-bottom: 0;
}
.ltx2__addkey {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.ltx2__keyrow {
  display: grid;
  grid-template-columns: 5rem 1fr auto;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.ltx2__keyframe-input {
  height: 34px;
}
.ltx2__count {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: var(--radius-pill);
  background: var(--bench);
  border: 1px solid var(--ce);
  font-size: 11px;
  font-weight: 600;
}
.ltx2__reset {
  border: none;
  background: transparent;
  color: var(--ink-2);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  padding: 0;
}
.ltx2__sublabel {
  display: block;
  font-size: 11.5px;
  color: var(--ink-2);
  font-weight: 600;
}
.ltx2__sublabel .ltx2__input {
  margin-top: 6px;
}
.ltx2__sublabel--spaced {
  margin-top: 8px;
}
.ltx2__pair--spaced {
  margin-top: 8px;
}
.ltx2__error {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--stop);
}
</style>
