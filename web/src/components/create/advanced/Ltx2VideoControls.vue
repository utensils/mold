<script setup lang="ts">
/*
 * LTX-2 advanced video suite — the family-specific controls the Advanced
 * drawer nests inside its "Video" section for LTX-2 / LTX-2.3 (the audio
 * families). Restores the full set the deleted GenerateParamsPanel offered:
 * pipeline mode, audio decode toggle + audio conditioning file, source video,
 * retake range, spatial/temporal upscale, and the keyframe editor. Every
 * control maps onto an existing GenerateFormState field (no renames); binary
 * uploads become SourceImage/SourceMediaState just like the source dropzone.
 * The component holds no local copy — it patches the parent form via v-model.
 */
import { computed, ref, watch } from "vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import type {
  GenerateFormState,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  SourceImageState,
  SourceMediaState,
} from "../../../types";
import { blobToBase64 } from "../../../lib/base64";

const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    audioOutputSupported?: boolean;
  }>(),
  { audioOutputSupported: true },
);
const emit = defineEmits<{ "update:modelValue": [value: GenerateFormState] }>();

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

// ── Audio decode toggle ───────────────────────────────────────────────
// enableAudio is boolean | null; null lets the server default (on for MP4).
// The switch reads on unless explicitly disabled, and writes true/false.
const audioOn = computed(
  () => props.audioOutputSupported && props.modelValue.enableAudio !== false,
);
function setAudio(on: boolean) {
  patch({ enableAudio: on });
}

// ── Pipeline mode ─────────────────────────────────────────────────────
const PIPELINE_OPTIONS: Ltx2PipelineMode[] = [
  "one-stage",
  "two-stage",
  "two-stage-hq",
  "distilled",
  "ic-lora",
  "keyframe",
  "a2vid",
  "retake",
];
const pipelineValue = computed(() => props.modelValue.pipeline ?? "");
function setPipeline(raw: string) {
  patch({ pipeline: (raw || null) as Ltx2PipelineMode | null });
}

// Camera-control LoRAs published for LTX-2 19B. LTX-2.3 uses a different
// architecture, so its users get the custom-path escape hatch only.
const CAMERA_MOTION_PRESETS = [
  { id: "dolly-in", label: "Dolly in" },
  { id: "dolly-left", label: "Dolly left" },
  { id: "dolly-out", label: "Dolly out" },
  { id: "dolly-right", label: "Dolly right" },
  { id: "jib-down", label: "Jib down" },
  { id: "jib-up", label: "Jib up" },
  { id: "static", label: "Static" },
] as const;
const isLtx23Model = computed(() => props.modelValue.model.includes("ltx-2.3"));
function cameraModeFor(value: string | null): string {
  if (!value) return "";
  return CAMERA_MOTION_PRESETS.some((preset) => preset.id === value)
    ? value
    : "custom";
}
const cameraMode = ref(cameraModeFor(props.modelValue.cameraControl));
watch(
  () => props.modelValue.cameraControl,
  (value) => {
    if (cameraMode.value === "custom" && cameraModeFor(value) === "custom")
      return;
    cameraMode.value = cameraModeFor(value);
  },
);
function setCameraMode(raw: string) {
  cameraMode.value = raw;
  if (raw === "custom") {
    if (cameraModeFor(props.modelValue.cameraControl) !== "custom")
      patch({ cameraControl: "" });
    return;
  }
  patch({ cameraControl: raw || null });
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
    <div class="ltx2__row">
      <span class="ltx2__label">Decode audio</span>
      <SwitchToggle
        :model-value="audioOn"
        :disabled="!audioOutputSupported"
        label="Decode audio"
        data-test="ltx2-enable-audio"
        @update:model-value="setAudio"
      />
    </div>
    <p
      v-if="!audioOutputSupported"
      class="ltx2__hint"
      data-test="ltx2-audio-unavailable"
    >
      Audio assets are not included with this checkpoint. Video generation
      remains available.
    </p>

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
          v-for="preset in CAMERA_MOTION_PRESETS"
          :key="preset.id"
          :value="preset.id"
          :disabled="isLtx23Model"
        >
          {{ preset.label }}
        </option>
        <option value="custom">Custom LoRA path…</option>
      </select>
      <input
        v-if="cameraMode === 'custom'"
        class="ltx2__input"
        data-test="ltx2-camera-motion-custom"
        placeholder="/path/to/lora.safetensors"
        :value="modelValue.cameraControl ?? ''"
        @input="
          patch({ cameraControl: ($event.target as HTMLInputElement).value })
        "
      />
      <p v-if="isLtx23Model" class="ltx2__hint">
        Presets are for LTX-2 19B; use a custom LoRA path for LTX-2.3.
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
</style>
