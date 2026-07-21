<script setup lang="ts">
import { computed, ref, watch } from "vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import SegmentedControl, { type SegmentOption } from "@ui/components/SegmentedControl.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Chip from "@ui/components/Chip.vue";
import {
  applyModelDefaults,
  buildRequest,
  newGenerateForm,
  seedMode,
  type GenerateForm,
  type PickedImage,
} from "../../lib/generateForm";
import type {
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
  OutputFormat,
} from "../../lib/api/types";
import { generationCapabilitiesForFamily, outputFormatsForFamily } from "../../lib/capabilities";
import { frames8n1Error, snapFrames } from "../../lib/chain";
import {
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
  type ChainRoutingDecision,
} from "../../lib/chainRouting";
import { fileToBase64 } from "../../lib/image";
import {
  advancedVideoValidationError,
  audioOutputValidationError,
  cameraControlValidationError,
  fpsValidationError,
} from "../../lib/generateValidation";
import { advancedActiveCount } from "../../lib/advancedCount";
import SourceImageWell from "../generate/SourceImageWell.vue";
import LoraStack from "../generate/LoraStack.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    form: GenerateForm;
    /** The picked model, used by Reset to restore its defaults. */
    selectedModel?: ModelEntry | null;
    upscalers?: ModelEntry[];
  }>(),
  { selectedModel: null, upscalers: () => [] },
);

const emit = defineEmits<{ close: []; "append-word": [word: string] }>();

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const formats = computed(() => outputFormatsForFamily(props.form.family));
const advancedCount = computed(() => advancedActiveCount(props.form));

// One section open at a time.
const openSection = ref<string | null>(null);
function toggle(id: string) {
  openSection.value = openSection.value === id ? null : id;
}

// ── Scheduler & sampling ─────────────────────────────────────────────────────
const schedulerLabels: Record<string, string> = {
  default: "Default",
  ddim: "DDIM",
  "euler-ancestral": "Euler ancestral",
  unipc: "UniPC",
};
const schedulerSummary = computed(
  () => schedulerLabels[props.form.scheduler] ?? props.form.scheduler,
);

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

// ── Video (ltx families) ─────────────────────────────────────────────────────
const framesError = computed(() => frames8n1Error(props.form.frames));
const generationRequest = computed(() => buildRequest(props.form));
const chainDecision = computed<ChainRoutingDecision>(() =>
  caps.value.supportsVideo
    ? decideGenerateRequestRouting(generationRequest.value, props.form.family)
    : { kind: "single" },
);
const chainCompatibilityError = computed(() => {
  if (chainDecision.value.kind !== "chain") return null;
  return unsupportedAutoChainFields(generationRequest.value).length > 0
    ? "Long-video chaining can’t preserve the selected advanced options. Remove them or reduce Frames to 97 or fewer."
    : null;
});
const fpsError = computed(() =>
  caps.value.supportsVideo ? fpsValidationError(props.form.fps) : null,
);
const cameraError = computed(() => cameraControlValidationError(props.form));
const audioFormatError = computed(() => audioOutputValidationError(props.form));
const advancedVideoError = computed(() => advancedVideoValidationError(props.form));

const CAMERA_MOTION_PRESETS = [
  { id: "dolly-in", label: "Dolly in" },
  { id: "dolly-left", label: "Dolly left" },
  { id: "dolly-out", label: "Dolly out" },
  { id: "dolly-right", label: "Dolly right" },
  { id: "jib-down", label: "Jib down" },
  { id: "jib-up", label: "Jib up" },
  { id: "static", label: "Static" },
] as const;
const isLtx23Model = computed(() => props.form.model.includes("ltx-2.3"));
function cameraModeFor(value: string | null): string {
  if (!value) return "";
  return CAMERA_MOTION_PRESETS.some((p) => p.id === value) ? value : "custom";
}
const uiCameraMode = ref(cameraModeFor(props.form.cameraControl));
watch(
  () => props.form.cameraControl,
  (value) => {
    if (uiCameraMode.value === "custom" && (value === "" || cameraModeFor(value) === "custom")) {
      return;
    }
    uiCameraMode.value = cameraModeFor(value);
  },
);
function setCameraMode(mode: string) {
  uiCameraMode.value = mode;
  if (mode === "custom") {
    if (cameraModeFor(props.form.cameraControl) !== "custom") props.form.cameraControl = "";
  } else {
    props.form.cameraControl = mode === "" ? null : mode;
  }
}

const advancedOpen = ref(false);
const keyframePickerOpen = ref(false);
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
function setPipeline(v: string) {
  props.form.pipeline = (v || null) as Ltx2PipelineMode | null;
  if (props.form.pipeline !== "retake") props.form.retakeRange = null;
}
function setSpatial(v: string) {
  props.form.spatialUpscale = (v || null) as Ltx2SpatialUpscale | null;
}
function setTemporal(v: string) {
  props.form.temporalUpscale = (v || null) as Ltx2TemporalUpscale | null;
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
async function setAudioFile(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  props.form.audioFile = { filename: file.name, base64: await fileToBase64(file) };
}
function clearAudioFile() {
  props.form.audioFile = null;
}
function snapFramesField() {
  props.form.frames = snapFrames(props.form.frames);
}

// ── Reset — restore the model's defaults, preserve prompt + prepared state ────
function reset() {
  const { prompt, originalPrompt, batchSize, model, family } = props.form;
  Object.assign(props.form, newGenerateForm());
  if (props.selectedModel) applyModelDefaults(props.form, props.selectedModel);
  else {
    props.form.model = model;
    props.form.family = family;
  }
  props.form.prompt = prompt;
  props.form.originalPrompt = originalPrompt;
  props.form.batchSize = generationCapabilitiesForFamily(props.form.family).forcesBatchSizeOne
    ? 1
    : batchSize;
}
</script>

<template>
  <DrawerPanel :open="open" :width="560" title="Advanced" @close="emit('close')">
    <template #header>
      <div class="ms-adv__head">
        <div>
          <div class="ms-adv__title">Advanced</div>
          <div class="ms-adv__subtitle">Fine controls, tucked away until you need them</div>
        </div>
        <BadgePill v-if="advancedCount > 0" tone="accent" data-test="advanced-active"
          >{{ advancedCount }} active</BadgePill
        >
      </div>
    </template>

    <div class="ms-adv__list">
      <!-- 1 · Scheduler & sampling -->
      <AccordionSection
        v-if="caps.supportsScheduler || caps.supportsCfgPlus"
        icon="scheduler"
        title="Scheduler &amp; sampling"
        :summary="schedulerSummary"
        :open="openSection === 'scheduler'"
        @toggle="toggle('scheduler')"
      >
        <template v-if="caps.supportsScheduler">
          <label class="ms-label">Scheduler</label>
          <select v-model="form.scheduler" class="ms-select" aria-label="Scheduler">
            <option v-for="opt in caps.schedulerOptions" :key="opt" :value="opt">
              {{ schedulerLabels[opt] ?? opt }}
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

      <!-- 2 · Negative prompt -->
      <AccordionSection
        v-if="caps.supportsNegativePrompt"
        icon="negative"
        title="Negative prompt"
        summary="What to steer away from"
        :open="openSection === 'negative'"
        @toggle="toggle('negative')"
      >
        <textarea
          v-model="form.negativePrompt"
          data-selectable
          rows="2"
          placeholder="blurry, low quality, deformed…"
          class="ms-textarea"
          aria-label="Negative prompt"
        />
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

      <!-- 3 · Source image -->
      <AccordionSection
        v-if="caps.supportsImg2img"
        icon="image"
        title="Source image"
        summary="Image-to-image &amp; inpainting"
        :open="openSection === 'source'"
        @toggle="toggle('source')"
      >
        <SourceImageWell :form="form" />
      </AccordionSection>

      <!-- 4 · LoRA stack -->
      <AccordionSection
        v-if="caps.supportsLora"
        icon="layers"
        title="LoRA stack"
        summary="Style adapters"
        :open="openSection === 'lora'"
        @toggle="toggle('lora')"
      >
        <LoraStack :form="form" :model="form.model" @append-word="emit('append-word', $event)" />
      </AccordionSection>

      <!-- 5 · Upscale after generate -->
      <AccordionSection
        v-if="!caps.supportsVideo && upscalers.length"
        icon="upscale"
        title="Upscale after generate"
        :summary="form.upscaleModel || 'Off'"
        :open="openSection === 'upscale'"
        @toggle="toggle('upscale')"
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
        :open="openSection === 'output'"
        @toggle="toggle('output')"
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
        :open="openSection === 'video'"
        @toggle="toggle('video')"
      >
        <label class="ms-label">Frames</label>
        <input
          v-model.number="form.frames"
          type="number"
          step="8"
          min="1"
          aria-label="Frames"
          :aria-invalid="framesError ? 'true' : undefined"
          class="ms-input data-mono"
          :class="framesError ? 'ms-input--error' : ''"
          @change="snapFramesField"
        />
        <p v-if="framesError" class="ms-error">{{ framesError }}</p>

        <div
          v-if="chainDecision.kind === 'chain' && !chainCompatibilityError"
          data-test="chain-cue"
          class="ms-cue"
        >
          Will render as
          <span class="data-mono font-semibold text-ink">{{ chainDecision.stageCount }}</span>
          chained clips of {{ chainDecision.clipFrames }} frames (motion-tail
          {{ chainDecision.motionTail }}) — expect this to take substantially longer than a single
          clip.
        </div>
        <p
          v-else-if="chainDecision.kind === 'reject'"
          data-test="chain-reject"
          class="ms-error ms-error--mt"
        >
          {{ chainDecision.reason }}
        </p>
        <p
          v-else-if="chainCompatibilityError"
          data-test="chain-compatibility-error"
          class="ms-error ms-error--mt"
        >
          {{ chainCompatibilityError }}
        </p>

        <label class="ms-label ms-label--mt">FPS</label>
        <input
          v-model.number="form.fps"
          type="number"
          min="1"
          max="60"
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
              v-for="p in CAMERA_MOTION_PRESETS"
              :key="p.id"
              :value="p.id"
              :disabled="isLtx23Model"
            >
              {{ p.label }}
            </option>
            <option value="custom">Custom LoRA path…</option>
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
            @input="form.cameraControl = ($event.target as HTMLInputElement).value"
          />
          <p v-if="isLtx23Model" data-test="camera-motion-23-hint" class="ms-hint">
            Presets are published for LTX-2 19B only — use a custom LoRA path for LTX-2.3.
          </p>
          <p v-if="cameraError" data-test="camera-motion-error" class="ms-error" role="alert">
            {{ cameraError }}
          </p>
        </template>

        <div v-if="caps.supportsAudio" class="ms-switch-row ms-switch-row--mt">
          <div class="ms-switch-row__title">Generate audio</div>
          <SwitchToggle
            :model-value="form.enableAudio"
            label="Generate audio"
            @update:model-value="form.enableAudio = $event"
          />
        </div>
        <p v-if="audioFormatError" class="ms-error" role="alert">{{ audioFormatError }}</p>

        <template v-if="caps.supportsAdvancedVideo">
          <button
            type="button"
            class="ms-disclosure"
            data-test="ltx2-disclosure"
            :aria-expanded="advancedOpen"
            @click="advancedOpen = !advancedOpen"
          >
            {{ advancedOpen ? "▾" : "▸" }} LTX-2 pipeline
          </button>
          <p
            v-if="advancedVideoError"
            data-test="ltx2-validation-error"
            class="ms-error"
            role="alert"
          >
            {{ advancedVideoError }}
          </p>
          <div v-if="advancedOpen">
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

            <template v-if="form.pipeline === 'a2vid'">
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
              <p class="ms-hint">a2vid needs an audio track to drive the video.</p>
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

    <template #footer>
      <div class="ms-adv__footer">
        <button type="button" class="ms-adv__reset" data-test="advanced-reset" @click="reset">
          Reset
        </button>
        <div class="ms-adv__spacer" />
        <button type="button" class="ms-adv__done" data-test="advanced-done" @click="emit('close')">
          Done
        </button>
      </div>
    </template>
  </DrawerPanel>
</template>

<style scoped>
.ms-adv__head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}
.ms-adv__title {
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 700;
}
.ms-adv__subtitle {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  margin-top: 1px;
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
.ms-disclosure {
  margin-top: 16px;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  cursor: pointer;
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
.ms-adv__footer {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}
.ms-adv__spacer {
  flex: 1;
}
.ms-adv__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px 16px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.ms-adv__done {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 11px 26px;
  border-radius: 10px;
  font-size: 13.5px;
  font-weight: 700;
  cursor: pointer;
}
</style>
