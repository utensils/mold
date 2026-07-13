<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { seedMode, type GenerateForm, type PickedImage } from "../../lib/generateForm";
import type {
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
} from "../../lib/api/types";
import { generationCapabilitiesForFamily, outputFormatsForFamily } from "../../lib/capabilities";
import { frames8n1Error, snapFrames } from "../../lib/chain";
import { fileToBase64 } from "../../lib/image";
import {
  aspectRatioLabel,
  matchPreset,
  orientationLabel,
  presetsForFamily,
} from "../../lib/resolutions";
import { randomSeed } from "../../stores/generation";
import ImagePickerModal from "./ImagePickerModal.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** Seed of the most recent finished print — powers "use last seed". */
    lastSeed?: number | null;
    /** Installed/known still-image upscalers for the post-generate select. */
    upscalers?: ModelEntry[];
  }>(),
  { lastSeed: null, upscalers: () => [] },
);

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const formats = computed(() => outputFormatsForFamily(props.form.family));
const framesError = computed(() => frames8n1Error(props.form.frames));

// ── Size presets ────────────────────────────────────────────────────────────
const presets = computed(() => presetsForFamily(props.form.family));
const presetValue = computed(
  () => matchPreset(props.form.width, props.form.height, props.form.family)?.label ?? "custom",
);
const aspectLabel = computed(() =>
  aspectRatioLabel(props.form.width, props.form.height, props.form.family),
);
const orientation = computed(() => orientationLabel(props.form.width, props.form.height));
const aspectShapeStyle = computed(() => ({
  aspectRatio: `${props.form.width} / ${props.form.height}`,
  ...(props.form.width >= props.form.height ? { width: "30px" } : { height: "26px" }),
}));
function applyPreset(event: Event) {
  const label = (event.target as HTMLSelectElement).value;
  const preset = presets.value.find((p) => p.label === label);
  if (!preset) return; // "Custom" — keep whatever the inputs say
  props.form.width = preset.width;
  props.form.height = preset.height;
}

// ── Seed mode ───────────────────────────────────────────────────────────────
// The segmented control owns its own mode instead of deriving it from the
// field: deriving would unmount the input the instant the user clears it
// mid-edit, yanking focus and silently flipping to Random.
const uiSeedMode = ref<"random" | "fixed">(seedMode(props.form.seed));
watch(
  () => props.form.seed,
  (seed) => {
    // External numeric writes (reuse settings, lock-last) flip to Fixed;
    // a cleared field while editing stays Fixed with a hint.
    if (seedMode(seed) === "fixed") uiSeedMode.value = "fixed";
  },
);
function setSeedMode(mode: "random" | "fixed") {
  uiSeedMode.value = mode;
  if (mode === "random") {
    props.form.seed = "";
  } else if (seedMode(props.form.seed) === "random") {
    // Locking with nothing entered picks up the last print's seed, else rolls.
    props.form.seed = String(props.lastSeed ?? randomSeed());
  }
}
/** The wire drops anything non-numeric — say so instead of lying "Fixed". */
const seedHint = computed(() => {
  if (uiSeedMode.value !== "fixed") return null;
  const raw = props.form.seed.trim();
  if (raw === "") return "Empty — a random seed will be used.";
  if (!Number.isFinite(Number(raw))) return "Not a number — a random seed will be used.";
  return null;
});

// ── LTX-2 advanced video (ltx2 only) ──────────────────────────────────────
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

/** Suggest the next keyframe index: +24 from the last, mirroring the web SPA
 * (keyframe frames are frame indices, not the 8n+1 total-count constraint). */
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

const snap16 = (v: number) => Math.max(64, Math.round(v / 16) * 16);

function snapWidth() {
  props.form.width = snap16(props.form.width);
}
function snapHeight() {
  props.form.height = snap16(props.form.height);
}
function swapSize() {
  const { form } = props;
  [form.width, form.height] = [form.height, form.width];
}
function randomize() {
  props.form.seed = String(randomSeed());
}
function stepBatch(delta: number) {
  props.form.batchSize = Math.min(8, Math.max(1, props.form.batchSize + delta));
}

const schedulerLabel: Record<string, string> = {
  default: "Default",
  ddim: "DDIM",
  "euler-ancestral": "Euler ancestral",
  unipc: "UniPC",
};
</script>

<template>
  <div>
    <div class="mb-2 flex items-center gap-2">
      <span class="edge-code">Print</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>

    <!-- Size -->
    <label class="text-caption text-ink-2">Size</label>
    <select
      data-test="size-preset"
      class="border-edge data-mono mt-1 h-7 w-full min-w-0 rounded-control border bg-bath px-1.5 text-ink"
      :value="presetValue"
      aria-label="Common sizes"
      @change="applyPreset"
    >
      <option v-for="preset in presets" :key="preset.label" :value="preset.label">
        {{ preset.label }}
      </option>
      <option value="custom">Custom…</option>
    </select>
    <div class="mt-1.5 flex min-w-0 items-center gap-1.5">
      <input
        v-model.number="form.width"
        type="number"
        step="16"
        min="64"
        aria-label="Width"
        class="border-edge data-mono h-7 w-full min-w-0 rounded-control border bg-bath px-1.5 text-ink"
        @change="snapWidth"
      />
      <button
        type="button"
        class="shrink-0 text-ink-3 hover:text-ink"
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
        class="border-edge data-mono h-7 w-full min-w-0 rounded-control border bg-bath px-1.5 text-ink"
        @change="snapHeight"
      />
    </div>
    <div
      data-test="aspect-helper"
      class="border-edge mt-2 flex items-center gap-2 rounded-control border bg-bath px-2 py-1.5"
      role="status"
      aria-live="polite"
    >
      <div class="flex h-8 w-10 shrink-0 items-center justify-center" aria-hidden="true">
        <div
          data-test="aspect-shape"
          class="border-halide/70 max-h-[26px] max-w-[30px] border bg-[color-mix(in_srgb,var(--halide)_12%,transparent)]"
          :style="aspectShapeStyle"
        />
      </div>
      <div class="min-w-0 leading-tight">
        <div class="data-mono text-ink">{{ aspectLabel }}</div>
        <div class="text-caption text-ink-2">{{ orientation }}</div>
      </div>
    </div>

    <!-- Steps -->
    <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
      Steps <span class="data-mono text-ink">{{ form.steps }}</span>
    </label>
    <input
      v-model.number="form.steps"
      type="range"
      min="1"
      max="60"
      class="mt-1 w-full accent-[var(--safelight)]"
    />

    <!-- Guidance -->
    <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
      Guidance <span class="data-mono text-ink">{{ form.guidance.toFixed(1) }}</span>
    </label>
    <input
      v-model.number="form.guidance"
      type="range"
      min="0"
      max="12"
      step="0.1"
      class="mt-1 w-full accent-[var(--safelight)]"
    />

    <!-- CFG++ (families per matrix) -->
    <label
      v-if="caps.supportsCfgPlus"
      class="mt-3 flex cursor-pointer items-center justify-between text-caption text-ink-2"
    >
      <span>
        CFG++
        <span class="block text-ink-3">Lower guidance to 1.5–2.5</span>
      </span>
      <input v-model="form.cfgPlus" type="checkbox" class="accent-[var(--safelight)]" />
    </label>

    <!-- Seed: the mode is explicit so "am I re-rolling or locked?" is
         answerable at a glance instead of implied by an empty field. -->
    <label class="mt-3 text-caption text-ink-2">Seed</label>
    <div
      class="border-edge mt-1 flex rounded-control border bg-bath p-0.5"
      role="group"
      aria-label="Seed mode"
    >
      <button
        type="button"
        data-test="seed-mode-random"
        :aria-pressed="uiSeedMode === 'random'"
        class="flex-1 rounded-control px-2 py-1 text-caption transition-colors"
        :class="
          uiSeedMode === 'random' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'
        "
        @click="setSeedMode('random')"
      >
        ⚄ Random
      </button>
      <button
        type="button"
        data-test="seed-mode-fixed"
        :aria-pressed="uiSeedMode === 'fixed'"
        class="flex-1 rounded-control px-2 py-1 text-caption transition-colors"
        :class="
          uiSeedMode === 'fixed' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'
        "
        @click="setSeedMode('fixed')"
      >
        🔒 Fixed
      </button>
    </div>
    <div v-if="uiSeedMode === 'fixed'" class="mt-1.5 flex min-w-0 items-center gap-1.5">
      <input
        v-model="form.seed"
        data-selectable
        data-test="seed-input"
        type="text"
        inputmode="numeric"
        aria-label="Seed value"
        class="border-edge data-mono h-7 w-full min-w-0 rounded-control border bg-bath px-1.5 text-ink placeholder:text-ink-3"
      />
      <button
        type="button"
        class="shrink-0 text-ink-3 hover:text-ink"
        title="Reroll this seed"
        aria-label="Reroll this seed"
        @click="randomize"
      >
        ⟳
      </button>
    </div>
    <p v-if="seedHint" data-test="seed-hint" class="mt-1 text-caption text-safelight">
      {{ seedHint }}
    </p>
    <p v-if="uiSeedMode === 'random'" class="mt-1 text-caption text-ink-3">
      New seed every print<template v-if="lastSeed !== null">
        ·
        <button
          type="button"
          data-test="lock-last-seed"
          class="text-halide hover:underline"
          @click="form.seed = String(lastSeed)"
        >
          lock last ({{ lastSeed }})
        </button></template
      >
    </p>

    <!-- Scheduler (sd15/sdxl only) -->
    <template v-if="caps.supportsScheduler">
      <label class="mt-3 text-caption text-ink-2">Scheduler</label>
      <select
        v-model="form.scheduler"
        class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
      >
        <option v-for="opt in caps.schedulerOptions" :key="opt" :value="opt">
          {{ schedulerLabel[opt] ?? opt }}
        </option>
      </select>
    </template>

    <!-- Negative prompt (CFG families only — absent otherwise) -->
    <template v-if="caps.supportsNegativePrompt">
      <label class="mt-3 text-caption text-ink-2">Negative prompt</label>
      <textarea
        v-model="form.negativePrompt"
        data-selectable
        rows="2"
        placeholder="blurry, low quality, watermark…"
        class="border-edge mt-1 w-full resize-none rounded-control border bg-bath px-1.5 py-1 text-body text-ink placeholder:text-ink-3"
      />
    </template>

    <!-- Batch -->
    <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
      Batch
      <span v-if="caps.forcesBatchSizeOne" class="text-ink-3">
        Locked to 1 — edit models render one at a time
      </span>
    </label>
    <div class="mt-1 flex items-center gap-2">
      <button
        type="button"
        class="border-edge h-7 w-7 rounded-control border bg-bath text-ink-2 hover:text-ink disabled:opacity-40"
        :disabled="caps.forcesBatchSizeOne || form.batchSize <= 1"
        @click="stepBatch(-1)"
      >
        ◂
      </button>
      <span class="data-mono w-6 text-center text-ink">{{ form.batchSize }}</span>
      <button
        type="button"
        class="border-edge h-7 w-7 rounded-control border bg-bath text-ink-2 hover:text-ink disabled:opacity-40"
        :disabled="caps.forcesBatchSizeOne || form.batchSize >= 8"
        @click="stepBatch(1)"
      >
        ▸
      </button>
    </div>

    <!-- Video params (ltx families) -->
    <template v-if="caps.supportsVideo">
      <div class="mt-5 mb-2 flex items-center gap-2">
        <span class="edge-code">Video</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>

      <label class="text-caption text-ink-2">Frames</label>
      <input
        v-model.number="form.frames"
        type="number"
        step="8"
        min="1"
        aria-label="Frames"
        :aria-invalid="framesError ? 'true' : undefined"
        class="border-edge data-mono mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
        :class="framesError ? 'border-stop' : ''"
        @change="snapFramesField"
      />
      <p v-if="framesError" class="mt-1 text-caption text-stop">{{ framesError }}</p>

      <label class="mt-3 text-caption text-ink-2">FPS</label>
      <input
        v-model.number="form.fps"
        type="number"
        min="1"
        max="60"
        aria-label="Frames per second"
        class="border-edge data-mono mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
      />

      <label
        v-if="caps.supportsAudio"
        class="mt-3 flex cursor-pointer items-center justify-between text-caption text-ink-2"
      >
        Generate audio
        <input v-model="form.enableAudio" type="checkbox" class="accent-[var(--safelight)]" />
      </label>
    </template>

    <!-- LTX-2 pipeline (ltx2 only) -->
    <template v-if="caps.supportsAdvancedVideo">
      <button
        type="button"
        class="mt-5 mb-2 flex w-full items-center gap-2 text-left"
        data-test="ltx2-disclosure"
        :aria-expanded="advancedOpen"
        @click="advancedOpen = !advancedOpen"
      >
        <span class="edge-code">{{ advancedOpen ? "▾" : "▸" }} LTX-2 pipeline</span>
        <div class="border-edge h-px flex-1 border-t" />
      </button>

      <div v-if="advancedOpen">
        <label class="text-caption text-ink-2">Pipeline</label>
        <select
          :value="form.pipeline ?? ''"
          aria-label="LTX-2 pipeline mode"
          class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
          data-test="ltx2-pipeline"
          @change="setPipeline(($event.target as HTMLSelectElement).value)"
        >
          <option value="">Auto</option>
          <option v-for="opt in pipelineOptions" :key="opt" :value="opt">{{ opt }}</option>
        </select>

        <div class="mt-3 grid grid-cols-2 gap-1.5">
          <div>
            <label class="text-caption text-ink-2">Spatial</label>
            <select
              :value="form.spatialUpscale ?? ''"
              aria-label="Spatial upscale"
              class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
              data-test="ltx2-spatial"
              @change="setSpatial(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Native</option>
              <option v-for="v in spatialOptions" :key="v" :value="v">{{ v }}</option>
            </select>
          </div>
          <div>
            <label class="text-caption text-ink-2">Temporal</label>
            <select
              :value="form.temporalUpscale ?? ''"
              aria-label="Temporal upscale"
              class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
              data-test="ltx2-temporal"
              @change="setTemporal(($event.target as HTMLSelectElement).value)"
            >
              <option value="">Native</option>
              <option v-for="v in temporalOptions" :key="v" :value="v">{{ v }}</option>
            </select>
          </div>
        </div>

        <!-- Retake window (retake pipeline only) -->
        <template v-if="form.pipeline === 'retake'">
          <label class="mt-3 text-caption text-ink-2">Retake range (seconds)</label>
          <div class="mt-1 grid grid-cols-2 gap-1.5">
            <input
              type="number"
              step="0.1"
              min="0"
              :value="form.retakeRange?.start_seconds ?? ''"
              placeholder="start"
              aria-label="Retake start (seconds)"
              class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink placeholder:text-ink-3"
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
              class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink placeholder:text-ink-3"
              data-test="ltx2-retake-end"
              @input="setRetake('end', ($event.target as HTMLInputElement).value)"
            />
          </div>
        </template>

        <!-- Conditioning audio (a2vid pipeline only) -->
        <template v-if="form.pipeline === 'a2vid'">
          <label class="mt-3 text-caption text-ink-2">Conditioning audio</label>
          <div v-if="form.audioFile" class="mt-1 flex items-center justify-between gap-2">
            <span class="data-mono truncate text-caption text-ink" :title="form.audioFile.filename">
              {{ form.audioFile.filename }}
            </span>
            <button
              type="button"
              class="text-caption text-ink-3 hover:text-stop"
              @click="clearAudioFile"
            >
              clear
            </button>
          </div>
          <input
            v-else
            type="file"
            accept="audio/*"
            aria-label="Conditioning audio"
            class="mt-1 block w-full text-caption text-ink-3"
            data-test="ltx2-audio-file"
            @change="setAudioFile"
          />
          <p class="mt-1 text-caption text-ink-3">a2vid needs an audio track to drive the video.</p>
        </template>

        <!-- Source video -->
        <label class="mt-3 text-caption text-ink-2">Source video</label>
        <div v-if="form.sourceVideo" class="mt-1 flex items-center justify-between gap-2">
          <span class="data-mono truncate text-caption text-ink" :title="form.sourceVideo.filename">
            {{ form.sourceVideo.filename }}
          </span>
          <button
            type="button"
            class="text-caption text-ink-3 hover:text-stop"
            @click="clearSourceVideo"
          >
            clear
          </button>
        </div>
        <input
          v-else
          type="file"
          accept="video/*"
          aria-label="Source video"
          class="mt-1 block w-full text-caption text-ink-3"
          data-test="ltx2-source-video"
          @change="setSourceVideo"
        />

        <!-- Keyframes -->
        <div class="mt-3 flex items-center justify-between">
          <label class="text-caption text-ink-2">Keyframes</label>
          <button
            type="button"
            class="text-caption text-safelight underline-offset-2 hover:underline"
            data-test="ltx2-add-keyframe"
            @click="keyframePickerOpen = true"
          >
            Add…
          </button>
        </div>
        <div
          v-for="(keyframe, index) in form.keyframes"
          :key="`${keyframe.image.filename}-${index}`"
          class="border-edge mt-1 grid grid-cols-[4rem_1fr_auto] items-center gap-2 rounded-control border bg-bath p-1.5"
        >
          <input
            type="number"
            min="0"
            :value="keyframe.frame"
            :aria-label="`Keyframe ${index + 1} frame`"
            class="border-edge data-mono h-6 w-full rounded-control border bg-bench px-1 text-caption text-ink"
            :data-test="`ltx2-keyframe-frame-${index}`"
            @input="updateKeyframeFrame(index, ($event.target as HTMLInputElement).value)"
          />
          <span class="truncate text-caption text-ink-2">{{ keyframe.image.filename }}</span>
          <button
            type="button"
            class="text-ink-3 hover:text-stop"
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

    <!-- Post-generate upscale (still images; the server auto-pulls the
         upscaler model on first use and runs it after the print develops) -->
    <template v-if="!caps.supportsVideo && upscalers.length">
      <label class="mt-3 text-caption text-ink-2">Upscale</label>
      <select
        v-model="form.upscaleModel"
        data-test="upscale-select"
        class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
      >
        <option value="">Off</option>
        <option v-for="u in upscalers" :key="u.name" :value="u.name">
          {{ u.name }}{{ u.downloaded ? "" : " (downloads on first use)" }}
        </option>
      </select>
    </template>

    <!-- Output format -->
    <label class="mt-3 text-caption text-ink-2">Format</label>
    <select
      v-model="form.outputFormat"
      class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-1.5 text-body text-ink"
    >
      <option v-for="f in formats" :key="f" :value="f">{{ f }}</option>
    </select>
  </div>
</template>
