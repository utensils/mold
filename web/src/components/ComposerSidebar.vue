<script setup lang="ts">
import { computed } from "vue";
import type { GenerateFormState, ModelInfoExtended } from "../types";

/*
 * Right-hand sidebar companion to <Composer>.
 *
 * Surfaces the generation knobs the prototype wanted a dedicated rail
 * for — model picker, dimension presets, steps/guidance/batch sliders,
 * seed control, and the big gradient Generate button — while leaving
 * the "advanced" long tail (scheduler, strength, output format, frame
 * counts, expand variations) in the existing SettingsModal so we don't
 * duplicate controls or remove functionality.
 *
 * The component is presentational: all state lives on the form and
 * flows in via `v-model`. Running/progress state comes from the
 * existing useGenerateStream composable via the two reactive props.
 */

const props = defineProps<{
  modelValue: GenerateFormState;
  models: ModelInfoExtended[];
  running: boolean;
  progressPct: number | null;
  progressStage: string | null;
  canSubmit: boolean;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
  (e: "submit"): void;
}>();

function patch(partial: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...partial });
}

/*
 * Only advertise models that are actually downloaded. Picking a model
 * whose weights aren't on disk would just bounce through the download
 * flow from the server — the picker in this panel is for quick
 * switching, not model management (that's what the Downloads drawer
 * is for).
 */
const availableModels = computed(() =>
  [...props.models]
    .filter((m) => m.downloaded)
    .sort((a, b) => a.name.localeCompare(b.name)),
);

function modelNameOnly(name: string): string {
  return name.split(":")[0] ?? name;
}
function modelTagOnly(name: string): string {
  const colon = name.indexOf(":");
  if (colon < 0) return "bf16";
  return name.slice(colon + 1);
}

function pickModel(m: ModelInfoExtended) {
  patch({
    model: m.name,
    width: m.default_width,
    height: m.default_height,
    steps: m.default_steps,
    guidance: m.default_guidance,
  });
}

/*
 * Common aspect ratios the user is most likely to want. Locked to a
 * small set because exposing arbitrary width/height combos invites
 * off-grid values that most diffusion models don't handle well — the
 * SettingsModal is still there for finer control.
 */
const DIMS: [number, number][] = [
  [512, 512],
  [768, 768],
  [1024, 1024],
  [1024, 1536],
  [1536, 1024],
  [1920, 1080],
];

function pickDim(w: number, h: number) {
  patch({ width: w, height: h });
}

function randomSeed() {
  patch({ seed: Math.floor(Math.random() * 1e9) });
}

function clearSeed() {
  patch({ seed: null });
}

function onSeedInput(e: Event) {
  const v = (e.target as HTMLInputElement).value;
  patch({ seed: v ? Number(v) : null });
}

function onStepsInput(e: Event) {
  patch({ steps: Number((e.target as HTMLInputElement).value) });
}
function onGuidanceInput(e: Event) {
  patch({ guidance: Number((e.target as HTMLInputElement).value) });
}
function onBatchInput(e: Event) {
  patch({ batchSize: Number((e.target as HTMLInputElement).value) });
}

function submit() {
  if (props.canSubmit && !props.running) emit("submit");
}

const dimPreviewSize = (w: number, h: number) => {
  const max = Math.max(w, h);
  return { width: `${(w / max) * 28}px`, height: `${(h / max) * 28}px` };
};
</script>

<template>
  <aside class="gen-side">
    <section class="gen-side-block">
      <div class="gen-side-head">Model</div>
      <div
        v-if="availableModels.length === 0"
        style="font-size: 12px; color: var(--fg-3)"
      >
        No models downloaded. Pull one from the Downloads drawer.
      </div>
      <div v-else class="gen-model-list">
        <button
          v-for="m in availableModels"
          :key="m.name"
          class="gen-model"
          :class="{ on: modelValue.model === m.name }"
          @click="pickModel(m)"
        >
          <span class="gen-model-name">{{ modelNameOnly(m.name) }}</span>
          <span class="gen-model-tag">{{ modelTagOnly(m.name) }}</span>
        </button>
      </div>
    </section>

    <section class="gen-side-block">
      <div class="gen-side-head">Dimensions</div>
      <div class="gen-dims">
        <button
          v-for="[w, h] in DIMS"
          :key="`${w}x${h}`"
          class="gen-dim"
          :class="{ on: modelValue.width === w && modelValue.height === h }"
          @click="pickDim(w, h)"
        >
          <div class="gen-dim-box" :style="dimPreviewSize(w, h)" />
          <span>{{ w }}×{{ h }}</span>
        </button>
      </div>
    </section>

    <section class="gen-side-block">
      <div class="gen-side-head">Sampling</div>

      <div class="slider">
        <div class="slider-row">
          <span class="slider-label">Steps</span>
          <span class="slider-value">{{ modelValue.steps }}</span>
        </div>
        <input
          type="range"
          min="4"
          max="50"
          step="1"
          :value="modelValue.steps"
          @input="onStepsInput"
        />
      </div>

      <div class="slider">
        <div class="slider-row">
          <span class="slider-label">Guidance</span>
          <span class="slider-value">
            {{ modelValue.guidance.toFixed(1) }}
          </span>
        </div>
        <input
          type="range"
          min="1"
          max="15"
          step="0.1"
          :value="modelValue.guidance"
          @input="onGuidanceInput"
        />
      </div>

      <div class="slider">
        <div class="slider-row">
          <span class="slider-label">Batch</span>
          <span class="slider-value">{{ modelValue.batchSize }}</span>
        </div>
        <input
          type="range"
          min="1"
          max="8"
          step="1"
          :value="modelValue.batchSize"
          @input="onBatchInput"
        />
      </div>
    </section>

    <section class="gen-side-block">
      <div class="gen-side-head">Seed</div>
      <div class="gen-seed-row">
        <input
          type="number"
          class="gen-seed-input"
          :value="modelValue.seed ?? ''"
          placeholder="random"
          @input="onSeedInput"
        />
        <button
          class="gen-seed-dice"
          aria-label="Random seed"
          title="Random seed"
          @click="randomSeed"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="18" height="18" rx="3" />
            <circle cx="8" cy="8" r="1.2" fill="currentColor" />
            <circle cx="16" cy="8" r="1.2" fill="currentColor" />
            <circle cx="12" cy="12" r="1.2" fill="currentColor" />
            <circle cx="8" cy="16" r="1.2" fill="currentColor" />
            <circle cx="16" cy="16" r="1.2" fill="currentColor" />
          </svg>
        </button>
        <button
          class="gen-seed-clear"
          :disabled="modelValue.seed === null"
          @click="clearSeed"
        >
          Random
        </button>
      </div>
    </section>

    <button
      class="gen-go gen-go-full"
      :disabled="!canSubmit || running"
      @click="submit"
    >
      <template v-if="running">
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          class="animate-spin"
          aria-hidden="true"
        >
          <circle
            cx="12"
            cy="12"
            r="9"
            stroke="currentColor"
            stroke-width="2"
            fill="none"
            opacity="0.2"
          />
          <path
            d="M21 12a9 9 0 0 0-9-9"
            stroke="currentColor"
            stroke-width="2"
            fill="none"
            stroke-linecap="round"
          />
        </svg>
        <span class="tabular">
          {{ progressPct !== null ? `${progressPct}%` : "…" }}
        </span>
      </template>
      <template v-else>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="m15 4 5 5-11 11H4v-5z" />
          <path d="m14 5 5 5" />
        </svg>
        Generate
      </template>
    </button>

    <div v-if="running" class="gen-progress" aria-hidden="true">
      <div
        class="gen-progress-fill"
        :style="{ width: `${progressPct ?? 0}%` }"
      />
    </div>
    <div v-if="running && progressStage" class="gen-running-stage">
      {{ progressStage }}
    </div>
  </aside>
</template>
