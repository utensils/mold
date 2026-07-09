<script setup lang="ts">
import { computed } from "vue";
import type { GenerateForm } from "../../lib/generateForm";
import { generationCapabilitiesForFamily, outputFormatsForFamily } from "../../lib/capabilities";
import { randomSeed } from "../../stores/generation";

const props = defineProps<{ form: GenerateForm }>();

const caps = computed(() => generationCapabilitiesForFamily(props.form.family));
const formats = computed(() => outputFormatsForFamily(props.form.family));

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
    <div class="mt-1 flex items-center gap-1.5">
      <input
        v-model.number="form.width"
        type="number"
        step="16"
        min="64"
        class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
        @change="snapWidth"
      />
      <button
        type="button"
        class="text-ink-3 hover:text-ink"
        title="Swap width and height"
        @click="swapSize"
      >
        ⇄
      </button>
      <input
        v-model.number="form.height"
        type="number"
        step="16"
        min="64"
        class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
        @change="snapHeight"
      />
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

    <!-- Seed -->
    <label class="mt-3 text-caption text-ink-2">Seed</label>
    <div class="mt-1 flex items-center gap-1.5">
      <input
        v-model="form.seed"
        data-selectable
        type="text"
        inputmode="numeric"
        placeholder="Random"
        class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink placeholder:text-ink-3"
      />
      <button
        type="button"
        class="text-ink-3 hover:text-ink"
        title="Randomize seed"
        @click="randomize"
      >
        ⟳
      </button>
    </div>

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
