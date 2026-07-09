<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerationStore, jobPhase, jobProgress } from "../stores/generation";
import { useGalleryStore } from "../stores/gallery";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { formatGB } from "../lib/format";
import type { ModelEntry } from "../lib/api/types";

const router = useRouter();
const conn = useConnectionStore();
const generation = useGenerationStore();
const gallery = useGalleryStore();
const models = useModelStore();
const composer = useComposerStore();
const toasts = useToastStore();

const prompt = ref("");
const promptEl = ref<HTMLTextAreaElement | null>(null);
const model = ref<string>("");
const width = ref(1024);
const height = ref(1024);
const steps = ref(4);
const guidance = ref(3.5);
const seed = ref<string>("");
const pickerOpen = ref(false);

const job = computed(() => generation.active);
const running = computed(
  () => job.value !== null && job.value.status !== "complete" && job.value.status !== "error",
);

const selectedModel = computed<ModelEntry | null>(
  () => models.installed.find((m) => m.name === model.value) ?? null,
);

const buttonLabel = computed(() => {
  const j = job.value;
  if (!j || !running.value) return "Generate";
  if (j.status === "denoising") return `Developing… ${j.step}/${j.total}`;
  if (j.status === "loading") return j.stage ? `${j.stage}…` : "Loading…";
  return j.queuePosition && j.queuePosition > 0 ? `Queued #${j.queuePosition}` : "Queued";
});

const edgeCode = computed(() => {
  const j = job.value;
  if (!j) return "";
  const name = j.model.toUpperCase().replace(":", "·");
  const s = j.result ? `S ${j.result.seed_used}` : `S ${j.visualSeed.slice(0, 12)}`;
  const stepPart = `${j.status === "complete" ? j.total : j.step}/${j.total}`;
  const size = j.result ? `${j.result.width}×${j.result.height}` : `${j.width}×${j.height}`;
  const time = j.result ? `${(j.result.generation_time_ms / 1000).toFixed(1)}s` : "";
  return [name, s, stepPart, size, time].filter(Boolean).join("  ");
});

function applyModelDefaults(m: ModelEntry) {
  width.value = m.default_width;
  height.value = m.default_height;
  steps.value = m.default_steps;
  guidance.value = m.default_guidance;
}

function pickModel(m: ModelEntry) {
  model.value = m.name;
  applyModelDefaults(m);
  pickerOpen.value = false;
}

function randomizeSeed() {
  seed.value = String(Math.floor(Math.random() * 0xffffffff));
}

function swapSize() {
  [width.value, height.value] = [height.value, width.value];
}

const snap16 = (v: number) => Math.max(64, Math.round(v / 16) * 16);

async function generate() {
  if (!prompt.value.trim() || !model.value || running.value) return;
  const parsedSeed = seed.value.trim() === "" ? undefined : Number(seed.value);
  const request = {
    prompt: prompt.value.trim(),
    model: model.value,
    width: snap16(width.value),
    height: snap16(height.value),
    steps: steps.value,
    guidance: guidance.value,
    ...(parsedSeed !== undefined && Number.isFinite(parsedSeed) ? { seed: parsedSeed } : {}),
  };
  await generation.generate(request);
  const j = generation.active;
  if (j?.status === "complete") {
    toasts.push("Generated — saved to Gallery");
    void gallery.fetch();
  } else if (j?.status === "error" && j.error) {
    toasts.push(j.error, "error");
  }
}

function onComposerKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && e.metaKey) {
    e.preventDefault();
    void generate();
  }
}

watch(
  () => models.installed,
  (installed) => {
    if (!model.value && installed.length > 0) {
      const preferred = installed.find((m) => m.family === "flux") ?? installed[0]!;
      model.value = preferred.name;
      applyModelDefaults(preferred);
    }
  },
  { immediate: true },
);

// Refetch on every visit — a pull may have finished since the last look.
watch(
  () => conn.ready,
  (ready) => {
    if (ready) void models.fetch();
  },
  { immediate: true },
);

onMounted(() => {
  promptEl.value?.focus();
  const prefill = composer.take();
  if (prefill) {
    prompt.value = prefill.prompt;
    model.value = prefill.model;
    seed.value = prefill.seed !== null ? String(prefill.seed) : "";
    width.value = prefill.width;
    height.value = prefill.height;
    steps.value = prefill.steps;
    guidance.value = prefill.guidance;
  }
});
</script>

<template>
  <EmptyState
    v-if="conn.ready && !models.loading && models.installed.length === 0"
    headline="Develop your first print."
    detail="mold runs models locally on your Mac's GPU. Pull one to start."
    action="Browse models"
    @action="router.push('/models')"
  />

  <div v-else class="grid h-full grid-cols-[1fr_300px]">
    <!-- Canvas + composer -->
    <div class="flex min-w-0 flex-col p-6">
      <div class="flex min-h-0 flex-1 items-center justify-center">
        <div class="flex max-h-full flex-col" style="width: min(100%, 62vh)">
          <div
            class="relative w-full overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
            :style="{ aspectRatio: `${job?.width ?? width} / ${job?.height ?? height}` }"
          >
            <video
              v-if="job?.resultUrl && job.result?.video_frames"
              :src="job.resultUrl"
              class="absolute inset-0 h-full w-full object-contain"
              autoplay
              loop
              controls
            />
            <img
              v-else-if="job?.resultUrl"
              :src="job.resultUrl"
              alt=""
              class="absolute inset-0 h-full w-full object-contain transition-opacity duration-500"
            />
            <DevelopCanvas
              v-if="job && job.status !== 'complete'"
              :seed="job.visualSeed"
              :progress="jobProgress(job)"
              :phase="jobPhase(job)"
              class="absolute inset-0"
            />
            <div
              v-if="!job"
              class="absolute inset-0 flex items-center justify-center text-caption text-ink-3"
            >
              The print develops here
            </div>
            <div
              v-if="job && job.status === 'denoising'"
              class="edge-code absolute bottom-2 left-3"
            >
              {{ job.step }}/{{ job.total }}
            </div>
          </div>
          <div v-if="job" class="edge-code mt-2 truncate" :title="edgeCode">{{ edgeCode }}</div>
          <p v-if="job?.status === 'error'" class="mt-2 text-caption text-stop">
            {{ job.error }}
          </p>
        </div>
      </div>

      <!-- Composer -->
      <div class="border-edge mt-4 rounded-chrome border bg-bench p-3">
        <textarea
          ref="promptEl"
          v-model="prompt"
          data-selectable
          rows="2"
          placeholder="Describe the print — a lighthouse at dusk, kodak portra…"
          class="w-full resize-none bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          @keydown="onComposerKeydown"
        />
        <div class="mt-2 flex items-center justify-between">
          <span class="edge-code">{{ model || "no model" }}</span>
          <button
            type="button"
            class="h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-[#141110] transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-60"
            :disabled="running || !prompt.trim() || !model"
            @click="generate"
          >
            {{ buttonLabel }}
            <kbd v-if="!running" class="data-mono ml-1 opacity-60">⌘↩</kbd>
          </button>
        </div>
      </div>
    </div>

    <!-- Inspector -->
    <aside class="border-edge overflow-y-auto border-l bg-bench p-4">
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">MODEL</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>
      <div class="relative">
        <button
          type="button"
          class="border-edge flex h-9 w-full items-center justify-between rounded-control border bg-bath px-2 text-body text-ink"
          @click="pickerOpen = !pickerOpen"
        >
          <span class="truncate">{{ selectedModel?.name ?? "Choose a model" }}</span>
          <span v-if="selectedModel?.disk_usage_bytes" class="data-mono ml-2 text-ink-3">
            {{ formatGB(selectedModel.disk_usage_bytes) }}
          </span>
        </button>
        <div
          v-if="pickerOpen"
          class="border-edge absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-chrome border bg-bench shadow-raised"
        >
          <template v-for="[family, list] in models.byFamily" :key="family">
            <div class="edge-code px-2 pt-2 pb-1">{{ family.toUpperCase() }}</div>
            <button
              v-for="m in list"
              :key="m.name"
              type="button"
              class="flex w-full items-center justify-between px-2 py-1.5 text-left text-body text-ink-2 hover:bg-bath hover:text-ink"
              @click="pickModel(m)"
            >
              <span class="truncate">{{ m.name }}</span>
              <span
                class="ml-2 h-1.5 w-1.5 shrink-0 rounded-full"
                :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
                :title="m.is_loaded ? 'On GPU' : ''"
              />
            </button>
          </template>
        </div>
      </div>

      <div class="mt-5 mb-2 flex items-center gap-2">
        <span class="edge-code">PRINT</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>

      <label class="text-caption text-ink-2">Size</label>
      <div class="mt-1 flex items-center gap-1.5">
        <input
          v-model.number="width"
          type="number"
          step="16"
          min="64"
          class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
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
          v-model.number="height"
          type="number"
          step="16"
          min="64"
          class="border-edge data-mono h-7 w-full rounded-control border bg-bath px-1.5 text-ink"
        />
      </div>

      <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
        Steps <span class="data-mono text-ink">{{ steps }}</span>
      </label>
      <input
        v-model.number="steps"
        type="range"
        min="1"
        max="60"
        class="mt-1 w-full accent-[var(--safelight)]"
      />

      <label class="mt-3 flex items-center justify-between text-caption text-ink-2">
        Guidance <span class="data-mono text-ink">{{ guidance.toFixed(1) }}</span>
      </label>
      <input
        v-model.number="guidance"
        type="range"
        min="0"
        max="12"
        step="0.1"
        class="mt-1 w-full accent-[var(--safelight)]"
      />

      <label class="mt-3 text-caption text-ink-2">Seed</label>
      <div class="mt-1 flex items-center gap-1.5">
        <input
          v-model="seed"
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
          @click="randomizeSeed"
        >
          ⟳
        </button>
      </div>
      <p class="mt-4 text-caption text-ink-3">
        The full per-family parameter panel (LoRAs, img2img, expand) arrives next milestone.
      </p>
    </aside>
  </div>
</template>
