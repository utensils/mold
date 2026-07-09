<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import ParamPanel from "../components/generate/ParamPanel.vue";
import LoraStack from "../components/generate/LoraStack.vue";
import SourceImageWell from "../components/generate/SourceImageWell.vue";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import { useConnectionStore } from "../stores/connection";
import { useGenerationStore, jobPhase, jobProgress, type Job } from "../stores/generation";
import { useGalleryStore } from "../stores/gallery";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { useUiStore } from "../stores/ui";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { applyModelDefaults, buildRequest, newGenerateForm } from "../lib/generateForm";
import { formatGB } from "../lib/format";
import { randomSeed } from "../stores/generation";
import type { ModelEntry } from "../lib/api/types";

const router = useRouter();
const conn = useConnectionStore();
const generation = useGenerationStore();
const gallery = useGalleryStore();
const models = useModelStore();
const composer = useComposerStore();
const toasts = useToastStore();
const ui = useUiStore();

const form = reactive(newGenerateForm());
const promptEl = ref<HTMLTextAreaElement | null>(null);
const expandControl = ref<InstanceType<typeof ExpandControl> | null>(null);
const pickerOpen = ref(false);

const job = computed(() => generation.active);
const siblings = computed(() => generation.siblings);
const running = computed(
  () => job.value !== null && job.value.status !== "complete" && job.value.status !== "error",
);

const caps = computed(() => generationCapabilitiesForFamily(form.family));
const selectedModel = computed<ModelEntry | null>(
  () => models.installed.find((m) => m.name === form.model) ?? null,
);

/** The request the estimate badge previews — null until a model is chosen. */
const estimateRequest = computed(() => (form.model ? buildRequest(form) : null));

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

function pickModel(m: ModelEntry) {
  applyModelDefaults(form, m);
  pickerOpen.value = false;
}

function siblingDot(s: Job): string {
  if (s.status === "complete") return "text-ink"; // ◉ developed
  if (s.status === "error") return "text-stop";
  return "text-ink-3"; // ◎ pending
}

function onExpandApply(payload: { expanded: string; original: string }) {
  form.prompt = payload.expanded;
  form.originalPrompt = payload.original;
}
function onExpandRestore(original: string) {
  form.prompt = original;
  form.originalPrompt = null;
}
function appendPromptWord(word: string) {
  const trimmed = word.trim();
  if (!trimmed) return;
  form.prompt = form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed;
}

async function generate() {
  if (!form.prompt.trim() || !form.model || running.value) return;
  const request = buildRequest(form);
  const batch = caps.value.forcesBatchSizeOne ? 1 : form.batchSize;
  await generation.generateBatch(request, batch);
  const done = generation.siblings;
  const ok = done.filter((s) => s.status === "complete").length;
  const failed = done.find((s) => s.status === "error");
  if (ok > 0) {
    toasts.push(
      ok === 1 ? "Generated — saved to Gallery" : `Generated ${ok} prints — saved to Gallery`,
    );
    void gallery.fetch();
  } else if (failed?.error) {
    toasts.push(failed.error, "error");
  }
}

function onComposerKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && e.metaKey) {
    e.preventDefault();
    void generate();
  } else if ((e.key === "e" || e.key === "E") && e.metaKey) {
    e.preventDefault();
    expandControl.value?.expand();
  }
}

watch(
  () => models.installed,
  (installed) => {
    if (!form.model && installed.length > 0) {
      const preferred = installed.find((m) => m.family === "flux") ?? installed[0]!;
      applyModelDefaults(form, preferred);
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

function applyPrefill() {
  const prefill = composer.take();
  if (!prefill) return;
  form.prompt = prefill.prompt;
  form.model = prefill.model;
  form.seed = prefill.seed !== null ? String(prefill.seed) : "";
  form.width = prefill.width;
  form.height = prefill.height;
  form.steps = prefill.steps;
  form.guidance = prefill.guidance;
  const m = models.installed.find((x) => x.name === prefill.model);
  if (m) form.family = m.family;
  void nextTick(() => promptEl.value?.focus());
}

// Apply a prefill whenever one arrives (Reuse settings, history, "Generate
// with <model>"), including one already queued before this view mounted.
watch(() => composer.prefill, applyPrefill, { immediate: true });

// ⌘N — clear the composer for a fresh generation, keeping the model.
watch(
  () => ui.newGenerationTick,
  () => {
    form.prompt = "";
    form.originalPrompt = null;
    form.negativePrompt = "";
    form.seed = "";
    form.sourceImage = null;
    form.maskImage = null;
    void nextTick(() => promptEl.value?.focus());
  },
);

// ⌘R — randomize the seed.
watch(
  () => ui.randomizeSeedTick,
  () => {
    form.seed = String(randomSeed());
  },
);

// Menu ▸ Generate / Expand Prompt reuse the composer actions.
watch(
  () => ui.generateTick,
  () => void generate(),
);
watch(
  () => ui.expandTick,
  () => expandControl.value?.expand(),
);

onMounted(() => {
  promptEl.value?.focus();
});
</script>

<template>
  <StarterCards
    v-if="conn.ready && !models.loading && models.installed.length === 0"
    @browse="router.push('/models')"
  />

  <div v-else class="grid h-full grid-cols-[1fr_320px]">
    <!-- Canvas + composer -->
    <div class="flex min-w-0 flex-col p-6">
      <div class="flex min-h-0 flex-1 items-center justify-center">
        <div class="flex max-h-full flex-col" style="width: min(100%, 62vh)">
          <div
            class="relative w-full overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
            :style="{ aspectRatio: `${job?.width ?? form.width} / ${job?.height ?? form.height}` }"
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

          <!-- Batch dots -->
          <div v-if="siblings.length > 1" class="mt-2 flex items-center gap-1.5">
            <span
              v-for="(s, i) in siblings"
              :key="i"
              class="data-mono text-body"
              :class="siblingDot(s)"
              :title="`${i + 1} of ${siblings.length}`"
            >
              {{ s.status === "complete" ? "◉" : s.status === "error" ? "◉" : "◎" }}
            </span>
            <span class="edge-code ml-1">
              {{ siblings.filter((s) => s.status === "complete").length }} of {{ siblings.length }}
            </span>
          </div>

          <p v-if="job?.status === 'error'" class="mt-2 text-caption text-stop">{{ job.error }}</p>
        </div>
      </div>

      <!-- Composer -->
      <div class="border-edge mt-4 rounded-chrome border bg-bench p-3">
        <textarea
          ref="promptEl"
          v-model="form.prompt"
          data-selectable
          rows="2"
          placeholder="Describe the print — a lighthouse at dusk, kodak portra…"
          class="w-full resize-none bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          @keydown="onComposerKeydown"
        />
        <div class="mt-2 flex items-center justify-between gap-2">
          <ExpandControl
            ref="expandControl"
            :prompt="form.prompt"
            :family="form.family"
            @apply="onExpandApply"
            @restore="onExpandRestore"
          />
          <div class="flex items-center gap-3">
            <EstimateBadge :request="estimateRequest" />
            <button
              type="button"
              class="h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-[#141110] transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-60"
              :disabled="running || !form.prompt.trim() || !form.model"
              @click="generate"
            >
              {{ buttonLabel }}
              <kbd v-if="!running" class="data-mono ml-1 opacity-60">⌘↩</kbd>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Inspector -->
    <aside class="border-edge overflow-y-auto border-l bg-bench p-4">
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">Model</span>
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

      <ParamPanel :form="form" class="mt-5" />
      <SourceImageWell :form="form" />
      <LoraStack
        v-if="caps.supportsLora"
        :form="form"
        :model="form.model"
        @append-word="appendPromptWord"
      />
    </aside>
  </div>
</template>
