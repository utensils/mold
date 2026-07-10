<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import EmptyState from "../components/shell/EmptyState.vue";
import StageCard from "../components/chains/StageCard.vue";
import SpliceMark from "../components/chains/SpliceMark.vue";
import ChainJobsList from "../components/chains/ChainJobsList.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useChainJobsStore } from "../stores/chainJobs";
import { useToastStore } from "../stores/toasts";
import { isVideoFamily } from "../lib/capabilities";
import {
  MAX_CHAIN_STAGES,
  chainToToml,
  durationSeconds,
  estimatedTotalFrames,
  isLtx2FrameCount,
} from "../lib/chain";
import {
  chainFormToRequest,
  chainFormToScript,
  newChainForm,
  newStage,
  tomlToChainForm,
} from "../lib/chainForm";
import { fetchChainLimits } from "../lib/api/chains";
import { apiJson } from "../lib/api/client";
import { randomSeed } from "../stores/generation";
import type { ChainLimits, ModelEntry, ResourceSnapshot } from "../lib/api/types";

const router = useRouter();
const conn = useConnectionStore();
const models = useModelStore();
const chains = useChainJobsStore();
const toasts = useToastStore();

const form = reactive(newChainForm());
const limits = ref<ChainLimits | null>(null);
const backend = ref<string | null>(null);
const showToml = ref(false);
const rendering = ref(false);
const tomlInput = ref<HTMLInputElement | null>(null);

const videoModels = computed<ModelEntry[]>(() =>
  models.installed.filter((m) => isVideoFamily(m.family)),
);
// LTX-2 needs CUDA; on a Metal Mac the option shows but can't be selected.
const isCudaOnlyOnThisMachine = (m: ModelEntry) =>
  (m.family === "ltx2" || m.family === "ltx-2") && backend.value === "metal";

const wireStages = computed(() => chainFormToScript(form).stage);
const totalFrames = computed(() => estimatedTotalFrames(wireStages.value, form.motionTailFrames));
const duration = computed(() => durationSeconds(totalFrames.value, form.fps));

const stageErrors = computed(() => form.stages.filter((s) => !isLtx2FrameCount(s.frames)).length);
const tomlText = computed(() => chainToToml(chainFormToScript(form)));

const fit = computed(() => {
  if (stageErrors.value > 0) return { ok: false, text: "Fix the frame counts above." };
  if (!limits.value)
    return { ok: true, text: `${totalFrames.value} frames · ${duration.value.toFixed(1)}s` };
  if (form.stages.length > limits.value.max_stages) {
    return { ok: false, text: `Too many stages (max ${limits.value.max_stages}).` };
  }
  if (totalFrames.value > limits.value.max_total_frames) {
    return {
      ok: false,
      text: `Needs ${totalFrames.value} frames; cap is ${limits.value.max_total_frames}.`,
    };
  }
  return {
    ok: true,
    text: `✓ fits · ${totalFrames.value} frames · ${duration.value.toFixed(1)}s @ ${form.fps}fps`,
  };
});

const live = computed(() => chains.live);
const canRender = computed(
  () =>
    !!form.model &&
    !rendering.value &&
    stageErrors.value === 0 &&
    form.stages.length > 0 &&
    !!form.stages[0]!.prompt.trim(),
);

function pickModel(name: string) {
  form.model = name;
  const m = videoModels.value.find((x) => x.name === name);
  if (m) {
    form.width = m.default_width;
    form.height = m.default_height;
    form.steps = m.default_steps;
  }
  void loadLimits(name);
}

async function loadLimits(model: string) {
  try {
    limits.value = await fetchChainLimits(model);
    if (!limits.value.supports_audio) form.enableAudio = false;
  } catch {
    limits.value = null;
  }
}

function addStage() {
  if (form.stages.length >= MAX_CHAIN_STAGES) return;
  form.stages.push(newStage());
}
function removeStage(i: number) {
  if (form.stages.length <= 1) return;
  form.stages.splice(i, 1);
}
function moveStage(i: number, delta: number) {
  const j = i + delta;
  if (j < 0 || j >= form.stages.length) return;
  const [s] = form.stages.splice(i, 1);
  form.stages.splice(j, 0, s!);
}
function randomizeSeed() {
  form.seed = String(randomSeed());
}

function openToml() {
  tomlInput.value?.click();
}

async function onTomlFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  (e.target as HTMLInputElement).value = ""; // allow re-picking the same file
  if (!file) return;
  try {
    const parsed = tomlToChainForm(await file.text());
    Object.assign(form, parsed);
    if (parsed.model && videoModels.value.some((m) => m.name === parsed.model)) {
      void loadLimits(parsed.model);
    } else if (parsed.model) {
      limits.value = null;
      toasts.push(`Pull ${parsed.model} first`);
    }
    toasts.push(`Loaded ${file.name}`);
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

function exportToml() {
  const blob = new Blob([tomlText.value], { type: "application/toml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chain.toml";
  a.click();
  URL.revokeObjectURL(url);
}

async function render() {
  if (!canRender.value) return;
  rendering.value = true;
  try {
    await chains.create(chainFormToRequest(form));
    toasts.push("Chain queued");
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    rendering.value = false;
  }
}

watch(
  () => conn.ready,
  (ready) => {
    if (ready) {
      void models.fetch();
      void chains.fetchJobs();
      apiJson<ResourceSnapshot>("/api/resources")
        .then((r) => (backend.value = r.gpus[0]?.backend ?? null))
        .catch(() => {});
    }
  },
  { immediate: true },
);

// Auto-select the first usable video model.
watch(
  videoModels,
  (list) => {
    if (!form.model && list.length > 0) {
      const usable = list.find((m) => !isCudaOnlyOnThisMachine(m)) ?? list[0]!;
      pickModel(usable.name);
    }
  },
  { immediate: true },
);

onMounted(() => {
  if (chains.watchingId === null && chains.jobs.length === 0) void chains.fetchJobs();
});
</script>

<template>
  <EmptyState
    v-if="conn.ready && !models.loading && videoModels.length === 0"
    headline="No video models yet"
    detail="Chains stitch clips from LTX video models. Pull one to start."
    action="Browse models"
    @action="router.push('/models')"
  />

  <div v-else class="flex h-full flex-col overflow-y-auto">
    <!-- Chain header -->
    <header class="border-edge flex flex-wrap items-end gap-3 border-b bg-bench px-4 py-3">
      <div>
        <label class="edge-code">Model</label>
        <select
          :value="form.model"
          class="border-edge mt-1 block h-8 w-56 rounded-control border bg-bath px-2 text-body text-ink"
          @change="pickModel(($event.target as HTMLSelectElement).value)"
        >
          <option
            v-for="m in videoModels"
            :key="m.name"
            :value="m.name"
            :disabled="isCudaOnlyOnThisMachine(m)"
          >
            {{ m.name }}{{ isCudaOnlyOnThisMachine(m) ? " — CUDA only" : "" }}
          </option>
        </select>
      </div>
      <div>
        <label class="edge-code">Size</label>
        <div class="mt-1 flex items-center gap-1">
          <input
            v-model.number="form.width"
            type="number"
            step="16"
            class="border-edge data-mono h-8 w-20 rounded-control border bg-bath px-1.5 text-ink"
          />
          <span class="text-ink-3">×</span>
          <input
            v-model.number="form.height"
            type="number"
            step="16"
            class="border-edge data-mono h-8 w-20 rounded-control border bg-bath px-1.5 text-ink"
          />
        </div>
      </div>
      <div>
        <label class="edge-code">FPS</label>
        <input
          v-model.number="form.fps"
          type="number"
          min="1"
          max="60"
          class="border-edge data-mono mt-1 block h-8 w-16 rounded-control border bg-bath px-1.5 text-ink"
        />
      </div>
      <div>
        <label class="edge-code">Seed</label>
        <div class="mt-1 flex items-center gap-1">
          <input
            v-model="form.seed"
            data-selectable
            type="text"
            inputmode="numeric"
            placeholder="Random"
            class="border-edge data-mono h-8 w-28 rounded-control border bg-bath px-1.5 text-ink placeholder:text-ink-3"
          />
          <button
            type="button"
            class="text-ink-3 hover:text-ink"
            title="Randomize"
            aria-label="Randomize seed"
            @click="randomizeSeed"
          >
            ⟳
          </button>
        </div>
      </div>
      <label v-if="limits?.supports_audio" class="flex items-center gap-1 text-caption text-ink-2">
        <input v-model="form.enableAudio" type="checkbox" class="accent-[var(--safelight)]" />
        Audio
      </label>

      <p
        v-if="
          form.model && isCudaOnlyOnThisMachine(videoModels.find((m) => m.name === form.model)!)
        "
        class="w-full text-caption text-halide"
      >
        LTX-2 generates on CUDA GPUs only. Connect a remote Linux engine to use it.
      </p>
    </header>

    <!-- Filmstrip -->
    <div class="flex items-stretch gap-0 overflow-x-auto p-4">
      <template v-for="(stage, i) in form.stages" :key="i">
        <SpliceMark
          v-if="i > 0"
          :stage="stage"
          :motion-tail="form.motionTailFrames"
          :fade-max="limits?.fade_frames_max ?? 32"
        />
        <StageCard
          :stage="stage"
          :index="i"
          :base-seed="form.seed || form.model"
          :job-stage="chains.watchingId ? (live.detail?.stages[i] ?? null) : null"
          :progress="chains.watchingId ? (live.progress[i] ?? null) : null"
          :job-id="chains.watchingId"
          :can-move-left="i > 0"
          :can-move-right="i < form.stages.length - 1"
          :can-remove="form.stages.length > 1"
          @remove="removeStage(i)"
          @move-left="moveStage(i, -1)"
          @move-right="moveStage(i, 1)"
        />
      </template>
      <button
        type="button"
        class="border-edge ml-1 flex w-24 shrink-0 items-center justify-center rounded-chrome border border-dashed text-caption text-ink-3 hover:text-ink disabled:opacity-40"
        :disabled="form.stages.length >= MAX_CHAIN_STAGES"
        @click="addStage"
      >
        + Add stage
      </button>
    </div>

    <!-- Footer: totals + fit + actions -->
    <div class="border-edge flex flex-wrap items-center gap-3 border-t border-b bg-bench px-4 py-2">
      <span class="edge-code" :class="fit.ok ? 'text-halide' : 'text-stop'">{{ fit.text }}</span>
      <div class="ml-auto flex items-center gap-2">
        <input
          ref="tomlInput"
          type="file"
          accept=".toml,text/plain"
          class="hidden"
          @change="onTomlFile"
        />
        <button
          type="button"
          class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink"
          @click="openToml"
        >
          Open .toml…
        </button>
        <button
          type="button"
          class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink"
          @click="showToml = !showToml"
        >
          {{ showToml ? "Hide TOML" : "Edit as TOML" }}
        </button>
        <button
          type="button"
          class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink"
          @click="exportToml"
        >
          Export .toml
        </button>
        <button
          type="button"
          class="h-8 rounded-chrome bg-safelight px-4 text-body font-semibold text-[#141110] hover:brightness-105 disabled:opacity-50"
          :disabled="!canRender"
          @click="render"
        >
          Render chain
        </button>
      </div>
    </div>

    <!-- TOML view (read-only; import needs a TOML parser dep — see report) -->
    <div v-if="showToml" class="border-edge border-b p-4">
      <textarea
        :value="tomlText"
        readonly
        rows="12"
        data-selectable
        class="border-edge data-mono w-full rounded-control border bg-bath p-2 text-caption text-ink"
      />
    </div>

    <!-- Jobs list -->
    <div class="p-4">
      <ChainJobsList />
    </div>
  </div>
</template>
