<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import EmptyState from "../components/shell/EmptyState.vue";
import ImagePickerModal from "../components/generate/ImagePickerModal.vue";
import StageCard from "../components/chains/StageCard.vue";
import SpliceMark from "../components/chains/SpliceMark.vue";
import ChainJobsList from "../components/chains/ChainJobsList.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
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
import { base64ToDataUrl } from "../lib/image";
import { randomSeed } from "../stores/generation";
import type { PickedImage } from "../lib/generateForm";
import { mergeInstalledModels } from "../lib/generateModels";
import type { ChainLimits, ModelEntry, ResourceSnapshot } from "../lib/api/types";

const router = useRouter();
const conn = useConnectionStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const chains = useChainJobsStore();
const toasts = useToastStore();

const form = reactive(newChainForm());
const limits = ref<ChainLimits | null>(null);
const backend = ref<string | null>(null);
const showToml = ref(false);
const rendering = ref(false);
const tomlInput = ref<HTMLInputElement | null>(null);

const installedModels = computed(() =>
  mergeInstalledModels(models.installed, hostModels.unionInstalled),
);
const videoModels = computed<ModelEntry[]>(() =>
  installedModels.value.filter((m) => isVideoFamily(m.family)),
);
// LTX-2 needs CUDA; on a Metal Mac the option shows but can't be selected.
const isCudaOnlyOnThisMachine = (m: ModelEntry) => {
  if (m.family !== "ltx2" && m.family !== "ltx-2") return false;
  const ownerIds = hostModels.hostsFor(m.name);
  if (ownerIds.some((id) => id !== "local")) return false;
  return backend.value === "metal";
};

function routeForModel(model: ModelEntry) {
  const cudaOnly = model.family === "ltx2" || model.family === "ltx-2";
  return hosts.resolveRoute(cudaOnly ? "capable" : null, model.name);
}

const selectedRoute = computed(() => {
  const model = videoModels.value.find((entry) => entry.name === form.model);
  return model ? routeForModel(model) : null;
});
const watchedHostId = computed(
  () => hosts.all.find((host) => host.baseUrl === chains.target?.baseUrl)?.id ?? null,
);

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
    const entry = videoModels.value.find((candidate) => candidate.name === model);
    const route = entry ? routeForModel(entry) : hosts.resolveRoute(null, model);
    const nextLimits = await fetchChainLimits(model, route?.target ?? null);
    if (form.model !== model) return;
    limits.value = nextLimits;
    await chains.fetchJobs(route?.target ?? null);
    if (!limits.value.supports_audio) form.enableAudio = false;
  } catch {
    if (form.model === model) limits.value = null;
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

// Image picker target: "start" = the chain-level starting image, a number =
// that stage's source image, null = closed. One modal serves both.
const pickerFor = ref<"start" | number | null>(null);
const pickerTitle = computed(() =>
  pickerFor.value === "start" ? "Chain start image" : `Stage ${Number(pickerFor.value) + 1} image`,
);

function onImagePicked(picked: PickedImage[]) {
  const first = picked[0];
  const target = pickerFor.value;
  if (!first || target === null) return;
  if (target === "start") {
    form.startImage = first.base64;
  } else if (form.stages[target]) {
    form.stages[target]!.sourceImage = first.base64;
  }
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
    await chains.create(chainFormToRequest(form), selectedRoute.value?.target ?? null);
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

watch(
  () =>
    hosts.all
      .filter((host) => host.status === "ready")
      .map((host) => host.id)
      .join("\n"),
  () => void hostModels.refresh(),
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
    @action="router.push('/models?type=video')"
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
      <div>
        <label class="edge-code">Start image</label>
        <div class="mt-1 flex items-center gap-1">
          <button
            v-if="form.startImage"
            type="button"
            data-test="chain-start-thumb"
            class="border-edge block h-8 w-8 overflow-hidden rounded-media border hover:brightness-110"
            title="Replace the chain's starting image"
            aria-label="Replace start image"
            @click="pickerFor = 'start'"
          >
            <img
              :src="base64ToDataUrl(form.startImage)"
              alt=""
              class="h-full w-full object-cover"
            />
          </button>
          <button
            v-else
            type="button"
            data-test="chain-start-attach"
            class="border-edge h-8 rounded-control border border-dashed px-2 text-caption text-ink-3 hover:text-ink"
            title="Seed the film from a still — it conditions stage 1"
            @click="pickerFor = 'start'"
          >
            Attach…
          </button>
          <button
            v-if="form.startImage"
            type="button"
            data-test="chain-start-clear"
            class="text-ink-3 hover:text-stop"
            title="Remove start image"
            aria-label="Remove start image"
            @click="form.startImage = null"
          >
            ✕
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
          :api-target="chains.target"
          :host-id="watchedHostId"
          :can-move-left="i > 0"
          :can-move-right="i < form.stages.length - 1"
          :can-remove="form.stages.length > 1"
          @remove="removeStage(i)"
          @move-left="moveStage(i, -1)"
          @move-right="moveStage(i, 1)"
          @pick-image="pickerFor = i"
          @clear-image="stage.sourceImage = null"
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
          class="h-8 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent hover:brightness-105 disabled:opacity-50"
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

    <ImagePickerModal
      :open="pickerFor !== null"
      :title="pickerTitle"
      @pick="onImagePicked"
      @close="pickerFor = null"
    />
  </div>
</template>
