<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { modelSupportsSequence } from "@mold/studio";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import ModelPicker from "../components/create/ModelPicker.vue";
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
import { autoGrowRows } from "../lib/autogrow";
import { highlightToml } from "../lib/tomlHighlight";
import { randomSeed } from "../stores/generation";
import type { PickedImage } from "../lib/generateForm";
import { mergeInstalledModels } from "../lib/generateModels";
import type { ChainLimits, ModelEntry, ResourceSnapshot } from "../lib/api/types";

const router = useRouter();

/** The chain composer is the Sequence side; Single returns to /create. */
function setComposerMode(mode: string | number) {
  if (mode === "single") void router.push("/create");
}

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
  installedModels.value.filter(
    (model) => isVideoFamily(model.family) && modelSupportsSequence(model),
  ),
);
// The picker lists every installed video checkpoint — incompatible ones stay
// visible but disabled with the reason, instead of silently vanishing.
const pickerVideoModels = computed<ModelEntry[]>(() =>
  installedModels.value.filter((model) => isVideoFamily(model.family)),
);
const selectedVideoModel = computed<ModelEntry | null>(
  () => installedModels.value.find((m) => m.name === form.model) ?? null,
);
function sequenceDisabledReason(m: ModelEntry): string | null {
  if (!modelSupportsSequence(m)) return "Two-stage checkpoint — can't chain clips";
  if (isCudaOnlyOnThisMachine(m)) return "CUDA only — connect a CUDA machine";
  return null;
}
// LTX-2 needs CUDA; on a Metal Mac the option shows but can't be selected.
const isCudaOnlyOnThisMachine = (m?: ModelEntry) => {
  if (!m) return false;
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
const selectedModel = computed(() => videoModels.value.find((entry) => entry.name === form.model));
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
    !!selectedModel.value &&
    !isCudaOnlyOnThisMachine(selectedModel.value) &&
    !rendering.value &&
    stageErrors.value === 0 &&
    form.stages.length >= 2 &&
    form.stages.every((stage) => !!stage.prompt.trim()),
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

// ── TOML editor ──────────────────────────────────────────────────────────────
// The panel edits a draft of the canonical script; Apply round-trips it
// through the same parser as Open .toml. Form edits keep resyncing the draft
// until the user actually types, so the panel always mirrors the filmstrip.
const tomlEditor = ref<HTMLTextAreaElement | null>(null);
const tomlDraft = ref("");
const tomlError = ref<string | null>(null);
let tomlSynced = "";
const tomlDirty = computed(() => tomlDraft.value !== tomlSynced);
const tomlHighlighted = computed(() => `${highlightToml(tomlDraft.value)}\n`);

watch(
  [showToml, tomlText],
  ([open, text]) => {
    if (!open || tomlDirty.value) return;
    tomlDraft.value = text;
    tomlSynced = text;
    tomlError.value = null;
  },
  { immediate: true },
);
watch(
  [tomlDraft, showToml],
  () => {
    if (showToml.value && tomlEditor.value) autoGrowRows(tomlEditor.value);
  },
  { flush: "post" },
);

function applyToml() {
  try {
    const parsed = tomlToChainForm(tomlDraft.value);
    Object.assign(form, parsed);
    if (parsed.model && videoModels.value.some((m) => m.name === parsed.model)) {
      void loadLimits(parsed.model);
    }
    tomlSynced = tomlText.value;
    tomlDraft.value = tomlSynced;
    tomlError.value = null;
    toasts.push("Applied to the sequence");
  } catch (err) {
    tomlError.value = err instanceof Error ? err.message : String(err);
  }
}

function revertToml() {
  tomlDraft.value = tomlText.value;
  tomlSynced = tomlDraft.value;
  tomlError.value = null;
}

async function copyToml() {
  try {
    await navigator.clipboard.writeText(tomlDraft.value);
    toasts.push("Copied chain.toml");
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

// File tools popover: a controlled menu that dismisses on outside pointerdown,
// Escape, and after any action (native <details> never closes itself).
const fileToolsEl = ref<HTMLDivElement | null>(null);
const fileToolsOpen = ref(false);
function fileTool(action: () => void) {
  fileToolsOpen.value = false;
  action();
}
function onFileToolsPointerDown(event: PointerEvent) {
  if (!fileToolsOpen.value || !fileToolsEl.value) return;
  if (!event.composedPath().includes(fileToolsEl.value)) fileToolsOpen.value = false;
}
function onFileToolsKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") fileToolsOpen.value = false;
}
onMounted(() => {
  document.addEventListener("pointerdown", onFileToolsPointerDown);
  document.addEventListener("keydown", onFileToolsKeydown);
});
onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onFileToolsPointerDown);
  document.removeEventListener("keydown", onFileToolsKeydown);
});
</script>

<template>
  <EmptyState
    v-if="conn.ready && !models.loading && videoModels.length === 0"
    headline="Sequences need a video model"
    detail="Pull a chain-capable LTX Video or distilled LTX-2 checkpoint, then tell the story one clip at a time."
    action="Browse video models"
    @action="router.push('/models?tab=discover&type=video&kind=checkpoint&intent=sequence')"
  />

  <div v-else class="flex h-full flex-col overflow-y-auto">
    <!-- Same 52px header anatomy as Create: title · summary · mode switch ·
         spacer · routing control (the model picker plays the host chip's role,
         since a sequence routes by its model's host). -->
    <header
      data-test="chain-header"
      class="border-edge flex h-[52px] shrink-0 items-center gap-3 border-b px-[22px]"
    >
      <span class="font-display text-[15px] font-semibold text-ink">Untitled sequence</span>
      <span
        data-test="chain-summary"
        class="edge-code rounded-full border border-edge px-2 py-[3px] text-ink-3"
      >
        {{ form.stages.length }} clips · {{ form.width }}×{{ form.height }} · {{ form.fps }} fps
      </span>
      <SegmentedControl
        model-value="sequence"
        compact
        :options="[
          { value: 'single', label: 'Single' },
          { value: 'sequence', label: 'Sequence' },
        ]"
        label="Composer mode"
        data-test="composer-mode"
        @update:model-value="setComposerMode"
      />
      <div class="flex-1" />
      <div class="w-72 min-w-0">
        <ModelPicker
          :models="pickerVideoModels"
          :selected="selectedVideoModel"
          :disabled-reason="sequenceDisabledReason"
          browse-target="/models?tab=discover&type=video&kind=checkpoint&intent=sequence"
          browse-label="Browse video models →"
          @pick="(m) => pickModel(m.name)"
        />
      </div>
    </header>

    <div class="border-edge border-b bg-bench px-4 py-3">
      <details class="rounded-control border border-edge bg-bath/60 px-3 py-2">
        <summary class="cursor-pointer text-caption font-medium text-ink-2">Sequence tools</summary>
        <div class="mt-3 flex flex-wrap items-end gap-4">
          <div>
            <label class="edge-code">Size</label>
            <div class="mt-1 flex items-center gap-1">
              <input
                v-model.number="form.width"
                type="number"
                step="16"
                class="border-edge data-mono h-8 w-20 rounded-control border bg-bench px-1.5 text-ink"
              />
              <span class="text-ink-3">×</span>
              <input
                v-model.number="form.height"
                type="number"
                step="16"
                class="border-edge data-mono h-8 w-20 rounded-control border bg-bench px-1.5 text-ink"
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
              class="border-edge data-mono mt-1 block h-8 w-16 rounded-control border bg-bench px-1.5 text-ink"
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
                class="border-edge data-mono h-8 w-28 rounded-control border bg-bench px-1.5 text-ink placeholder:text-ink-3"
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
            <label class="edge-code">Opening image</label>
            <div class="mt-1 flex items-center gap-1">
              <button
                type="button"
                data-test="chain-start-attach"
                class="border-edge h-8 rounded-control border border-dashed px-2 text-caption text-ink-3 hover:text-ink"
                @click="pickerFor = 'start'"
              >
                {{ form.startImage ? "Replace…" : "Attach…" }}
              </button>
              <button
                v-if="form.startImage"
                type="button"
                data-test="chain-start-clear"
                class="text-ink-3 hover:text-stop"
                aria-label="Remove start image"
                @click="form.startImage = null"
              >
                ✕
              </button>
            </div>
          </div>
          <label
            v-if="limits?.supports_audio"
            class="flex h-8 items-center gap-1 text-caption text-ink-2"
          >
            <input v-model="form.enableAudio" type="checkbox" class="accent-[var(--safelight)]" />
            Generate audio
          </label>
        </div>
      </details>

      <p
        v-if="selectedVideoModel && isCudaOnlyOnThisMachine(selectedVideoModel)"
        class="w-full text-caption text-halide"
      >
        LTX-2 generates on CUDA GPUs only. Connect a remote Linux engine to use it.
      </p>
    </div>

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
        + Add clip
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
        <div ref="fileToolsEl" class="relative">
          <button
            type="button"
            data-test="file-tools-toggle"
            class="border-edge flex h-8 cursor-pointer items-center rounded-control border px-3 text-body text-ink-2 hover:text-ink"
            :aria-expanded="fileToolsOpen"
            aria-haspopup="menu"
            @click="fileToolsOpen = !fileToolsOpen"
          >
            File tools
          </button>
          <div
            v-if="fileToolsOpen"
            data-test="file-tools-menu"
            role="menu"
            class="absolute right-0 bottom-10 z-10 flex w-40 flex-col gap-1 rounded-control border border-edge bg-bath p-2 shadow-xl"
          >
            <button
              type="button"
              role="menuitem"
              class="h-8 text-left text-caption text-ink-2 hover:text-ink"
              @click="fileTool(openToml)"
            >
              Open .toml…
            </button>
            <button
              type="button"
              role="menuitem"
              class="h-8 text-left text-caption text-ink-2 hover:text-ink"
              @click="fileTool(() => (showToml = !showToml))"
            >
              {{ showToml ? "Hide TOML" : "View as TOML" }}
            </button>
            <button
              type="button"
              role="menuitem"
              class="h-8 text-left text-caption text-ink-2 hover:text-ink"
              @click="fileTool(exportToml)"
            >
              Export .toml
            </button>
          </div>
        </div>
        <button
          type="button"
          data-test="generate-sequence"
          class="h-8 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent hover:brightness-105 disabled:opacity-50"
          :disabled="!canRender"
          @click="render"
        >
          {{ rendering ? "Starting…" : "Generate sequence" }}
        </button>
      </div>
    </div>

    <!-- TOML editor: syntax-highlighted overlay; Apply parses via the same
         tomlToChainForm path as Open .toml. -->
    <div v-if="showToml" data-test="toml-panel" class="border-edge border-b p-4">
      <div class="flex items-center justify-between pb-2">
        <span class="edge-code">chain.toml</span>
        <div class="flex items-center gap-2">
          <button
            type="button"
            class="border-edge rounded-control border px-2.5 py-1 text-caption text-ink-2 hover:text-ink"
            @click="copyToml"
          >
            Copy
          </button>
          <button
            v-if="tomlDirty"
            type="button"
            data-test="toml-revert"
            class="border-edge rounded-control border px-2.5 py-1 text-caption text-ink-2 hover:text-ink"
            @click="revertToml"
          >
            Revert
          </button>
          <button
            type="button"
            data-test="toml-apply"
            class="rounded-control bg-safelight px-2.5 py-1 text-caption font-semibold text-on-accent disabled:opacity-50"
            :disabled="!tomlDirty"
            @click="applyToml"
          >
            Apply
          </button>
        </div>
      </div>
      <div class="ms-toml">
        <pre aria-hidden="true" class="ms-toml__hl" v-html="tomlHighlighted"></pre>
        <textarea
          ref="tomlEditor"
          v-model="tomlDraft"
          data-test="toml-editor"
          data-selectable
          rows="4"
          spellcheck="false"
          autocapitalize="off"
          autocomplete="off"
          aria-label="Chain TOML script"
          class="ms-toml__input"
        />
      </div>
      <p v-if="tomlError" data-test="toml-error" role="alert" class="mt-2 text-caption text-stop">
        {{ tomlError }}
      </p>
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

<style scoped>
/* Overlay editor: the highlighted <pre> paints underneath a transparent-text
   textarea; identical font metrics + padding keep the layers aligned. */
.ms-toml {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  transition: border-color 0.1s;
}
.ms-toml:focus-within {
  border-color: var(--safelight);
}
.ms-toml__hl,
.ms-toml__input {
  margin: 0;
  padding: 12px 14px;
  font-family: var(--f-mono);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  tab-size: 2;
}
.ms-toml__hl {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  color: var(--ink-2);
}
.ms-toml__input {
  position: relative;
  display: block;
  width: 100%;
  border: 0;
  background: transparent;
  color: transparent;
  caret-color: var(--rebate);
  outline: none;
  resize: none;
  overflow: hidden;
}
.ms-toml__hl :deep(.tk-table) {
  color: var(--safelight);
  font-weight: 700;
}
.ms-toml__hl :deep(.tk-key) {
  color: var(--rebate);
}
.ms-toml__hl :deep(.tk-str) {
  color: var(--success);
}
.ms-toml__hl :deep(.tk-num),
.ms-toml__hl :deep(.tk-bool) {
  color: var(--halide);
}
.ms-toml__hl :deep(.tk-comment) {
  color: var(--ink-3);
  font-style: italic;
}
</style>
