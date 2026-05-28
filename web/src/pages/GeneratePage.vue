<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import Composer from "../components/Composer.vue";
import GenerateParamsPanel from "../components/GenerateParamsPanel.vue";
type GenerateParamsPanelInstance = InstanceType<typeof GenerateParamsPanel>;
import LoraPicker from "../components/LoraPicker.vue";
import ModelPicker from "../components/ModelPicker.vue";
import PreferencesModal from "../components/PreferencesModal.vue";
import ExpandModal from "../components/ExpandModal.vue";
import ImagePickerModal from "../components/ImagePickerModal.vue";
import RunningStrip from "../components/RunningStrip.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import DetailDrawer from "../components/DetailDrawer.vue";
import TopBar from "../components/TopBar.vue";
import {
  deleteGalleryImage,
  fetchModels,
  listGallery,
  updateQueueJobTargetGpu,
} from "../api";
import {
  applyMetadataToForm,
  useGenerateForm,
} from "../composables/useGenerateForm";
import { isQwenImageEditFamily } from "../composables/useGenerateForm";
import { useGenerateStream, type Job } from "../composables/useGenerateStream";
import { useQueue } from "../composables/useQueue";
import { decideChainRouting } from "../lib/chainRouting";
import { isStandaloneGenerationModel } from "../lib/modelFilters";
import { useStatusPoll } from "../composables/useStatusPoll";
import type {
  ChainRequestWire,
  ChainStageWire,
  ExpandFormState,
  GalleryImage,
  LoraSelection,
  ModelInfoExtended,
  SourceImageState,
} from "../types";
import { supportsLora } from "../types";
import type { ChainScriptToml } from "../lib/chainToml";
import type { ComposerMode } from "../components/Composer.vue";

type ViewMode = "feed" | "grid";

function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem("mold.gallery.view");
    return v === "grid" ? "grid" : "feed";
  } catch {
    return "feed";
  }
}
function loadMuted(): boolean {
  try {
    return localStorage.getItem("mold.gallery.muted") !== "false";
  } catch {
    return true;
  }
}
function persistViewMode(v: ViewMode) {
  try {
    localStorage.setItem("mold.gallery.view", v);
  } catch {
    /* ignore */
  }
}
function persistMuted(v: boolean) {
  try {
    localStorage.setItem("mold.gallery.muted", String(v));
  } catch {
    /* ignore */
  }
}

const form = useGenerateForm();
const { status } = useStatusPoll();
const queue = useQueue();
const models = ref<ModelInfoExtended[]>([]);
const galleryEntries = ref<GalleryImage[]>([]);
const view = ref<ViewMode>(loadViewMode());
const muted = ref(loadMuted());

const showPreferences = ref(false);
const showExpand = ref(false);
const showPicker = ref(false);
const composerError = ref<string | null>(null);

function loadComposerMode(): ComposerMode {
  try {
    const v = localStorage.getItem("mold.composer.mode");
    return v === "script" ? "script" : "single";
  } catch {
    return "single";
  }
}
const composerMode = ref<ComposerMode>(loadComposerMode());
function setComposerMode(v: ComposerMode) {
  composerMode.value = v;
  try {
    localStorage.setItem("mold.composer.mode", v);
  } catch {
    /* ignore */
  }
}

const expandStageIndex = ref<number | null>(null);
const composerRef = ref<InstanceType<typeof Composer> | null>(null);
const paramsPanelRef = ref<GenerateParamsPanelInstance | null>(null);

// Drawer state (mirrors GalleryPage).
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

// The running-strip card auto-dismisses inside the SSE singleton (see
// `scheduleAutoRemoveOnDone` in useGenerateStream); here we only need to
// pull in the freshly-saved gallery row when a job transitions to "done".
//
// We deliberately *don't* use the singleton's `onComplete` listener for
// this — that listener is only registered while GeneratePage is mounted,
// so a job that completes while the user is sitting on /catalog or
// /gallery would never trigger a refresh. A watcher on `stream.jobs`
// keyed on the set of done ids reconciles immediately on mount and
// fires for any subsequent transitions, regardless of which route was
// active when the SSE complete event arrived.
const stream = useGenerateStream();

async function refreshModels() {
  try {
    models.value = await fetchModels();
  } catch (e) {
    console.error(e);
  }
}

async function refreshGallery() {
  try {
    galleryEntries.value = await listGallery();
  } catch {
    /* ignore */
  }
}

// Computed of done ids joined into a stable string: changes only when
// the *set* of done ids changes, not on every progress event. Avoids
// `{ deep: true }` which would refire on every denoise tick.
const doneJobIds = computed(() =>
  stream.jobs.value
    .filter((j) => j.state === "done")
    .map((j) => j.id)
    .join(","),
);

// Track ids we've already triggered a refresh for so the same completion
// can't fire `refreshGallery()` twice (e.g. on remount when the job is
// still in the "done" pre-auto-remove window).
const seenDoneIds = new Set<string>();
watch(
  doneJobIds,
  () => {
    let added = false;
    for (const j of stream.jobs.value) {
      if (j.state !== "done") continue;
      if (seenDoneIds.has(j.id)) continue;
      seenDoneIds.add(j.id);
      added = true;
    }
    if (added) void refreshGallery();
  },
  { immediate: true },
);

// ── Auto-refresh ──────────────────────────────────────────────────────────
// Gallery gets new entries whenever a job completes (via stream onComplete)
// or whenever someone drops a file into MOLD_OUTPUT_DIR out-of-band (e.g.
// `mold run --local` on the same host). Polling every 10 s catches the
// second case without hammering the server. Models poll less often; new
// entries show up when `mold pull` finishes or a new weight is dropped in.
let galleryTimer: ReturnType<typeof setInterval> | null = null;
let modelsTimer: ReturnType<typeof setInterval> | null = null;

function startAutoRefresh() {
  stopAutoRefresh();
  galleryTimer = setInterval(() => {
    if (!document.hidden) void refreshGallery();
  }, 10_000);
  modelsTimer = setInterval(() => {
    if (!document.hidden) void refreshModels();
  }, 15_000);
}

function stopAutoRefresh() {
  if (galleryTimer) {
    clearInterval(galleryTimer);
    galleryTimer = null;
  }
  if (modelsTimer) {
    clearInterval(modelsTimer);
    modelsTimer = null;
  }
}

const currentModel = computed(
  () => models.value.find((m) => m.name === form.state.value.model) ?? null,
);

const gpus = computed(
  () =>
    status.value?.gpus?.map((g) => ({ ordinal: g.ordinal, state: g.state })) ??
    [],
);

const gpuListForPlacement = computed(
  () =>
    status.value?.gpus?.map((g) => ({
      ordinal: g.ordinal,
      name: `GPU ${g.ordinal}`,
    })) ?? [],
);

// Placeholder TopBar props — filters/search/mute/refresh/counts are all
// hidden on /generate by TopBar's `v-if="$route.name === 'gallery'"`, but
// the props are still declared required on the component.
const topBarCounts = computed(() => ({
  total: galleryEntries.value.length,
  images: galleryEntries.value.length,
  video: 0,
  filtered: galleryEntries.value.length,
}));

const chainDecision = computed(() =>
  decideChainRouting(
    form.state.value.frames,
    currentModel.value?.family ?? null,
    form.state.value.model,
  ),
);

const currentFamily = computed(
  () => currentModel.value?.family ?? form.state.value.modelFamily,
);

const familySupportsLora = computed(() => supportsLora(currentFamily.value));

function selectModel(model: ModelInfoExtended) {
  form.applyModelDefaults(model);
}

function onLorasChange(loras: LoraSelection[]) {
  form.state.value.loras = loras;
}

function onAppendPromptPhrase(phrase: string) {
  form.appendPromptPhrase(phrase);
}

function onSubmit() {
  composerError.value = null;
  if (!form.state.value.model) {
    // First-run / nothing-downloaded case. The user has to pick a model
    // before submit can do anything; force-expand the params panel so
    // the ModelPicker is reachable without hunting for the toggle.
    paramsPanelRef.value?.setExpanded(true);
    return;
  }
  const qwenImageEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  if (qwenImageEdit && form.state.value.imageAttachments.length === 0) {
    composerError.value = "Qwen image edit needs a target image.";
    return;
  }
  if (
    !qwenImageEdit &&
    form.state.value.maskImage &&
    form.state.value.imageAttachments.length === 0
  ) {
    composerError.value = "Mask image needs a source image.";
    return;
  }
  const decision = chainDecision.value;
  if (decision.kind === "reject") {
    // Block submit on a well-defined routing rejection (non-chainable
    // family over its per-clip budget). Keeping this as alert() matches
    // the existing terse validation UX — a toast system would be a
    // separate piece of work.
    alert(decision.reason);
    return;
  }
  const req = form.toRequest();
  stream.submit(req, decision);
  if (
    form.state.value.seedMode === "increment" &&
    form.state.value.seed !== null
  ) {
    form.state.value.seed += Math.max(1, form.state.value.batchSize);
  }
}

function onSubmitScript(script: ChainScriptToml) {
  const stages: ChainStageWire[] = script.stage.map((s) => ({
    prompt: s.prompt,
    frames: s.frames,
    transition: s.transition,
    fade_frames: s.fade_frames,
    negative_prompt: s.negative_prompt,
    seed_offset: s.seed_offset,
    source_image: s.source_image_b64 ?? null,
  }));
  const req: ChainRequestWire = {
    model: script.chain.model,
    stages,
    motion_tail_frames: script.chain.motion_tail_frames,
    width: script.chain.width,
    height: script.chain.height,
    fps: script.chain.fps,
    seed: script.chain.seed ?? null,
    steps: script.chain.steps,
    guidance: script.chain.guidance,
    strength: script.chain.strength,
    output_format: script.chain.output_format,
    enable_audio: script.chain.enable_audio,
  };
  const decision = {
    kind: "chain" as const,
    clipFrames: stages[0]?.frames ?? 97,
    motionTail: script.chain.motion_tail_frames,
    stageCount: stages.length,
  };
  stream.submit(req, decision);
}

const expandStagePrompt = ref("");

function onExpandStage(stageIndex: number, prompt: string) {
  expandStageIndex.value = stageIndex;
  expandStagePrompt.value = prompt;
  showExpand.value = true;
}

function onClearSource() {
  form.state.value.imageAttachments = [];
}

function onPickSource(v: SourceImageState[]) {
  const qwenEdit =
    isQwenImageEditFamily(
      currentModel.value?.family ?? form.state.value.modelFamily,
    ) || form.state.value.model.startsWith("qwen-image-edit:");
  form.state.value.imageAttachments = qwenEdit
    ? [...form.state.value.imageAttachments, ...v]
    : v.slice(0, 1);
  composerError.value = null;
}

function openItem(item: GalleryImage) {
  const idx = galleryEntries.value.findIndex(
    (e) => e.filename === item.filename,
  );
  selectedIndex.value = idx;
  selected.value = item;
}

function recreateFromGallery(item: GalleryImage) {
  form.state.value = applyMetadataToForm(form.state.value, item.metadata, {
    format: item.format,
    models: models.value,
  });
}

// Map a finished Job back to its saved GalleryImage. The SSE complete
// event doesn't echo the on-disk filename, so we match on seed + model
// against the freshly-refreshed gallery (sorted newest-first, so the
// first match is the right one if a seed happens to repeat).
function openJob(job: Job) {
  const r = job.result;
  if (!r) return;
  const match = galleryEntries.value.find(
    (e) => e.metadata.seed === r.seed_used && e.metadata.model === r.model,
  );
  if (match) openItem(match);
}
function closeDrawer() {
  selected.value = null;
  selectedIndex.value = -1;
}
function stepDrawer(delta: number) {
  if (selectedIndex.value < 0) return;
  const list = galleryEntries.value;
  const next = selectedIndex.value + delta;
  if (next < 0 || next >= list.length) return;
  selectedIndex.value = next;
  selected.value = list[next] ?? null;
}
async function handleDelete(item: GalleryImage) {
  try {
    await deleteGalleryImage(item.filename);
    galleryEntries.value = galleryEntries.value.filter(
      (e) => e.filename !== item.filename,
    );
    if (selected.value && selected.value.filename === item.filename) {
      closeDrawer();
    }
  } catch (e) {
    console.error(e);
  }
}

async function onQueueLaneChange(id: string, targetGpu: number | null) {
  try {
    await updateQueueJobTargetGpu(id, targetGpu);
    await queue.refresh();
  } catch (e) {
    console.error(e);
  }
}

function setView(v: ViewMode) {
  view.value = v;
  persistViewMode(v);
}
function setMuted(v: boolean) {
  muted.value = v;
  persistMuted(v);
}

const queueBusy = computed(() =>
  stream.jobs.value.some((j) => j.state === "running"),
);

onMounted(async () => {
  await refreshModels();
  try {
    galleryEntries.value = await listGallery();
  } catch (e) {
    console.error(e);
  }
  if (!form.state.value.model) {
    const first = models.value.find(
      (m) => m.downloaded && isStandaloneGenerationModel(m),
    );
    if (first) form.applyModelDefaults(first);
  }
  startAutoRefresh();
});

onBeforeUnmount(() => {
  stopAutoRefresh();
  queue.stop();
});
</script>

<template>
  <div
    data-test="generate-shell"
    class="mx-auto max-w-[2400px] px-3 pb-40 pt-4 sm:px-5 sm:pt-6 lg:px-6 2xl:px-8"
  >
    <TopBar
      :filter="'all'"
      :search="''"
      :view="view"
      :muted="muted"
      :counts="topBarCounts"
      :loading="false"
      @update:filter="() => {}"
      @update:search="() => {}"
      @update:view="setView"
      @update:muted="setMuted"
      @refresh="refreshGallery"
    />

    <div
      data-test="generate-workspace"
      class="mt-4 grid gap-4 sm:mt-6 xl:grid-cols-[21rem_minmax(0,1fr)_28rem] 2xl:grid-cols-[22rem_minmax(0,1fr)_30rem]"
    >
      <aside class="space-y-4 xl:sticky xl:top-4 xl:self-start">
        <section class="glass rounded-2xl p-4">
          <h2 class="text-sm font-semibold text-slate-100">Models</h2>
          <div class="mt-3">
            <ModelPicker
              :models="models"
              :model-value="form.state.value.model"
              @update:model-value="() => {}"
              @select="selectModel"
            />
          </div>
        </section>

        <section v-if="familySupportsLora" class="glass rounded-2xl p-4">
          <h2 class="text-sm font-semibold text-slate-100">LoRA Stack</h2>
          <LoraPicker
            :family="currentFamily"
            :model-value="form.state.value.loras"
            @update:model-value="onLorasChange"
            @append-prompt="onAppendPromptPhrase"
          />
        </section>
      </aside>

      <main class="min-w-0">
        <Composer
          ref="composerRef"
          v-model="form.state.value"
          :mode="composerMode"
          :queue-depth="status?.queue_depth ?? null"
          :queue-capacity="status?.queue_capacity ?? null"
          :gpus="gpus"
          :expand-active="form.state.value.expand.enabled"
          :family="currentFamily"
          :placement-gpus="gpuListForPlacement"
          :chain-decision="chainDecision"
          :submit-error="composerError"
          @submit="onSubmit"
          @submit-script="onSubmitScript"
          @update:mode="setComposerMode"
          @open-preferences="showPreferences = true"
          @open-expand="showExpand = true"
          @open-expand-stage="(idx: number, p: string) => onExpandStage(idx, p)"
          @open-image-picker="showPicker = true"
          @clear-source="onClearSource"
        />

        <RunningStrip
          :jobs="stream.jobs.value"
          :queue-entries="queue.entries.value"
          :gpus="gpus"
          @cancel="stream.cancel"
          @open="openJob"
          @dismiss="stream.remove"
          @clear-finished="stream.clearDone"
          @lane-change="onQueueLaneChange"
        />

        <section class="mt-4">
          <div class="mb-2 flex items-center justify-between">
            <h2 class="text-sm font-semibold text-slate-100">Recent</h2>
            <span class="text-xs text-slate-500"
              >{{ galleryEntries.length }} items</span
            >
          </div>
          <div class="max-h-[52rem] overflow-y-auto pr-1">
            <GalleryFeed
              :entries="galleryEntries"
              :loading="false"
              :view="'grid'"
              :muted="muted"
              :show-recreate="true"
              @open="openItem"
              @recreate="recreateFromGallery"
            />
          </div>
        </section>
      </main>

      <aside class="xl:sticky xl:top-4 xl:self-start">
        <GenerateParamsPanel
          ref="paramsPanelRef"
          v-model="form.state.value"
          :models="models"
          :placement-gpus="gpuListForPlacement"
          :force-expanded="true"
          :show-model-picker="false"
          :show-loras="false"
          title="Controls"
        />
      </aside>
    </div>

    <PreferencesModal
      :open="showPreferences"
      @close="showPreferences = false"
    />
    <ExpandModal
      :open="showExpand"
      :prompt="
        expandStageIndex !== null ? expandStagePrompt : form.state.value.prompt
      "
      :expand="form.state.value.expand"
      :current-model="currentModel"
      :queue-busy="queueBusy"
      @update:expand="(v: ExpandFormState) => (form.state.value.expand = v)"
      @apply-prompt="
        (v: string) => {
          if (expandStageIndex !== null) {
            composerRef?.scriptComposerRef?.setStagePrompt(expandStageIndex, v);
          } else {
            form.state.value.prompt = v;
          }
        }
      "
      @close="
        showExpand = false;
        expandStageIndex = null;
      "
    />
    <ImagePickerModal
      :open="showPicker"
      @pick="onPickSource"
      @close="showPicker = false"
    />

    <DetailDrawer
      :item="selected"
      :has-prev="selectedIndex > 0"
      :has-next="
        selectedIndex >= 0 && selectedIndex < galleryEntries.length - 1
      "
      :index="selectedIndex"
      :total="galleryEntries.length"
      :muted="muted"
      @close="closeDrawer"
      @prev="stepDrawer(-1)"
      @next="stepDrawer(1)"
      @delete="handleDelete"
    />
  </div>
</template>
