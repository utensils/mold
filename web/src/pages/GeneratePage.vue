<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import Composer from "../components/Composer.vue";
import SettingsModal from "../components/SettingsModal.vue";
import ExpandModal from "../components/ExpandModal.vue";
import ImagePickerModal from "../components/ImagePickerModal.vue";
import RunningStrip from "../components/RunningStrip.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import DetailDrawer from "../components/DetailDrawer.vue";
import TopBar from "../components/TopBar.vue";
import {
  deleteGalleryImage,
  fetchCapabilities,
  fetchModels,
  listGallery,
} from "../api";
import { useGenerateForm } from "../composables/useGenerateForm";
import { useGenerateStream, type Job } from "../composables/useGenerateStream";
import { useHideMode } from "../composables/useHideMode";
import { decideChainRouting } from "../lib/chainRouting";
import { useStatusPoll } from "../composables/useStatusPoll";
import type {
  ChainStageWire,
  ExpandFormState,
  GalleryImage,
  ModelInfoExtended,
  ServerCapabilities,
  SourceImageState,
} from "../types";
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
// Shared privacy toggle with Gallery — see useHideMode for button semantics.
const hide = useHideMode();
const models = ref<ModelInfoExtended[]>([]);
const galleryEntries = ref<GalleryImage[]>([]);
const view = ref<ViewMode>(loadViewMode());
const muted = ref(loadMuted());
const capabilities = ref<ServerCapabilities>({
  gallery: { can_delete: false },
});

const showSettings = ref(false);
const showExpand = ref(false);
const showPicker = ref(false);

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

// Drawer state (mirrors GalleryPage).
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

const stream = useGenerateStream(async () => {
  try {
    galleryEntries.value = await listGallery();
  } catch {
    /* leave previous */
  }
});

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

// Faster model polling while the advanced/settings modal is open — the user
// is likely waiting for a download to complete so they can pick the new
// variant without hitting a manual refresh.
let settingsModelsTimer: ReturnType<typeof setInterval> | null = null;
watch(
  () => showSettings.value,
  (open) => {
    if (settingsModelsTimer) {
      clearInterval(settingsModelsTimer);
      settingsModelsTimer = null;
    }
    if (open) {
      settingsModelsTimer = setInterval(() => {
        if (!document.hidden) void refreshModels();
      }, 3_000);
    }
  },
);

const currentModel = computed(
  () => models.value.find((m) => m.name === form.state.value.model) ?? null,
);

const gpus = computed(
  () =>
    status.value?.gpus?.map((g) => ({ ordinal: g.ordinal, state: g.state })) ??
    null,
);

// TODO-REMOVE-AFTER-MERGE: use useResources().gpuList once Agent B merges.
const gpuListForPlacement = computed(
  () =>
    status.value?.gpus?.map((g) => ({
      ordinal: g.ordinal,
      name: `GPU ${g.ordinal}`,
    })) ?? [],
);

const settingsDirty = computed(() => {
  const s = form.state.value;
  const m = currentModel.value;
  if (!m) return false;
  return (
    s.width !== m.default_width ||
    s.height !== m.default_height ||
    s.steps !== m.default_steps ||
    Math.abs(s.guidance - m.default_guidance) > 0.001 ||
    s.batchSize !== 1 ||
    s.seed !== null ||
    s.negativePrompt.length > 0
  );
});

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

function onSubmit() {
  if (!form.state.value.model) {
    showSettings.value = true;
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
}

function onSubmitScript(script: ChainScriptToml) {
  const stages: ChainStageWire[] = script.stage.map((s) => ({
    prompt: s.prompt,
    frames: s.frames,
    transition: s.transition,
    fade_frames: s.fade_frames,
    negative_prompt: s.negative_prompt,
    seed_offset: s.seed_offset,
  }));
  const req = {
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
  };
  const decision = {
    kind: "chain" as const,
    clipFrames: stages[0]?.frames ?? 97,
    motionTail: script.chain.motion_tail_frames,
    stageCount: stages.length,
  };
  stream.submit(req as never, decision);
}

const expandStagePrompt = ref("");

function onExpandStage(stageIndex: number, prompt: string) {
  expandStageIndex.value = stageIndex;
  expandStagePrompt.value = prompt;
  showExpand.value = true;
}

function onClearSource() {
  form.state.value.sourceImage = null;
}

function onPickSource(v: SourceImageState) {
  form.state.value.sourceImage = v;
}

function openItem(item: GalleryImage) {
  const idx = galleryEntries.value.findIndex(
    (e) => e.filename === item.filename,
  );
  selectedIndex.value = idx;
  selected.value = item;
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
  try {
    capabilities.value = await fetchCapabilities();
  } catch {
    /* keep default */
  }
  if (!form.state.value.model) {
    const first = models.value.find((m) => m.downloaded);
    if (first) form.applyModelDefaults(first);
  }
  startAutoRefresh();
});

onBeforeUnmount(() => {
  stopAutoRefresh();
  if (settingsModelsTimer) {
    clearInterval(settingsModelsTimer);
    settingsModelsTimer = null;
  }
});
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-4 sm:px-6 sm:pt-6 lg:px-10">
    <TopBar
      :filter="'all'"
      :search="''"
      :view="view"
      :muted="muted"
      :counts="topBarCounts"
      :loading="false"
      :hide-mode="!hide.anyVisible.value"
      @update:filter="() => {}"
      @update:search="() => {}"
      @update:view="setView"
      @update:muted="setMuted"
      @update:hide-mode="hide.toggle"
      @refresh="refreshGallery"
    />

    <div class="mt-4 sm:mt-6">
      <Composer
        ref="composerRef"
        v-model="form.state.value"
        :mode="composerMode"
        :queue-depth="status?.queue_depth ?? null"
        :queue-capacity="status?.queue_capacity ?? null"
        :gpus="gpus"
        :expand-active="form.state.value.expand.enabled"
        :settings-dirty="settingsDirty"
        :family="currentModel?.family ?? ''"
        :placement-gpus="gpuListForPlacement"
        :chain-decision="chainDecision"
        @submit="onSubmit"
        @submit-script="onSubmitScript"
        @update:mode="setComposerMode"
        @open-settings="showSettings = true"
        @open-expand="showExpand = true"
        @open-expand-stage="(idx: number, p: string) => onExpandStage(idx, p)"
        @open-image-picker="showPicker = true"
        @clear-source="onClearSource"
      />

      <RunningStrip
        :jobs="stream.jobs.value"
        :hide-mode="hide.hideMode.value"
        :revealed="hide.revealed.value"
        @cancel="stream.cancel"
        @open="openJob"
        @dismiss="stream.remove"
        @clear-finished="stream.clearDone"
        @reveal="(id: string) => hide.revealOne(id)"
      />

      <div class="mt-6">
        <GalleryFeed
          :entries="galleryEntries"
          :loading="false"
          :view="view"
          :muted="muted"
          :hide-mode="hide.hideMode.value"
          :revealed="hide.revealed.value"
          @open="openItem"
          @reveal="(item: GalleryImage) => hide.revealOne(item.filename)"
        />
      </div>
    </div>

    <SettingsModal
      :open="showSettings"
      v-model="form.state.value"
      :models="models"
      @close="showSettings = false"
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
      :can-delete="capabilities.gallery.can_delete"
      :muted="muted"
      @close="closeDrawer"
      @prev="stepDrawer(-1)"
      @next="stepDrawer(1)"
      @delete="handleDelete"
    />
  </div>
</template>
