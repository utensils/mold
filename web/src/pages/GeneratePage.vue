<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Composer from "../components/Composer.vue";
import ComposerSidebar from "../components/ComposerSidebar.vue";
import ResourceStrip from "../components/ResourceStrip.vue";
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
import { useStatusPoll } from "../composables/useStatusPoll";
import { useStarred } from "../composables/useStarred";
import type {
  ExpandFormState,
  GalleryImage,
  ModelInfoExtended,
  ServerCapabilities,
  SourceImageState,
} from "../types";

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
const { starred, toggle: toggleStar } = useStarred();

/*
 * Re-run: when the user clicks "Re-run" on a gallery item (from any
 * surface — feed card, detail drawer, or inline feed under /generate),
 * we populate the composer's form state from the item's metadata.
 *
 * Cross-page transitions hand off through a `window.__moldRerun` stash
 * set by GalleryPage; same-page transitions (inline feed / detail drawer
 * under /generate) call `applyRerun` directly.
 */
type RerunPayload = {
  prompt?: string | null;
  negative_prompt?: string | null;
  model?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  seed?: number | null;
  scheduler?: unknown;
  strength?: number | null;
  frames?: number | null;
  fps?: number | null;
};

function applyRerun(p: RerunPayload) {
  const s = form.state.value;
  s.prompt = p.prompt || "";
  s.negativePrompt = p.negative_prompt || "";
  if (p.model) s.model = p.model;
  if (p.width) s.width = p.width;
  if (p.height) s.height = p.height;
  if (p.steps !== undefined && p.steps !== null) s.steps = p.steps;
  if (p.guidance !== undefined && p.guidance !== null) s.guidance = p.guidance;
  s.seed = p.seed ?? null;
  if (p.strength !== undefined && p.strength !== null) s.strength = p.strength;
  if (p.frames !== undefined) s.frames = p.frames;
  if (p.fps !== undefined) s.fps = p.fps;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function rerunItem(item: GalleryImage) {
  applyRerun(item.metadata);
}
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

/*
 * Latest running job. We surface its progress to the sidebar so the big
 * Generate button can show a percentage, the progress bar underneath
 * can animate, and a short stage string can describe what the server
 * is doing. RunningStrip still shows the full queue.
 */
const latestJob = computed<Job | null>(() => {
  const js = stream.jobs.value;
  return js.find((j) => j.state === "running") ?? null;
});

const latestProgressPct = computed<number | null>(() => {
  const j = latestJob.value;
  if (!j) return null;
  const p = j.progress;
  if (p.totalSteps && p.step !== null) {
    return Math.round((p.step / p.totalSteps) * 100);
  }
  if (p.weightBytesTotal && p.weightBytesLoaded !== null) {
    return Math.round((p.weightBytesLoaded / p.weightBytesTotal) * 100);
  }
  return null;
});

const latestStage = computed<string | null>(
  () => latestJob.value?.progress.stage ?? null,
);

const canSubmit = computed(
  () =>
    form.state.value.prompt.trim().length > 0 &&
    form.state.value.model.length > 0,
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

function onSubmit() {
  if (!form.state.value.model) {
    showSettings.value = true;
    return;
  }
  const req = form.toRequest();
  stream.submit(req);
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

onMounted(async () => {
  // Pull rerun stash set by GalleryPage before we touch the form, so the
  // navigated-in prefill wins over a stale localStorage snapshot.
  const w = window as unknown as { __moldRerun?: RerunPayload };
  if (w.__moldRerun) {
    applyRerun(w.__moldRerun);
    delete w.__moldRerun;
  }
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
});
</script>

<template>
  <div class="mold-shell">
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

    <div class="gen mt-4 sm:mt-6">
      <div class="gen-main">
        <header class="gen-header">
          <div>
            <div class="gen-eyebrow">Compose</div>
            <h1 class="gen-title">New generation</h1>
          </div>
          <div class="gen-status">
            <span
              v-if="status?.queue_depth !== undefined"
              class="gen-queue"
              :title="`queue ${status.queue_depth} / ${status.queue_capacity}`"
            >
              queue {{ status.queue_depth }}/{{ status.queue_capacity }}
            </span>
            <span
              v-for="g in gpus ?? []"
              :key="g.ordinal"
              class="gen-gpu"
              :title="`GPU ${g.ordinal} · ${g.state}`"
            >
              <span
                style="
                  width: 6px;
                  height: 6px;
                  border-radius: 999px;
                  background: var(--accent);
                "
                :style="{
                  opacity: g.state === 'idle' ? 0.3 : 1,
                }"
              />
              GPU {{ g.ordinal }} {{ g.state }}
            </span>
          </div>
        </header>

        <Composer
          v-model="form.state.value"
          :queue-depth="status?.queue_depth ?? null"
          :queue-capacity="status?.queue_capacity ?? null"
          :gpus="gpus"
          :expand-active="form.state.value.expand.enabled"
          :settings-dirty="settingsDirty"
          :family="currentModel?.family ?? ''"
          :placement-gpus="gpuListForPlacement"
          @submit="onSubmit"
          @open-settings="showSettings = true"
          @open-expand="showExpand = true"
          @open-image-picker="showPicker = true"
          @clear-source="onClearSource"
        />

        <!-- Agent B: always-visible VRAM + RAM telemetry -->
        <ResourceStrip class="hidden lg:block" variant="full" />

        <RunningStrip
          :jobs="stream.jobs.value"
          @cancel="stream.cancel"
          @open="openJob"
        />
      </div>

      <ComposerSidebar
        v-model="form.state.value"
        :models="models"
        :running="latestJob !== null"
        :progress-pct="latestProgressPct"
        :progress-stage="latestStage"
        :can-submit="canSubmit"
        @submit="onSubmit"
      />
    </div>

    <div class="mt-6">
      <GalleryFeed
        :entries="galleryEntries"
        :loading="false"
        :view="view"
        :muted="muted"
        :starred="starred"
        @open="openItem"
        @star="(i: GalleryImage) => toggleStar(i.filename)"
        @rerun="rerunItem"
      />
    </div>

    <SettingsModal
      :open="showSettings"
      v-model="form.state.value"
      :models="models"
      @close="showSettings = false"
    />
    <ExpandModal
      :open="showExpand"
      :prompt="form.state.value.prompt"
      :expand="form.state.value.expand"
      :current-model="currentModel"
      @update:expand="(v: ExpandFormState) => (form.state.value.expand = v)"
      @apply-prompt="(v: string) => (form.state.value.prompt = v)"
      @close="showExpand = false"
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
      @rerun="rerunItem"
    />
  </div>
</template>
