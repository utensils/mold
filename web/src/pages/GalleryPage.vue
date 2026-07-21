<script setup lang="ts">
/*
 * Gallery workspace (Mold Studio W5, spec §06 + prototype WEB GALLERY /
 * LIGHTBOX). Grid-first: session prints render as MediaTiles on a responsive
 * grid, with a secondary feed view kept for existing users. All / Images /
 * Video filter, a search that narrows prompt + model + filename (synced to
 * `?q=`), marquee multi-select + bulk delete, and a two-pane / full-screen
 * Lightbox with reuse / use-as-source / download / delete.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import { deleteGalleryImage, imageUrl } from "../api";
import { blobToBase64 } from "../lib/base64";
import { requestConfirm, toast, undoableAction } from "../lib/toasts";
import {
  applyMetadataToForm,
  useGenerateForm,
} from "../composables/useGenerateForm";
import {
  fetchMergedGallery,
  type HostGalleryImage,
} from "../lib/multiHostGallery";
import {
  ORIGIN_HOST_ID,
  getHost,
  listHosts,
  originHost,
} from "../lib/hostRegistry";
import { hostDeleteGalleryImage } from "../components/machines/hostClient";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import GalleryGrid from "../components/gallery/GalleryGrid.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import Lightbox from "../components/gallery/Lightbox.vue";

type FilterKind = "all" | "images" | "video";
type ViewMode = "feed" | "grid";

// Persist the layout. The redesign is grid-first, so `grid` is the default;
// `feed` stays available as a single-column, prompt-forward stream.
const VIEW_STORAGE_KEY = "mold.gallery.view";
function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem(VIEW_STORAGE_KEY);
    if (v === "feed" || v === "grid") return v;
  } catch {
    /* localStorage may be blocked */
  }
  return "grid";
}

const entries = ref<HostGalleryImage[]>([]);
// Hosts whose /api/gallery failed this refresh (surfaced, not hidden), and
// how many non-origin hosts were attempted (drives the honest count line).
const unreachableHostIds = ref<string[]>([]);
const remoteHostCount = ref(0);
const loading = ref(true);
const errorMessage = ref<string | null>(null);
const filter = ref<FilterKind>("all");
const search = ref("");
const view = ref<ViewMode>(loadViewMode());

const form = useGenerateForm();
const route = useRoute();
const router = useRouter();

const filterOptions: SegmentOption<FilterKind>[] = [
  { value: "all", label: "All" },
  { value: "images", label: "Images" },
  { value: "video", label: "Video" },
];

// Seed the search from the global nav's `?q=` and keep the two in sync.
watch(
  () => route.query.q,
  (q) => {
    const next = typeof q === "string" ? q : "";
    if (next !== search.value) search.value = next;
  },
  { immediate: true },
);

function syncSearchToUrl(value: string) {
  const q = value.trim();
  const current = typeof route.query.q === "string" ? route.query.q : "";
  if (q === current) return;
  const query = { ...route.query };
  if (q) query.q = q;
  else delete query.q;
  void router.replace({ query });
}
function onSearchInput(value: string) {
  search.value = value;
  syncSearchToUrl(value);
}
function clearSearch() {
  search.value = "";
  syncSearchToUrl("");
}
function setFilter(next: FilterKind) {
  filter.value = next;
}

/*
 * Audio preference. Browsers forbid unmuted autoplay without a gesture, so we
 * default muted; the first toggle click doubles as that gesture. Persisted.
 */
const MUTED_STORAGE_KEY = "mold.gallery.muted";
function loadMuted(): boolean {
  try {
    const v = localStorage.getItem(MUTED_STORAGE_KEY);
    if (v === "false") return false;
    if (v === "true") return true;
  } catch {
    /* localStorage may be blocked */
  }
  return true;
}
const muted = ref<boolean>(loadMuted());
function setMuted(next: boolean) {
  muted.value = next;
  try {
    localStorage.setItem(MUTED_STORAGE_KEY, String(next));
  } catch {
    /* ignore */
  }
}

// ── NEW badge tracking ──────────────────────────────────────────────────────
// Filenames present on first load are "seen" (never badged). Anything that
// arrives on a later refresh is fresh until the next reload.
const seen = new Set<string>();
const fresh = ref<Set<string>>(new Set());
let firstLoadDone = false;
function reconcileFresh(list: GalleryImage[]) {
  if (!firstLoadDone) {
    for (const e of list) seen.add(e.filename);
    firstLoadDone = true;
    return;
  }
  let changed = false;
  const next = new Set(fresh.value);
  for (const e of list) {
    if (!seen.has(e.filename)) {
      seen.add(e.filename);
      next.add(e.filename);
      changed = true;
    }
  }
  if (changed) fresh.value = next;
}

// ── Multi-select ────────────────────────────────────────────────────────────
const selectMode = ref(false);
const selection = ref<Set<string>>(new Set());
const selectionAnchor = ref<string | null>(null);

function setSelectMode(next: boolean) {
  selectMode.value = next;
  if (!next) {
    selection.value = new Set();
    selectionAnchor.value = null;
  }
}

function toggleSelect(payload: {
  item: GalleryImage;
  shift: boolean;
  meta: boolean;
}) {
  const { item, shift, meta } = payload;
  const name = item.filename;
  if (shift && selectionAnchor.value) {
    const list = filtered.value;
    const a = list.findIndex((e) => e.filename === selectionAnchor.value);
    const b = list.findIndex((e) => e.filename === name);
    if (a === -1 || b === -1) {
      toggleOne(name, meta);
      return;
    }
    const [lo, hi] = a < b ? [a, b] : [b, a];
    const next = new Set(selection.value);
    for (let i = lo; i <= hi; i++) {
      const f = list[i]?.filename;
      if (f) next.add(f);
    }
    selection.value = next;
    return;
  }
  toggleOne(name, meta);
  selectionAnchor.value = name;
}

function toggleOne(name: string, _meta: boolean) {
  const next = new Set(selection.value);
  if (next.has(name)) next.delete(name);
  else next.add(name);
  selection.value = next;
}

function onDragSelect(payload: { filenames: string[] }) {
  selection.value = new Set(payload.filenames);
}

function selectAllVisible() {
  const next = new Set<string>();
  for (const e of filtered.value) next.add(e.filename);
  selection.value = next;
}

function clearSelection() {
  selection.value = new Set();
  selectionAnchor.value = null;
}

// Delete routes to the host that owns the print — a remote print is deleted on
// its host, not the origin.
function deleteRouted(filename: string): Promise<void> {
  const entry = entries.value.find((e) => e.filename === filename);
  const host = entry ? getHost(entry.hostId) : null;
  if (!host || host.id === ORIGIN_HOST_ID) return deleteGalleryImage(filename);
  return hostDeleteGalleryImage(host, filename);
}

async function handleDeleteMany(names: string[]): Promise<number> {
  const results = await Promise.allSettled(names.map((n) => deleteRouted(n)));
  const deleted = new Set<string>();
  let failed = 0;
  names.forEach((n, i) => {
    if (results[i]?.status === "fulfilled") deleted.add(n);
    else failed++;
  });
  entries.value = entries.value.filter((e) => !deleted.has(e.filename));
  if (deleted.size > 0) {
    const next = new Set(selection.value);
    for (const n of deleted) next.delete(n);
    selection.value = next;
  }
  if (failed > 0) {
    errorMessage.value = `Deleted ${deleted.size} of ${names.length}. ${failed} failed.`;
  }
  return deleted.size;
}

async function deleteSelected() {
  const names = Array.from(selection.value);
  if (names.length === 0) return;
  const accepted = await requestConfirm({
    title:
      names.length === 1 ? "Delete print?" : `Delete ${names.length} prints?`,
    body: "This can't be undone.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!accepted) return;
  await handleDeleteMany(names);
}

async function deleteAllFiltered() {
  const list = filtered.value;
  if (list.length === 0) return;
  const everything = list.length === entries.value.length;
  const accepted = await requestConfirm({
    title: everything
      ? `Delete all ${list.length} prints?`
      : `Delete ${list.length} filtered prints?`,
    body: "This can't be undone.",
    confirmLabel: "Delete",
    danger: true,
    typedPhrase: "delete",
  });
  if (!accepted) return;
  const names = list.map((e) => e.filename);
  await handleDeleteMany(names);
}

// ── Filtering ────────────────────────────────────────────────────────────────
const kindFiltered = computed(() => {
  if (filter.value === "all") return entries.value;
  return entries.value.filter((e) => {
    const k = mediaKind(e.format, e.filename);
    if (filter.value === "video") return k === "video" || k === "animated";
    return k === "image";
  });
});

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return kindFiltered.value;
  return kindFiltered.value.filter((e) => {
    if (e.filename.toLowerCase().includes(q)) return true;
    const m = e.metadata;
    if (m.model.toLowerCase().includes(q)) return true;
    if (m.prompt && m.prompt.toLowerCase().includes(q)) return true;
    return false;
  });
});

const total = computed(() => entries.value.length);
const searchActive = computed(() => search.value.trim().length > 0);

// The empty-state variant to show when nothing is visible and we're not
// mid-load. `null` means render the grid/feed (skeletons handle first load).
const emptyKind = computed<null | "none" | "search" | "video" | "images">(
  () => {
    if (loading.value || filtered.value.length > 0) return null;
    if (searchActive.value) return "search";
    if (filter.value === "video") return "video";
    if (filter.value === "images") return "images";
    return "none";
  },
);

async function refresh() {
  loading.value = true;
  errorMessage.value = null;
  try {
    const merged = await fetchMergedGallery(listHosts());
    entries.value = merged.entries;
    unreachableHostIds.value = merged.unreachableHostIds;
    remoteHostCount.value = merged.remoteHostCount;
    reconcileFresh(merged.entries);
    // Only a total wipe-out (no host answered) is an error; one box down just
    // shows an "unreachable" note while the rest render.
    if (
      merged.reachableHostIds.length === 0 &&
      merged.unreachableHostIds.length > 0
    ) {
      errorMessage.value = "Couldn't reach any host's gallery.";
    }
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

// Honest count line: "all hosts" only when remotes are actually connected,
// otherwise "this server". Names the unreachable hosts rather than hiding them.
const scopeLabel = computed(() =>
  remoteHostCount.value > 0 ? "all hosts" : "this server",
);
const unreachableLabel = computed(() => {
  const names = unreachableHostIds.value
    .map((id) => getHost(id)?.name ?? id)
    .filter((n) => n !== originHost().name);
  return names.length ? `${names.join(", ")} unreachable` : "";
});

// ── Lightbox ─────────────────────────────────────────────────────────────────
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

function openItem(item: GalleryImage) {
  selectedIndex.value = filtered.value.findIndex(
    (e) => e.filename === item.filename,
  );
  selected.value = item;
}
function closeLightbox() {
  selected.value = null;
  selectedIndex.value = -1;
}
function stepLightbox(delta: number) {
  if (selectedIndex.value < 0) return;
  const list = filtered.value;
  const next = selectedIndex.value + delta;
  if (next < 0 || next >= list.length) return;
  selectedIndex.value = next;
  selected.value = list[next] ?? null;
}

function onReuse(item: GalleryImage) {
  // Existing recreate flow — restore serialized knobs, then land on Create.
  form.state.value = applyMetadataToForm(form.state.value, item.metadata, {
    format: item.format,
  });
  closeLightbox();
  void router.push({ name: "create" });
}

async function setAsSource(item: GalleryImage): Promise<boolean> {
  try {
    const res = await fetch(imageUrl(item.filename));
    if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
    const blob = await res.blob();
    const base64 = await blobToBase64(blob);
    form.state.value.imageAttachments = [
      { kind: "gallery", filename: item.filename, base64 },
    ];
    return true;
  } catch (err) {
    toast("error", err instanceof Error ? err.message : String(err));
    return false;
  }
}

async function onUseAsSource(item: GalleryImage) {
  if (!(await setAsSource(item))) return;
  closeLightbox();
  void router.push({ name: "create" });
}

async function onUpscale(item: GalleryImage) {
  if (!(await setAsSource(item))) return;
  closeLightbox();
  toast("info", "Added as source — pick an upscaler in Controls.");
  void router.push({ name: "create" });
}

async function onLightboxDelete(item: GalleryImage) {
  const accepted = await requestConfirm({
    title: "Delete print?",
    body: `${item.filename} — you can undo for a few seconds.`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!accepted) return;
  const entryIdx = entries.value.findIndex((e) => e.filename === item.filename);
  if (entryIdx === -1) return;
  const removed = entries.value[entryIdx]!;

  // Optimistic removal; commit the DELETE only once the undo window elapses.
  entries.value = entries.value.filter((e) => e.filename !== item.filename);
  if (filtered.value.length === 0) {
    closeLightbox();
  } else {
    selectedIndex.value = Math.min(
      selectedIndex.value,
      filtered.value.length - 1,
    );
    selected.value = filtered.value[selectedIndex.value] ?? null;
    if (!selected.value) closeLightbox();
  }

  undoableAction({
    text: "Print deleted",
    undo: () => {
      const next = entries.value.slice();
      next.splice(Math.min(entryIdx, next.length), 0, removed);
      entries.value = next;
    },
    commit: async () => {
      try {
        await deleteRouted(item.filename);
      } catch (err) {
        // Roll the print back into the list so the failure isn't silent.
        const next = entries.value.slice();
        next.splice(Math.min(entryIdx, next.length), 0, removed);
        entries.value = next;
        toast("error", err instanceof Error ? err.message : String(err));
      }
    },
  });
}

function setView(next: ViewMode) {
  view.value = next;
  try {
    localStorage.setItem(VIEW_STORAGE_KEY, next);
  } catch {
    /* ignore */
  }
}

// ── Back-to-top FAB ──────────────────────────────────────────────────────────
const showBackToTop = ref(false);
function onScroll() {
  showBackToTop.value = window.scrollY > window.innerHeight;
}
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}
onMounted(() => {
  window.addEventListener("scroll", onScroll, { passive: true });
});
onBeforeUnmount(() => {
  window.removeEventListener("scroll", onScroll);
});

onMounted(async () => {
  await refresh();
});
</script>

<template>
  <div class="gal">
    <header class="gal__head">
      <h1 class="gal__title">Gallery</h1>
      <span class="gal__count" data-test="gallery-count"
        >{{ total }} prints · {{ scopeLabel
        }}<span v-if="unreachableLabel" class="gal__unreachable">
          · {{ unreachableLabel }}</span
        ></span
      >
      <span class="gal__flex"></span>

      <SegmentedControl
        class="gal__filter"
        data-test="gallery-filter"
        :model-value="filter"
        :options="filterOptions"
        label="Filter prints"
        @update:model-value="setFilter"
      />

      <label class="gal__search">
        <Icon name="search" :size="15" />
        <input
          :value="search"
          type="search"
          placeholder="Search prompts, models…"
          aria-label="Search gallery"
          data-test="gallery-search"
          @input="onSearchInput(($event.target as HTMLInputElement).value)"
        />
      </label>

      <div class="gal__tools">
        <div class="gal__viewtoggle" role="group" aria-label="View mode">
          <button
            type="button"
            class="gal__vbtn"
            :data-on="view === 'grid' ? 'true' : undefined"
            :aria-pressed="view === 'grid'"
            aria-label="Grid view"
            @click="setView('grid')"
          >
            <Icon name="library" :size="16" />
          </button>
          <button
            type="button"
            class="gal__vbtn"
            :data-on="view === 'feed' ? 'true' : undefined"
            :aria-pressed="view === 'feed'"
            aria-label="Feed view"
            @click="setView('feed')"
          >
            <Icon name="menu" :size="16" />
          </button>
        </div>

        <button
          type="button"
          class="gal__icon"
          :aria-pressed="!muted"
          :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
          :title="muted ? 'Unmute videos' : 'Mute videos'"
          @click="setMuted(!muted)"
        >
          <svg
            v-if="muted"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="m22 9-6 6" />
            <path d="m16 9 6 6" />
          </svg>
          <svg
            v-else
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="M15.5 8.5a5 5 0 0 1 0 7" />
            <path d="M18.5 5.5a9 9 0 0 1 0 13" />
          </svg>
        </button>

        <button
          type="button"
          class="gal__select"
          :class="{ 'gal__select--on': selectMode }"
          :aria-pressed="selectMode"
          data-test="gallery-select"
          @click="setSelectMode(!selectMode)"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"
            />
          </svg>
          <span class="gal__select-label">
            {{ selectMode ? `${selection.size} selected` : "Select" }}
          </span>
        </button>

        <button
          type="button"
          class="gal__icon"
          :disabled="loading"
          :aria-busy="loading"
          aria-label="Refresh gallery"
          @click="refresh"
        >
          <Icon name="refresh" :size="16" :class="{ gal__spin: loading }" />
        </button>
      </div>
    </header>

    <main class="gal__main">
      <div v-if="errorMessage" class="gal__error" role="alert">
        <p class="gal__error-title">Couldn't load the gallery.</p>
        <p class="gal__error-body">{{ errorMessage }}</p>
      </div>

      <template v-else>
        <EmptyStateBlock
          v-if="emptyKind === 'search'"
          icon="search"
          headline="No prints match"
          guidance="Nothing matches your search. Clear it to browse the full library."
        >
          <template #action>
            <button
              type="button"
              class="gal__emptybtn"
              data-test="clear-search"
              @click="clearSearch"
            >
              Clear search
            </button>
          </template>
        </EmptyStateBlock>

        <EmptyStateBlock
          v-else-if="emptyKind === 'video'"
          icon="video"
          headline="No video clips yet"
          guidance="Generate with an LTX Video model to see clips here."
        />

        <EmptyStateBlock
          v-else-if="emptyKind === 'images'"
          icon="image"
          headline="No images yet"
          guidance="Generate an image and it will appear in your library."
        />

        <EmptyStateBlock
          v-else-if="emptyKind === 'none'"
          icon="image"
          headline="No prints yet"
          guidance="Head to Create — every image and clip you make lands here."
        />

        <GalleryGrid
          v-else-if="view === 'grid'"
          :entries="filtered"
          :loading="loading"
          :select-mode="selectMode"
          :selection="selection"
          :fresh="fresh"
          @open="openItem"
          @toggle-select="toggleSelect"
          @drag-select="onDragSelect"
        />

        <GalleryFeed
          v-else
          :entries="filtered"
          :loading="loading"
          :view="'feed'"
          :muted="muted"
          :select-mode="selectMode"
          :selection="selection"
          @open="openItem"
          @toggle-select="toggleSelect"
          @drag-select="onDragSelect"
        />
      </template>
    </main>

    <!-- Selection action bar (bulk delete). -->
    <Transition name="fade">
      <div
        v-if="selectMode"
        class="gal__bar-wrap"
        :style="{
          bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
        }"
      >
        <div class="gal__bar" role="toolbar" aria-label="Selection actions">
          <span class="gal__bar-count">
            {{ selection.size }}
            <span class="gal__bar-of">/ {{ filtered.length }} selected</span>
          </span>
          <button
            type="button"
            class="gal__bar-btn"
            :disabled="filtered.length === 0"
            @click="selectAllVisible"
          >
            Select all
          </button>
          <button
            type="button"
            class="gal__bar-btn"
            :disabled="selection.size === 0"
            @click="clearSelection"
          >
            Clear
          </button>
          <button
            type="button"
            class="gal__bar-danger"
            :disabled="selection.size === 0"
            @click="deleteSelected"
          >
            Delete selected
          </button>
          <button
            type="button"
            class="gal__bar-danger gal__bar-danger--soft"
            :disabled="filtered.length === 0"
            @click="deleteAllFiltered"
          >
            Delete all
          </button>
          <button
            type="button"
            class="gal__bar-x"
            aria-label="Exit select mode"
            @click="setSelectMode(false)"
          >
            <Icon name="close" :size="15" />
          </button>
        </div>
      </div>
    </Transition>

    <Transition name="fade">
      <button
        v-if="showBackToTop"
        type="button"
        aria-label="Scroll to top"
        class="gal__fab"
        :style="{
          bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
        }"
        @click="scrollToTop"
      >
        <Icon name="chevron-up" :size="20" :stroke-width="2.2" />
      </button>
    </Transition>

    <Lightbox
      :item="selected"
      :index="selectedIndex"
      :total="filtered.length"
      :has-prev="selectedIndex > 0"
      :has-next="selectedIndex >= 0 && selectedIndex < filtered.length - 1"
      :muted="muted"
      @close="closeLightbox"
      @prev="stepLightbox(-1)"
      @next="stepLightbox(1)"
      @reuse="onReuse"
      @use-source="onUseAsSource"
      @upscale="onUpscale"
      @delete="onLightboxDelete"
    />
  </div>
</template>

<style scoped>
.gal {
  position: relative;
  max-width: 1800px;
  margin: 0 auto;
  padding: 22px 20px 160px;
}

.gal__head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 14px;
  margin-bottom: 18px;
}
.gal__title {
  margin: 0;
  font-family: var(--f-display);
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}
.gal__count {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.gal__flex {
  flex: 1;
  min-width: 0;
}

.gal__filter {
  flex: 0 0 auto;
}

.gal__search {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-3);
}
.gal__search input {
  width: 180px;
  max-width: 42vw;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13px;
  outline: none;
}
.gal__search input::placeholder {
  color: var(--ink-3);
}
.gal__search:focus-within {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

.gal__tools {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.gal__viewtoggle {
  display: inline-flex;
  gap: 3px;
  padding: 3px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
}
.gal__vbtn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 28px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.gal__vbtn[data-on="true"] {
  background: var(--sel-bg);
  color: var(--sel-ink);
}
.gal__vbtn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.gal__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}
.gal__icon svg {
  width: 16px;
  height: 16px;
}
.gal__icon:hover:not(:disabled) {
  color: var(--rebate);
}
.gal__icon:disabled {
  opacity: 0.6;
  cursor: default;
}
.gal__icon:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.gal__spin {
  animation: gal-spin 0.9s linear infinite;
}
@keyframes gal-spin {
  to {
    transform: rotate(360deg);
  }
}

.gal__select {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 13px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}
.gal__select svg {
  width: 16px;
  height: 16px;
}
.gal__select:hover {
  color: var(--rebate);
}
.gal__select--on {
  background: var(--sel-bg);
  color: var(--sel-ink);
  border-color: var(--sel-border);
}
.gal__select:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.gal__main {
  margin-top: 4px;
}

.gal__error {
  background: color-mix(in srgb, var(--stop) 12%, var(--bench));
  border: 1px solid color-mix(in srgb, var(--stop) 40%, transparent);
  border-radius: var(--radius-card);
  padding: 14px 16px;
}
.gal__error-title {
  margin: 0;
  font-weight: 600;
  color: var(--rebate);
}
.gal__error-body {
  margin: 4px 0 0;
  font-size: 13px;
  color: var(--ink-2);
}

.gal__emptybtn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 9px 16px;
  border-radius: var(--radius-control);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}
.gal__emptybtn:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}
.gal__emptybtn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* Selection action bar. */
.gal__bar-wrap {
  position: fixed;
  inset-inline: 0;
  z-index: 40;
  display: flex;
  justify-content: center;
  padding: 0 16px;
  pointer-events: none;
}
.gal__bar {
  pointer-events: auto;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  max-width: 100%;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-pill);
  box-shadow: var(--shadow-raised);
  padding: 8px 12px;
  font-size: 13px;
  color: var(--rebate);
}
.gal__bar-count {
  padding: 0 6px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.gal__bar-of {
  color: var(--ink-3);
}
.gal__bar-btn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 6px 12px;
  border-radius: var(--radius-pill);
  font-weight: 600;
  cursor: pointer;
}
.gal__bar-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.gal__bar-danger {
  border: 0;
  background: var(--stop);
  color: #fff;
  padding: 6px 12px;
  border-radius: var(--radius-pill);
  font-weight: 700;
  cursor: pointer;
}
.gal__bar-danger--soft {
  background: color-mix(in srgb, var(--stop) 22%, transparent);
  color: var(--stop);
}
.gal__bar-danger:disabled {
  opacity: 0.5;
  cursor: default;
}
.gal__bar-x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
}
.gal__bar-x:hover {
  background: color-mix(in srgb, var(--rebate) 8%, transparent);
  color: var(--rebate);
}

.gal__fab {
  position: fixed;
  right: 20px;
  z-index: 20;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 46px;
  height: 46px;
  border: 0;
  border-radius: 50%;
  background: var(--safelight);
  color: var(--on-accent);
  box-shadow: var(--shadow-raised);
  cursor: pointer;
}
.gal__fab:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
