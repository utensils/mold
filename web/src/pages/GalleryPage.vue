<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { listGallery, deleteGalleryImage } from "../api";
import { requestConfirm } from "../lib/toasts";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import GalleryFeed from "../components/GalleryFeed.vue";
import DetailDrawer from "../components/DetailDrawer.vue";

type FilterKind = "all" | "images" | "video";
type ViewMode = "feed" | "grid";

// Persist the preferred layout across reloads. `feed` is the default —
// a single-column, Instagram-style stream with a visible caption — because
// it's friendlier for reading prompts. Users can flip to `grid` for a
// dense masonry view when scanning hundreds of items.
const VIEW_STORAGE_KEY = "mold.gallery.view";
function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem(VIEW_STORAGE_KEY);
    if (v === "feed" || v === "grid") return v;
  } catch {
    /* localStorage may be blocked (private mode, SSR, etc.) */
  }
  return "feed";
}

const entries = ref<GalleryImage[]>([]);
const loading = ref(true);
const errorMessage = ref<string | null>(null);
const filter = ref<FilterKind>("all");
const search = ref("");
const view = ref<ViewMode>(loadViewMode());

// Seed the search from the global nav's `?q=` and keep it in sync when the
// query param changes (the nav search routes here with `q`). The existing
// filtering logic below reads `search` unchanged.
const route = useRoute();
watch(
  () => route.query.q,
  (q) => {
    const next = typeof q === "string" ? q : "";
    if (next !== search.value) search.value = next;
  },
  { immediate: true },
);

/*
 * Global audio preference.
 *
 * Browser autoplay policies forbid unmuted autoplay without a user gesture
 * (Chrome/Safari/Firefox all enforce this). We default to muted so the feed
 * silently autoplays on load, then expose a header toggle. The first click
 * on the toggle _is_ the user gesture — from that point on every <video>
 * element picks up the preference and plays with sound.
 *
 * Persisted in localStorage so the choice sticks across reloads.
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
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

/*
 * Multi-select.
 *
 * `selectMode` flips the gallery into bulk-edit: clicks on cards toggle
 * their selection instead of opening the detail drawer. `selection` holds
 * filenames (stable id — survives re-fetches). `selectionAnchor` is the
 * last single-clicked filename; shift-clicking another tile selects the
 * inclusive range between them in filter order (Finder-style).
 *
 * Exiting select mode clears both the mode flag and the current selection
 * so the next entry starts from a clean slate.
 */
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
    // Shift-click: select the contiguous range in the currently-filtered
    // list between the anchor and the clicked item. Doesn't touch
    // selections outside that range — matches macOS Finder behavior.
    const list = filtered.value;
    const a = list.findIndex((e) => e.filename === selectionAnchor.value);
    const b = list.findIndex((e) => e.filename === name);
    if (a === -1 || b === -1) {
      // Anchor is no longer in the filtered list (filter changed). Fall
      // back to single toggle.
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

async function handleDeleteMany(names: string[]): Promise<number> {
  // Fire deletes in parallel — the server is local and individual DELETEs
  // are cheap. `Promise.allSettled` lets partial failures not take down
  // the whole batch; we surface the error count to the user.
  const results = await Promise.allSettled(
    names.map((n) => deleteGalleryImage(n)),
  );
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

// Filter pass #1: kind (all / images / video)
const kindFiltered = computed(() => {
  if (filter.value === "all") return entries.value;
  return entries.value.filter((e) => {
    const k = mediaKind(e.format, e.filename);
    if (filter.value === "video") return k === "video" || k === "animated";
    return k === "image";
  });
});

// Filter pass #2: search. Matches prompt, model, and filename for now.
// We normalize both sides to lowercase; fuzzy ranking would be over-engineering
// at this scale.
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

const counts = computed(() => {
  const total = entries.value.length;
  let images = 0;
  let video = 0;
  for (const e of entries.value) {
    const k = mediaKind(e.format, e.filename);
    if (k === "image") images++;
    else video++;
  }
  return { total, images, video, filtered: filtered.value.length };
});

async function refresh() {
  loading.value = true;
  errorMessage.value = null;
  try {
    entries.value = await listGallery();
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

function openItem(item: GalleryImage) {
  const idx = filtered.value.findIndex((e) => e.filename === item.filename);
  selectedIndex.value = idx;
  selected.value = item;
}

function closeDrawer() {
  selected.value = null;
  selectedIndex.value = -1;
}

function stepDrawer(delta: number) {
  if (selectedIndex.value < 0) return;
  const list = filtered.value;
  const next = selectedIndex.value + delta;
  if (next < 0 || next >= list.length) return;
  selectedIndex.value = next;
  selected.value = list[next] ?? null;
}

async function handleDelete(item: GalleryImage) {
  try {
    await deleteGalleryImage(item.filename);
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : String(err);
    return;
  }
  entries.value = entries.value.filter((e) => e.filename !== item.filename);
  if (selected.value && selected.value.filename === item.filename) {
    closeDrawer();
  }
}

function setView(next: ViewMode) {
  view.value = next;
  try {
    localStorage.setItem(VIEW_STORAGE_KEY, next);
  } catch {
    /* ignore */
  }
}

/*
 * "Back to top" floating action button.
 *
 * On mobile the header no longer sticks to the top of the viewport (it
 * would steal 10-15 % of vertical space on phones). To compensate, once
 * the user scrolls far enough that the header is off-screen we surface a
 * FAB in the bottom-right corner; tapping it smooth-scrolls back to top.
 *
 * The threshold is deliberately larger than `window.innerHeight` so the
 * button doesn't flicker in and out while the user is still near the
 * top of the feed.
 */
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
  <div
    class="relative mx-auto flex min-h-[100svh] max-w-[1800px] flex-col px-4 pb-40 sm:px-6 lg:px-10"
  >
    <!--
      Page header. W2 relocates the gallery-only toolbar (view / kind filter /
      mute / select / refresh) out of the former global TopBar; search now
      lives in the app nav and seeds this page via `?q=`. The controls keep
      their prior behaviors and aria-labels — the page gets a full redesign in
      a later milestone.
    -->
    <header class="mt-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:gap-4">
      <div class="min-w-0 flex-1">
        <h1 class="font-display text-2xl font-bold tracking-tight text-ink">
          Gallery
        </h1>
        <p class="mt-1 font-mono text-xs text-ink-3">
          {{ counts.filtered }} / {{ counts.total }}
        </p>
      </div>

      <div class="flex shrink-0 flex-wrap items-center gap-2">
        <!-- View-mode toggle -->
        <div
          class="flex items-center gap-0.5 rounded-full border border-white/5 bg-white/5 p-0.5"
          role="group"
          aria-label="View mode"
        >
          <button
            class="inline-flex h-9 items-center gap-1.5 rounded-full px-3 text-[13px] font-medium transition"
            :class="
              view === 'feed'
                ? 'bg-brand-500 text-white shadow-sm'
                : 'text-ink-200 hover:text-white'
            "
            :aria-pressed="view === 'feed'"
            @click="setView('feed')"
          >
            Feed
          </button>
          <button
            class="inline-flex h-9 items-center gap-1.5 rounded-full px-3 text-[13px] font-medium transition"
            :class="
              view === 'grid'
                ? 'bg-brand-500 text-white shadow-sm'
                : 'text-ink-200 hover:text-white'
            "
            :aria-pressed="view === 'grid'"
            @click="setView('grid')"
          >
            Grid
          </button>
        </div>

        <!-- Kind filter -->
        <nav
          class="flex items-center gap-0.5 rounded-full border border-white/5 bg-white/5 p-0.5 text-[13px] font-medium text-ink-200"
        >
          <button
            class="rounded-full px-3 py-1.5 transition"
            :class="
              filter === 'all'
                ? 'bg-brand-500 text-white shadow-sm'
                : 'hover:text-white'
            "
            @click="filter = 'all'"
          >
            All
            <span class="ml-1 text-[11px] tabular-nums opacity-70">{{
              counts.total
            }}</span>
          </button>
          <button
            class="hidden rounded-full px-3 py-1.5 transition sm:inline-flex"
            :class="
              filter === 'images'
                ? 'bg-brand-500 text-white shadow-sm'
                : 'hover:text-white'
            "
            @click="filter = 'images'"
          >
            Images
            <span class="ml-1 text-[11px] tabular-nums opacity-70">{{
              counts.images
            }}</span>
          </button>
          <button
            class="rounded-full px-3 py-1.5 transition"
            :class="
              filter === 'video'
                ? 'bg-brand-500 text-white shadow-sm'
                : 'hover:text-white'
            "
            @click="filter = 'video'"
          >
            Video
            <span class="ml-1 text-[11px] tabular-nums opacity-70">{{
              counts.video
            }}</span>
          </button>
        </nav>

        <!-- Sound toggle: the click doubles as the user gesture browsers
             require before unmuted autoplay is allowed. -->
        <button
          class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-white/5 text-ink-200 transition hover:text-white"
          :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
          :title="muted ? 'Unmute videos' : 'Mute videos'"
          :aria-pressed="!muted"
          @click="setMuted(!muted)"
        >
          <svg
            v-if="muted"
            class="h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
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
            class="h-4 w-4 text-brand-300"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="M15.5 8.5a5 5 0 0 1 0 7" />
            <path d="M18.5 5.5a9 9 0 0 1 0 13" />
          </svg>
        </button>

        <!-- Select mode toggle -->
        <button
          class="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/5 px-3 text-[13px] font-medium transition hover:text-white"
          :class="
            selectMode ? 'bg-brand-500 text-white' : 'bg-white/5 text-ink-200'
          "
          :aria-pressed="selectMode"
          :title="selectMode ? 'Exit select mode' : 'Select multiple'"
          @click="setSelectMode(!selectMode)"
        >
          <span class="hidden sm:inline">
            {{ selectMode ? `Selected ${selection.size}` : "Select" }}
          </span>
          <span v-if="selectMode && selection.size > 0" class="sm:hidden">
            {{ selection.size }}
          </span>
          <span v-else class="sm:hidden">Select</span>
        </button>

        <!-- Refresh -->
        <button
          class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-white/5 text-ink-200 transition hover:text-white disabled:opacity-60"
          :disabled="loading"
          :aria-busy="loading"
          aria-label="Refresh gallery"
          @click="refresh"
        >
          <svg
            class="h-4 w-4"
            :class="{ 'animate-spin': loading }"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
            <path d="M3 21v-5h5" />
          </svg>
        </button>
      </div>
    </header>

    <main class="mt-4 sm:mt-6">
      <div
        v-if="errorMessage"
        class="glass flex items-start gap-3 rounded-2xl px-4 py-3 text-sm text-rose-200"
      >
        <span class="mt-0.5">⚠</span>
        <div>
          <p class="font-medium text-rose-100">Couldn't load the gallery.</p>
          <p class="text-rose-200/80">{{ errorMessage }}</p>
        </div>
      </div>

      <GalleryFeed
        v-else
        :entries="filtered"
        :loading="loading"
        :view="view"
        :muted="muted"
        :select-mode="selectMode"
        :selection="selection"
        @open="openItem"
        @toggle-select="toggleSelect"
        @drag-select="onDragSelect"
      />
    </main>

    <!--
      Floating selection action bar.
      Appears when the user enters select mode. Surfaces counts and the
      four bulk actions: select-all-in-filter, clear, delete selected,
      delete all (the "nuke this filter" escape hatch). Positioned above
      the back-to-top FAB so both can coexist. We intentionally use
      window.confirm() inside the delete handlers — consistent with the
      detail drawer's single-item delete, and avoids shipping a modal.
    -->
    <Transition name="fade">
      <div
        v-if="selectMode"
        class="pointer-events-none fixed inset-x-0 z-40 flex justify-center px-4"
        :style="{
          bottom:
            'calc(var(--mold-tray-height, 0px) + max(0.75rem, env(safe-area-inset-bottom)))',
        }"
      >
        <div
          class="glass pointer-events-auto flex max-w-full flex-wrap items-center gap-2 rounded-full border border-white/10 bg-ink-900/80 px-3 py-2 text-[13px] text-ink-100 shadow-xl backdrop-blur"
          role="toolbar"
          aria-label="Selection actions"
        >
          <span class="px-2 font-medium tabular-nums">
            {{ selection.size }}
            <span class="text-ink-400">/ {{ filtered.length }} selected</span>
          </span>
          <button
            class="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-medium transition hover:bg-white/10 disabled:opacity-60"
            :disabled="filtered.length === 0"
            @click="selectAllVisible"
          >
            Select all
          </button>
          <button
            class="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-medium transition hover:bg-white/10 disabled:opacity-60"
            :disabled="selection.size === 0"
            @click="clearSelection"
          >
            Clear
          </button>
          <button
            class="rounded-full bg-rose-500/90 px-3 py-1 font-semibold text-white transition hover:bg-rose-500 disabled:opacity-50"
            :disabled="selection.size === 0"
            @click="deleteSelected"
          >
            Delete selected
          </button>
          <button
            class="rounded-full border border-rose-400/40 bg-rose-500/20 px-3 py-1 font-semibold text-rose-100 transition hover:bg-rose-500/30 disabled:opacity-50"
            :disabled="filtered.length === 0"
            :title="
              filtered.length === counts.total
                ? 'Delete every item in the gallery'
                : 'Delete every item that matches the current filter'
            "
            @click="deleteAllFiltered"
          >
            Delete all
          </button>
          <button
            class="ml-1 inline-flex h-7 w-7 items-center justify-center rounded-full text-ink-300 transition hover:bg-white/10 hover:text-white"
            aria-label="Exit select mode"
            @click="setSelectMode(false)"
          >
            <svg
              class="h-3.5 w-3.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.4"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M6 6l12 12" />
              <path d="M18 6 6 18" />
            </svg>
          </button>
        </div>
      </div>
    </Transition>

    <!-- Back-to-top FAB. Appears once the user has scrolled more than one
         viewport down, replacing the on-desktop convenience of the sticky
         header on mobile. Sits above the tray so an expanded CPU/GPU tray
         doesn't swallow it. Fade/translate transition keeps it unobtrusive. -->
    <Transition name="fade">
      <button
        v-if="showBackToTop"
        type="button"
        aria-label="Scroll to top"
        class="fixed right-4 z-20 inline-flex h-12 w-12 items-center justify-center rounded-full bg-brand-500 text-white shadow-[0_12px_30px_-10px_rgba(99,102,241,0.7)] backdrop-blur transition hover:bg-brand-400 active:scale-95 sm:right-6"
        :style="{
          bottom:
            'calc(var(--mold-tray-height, 0px) + max(0.75rem, env(safe-area-inset-bottom)))',
        }"
        @click="scrollToTop"
      >
        <svg
          class="h-5 w-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M12 19V5" />
          <path d="m5 12 7-7 7 7" />
        </svg>
      </button>
    </Transition>

    <DetailDrawer
      :item="selected"
      :has-prev="selectedIndex > 0"
      :has-next="selectedIndex >= 0 && selectedIndex < filtered.length - 1"
      :index="selectedIndex"
      :total="filtered.length"
      :muted="muted"
      @close="closeDrawer"
      @prev="stepDrawer(-1)"
      @next="stepDrawer(1)"
      @delete="handleDelete"
    />
  </div>
</template>
