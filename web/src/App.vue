<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { listGallery, deleteGalleryImage, fetchCapabilities } from "./api";
import type { GalleryImage, ServerCapabilities } from "./types";
import { mediaKind } from "./types";
import GalleryFeed from "./components/GalleryFeed.vue";
import DetailDrawer from "./components/DetailDrawer.vue";
import TopBar from "./components/TopBar.vue";

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
 * Server-reported feature toggles. We fetch these once on mount so the UI
 * can hide affordances the operator hasn't opted in to — most notably the
 * gallery delete button, which requires MOLD_GALLERY_ALLOW_DELETE=1 on the
 * host.
 */
const capabilities = ref<ServerCapabilities>({
  gallery: { can_delete: false },
});

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
  // Fire capabilities + listing in parallel. Capabilities fail-closed — an
  // older server that doesn't know the endpoint simply gets the default
  // `can_delete: false`, which is the safe behavior.
  const [caps] = await Promise.all([fetchCapabilities(), refresh()]);
  capabilities.value = caps;
});
</script>

<template>
  <div
    class="relative mx-auto flex min-h-[100svh] max-w-[1800px] flex-col px-4 pb-32 sm:px-6 lg:px-10"
  >
    <TopBar
      :filter="filter"
      :search="search"
      :view="view"
      :muted="muted"
      :counts="counts"
      :loading="loading"
      @update:filter="(f) => (filter = f)"
      @update:search="(s) => (search = s)"
      @update:view="setView"
      @update:muted="setMuted"
      @refresh="refresh"
    />

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
        @open="openItem"
      />
    </main>

    <!-- Back-to-top FAB. Appears once the user has scrolled more than one
         viewport down, replacing the on-desktop convenience of the sticky
         header on mobile. Fade/translate transition keeps it unobtrusive. -->
    <Transition name="fade">
      <button
        v-if="showBackToTop"
        type="button"
        aria-label="Scroll to top"
        class="fixed bottom-[max(1.25rem,env(safe-area-inset-bottom))] right-4 z-20 inline-flex h-12 w-12 items-center justify-center rounded-full bg-brand-500 text-white shadow-[0_12px_30px_-10px_rgba(99,102,241,0.7)] backdrop-blur transition hover:bg-brand-400 active:scale-95 sm:right-6"
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
      :can-delete="capabilities.gallery.can_delete"
      :muted="muted"
      @close="closeDrawer"
      @prev="stepDrawer(-1)"
      @next="stepDrawer(1)"
      @delete="handleDelete"
    />
  </div>
</template>
