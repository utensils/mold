<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { useDownloads } from "../composables/useDownloads";
import { useCatalog } from "../composables/useCatalog";
import ResourceStrip from "./ResourceStrip.vue";

const route = useRoute();

type FilterKind = "all" | "images" | "video";
type ViewMode = "feed" | "grid";

// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
const downloads = useDownloads();
const badgeCount = computed(
  () => (downloads.active.value ? 1 : 0) + downloads.queued.value.length,
);

function openDownloadsDrawer() {
  window.dispatchEvent(new CustomEvent("mold:open-downloads"));
}

// ─── Catalog refresh (visible on /generate + /catalog) ────────────────────────
const cat = useCatalog();
const showCatalogControls = computed(
  () => route.name === "generate" || route.name === "catalog",
);
const refreshState = computed(() => cat.refreshStatus.value?.state ?? null);
const refreshBusy = computed(
  () => refreshState.value === "pending" || refreshState.value === "running",
);
const refreshLabel = computed(() => {
  const s = cat.refreshStatus.value;
  if (!s) return "Refresh catalog";
  if (s.state === "pending") return "Queued…";
  if (s.state === "running") return "Scanning…";
  if (s.state === "done") return `Done · ${s.total_entries}`;
  if (s.state === "failed") {
    // "Busy" reads better than "Failed" when the actual reason is the
    // server's 409 conflict (another scan is mid-flight).
    return /in progress/i.test(s.message) ? "Busy — try later" : "Failed";
  }
  return "Refresh catalog";
});

const refreshTitle = computed(() => {
  const s = cat.refreshStatus.value;
  if (s?.state === "failed") return s.message;
  return refreshLabel.value;
});

async function onRefreshCatalog() {
  if (refreshBusy.value) return;
  try {
    await cat.startRefresh();
  } catch (err) {
    console.error("catalog refresh failed", err);
  }
}

// Gallery-only props (`hideMode`, `selectMode`, `selectionCount`,
// `canDelete`) are optional. The Generate page mounts this same TopBar
// without a gallery underneath it, so their defaults keep the toolbar
// in its "no bulk actions" state.
const props = withDefaults(
  defineProps<{
    filter: FilterKind;
    search: string;
    view: ViewMode;
    muted: boolean;
    counts: { total: number; images: number; video: number; filtered: number };
    loading: boolean;
    hideMode?: boolean;
    selectMode?: boolean;
    selectionCount?: number;
    canDelete?: boolean;
  }>(),
  {
    hideMode: false,
    selectMode: false,
    selectionCount: 0,
    canDelete: false,
  },
);

const emit = defineEmits<{
  (e: "update:filter", value: FilterKind): void;
  (e: "update:search", value: string): void;
  (e: "update:view", value: ViewMode): void;
  (e: "update:muted", value: boolean): void;
  (e: "update:hide-mode", value: boolean): void;
  (e: "update:select-mode", value: boolean): void;
  (e: "refresh"): void;
}>();

// Simple 180ms debounce so the search doesn't re-filter 1500 rows on every
// keystroke. We keep a local ref for the input and only push upwards once
// the user pauses.
const local = ref(props.search);
let t: ReturnType<typeof setTimeout> | null = null;

watch(local, (v) => {
  if (t) clearTimeout(t);
  t = setTimeout(() => emit("update:search", v), 180);
});

// If the parent resets the search (e.g. via a "clear all" button later),
// keep the local input in sync.
watch(
  () => props.search,
  (v) => {
    if (v !== local.value) local.value = v;
  },
);

function pick(next: FilterKind) {
  emit("update:filter", next);
}

function setView(next: ViewMode) {
  emit("update:view", next);
}

function clearSearch() {
  local.value = "";
  emit("update:search", "");
}
</script>

<template>
  <!--
    Sticky positioning is sm-and-up only AND gallery-only. On mobile the
    header scrolls away with the content so the feed has the full viewport
    height (a persistent bar eats 10-15 % of vertical space, which matters
    on phones). On /generate the header always scrolls — the Composer owns
    the top of the viewport once the user starts filling it out, and a
    sticky bar there competes with the ResourceTray at the bottom.
  -->
  <header
    class="glass relative z-30 flex flex-col gap-3 rounded-3xl px-4 py-3 sm:flex-row sm:items-center sm:gap-4 sm:px-5 sm:py-3.5"
    :class="route.name === 'gallery' ? 'sm:sticky sm:top-4' : ''"
  >
    <!-- Brand -->
    <div class="flex shrink-0 items-center gap-3">
      <img
        src="/logo.png"
        alt="mold"
        width="40"
        height="40"
        class="h-10 w-10 shrink-0 rounded-lg object-contain drop-shadow-[0_6px_18px_rgba(99,102,241,0.4)]"
      />
      <div class="leading-tight">
        <div class="text-base font-semibold tracking-tight text-ink-50">
          mold
        </div>
        <div
          class="hidden text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400 sm:block"
        >
          gallery · {{ counts.filtered }}/{{ counts.total }}
        </div>
      </div>
    </div>

    <nav
      class="flex items-center gap-1 rounded-full border border-white/5 bg-white/5 p-1 text-[13px] font-medium"
      aria-label="Primary navigation"
    >
      <router-link
        to="/"
        class="rounded-full px-3 py-1 text-ink-200 transition hover:text-white"
        active-class="bg-brand-500 text-white shadow-sm"
        exact-active-class="bg-brand-500 text-white shadow-sm"
      >
        Gallery
      </router-link>
      <router-link
        to="/generate"
        class="rounded-full px-3 py-1 text-ink-200 transition hover:text-white"
        active-class="bg-brand-500 text-white shadow-sm"
      >
        Generate
      </router-link>
      <router-link
        to="/catalog"
        class="rounded-full px-3 py-1 text-ink-200 transition hover:text-white"
        active-class="bg-brand-500 text-white shadow-sm"
      >
        Model Catalog
      </router-link>
    </nav>

    <!-- Catalog refresh button + status. Mirrors the Downloads slot so the
         operator can kick off a `mold catalog refresh` and watch progress
         from either /generate or /catalog without leaving their workflow.
         Hidden on /gallery (route is read-only with respect to model state). -->
    <button
      v-if="showCatalogControls"
      type="button"
      class="relative inline-flex h-10 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-sm text-ink-200 transition hover:text-white disabled:opacity-60"
      :class="{
        'bg-brand-500/20 border-brand-400/40 text-brand-100': refreshBusy,
        'bg-rose-500/20 border-rose-400/40 text-rose-100':
          refreshState === 'failed',
      }"
      :disabled="refreshBusy"
      :aria-busy="refreshBusy"
      :aria-label="refreshTitle"
      :title="refreshTitle"
      @click="onRefreshCatalog"
    >
      <svg
        class="h-4 w-4"
        :class="{ 'animate-spin': refreshBusy }"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
        <path d="M21 3v5h-5" />
        <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
        <path d="M3 21v-5h5" />
      </svg>
      <span class="hidden sm:inline">{{ refreshLabel }}</span>
    </button>

    <!-- Downloads drawer opener + badge (Agent A). Visible on every page. -->
    <button
      type="button"
      class="relative inline-flex h-10 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-sm text-ink-200 hover:text-white"
      aria-label="Open downloads"
      @click="openDownloadsDrawer"
    >
      <svg
        class="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M12 3v12" />
        <path d="m7 10 5 5 5-5" />
        <path d="M5 21h14" />
      </svg>
      <span class="hidden sm:inline">Downloads</span>
      <span
        v-if="badgeCount > 0"
        class="absolute -right-1 -top-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-brand-500 px-1 text-[11px] font-medium text-white"
        aria-label="Pending download count"
      >
        {{ badgeCount }}
      </span>
    </button>

    <!-- Search -->
    <label class="relative flex-1">
      <svg
        class="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-400"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      <input
        v-model="local"
        type="search"
        placeholder="Search prompts, models, filenames…"
        autocomplete="off"
        spellcheck="false"
        class="h-11 w-full rounded-full border border-white/5 bg-white/5 pl-10 pr-10 text-[14px] text-ink-100 placeholder:text-ink-400 focus:border-brand-400/40 focus:outline-none focus:ring-2 focus:ring-brand-400/25"
      />
      <button
        v-if="local"
        type="button"
        class="absolute right-2.5 top-1/2 inline-flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-full text-ink-300 transition hover:bg-white/10 hover:text-white"
        aria-label="Clear search"
        @click="clearSearch"
      >
        <svg
          class="h-3.5 w-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M6 6l12 12" />
          <path d="M18 6 6 18" />
        </svg>
      </button>
    </label>

    <div
      v-if="$route.name === 'gallery'"
      class="flex shrink-0 flex-wrap items-center gap-2"
    >
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
          <svg
            class="h-3.5 w-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            aria-hidden="true"
          >
            <path d="M4 6h16" />
            <path d="M4 12h16" />
            <path d="M4 18h16" />
          </svg>
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
          <svg
            class="h-3.5 w-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="7" height="7" rx="1.2" />
            <rect x="14" y="3" width="7" height="7" rx="1.2" />
            <rect x="3" y="14" width="7" height="7" rx="1.2" />
            <rect x="14" y="14" width="7" height="7" rx="1.2" />
          </svg>
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
          @click="pick('all')"
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
          @click="pick('images')"
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
          @click="pick('video')"
        >
          Video
          <span class="ml-1 text-[11px] tabular-nums opacity-70">{{
            counts.video
          }}</span>
        </button>
      </nav>

      <!-- Sound toggle: clicking this counts as the user gesture browsers
           require before they'll autoplay unmuted video, so after the first
           click every subsequent in-view video plays with sound. -->
      <button
        class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-white/5 text-ink-200 transition hover:text-white"
        :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
        :title="muted ? 'Unmute videos' : 'Mute videos'"
        :aria-pressed="!muted"
        @click="emit('update:muted', !muted)"
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

      <!-- Hide/blur toggle. Flipping it on blurs every gallery item; flipping
           it back off reveals everything. Per-item "Reveal" still lets users
           peek a single tile without disabling the global shroud. -->
      <button
        class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 text-ink-200 transition hover:text-white"
        :class="hideMode ? 'bg-brand-500 text-white' : 'bg-white/5'"
        :aria-pressed="hideMode"
        :aria-label="hideMode ? 'Unhide gallery' : 'Hide gallery'"
        :title="hideMode ? 'Unhide gallery' : 'Hide gallery'"
        @click="emit('update:hide-mode', !hideMode)"
      >
        <svg
          v-if="!hideMode"
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        <svg
          v-else
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path
            d="M10.6 5.1A10 10 0 0 1 12 5c6 0 10 7 10 7a17 17 0 0 1-3.3 4.2"
          />
          <path d="M6.7 6.7A17 17 0 0 0 2 12s4 7 10 7a9.7 9.7 0 0 0 5.3-1.7" />
          <path d="m3 3 18 18" />
          <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
        </svg>
      </button>

      <!-- Select mode toggle. Opens a bulk-edit mode where clicks select
           instead of opening the detail drawer, shift-clicks extend the
           range, and drag draws a marquee. Only shown when the server
           actually allows deletion — the only destructive action we gate
           behind selection today. -->
      <button
        v-if="canDelete"
        class="inline-flex h-10 items-center gap-1.5 rounded-full border border-white/5 px-3 text-[13px] font-medium transition hover:text-white"
        :class="
          selectMode ? 'bg-brand-500 text-white' : 'bg-white/5 text-ink-200'
        "
        :aria-pressed="selectMode"
        :title="selectMode ? 'Exit select mode' : 'Select multiple'"
        @click="emit('update:select-mode', !selectMode)"
      >
        <svg
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <rect x="3" y="3" width="7" height="7" rx="1.2" />
          <path d="m14 5 3 3 5-5" />
          <rect x="3" y="14" width="7" height="7" rx="1.2" />
          <rect x="14" y="14" width="7" height="7" rx="1.2" />
        </svg>
        <span class="hidden sm:inline">
          {{ selectMode ? `Selected ${selectionCount}` : "Select" }}
        </span>
        <span
          v-if="selectMode && selectionCount > 0"
          class="sm:hidden"
          aria-hidden="true"
        >
          {{ selectionCount }}
        </span>
      </button>

      <button
        class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-white/5 text-ink-200 transition hover:text-white disabled:opacity-60"
        :disabled="loading"
        :aria-busy="loading"
        aria-label="Refresh gallery"
        @click="emit('refresh')"
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

    <!-- Generate-route extras: hide toggle mirrors the Gallery button so
         preview images below the composer (and running-job tiles) can be
         shrouded without a trip to the Gallery tab. The narrow-viewport
         resource chip lives here too — desktop uses the full ResourceStrip
         inside the page. -->
    <div
      v-if="route.name === 'generate'"
      class="flex shrink-0 items-center gap-2"
    >
      <button
        class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/5 text-ink-200 transition hover:text-white"
        :class="hideMode ? 'bg-brand-500 text-white' : 'bg-white/5'"
        :aria-pressed="hideMode"
        :aria-label="hideMode ? 'Reveal previews' : 'Hide previews'"
        :title="hideMode ? 'Reveal previews' : 'Hide previews'"
        @click="emit('update:hide-mode', !hideMode)"
      >
        <svg
          v-if="!hideMode"
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        <svg
          v-else
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path
            d="M10.6 5.1A10 10 0 0 1 12 5c6 0 10 7 10 7a17 17 0 0 1-3.3 4.2"
          />
          <path d="M6.7 6.7A17 17 0 0 0 2 12s4 7 10 7a9.7 9.7 0 0 0 5.3-1.7" />
          <path d="m3 3 18 18" />
          <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
        </svg>
      </button>
      <div class="lg:hidden">
        <ResourceStrip variant="chip" />
      </div>
    </div>
  </header>
</template>
