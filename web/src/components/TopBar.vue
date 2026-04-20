<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { useDownloads } from "../composables/useDownloads";
import { useTweaks } from "../composables/useTweaks";
import ResourceStrip from "./ResourceStrip.vue";

/*
 * Top bar has two distinct looks, driven by Tweaks:
 *
 *   - Studio (default): a sticky glass pill — brand, nav, search, filters,
 *     view toggle, sound, refresh — all on one line.
 *   - Lab: an editorial masthead — big gradient wordmark + nav on the
 *     first row, search/filters/view on a second "controls" row.
 *
 * Both variants speak the same `update:*` event API, so the parent doesn't
 * care which is rendered. Downloads and ResourceStrip hooks are shared.
 */

const route = useRoute();
const { tweaks } = useTweaks();

type FilterKind = "all" | "images" | "video";
type ViewMode = "feed" | "grid";

const downloads = useDownloads();
const badgeCount = computed(
  () => (downloads.active.value ? 1 : 0) + downloads.queued.value.length,
);

function openDownloadsDrawer() {
  window.dispatchEvent(new CustomEvent("mold:open-downloads"));
}

const props = defineProps<{
  filter: FilterKind;
  search: string;
  view: ViewMode;
  muted: boolean;
  counts: { total: number; images: number; video: number; filtered: number };
  loading: boolean;
}>();

const emit = defineEmits<{
  (e: "update:filter", value: FilterKind): void;
  (e: "update:search", value: string): void;
  (e: "update:view", value: ViewMode): void;
  (e: "update:muted", value: boolean): void;
  (e: "refresh"): void;
}>();

// 180ms input debounce so a 1500-item gallery doesn't re-filter on every
// keystroke.
const local = ref(props.search);
let t: ReturnType<typeof setTimeout> | null = null;

watch(local, (v) => {
  if (t) clearTimeout(t);
  t = setTimeout(() => emit("update:search", v), 180);
});

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

const isGallery = computed(() => route.name === "gallery");
const isGenerate = computed(() => route.name === "generate");
const direction = computed(() => tweaks.value.direction);
</script>

<template>
  <!-- Studio direction -->
  <header
    v-if="direction === 'studio'"
    class="studio-topbar glass-panel"
    :class="{ 'flex-col sm:flex-row': true }"
  >
    <div class="studio-brand">
      <img src="/logo.png" alt="mold" />
      <div>
        <div class="studio-brand-name">mold</div>
        <div class="studio-brand-sub">
          gallery · {{ counts.filtered }}/{{ counts.total }}
        </div>
      </div>
    </div>

    <nav class="studio-nav" aria-label="Primary">
      <router-link to="/" :class="{ on: isGallery }">Gallery</router-link>
      <router-link to="/generate" :class="{ on: isGenerate }">
        Generate
      </router-link>
    </nav>

    <label class="studio-search">
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
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      <input
        v-model="local"
        type="search"
        placeholder="Search prompts, models, filenames…"
        autocomplete="off"
        spellcheck="false"
      />
      <button v-if="local" aria-label="Clear search" @click="clearSearch">
        <svg
          width="14"
          height="14"
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

    <template v-if="isGallery">
      <div class="studio-seg" role="group" aria-label="View mode">
        <button
          :class="{ on: view === 'feed' }"
          :aria-pressed="view === 'feed'"
          @click="setView('feed')"
        >
          <svg
            width="13"
            height="13"
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
          :class="{ on: view === 'grid' }"
          :aria-pressed="view === 'grid'"
          @click="setView('grid')"
        >
          <svg
            width="13"
            height="13"
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

      <div class="studio-seg" role="group" aria-label="Kind filter">
        <button :class="{ on: filter === 'all' }" @click="pick('all')">
          All <span class="studio-seg-count">{{ counts.total }}</span>
        </button>
        <button :class="{ on: filter === 'images' }" @click="pick('images')">
          Images <span class="studio-seg-count">{{ counts.images }}</span>
        </button>
        <button :class="{ on: filter === 'video' }" @click="pick('video')">
          Video <span class="studio-seg-count">{{ counts.video }}</span>
        </button>
      </div>

      <button
        class="studio-iconbtn"
        :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
        :aria-pressed="!muted"
        @click="emit('update:muted', !muted)"
      >
        <svg
          v-if="muted"
          width="15"
          height="15"
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
          width="15"
          height="15"
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

      <button
        class="studio-iconbtn"
        :disabled="loading"
        :aria-busy="loading"
        aria-label="Refresh gallery"
        @click="emit('refresh')"
      >
        <svg
          width="15"
          height="15"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
          :class="{ 'animate-spin': loading }"
        >
          <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
      </button>
    </template>

    <button
      class="studio-iconbtn"
      aria-label="Open downloads"
      @click="openDownloadsDrawer"
    >
      <svg
        width="15"
        height="15"
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
      <span v-if="badgeCount > 0" class="studio-iconbtn-badge">
        {{ badgeCount }}
      </span>
    </button>

    <div
      v-if="isGenerate"
      class="flex shrink-0 items-center lg:hidden"
      style="margin-left: auto"
    >
      <ResourceStrip variant="chip" />
    </div>
  </header>

  <!-- Lab direction — editorial masthead -->
  <header v-else class="lab-masthead">
    <div class="lab-masthead-row">
      <div class="lab-brand">
        <img src="/logo.png" alt="mold" class="lab-brand-mark" />
        <div class="lab-brand-type">
          <div class="lab-brand-name">mold</div>
          <div class="lab-brand-sub">local diffusion atelier</div>
        </div>
      </div>

      <nav class="lab-nav" aria-label="Primary">
        <router-link to="/" :class="{ on: isGallery }">Gallery</router-link>
        <router-link to="/generate" :class="{ on: isGenerate }">
          Generate
        </router-link>
        <button disabled>Models</button>
        <button disabled>Runs</button>
      </nav>

      <div class="lab-actions">
        <button
          class="lab-iconbtn"
          aria-label="Open downloads"
          @click="openDownloadsDrawer"
        >
          <svg
            width="16"
            height="16"
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
          <span v-if="badgeCount > 0" class="topbar-badge">
            {{ badgeCount }}
          </span>
        </button>
        <button
          v-if="isGallery"
          class="lab-iconbtn"
          :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
          @click="emit('update:muted', !muted)"
        >
          <svg
            v-if="muted"
            width="16"
            height="16"
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
            width="16"
            height="16"
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
          </svg>
        </button>
        <button
          v-if="isGallery"
          class="lab-iconbtn"
          :disabled="loading"
          aria-label="Refresh"
          @click="emit('refresh')"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
            :class="{ 'animate-spin': loading }"
          >
            <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
            <path d="M3 21v-5h5" />
          </svg>
        </button>
      </div>
    </div>

    <div v-if="isGallery" class="lab-controls">
      <label class="lab-search">
        <svg
          width="16"
          height="16"
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
          placeholder="Search prompts, models, seeds…"
          spellcheck="false"
        />
        <button v-if="local" aria-label="Clear" @click="clearSearch">
          <svg
            width="14"
            height="14"
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

      <div class="lab-segctl">
        <button :class="{ on: filter === 'all' }" @click="pick('all')">
          all <span class="lab-count">{{ counts.total }}</span>
        </button>
        <button :class="{ on: filter === 'images' }" @click="pick('images')">
          images <span class="lab-count">{{ counts.images }}</span>
        </button>
        <button :class="{ on: filter === 'video' }" @click="pick('video')">
          video <span class="lab-count">{{ counts.video }}</span>
        </button>
      </div>

      <div class="lab-segctl">
        <button :class="{ on: view === 'feed' }" @click="setView('feed')">
          feed
        </button>
        <button :class="{ on: view === 'grid' }" @click="setView('grid')">
          grid
        </button>
      </div>

      <div class="lab-counter">
        <span class="lab-counter-num">{{ counts.filtered }}</span>
        <span class="lab-counter-sep">/</span>
        <span class="lab-counter-total">{{ counts.total }}</span>
        <span class="lab-counter-label">shown</span>
      </div>
    </div>

    <div v-if="isGenerate" class="lab-controls">
      <ResourceStrip variant="chip" />
    </div>
  </header>
</template>
