<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import InstalledTab from "../components/models/InstalledTab.vue";
import CatalogTab from "../components/models/CatalogTab.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useDownloadsStore } from "../stores/downloads";
import { formatGB } from "../lib/format";
import { modelDiskBytes } from "../lib/models";
import { primaryModifierPressed } from "../lib/platform";
import { mediaTypeFromQuery, type MediaType } from "../lib/modelAvailability";

const conn = useConnectionStore();
const models = useModelStore();
const downloads = useDownloadsStore();
const route = useRoute();
const router = useRouter();
type Layout = "grid" | "table";
const layout = ref<Layout>("grid");
const query = ref("");
const searchEl = ref<HTMLInputElement | null>(null);
const catalogSection = ref<HTMLElement | null>(null);

const MEDIA_TYPES: { value: MediaType; label: string }[] = [
  { value: "all", label: "All" },
  { value: "image", label: "Images" },
  { value: "video", label: "Video" },
];

const mediaType = computed(() => mediaTypeFromQuery(route.query));

function setMediaType(type: MediaType) {
  // Legacy params from the split Models screen are dropped on first touch.
  const { type: _type, tab: _tab, availability: _availability, ...rest } = route.query;
  void router.replace({ query: type === "all" ? rest : { ...rest, type } });
}

const installedCount = computed(() => models.installed.length);
const installedBytes = computed(() =>
  models.installed.reduce((sum, m) => sum + modelDiskBytes(m), 0),
);

function scrollToCatalog() {
  catalogSection.value?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "f" && primaryModifierPressed(e) && !e.altKey) {
    e.preventDefault();
    searchEl.value?.focus();
  }
}

watch(
  () => conn.ready,
  (ready) => {
    if (ready) void models.fetch();
  },
  { immediate: true },
);

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  downloads.subscribe();
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
});
</script>

<template>
  <div class="flex h-full flex-col">
    <header class="border-edge flex flex-wrap items-center gap-3 border-b bg-bench px-4 py-2.5">
      <div class="flex items-center gap-1" aria-label="Media type">
        <button
          v-for="option in MEDIA_TYPES"
          :key="option.value"
          type="button"
          class="h-7 rounded-control px-2.5 text-body"
          :class="mediaType === option.value ? 'bg-bath text-ink' : 'text-ink-2 hover:text-ink'"
          :aria-pressed="mediaType === option.value"
          @click="setMediaType(option.value)"
        >
          {{ option.label }}
        </button>
      </div>

      <input
        ref="searchEl"
        v-model="query"
        data-selectable
        type="search"
        placeholder="Search models…"
        class="border-edge ml-auto h-7 min-w-48 flex-1 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3 sm:max-w-72"
      />

      <div
        class="flex h-7 items-center rounded-control border border-control-edge bg-bath p-0.5"
        aria-label="Catalog layout"
      >
        <button
          type="button"
          class="h-6 rounded-[3px] px-2 text-caption"
          :class="layout === 'grid' ? 'bg-bench text-ink' : 'text-ink-3 hover:text-ink'"
          :aria-pressed="layout === 'grid'"
          title="Grid view"
          @click="layout = 'grid'"
        >
          Grid
        </button>
        <button
          type="button"
          class="h-6 rounded-[3px] px-2 text-caption"
          :class="layout === 'table' ? 'bg-bench text-ink' : 'text-ink-3 hover:text-ink'"
          :aria-pressed="layout === 'table'"
          title="Table view"
          @click="layout = 'table'"
        >
          Table
        </button>
      </div>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <DownloadsTray />

      <section aria-labelledby="installed-models-heading">
        <div class="flex items-center gap-2 px-4 pt-4">
          <h2 id="installed-models-heading" class="text-body font-semibold text-ink">Installed</h2>
          <span class="data-mono text-caption text-ink-3">
            {{ installedCount }} · {{ formatGB(installedBytes) }}
          </span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <InstalledTab :query="query" :media-type="mediaType" @browse-catalog="scrollToCatalog" />
      </section>

      <section ref="catalogSection" aria-labelledby="available-models-heading">
        <div class="flex items-center gap-2 px-4 pt-4">
          <h2 id="available-models-heading" class="text-body font-semibold text-ink">Catalog</h2>
          <span class="text-caption text-ink-3">Hugging Face and Civitai</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <CatalogTab
          :query="query"
          :layout="layout"
          :exclude-installed="true"
          :media-type="mediaType"
        />
      </section>
    </div>
  </div>
</template>
