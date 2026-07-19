<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import CatalogTab from "../components/models/CatalogTab.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useUiStore } from "../stores/ui";
import { primaryModifierPressed } from "../lib/platform";
import { mediaTypeFromQuery, type MediaType } from "../lib/modelAvailability";

const conn = useConnectionStore();
const models = useModelStore();
const downloads = useDownloadsStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const route = useRoute();
const router = useRouter();
const ui = useUiStore();
const query = ref("");
const searchEl = ref<HTMLInputElement | null>(null);

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

const installedModels = computed(() => {
  const byName = new Map(
    models.installed.map((entry) => [entry.name, { ...entry, hostIds: ["local"] }]),
  );
  for (const entry of hostModels.unionInstalled) {
    const existing = byName.get(entry.name);
    if (existing) existing.hostIds = [...new Set([...existing.hostIds, ...entry.hostIds])];
    else byName.set(entry.name, { ...entry, hostIds: [...entry.hostIds] });
  }
  return [...byName.values()];
});

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

watch(
  () =>
    hosts.all
      .filter((host) => host.status === "ready" && host.baseUrl)
      .map((host) => `${host.id}:${host.baseUrl}:${host.status}:${host.apiKey ?? ""}`)
      .join("|"),
  () => {
    for (const host of hosts.all.filter((candidate) => candidate.status === "ready")) {
      void downloads.subscribe(host).catch(() => {
        // A pre-downloads-API host still participates in the model shelf.
      });
    }
    void hostModels.refresh();
  },
  { immediate: true },
);

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  void downloads.subscribe();
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

      <!-- View switcher — the Finder-style segmented control: the active
           segment carries the accent fill so the current view is unambiguous.
           Session-persisted in the ui store; table is the default. -->
      <div
        class="flex h-7 items-center gap-0.5 rounded-control border border-control-edge bg-bath p-0.5"
        role="radiogroup"
        aria-label="Catalog layout"
      >
        <button
          type="button"
          role="radio"
          data-test="layout-table"
          class="flex h-6 items-center gap-1.5 rounded-[3px] px-2 text-caption transition-colors duration-100"
          :class="
            ui.catalogLayout === 'table'
              ? 'bg-safelight text-on-accent'
              : 'text-ink-3 hover:text-ink'
          "
          :aria-checked="ui.catalogLayout === 'table'"
          title="Table view"
          @click="ui.setCatalogLayout('table')"
        >
          <svg
            viewBox="0 0 12 12"
            width="11"
            height="11"
            fill="none"
            stroke="currentColor"
            stroke-width="1.4"
            stroke-linecap="round"
            aria-hidden="true"
          >
            <path d="M1.75 2.5h8.5M1.75 6h8.5M1.75 9.5h8.5" />
          </svg>
          Table
        </button>
        <button
          type="button"
          role="radio"
          data-test="layout-grid"
          class="flex h-6 items-center gap-1.5 rounded-[3px] px-2 text-caption transition-colors duration-100"
          :class="
            ui.catalogLayout === 'grid'
              ? 'bg-safelight text-on-accent'
              : 'text-ink-3 hover:text-ink'
          "
          :aria-checked="ui.catalogLayout === 'grid'"
          title="Grid view"
          @click="ui.setCatalogLayout('grid')"
        >
          <svg
            viewBox="0 0 12 12"
            width="11"
            height="11"
            fill="none"
            stroke="currentColor"
            stroke-width="1.2"
            aria-hidden="true"
          >
            <rect x="1.5" y="1.5" width="3.6" height="3.6" rx="0.8" />
            <rect x="6.9" y="1.5" width="3.6" height="3.6" rx="0.8" />
            <rect x="1.5" y="6.9" width="3.6" height="3.6" rx="0.8" />
            <rect x="6.9" y="6.9" width="3.6" height="3.6" rx="0.8" />
          </svg>
          Grid
        </button>
      </div>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <DownloadsTray />

      <!-- One unified list: catalog results with installed models merged in
           and host-tagged; the Installed source tab scopes to what you have. -->
      <CatalogTab
        :query="query"
        :layout="ui.catalogLayout"
        :installed-entries="installedModels"
        :media-type="mediaType"
        @clear-media-filter="setMediaType('all')"
      />
    </div>
  </div>
</template>
