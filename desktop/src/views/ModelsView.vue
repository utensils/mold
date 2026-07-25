<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import SegmentedControl, { type SegmentOption } from "@ui/components/SegmentedControl.vue";
import CatalogTab from "../components/models/CatalogTab.vue";
import InstalledTab from "../components/models/InstalledTab.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { primaryModifierPressed } from "../lib/platform";
import { mediaTypeFromQuery, type MediaType } from "../lib/modelAvailability";
import { mergeModelPresentationMetadata } from "../lib/models";

const conn = useConnectionStore();
const models = useModelStore();
const downloads = useDownloadsStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const route = useRoute();
const router = useRouter();
const query = ref("");
const searchEl = ref<HTMLInputElement | null>(null);

/**
 * Installed vs Discover is the primary chrome now (Mold Studio Models). Both
 * views share the merged model set: Installed is the full-featured inventory
 * (load / unload / delete / per-host actions); Discover is the unified live
 * catalog with installed models still merged in, host-tagged and sorted first.
 */
type Segment = "installed" | "discover";
const SEGMENTS: SegmentOption<Segment>[] = [
  { value: "installed", label: "Installed" },
  { value: "discover", label: "Discover" },
];
// Legacy deep links (`?tab=catalog`) opened the browse view — honor them.
const legacyDiscover = route.query.tab === "catalog" || route.query.tab === "discover";
const segment = ref<Segment>(legacyDiscover ? "discover" : "installed");

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
    if (existing) {
      byName.set(entry.name, {
        ...mergeModelPresentationMetadata(existing, entry),
        hostIds: [...new Set([...existing.hostIds, ...entry.hostIds])],
      });
    } else {
      byName.set(entry.name, { ...entry, hostIds: [...entry.hostIds] });
    }
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
  <div class="flex h-full min-h-0 flex-col">
    <!-- Workspace header — Installed | Discover is the primary chrome -->
    <header class="border-edge flex h-13 shrink-0 items-center gap-3 border-b px-6">
      <h1 class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Models
      </h1>
      <div class="flex-1" />
      <SegmentedControl
        class="w-52 shrink-0"
        :model-value="segment"
        :options="SEGMENTS"
        label="Model view"
        @update:model-value="segment = $event"
      />
    </header>

    <!-- Shared filter strip: search + media type apply to both segments -->
    <div class="border-edge flex flex-wrap items-center gap-3 border-b bg-bench px-6 py-2.5">
      <input
        ref="searchEl"
        v-model="query"
        data-selectable
        type="search"
        placeholder="Search models…"
        class="border-edge h-7 min-w-48 flex-1 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3 sm:max-w-72"
      />

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
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <!-- Downloads pinned above the list in BOTH segments -->
      <DownloadsTray />

      <!-- Installed: the full-featured inventory scoped to what you have. -->
      <InstalledTab
        v-if="segment === 'installed'"
        :query="query"
        :media-type="mediaType"
        :entries="installedModels"
        @browse-catalog="segment = 'discover'"
      />

      <!-- Discover: one unified list — live catalog with installed models
           merged in, host-tagged and sorted first. -->
      <CatalogTab
        v-else
        :query="query"
        :installed-entries="installedModels"
        :media-type="mediaType"
        @clear-media-filter="setMediaType('all')"
      />
    </div>
  </div>
</template>
