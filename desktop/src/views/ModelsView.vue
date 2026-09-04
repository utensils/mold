<script setup lang="ts">
/*
 * Styles (README §02 lexicon: never "models" as the primary word). One 40px
 * view toolbar — Ready to use | Browse more, the kind filter, and Filter… —
 * over a download banner and either the shelf or the catalog. Both share the
 * merged model set: Ready to use is the full-featured inventory (load /
 * unload / remove / per-machine actions); Browse more is the live catalog
 * with ready styles merged in, machine-tagged and sorted first.
 */
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
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

type Segment = "installed" | "discover";
// Legacy deep links (`?tab=catalog`) opened the browse view — honor them.
const legacyDiscover = route.query.tab === "catalog" || route.query.tab === "discover";
const segment = ref<Segment>(legacyDiscover ? "discover" : "installed");

const MEDIA_TYPES: SegmentOption<MediaType>[] = [
  { value: "all", label: "All" },
  { value: "image", label: "Pictures" },
  { value: "video", label: "Clips" },
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
  for (const entry of hostModels.unionDownloaded) {
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

const segments = computed<SegmentOption<Segment>[]>(() => [
  { value: "installed", label: "Ready to use", sub: String(installedModels.value.length) },
  { value: "discover", label: "Browse more" },
]);

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
        // A pre-downloads-API host still participates in the shelf.
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
  <div class="flex h-full min-h-0 flex-col bg-bg">
    <!-- view toolbar -->
    <div
      class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2.5 border-b border-border bg-chrome px-3.5"
    >
      <SegmentedControl
        :model-value="segment"
        :options="segments"
        label="Styles view"
        compact
        inline
        @update:model-value="segment = $event"
      />
      <div class="flex-1" />
      <SegmentedControl
        :model-value="mediaType"
        :options="MEDIA_TYPES"
        label="Media type"
        compact
        @update:model-value="setMediaType"
      />
      <label
        class="flex h-[26px] w-[190px] items-center gap-1.5 rounded-control border border-border bg-bg px-2 focus-within:border-border-focus"
      >
        <Icon name="search" :size="14" class="shrink-0 text-fg-dim" />
        <input
          ref="searchEl"
          v-model="query"
          data-selectable
          type="search"
          placeholder="Filter…"
          aria-label="Filter styles"
          class="min-w-0 flex-1 bg-transparent text-xs text-fg outline-none placeholder:text-fg-dim"
        />
      </label>
    </div>

    <!-- Downloads on their way: pinned above the list on BOTH tabs, outside
         the scroll container so it stays put while the list moves. -->
    <DownloadsTray class="shrink-0" />

    <div class="min-h-0 flex-1 overflow-y-auto" data-test="models-scroll">
      <InstalledTab
        v-if="segment === 'installed'"
        :query="query"
        :media-type="mediaType"
        :entries="installedModels"
        @browse-catalog="segment = 'discover'"
      />

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
