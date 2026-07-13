<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import InstalledTab from "../components/models/InstalledTab.vue";
import CatalogTab from "../components/models/CatalogTab.vue";
import DownloadsTray from "../components/models/DownloadsTray.vue";
import { useConnectionStore } from "../stores/connection";
import { useModelStore } from "../stores/models";
import { useDownloadsStore } from "../stores/downloads";
import { formatGB } from "../lib/format";
import { modelDiskBytes } from "../lib/models";
import { primaryModifierPressed } from "../lib/platform";

const conn = useConnectionStore();
const models = useModelStore();
const downloads = useDownloadsStore();
const route = useRoute();

type Tab = "installed" | "catalog";
// "Browse all models" deep-links straight into the catalog (/models?tab=catalog).
const tab = ref<Tab>(route.query.tab === "catalog" ? "catalog" : "installed");
const query = ref("");
const searchEl = ref<HTMLInputElement | null>(null);

const installedCount = computed(() => models.installed.length);
const installedBytes = computed(() =>
  models.installed.reduce((sum, m) => sum + modelDiskBytes(m), 0),
);

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
    <!-- Header: tabs + search -->
    <header class="border-edge flex items-center gap-3 border-b bg-bench px-4 py-2">
      <div class="flex items-center gap-1">
        <button
          type="button"
          class="h-7 rounded-control px-2.5 text-body"
          :class="tab === 'installed' ? 'bg-bath text-ink' : 'text-ink-2 hover:text-ink'"
          @click="tab = 'installed'"
        >
          Installed
          <span class="data-mono ml-1 text-ink-3"
            >{{ installedCount }} · {{ formatGB(installedBytes) }}</span
          >
        </button>
        <button
          type="button"
          class="h-7 rounded-control px-2.5 text-body"
          :class="tab === 'catalog' ? 'bg-bath text-ink' : 'text-ink-2 hover:text-ink'"
          @click="tab = 'catalog'"
        >
          Catalog
        </button>
      </div>

      <input
        ref="searchEl"
        v-model="query"
        data-selectable
        type="search"
        :placeholder="tab === 'catalog' ? 'Search the catalog…' : 'Filter installed…'"
        class="border-edge ml-auto h-7 w-64 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
      />
    </header>

    <!-- Body -->
    <div class="min-h-0 flex-1 overflow-y-auto">
      <InstalledTab v-if="tab === 'installed'" :query="query" @browse-catalog="tab = 'catalog'" />
      <CatalogTab v-else :query="query" />
    </div>

    <DownloadsTray />
  </div>
</template>
