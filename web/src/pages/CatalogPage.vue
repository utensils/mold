<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted } from "vue";
import { useCatalog } from "../composables/useCatalog";
import CatalogSidebar from "../components/CatalogSidebar.vue";
import CatalogTopbar from "../components/CatalogTopbar.vue";
import CatalogCardGrid from "../components/CatalogCardGrid.vue";
import CatalogDetailDrawer from "../components/CatalogDetailDrawer.vue";
import TopBar from "../components/TopBar.vue";

const cat = useCatalog();

// TopBar requires gallery-shaped props even when those filters are hidden
// for the catalog route (see TopBar's `v-if="$route.name === 'gallery'"`).
// We feed it placeholders identical to GeneratePage's strategy.
const topBarCounts = computed(() => ({
  total: cat.entries.value.length,
  images: 0,
  video: 0,
  filtered: cat.entries.value.length,
}));

// Detect cross-client scans so the Refresh button stays disabled when a
// scan started by another tab or the CLI is in flight. We discover once
// on mount, then poll every 10 s while the user is on /catalog and the
// chip is idle. As soon as a scan id is discovered, useCatalog's own
// 1.5 s pollRefresh takes over.
let discoverTimer: ReturnType<typeof setInterval> | null = null;
onMounted(() => {
  void cat.refresh();
  void cat.discoverActiveRefresh();
  discoverTimer = setInterval(() => {
    if (!document.hidden) void cat.discoverActiveRefresh();
  }, 10_000);
});
onBeforeUnmount(() => {
  if (discoverTimer) clearInterval(discoverTimer);
});
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-4 sm:px-6 sm:pt-6 lg:px-10">
    <TopBar
      :filter="'all'"
      :search="''"
      :view="'feed'"
      :muted="true"
      :counts="topBarCounts"
      :loading="cat.loading.value"
      @update:filter="() => {}"
      @update:search="() => {}"
      @update:view="() => {}"
      @update:muted="() => {}"
      @update:hide-mode="() => {}"
      @refresh="() => cat.refresh()"
    />

    <div class="mt-4 sm:mt-6">
      <CatalogTopbar />
    </div>

    <div class="mt-4 flex flex-col gap-4 lg:flex-row lg:items-start">
      <CatalogSidebar />
      <CatalogCardGrid />
    </div>

    <CatalogDetailDrawer v-if="cat.detail.value" />
  </div>
</template>
