<script setup lang="ts">
import { computed, onMounted } from "vue";
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

onMounted(() => {
  void cat.refresh();
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
