<script setup lang="ts">
import { onMounted } from "vue";
import { useCatalog } from "../composables/useCatalog";
import CatalogSidebar from "../components/CatalogSidebar.vue";
import CatalogTopbar from "../components/CatalogTopbar.vue";
import CatalogCardGrid from "../components/CatalogCardGrid.vue";
import CatalogDetailDrawer from "../components/CatalogDetailDrawer.vue";

const cat = useCatalog();

onMounted(() => {
  void cat.refresh();
});
</script>

<template>
  <div class="mx-auto max-w-[1800px] px-4 pb-40 pt-4 sm:px-6 sm:pt-6 lg:px-10">
    <h1 class="font-display text-2xl font-bold tracking-tight text-ink">
      Models
    </h1>

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
