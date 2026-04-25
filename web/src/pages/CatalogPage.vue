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
  <div class="flex h-full">
    <CatalogSidebar />
    <main class="flex-1 flex flex-col overflow-hidden">
      <CatalogTopbar />
      <CatalogCardGrid />
    </main>
    <CatalogDetailDrawer v-if="cat.detail.value" />
  </div>
</template>
