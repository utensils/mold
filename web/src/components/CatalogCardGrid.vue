<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";
import CatalogCard from "./CatalogCard.vue";

const cat = useCatalog();

function openCard(id: string) {
  void cat.openDetail(id);
}
</script>

<template>
  <div class="flex-1 overflow-y-auto p-4">
    <!-- Loading state -->
    <div
      v-if="cat.loading.value"
      class="flex items-center justify-center py-12 text-zinc-500 text-sm"
    >
      Loading…
    </div>

    <!-- Error state -->
    <div
      v-else-if="cat.errorMsg.value"
      class="flex items-center justify-center py-12 text-rose-400 text-sm"
    >
      {{ cat.errorMsg.value }}
    </div>

    <!-- Empty state -->
    <div
      v-else-if="cat.entries.value.length === 0"
      class="flex flex-col items-center justify-center py-16 text-zinc-500 gap-2"
    >
      <p class="text-sm">No models found.</p>
      <p class="text-xs">
        Try adjusting filters or run
        <code class="bg-zinc-800 px-1 rounded">mold catalog refresh</code>.
      </p>
    </div>

    <!-- Grid -->
    <div
      v-else
      class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6"
    >
      <button
        v-for="entry in cat.entries.value"
        :key="entry.id"
        class="text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 rounded-lg"
        @click="openCard(entry.id)"
      >
        <CatalogCard :entry="entry" />
      </button>
    </div>
  </div>
</template>
