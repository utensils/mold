<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";
import CatalogCard from "./CatalogCard.vue";

const cat = useCatalog();

function openCard(id: string) {
  void cat.openDetail(id);
}
</script>

<template>
  <div class="min-w-0 flex-1">
    <!-- Loading state -->
    <div
      v-if="cat.loading.value"
      class="flex items-center justify-center py-12 text-sm text-ink-400"
    >
      Loading…
    </div>

    <!-- Error state -->
    <div
      v-else-if="cat.errorMsg.value"
      class="glass flex items-start gap-3 rounded-2xl px-4 py-3 text-sm text-rose-200"
    >
      <span class="mt-0.5">⚠</span>
      <div>
        <p class="font-medium text-rose-100">Couldn't load the catalog.</p>
        <p class="text-rose-200/80">{{ cat.errorMsg.value }}</p>
      </div>
    </div>

    <!-- Empty state -->
    <div
      v-else-if="cat.entries.value.length === 0"
      class="flex flex-col items-center justify-center gap-2 py-16 text-ink-400"
    >
      <p class="text-sm">No models found.</p>
      <p class="text-xs">
        Try adjusting filters or click
        <span class="text-ink-200">Refresh catalog</span>
        in the top bar.
      </p>
    </div>

    <!-- Grid -->
    <div
      v-else
      class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6"
    >
      <button
        v-for="entry in cat.entries.value"
        :key="entry.id"
        class="rounded-lg text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400/60"
        @click="openCard(entry.id)"
      >
        <CatalogCard :entry="entry" />
      </button>
    </div>
  </div>
</template>
