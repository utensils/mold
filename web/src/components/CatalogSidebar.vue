<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";

const cat = useCatalog();

function clearFamily() {
  cat.setFilter({ family: undefined });
}
</script>

<template>
  <aside
    class="bg-bench border border-edge w-full shrink-0 rounded-3xl p-3 sm:p-4 lg:w-60 lg:max-h-[calc(100svh-12rem)] lg:overflow-y-auto"
    aria-label="Family filter"
  >
    <div class="mb-2 flex items-center justify-between">
      <h2
        class="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-3"
      >
        Families
      </h2>
      <button
        v-if="cat.filter.value.family"
        type="button"
        class="rounded-full px-2 py-0.5 text-[11px] text-ink-3 transition hover:bg-white/5 hover:text-white"
        @click="clearFamily"
      >
        Clear
      </button>
    </div>
    <ul class="flex flex-wrap gap-1.5 lg:flex-col lg:gap-1">
      <li v-for="row in cat.families.value" :key="row.family">
        <button
          type="button"
          class="min-h-11 w-full rounded-2xl border border-transparent px-3 py-1.5 text-left text-[13px] font-medium text-ink-2 transition hover:bg-white/5 hover:text-white"
          :class="{
            'border-safelight/40 bg-safelight/15 text-white':
              cat.filter.value.family === row.family,
          }"
          :data-family="row.family"
          :aria-pressed="cat.filter.value.family === row.family"
          @click="cat.setFilter({ family: row.family })"
          @keydown.enter.prevent="cat.setFilter({ family: row.family })"
          @keydown.space.prevent="cat.setFilter({ family: row.family })"
        >
          {{ row.family }}
        </button>
      </li>
    </ul>
  </aside>
</template>
