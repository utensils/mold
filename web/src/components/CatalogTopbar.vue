<script setup lang="ts">
import { ref, watch } from "vue";
import { useCatalog } from "../composables/useCatalog";
import type { CatalogListParams } from "../types";

const cat = useCatalog();

type Modality = "image" | "video";
type SortOption = CatalogListParams["sort"];

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: "downloads", label: "Downloads" },
  { value: "rating", label: "Rating" },
  { value: "recent", label: "Recent" },
  { value: "name", label: "Name" },
];

const searchLocal = ref(cat.filter.value.q ?? "");
let debounceHandle: ReturnType<typeof setTimeout> | null = null;

watch(searchLocal, (v) => {
  if (debounceHandle) clearTimeout(debounceHandle);
  debounceHandle = setTimeout(() => {
    cat.setFilter({ q: v || undefined });
  }, 250);
});

// Keep local in sync if filter is reset externally
watch(
  () => cat.filter.value.q,
  (v) => {
    if ((v ?? "") !== searchLocal.value) searchLocal.value = v ?? "";
  },
);

function setModality(m: Modality | undefined) {
  cat.setFilter({ modality: m });
}

function setSort(s: SortOption) {
  cat.setFilter({ sort: s });
}

function setSource(s: "hf" | "civitai" | undefined) {
  cat.setFilter({ source: s });
}

function toggleNsfw() {
  cat.setFilter({ include_nsfw: !cat.filter.value.include_nsfw });
}
</script>

<template>
  <div class="flex flex-wrap items-center gap-2 border-b border-zinc-800 p-3">
    <!-- Modality chips -->
    <div
      class="flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900 p-0.5 text-xs font-medium"
    >
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          !cat.filter.value.modality
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setModality(undefined)"
      >
        All
      </button>
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          cat.filter.value.modality === 'image'
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setModality('image')"
      >
        Image
      </button>
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          cat.filter.value.modality === 'video'
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setModality('video')"
      >
        Video
      </button>
    </div>

    <!-- Source chips -->
    <div
      class="flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900 p-0.5 text-xs font-medium"
    >
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          !cat.filter.value.source
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setSource(undefined)"
      >
        All sources
      </button>
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          cat.filter.value.source === 'hf'
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setSource('hf')"
      >
        HF
      </button>
      <button
        class="rounded-full px-3 py-1 transition"
        :class="
          cat.filter.value.source === 'civitai'
            ? 'bg-zinc-700 text-zinc-100'
            : 'text-zinc-400 hover:text-zinc-100'
        "
        @click="setSource('civitai')"
      >
        CivitAI
      </button>
    </div>

    <!-- Search -->
    <div class="relative flex-1 min-w-[160px]">
      <svg
        class="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      <input
        v-model="searchLocal"
        type="search"
        placeholder="Search catalog…"
        autocomplete="off"
        spellcheck="false"
        class="h-8 w-full rounded-full border border-zinc-700 bg-zinc-900 pl-9 pr-3 text-xs text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-500 focus:outline-none"
      />
    </div>

    <!-- Sort -->
    <select
      :value="cat.filter.value.sort ?? 'downloads'"
      class="h-8 rounded-full border border-zinc-700 bg-zinc-900 px-3 text-xs text-zinc-200 focus:outline-none"
      @change="
        setSort(($event.target as HTMLSelectElement).value as SortOption)
      "
    >
      <option v-for="opt in SORT_OPTIONS" :key="opt.value" :value="opt.value">
        {{ opt.label }}
      </option>
    </select>

    <!-- NSFW toggle -->
    <button
      class="rounded-full border border-zinc-700 px-3 py-1 text-xs transition"
      :class="
        cat.filter.value.include_nsfw
          ? 'bg-amber-700/40 text-amber-200 border-amber-700'
          : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
      "
      @click="toggleNsfw"
    >
      NSFW
    </button>
  </div>
</template>
