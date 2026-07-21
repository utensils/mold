<script setup lang="ts">
import { ref, watch } from "vue";
import { useCatalog } from "../composables/useCatalog";
import type { CatalogKind, CatalogListParams } from "../types";

const cat = useCatalog();

type Modality = "image" | "video";
type SortOption = CatalogListParams["sort"];
type KindFilter = CatalogKind;

const KIND_OPTIONS: { value: KindFilter; label: string }[] = [
  { value: "checkpoint", label: "Models" },
  { value: "lora", label: "LoRAs" },
  { value: "clip", label: "CLIP" },
  { value: "text-encoder", label: "Text encoders" },
  { value: "vae", label: "VAEs" },
  { value: "tokenizer", label: "Tokenizers" },
  { value: "control-net", label: "ControlNet" },
];

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: "downloads", label: "Downloads" },
  { value: "rating", label: "Rating" },
  { value: "recent", label: "Recent" },
];

const searchLocal = ref(cat.filter.value.q ?? "");
let debounceHandle: ReturnType<typeof setTimeout> | null = null;

watch(searchLocal, (v) => {
  if (debounceHandle) clearTimeout(debounceHandle);
  debounceHandle = setTimeout(() => {
    cat.setFilter({ q: v || undefined });
  }, 250);
});

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

function setKind(k: KindFilter | undefined) {
  cat.setFilter({ kind: k });
}

function setNsfw(on: boolean) {
  cat.setFilter({ include_nsfw: on || undefined });
}

function clearSearch() {
  searchLocal.value = "";
  cat.setFilter({ q: undefined });
}
</script>

<template>
  <div
    class="bg-bench border border-edge flex flex-wrap items-center gap-2 rounded-3xl px-4 py-3 sm:gap-3 sm:px-5 sm:py-3.5"
  >
    <!-- Modality chips — match TopBar's filter-pill pattern -->
    <nav
      class="flex items-center gap-0.5 rounded-full border border-white/5 bg-white/5 p-0.5 text-[13px] font-medium text-ink-2"
      aria-label="Modality filter"
    >
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          !cat.filter.value.modality
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setModality(undefined)"
      >
        All
      </button>
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          cat.filter.value.modality === 'image'
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setModality('image')"
      >
        Image
      </button>
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          cat.filter.value.modality === 'video'
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setModality('video')"
      >
        Video
      </button>
    </nav>

    <!-- Kind chips -->
    <nav
      class="flex items-center gap-0.5 rounded-full border border-white/5 bg-white/5 p-0.5 text-[13px] font-medium text-ink-2"
      aria-label="Kind filter"
    >
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          !cat.filter.value.kind
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setKind(undefined)"
      >
        All
      </button>
      <button
        v-for="opt in KIND_OPTIONS"
        :key="opt.value"
        class="rounded-full px-3 py-1.5 transition"
        :class="
          cat.filter.value.kind === opt.value
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setKind(opt.value)"
      >
        {{ opt.label }}
      </button>
    </nav>

    <!-- Source chips -->
    <nav
      class="flex items-center gap-0.5 rounded-full border border-white/5 bg-white/5 p-0.5 text-[13px] font-medium text-ink-2"
      aria-label="Source filter"
    >
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          !cat.filter.value.source
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setSource(undefined)"
      >
        All sources
      </button>
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          cat.filter.value.source === 'hf'
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setSource('hf')"
      >
        HF
      </button>
      <button
        class="rounded-full px-3 py-1.5 transition"
        :class="
          cat.filter.value.source === 'civitai'
            ? 'bg-safelight text-white shadow-sm'
            : 'hover:text-white'
        "
        @click="setSource('civitai')"
      >
        CivitAI
      </button>
    </nav>

    <!-- Search -->
    <label class="relative min-w-[180px] flex-1">
      <svg
        class="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-3"
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
        class="h-10 w-full rounded-full border border-white/5 bg-white/5 pl-10 pr-10 text-[13px] text-rebate placeholder:text-ink-3 focus:border-safelight/40 focus:outline-none focus:ring-2 focus:ring-safelight/25"
      />
      <button
        v-if="searchLocal"
        type="button"
        class="absolute right-2.5 top-1/2 inline-flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-full text-ink-3 transition hover:bg-white/10 hover:text-white"
        aria-label="Clear search"
        @click="clearSearch"
      >
        <svg
          class="h-3.5 w-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M6 6l12 12" />
          <path d="M18 6 6 18" />
        </svg>
      </button>
    </label>

    <!-- Sort -->
    <select
      :value="cat.filter.value.sort ?? 'downloads'"
      class="h-10 rounded-full border border-white/5 bg-white/5 px-4 text-[13px] font-medium text-rebate focus:border-safelight/40 focus:outline-none focus:ring-2 focus:ring-safelight/25"
      aria-label="Sort by"
      @change="
        setSort(($event.target as HTMLSelectElement).value as SortOption)
      "
    >
      <option v-for="opt in SORT_OPTIONS" :key="opt.value" :value="opt.value">
        {{ opt.label }}
      </option>
    </select>

    <!-- Include NSFW -->
    <label
      class="flex items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3.5 py-1.5 text-[13px] font-medium text-ink-2"
    >
      <input
        type="checkbox"
        class="h-3.5 w-3.5 accent-safelight"
        aria-label="Include NSFW"
        :checked="cat.filter.value.include_nsfw ?? false"
        @change="setNsfw(($event.target as HTMLInputElement).checked)"
      />
      Include NSFW
    </label>
  </div>
</template>
