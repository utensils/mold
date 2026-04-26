<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";

const cat = useCatalog();

function formatGB(bytes: number | null): string {
  if (!bytes) return "—";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

async function handleDownload() {
  if (!cat.detail.value) return;
  await cat.startDownload(cat.detail.value.id);
  cat.closeDetail();
}
</script>

<template>
  <div
    v-if="cat.detail.value"
    class="fixed inset-y-0 right-0 z-40 flex w-80 flex-col border-l border-zinc-800 bg-zinc-950 shadow-2xl"
    role="dialog"
    aria-modal="true"
    :aria-label="cat.detail.value.name"
  >
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-zinc-800 p-4">
      <h2 class="text-sm font-semibold text-zinc-100 truncate pr-2">
        {{ cat.detail.value.name }}
      </h2>
      <button
        data-test="close-btn"
        type="button"
        class="inline-flex h-7 w-7 items-center justify-center rounded-full text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition"
        aria-label="Close"
        @click="cat.closeDetail()"
      >
        <svg
          class="h-4 w-4"
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
    </div>

    <!-- Thumbnail -->
    <div
      class="aspect-video bg-zinc-900 flex items-center justify-center overflow-hidden"
    >
      <img
        v-if="cat.detail.value.thumbnail_url"
        :src="cat.detail.value.thumbnail_url"
        :alt="cat.detail.value.name"
        class="object-cover w-full h-full"
      />
      <span v-else class="text-zinc-600 text-xs">no thumbnail</span>
    </div>

    <!-- Meta -->
    <div class="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
      <!-- Phase badge -->
      <div v-if="cat.detail.value.engine_phase >= 3" class="inline-flex">
        <span
          class="text-[11px] px-2 py-0.5 rounded bg-amber-700/30 text-amber-200"
        >
          Coming in phase {{ cat.detail.value.engine_phase }}
        </span>
      </div>

      <!-- Author + family -->
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div>
          <div class="text-zinc-500 mb-0.5">Author</div>
          <div class="text-zinc-200">
            {{ cat.detail.value.author ?? "unknown" }}
          </div>
        </div>
        <div>
          <div class="text-zinc-500 mb-0.5">Family</div>
          <div class="text-zinc-200">{{ cat.detail.value.family }}</div>
        </div>
        <div>
          <div class="text-zinc-500 mb-0.5">Size</div>
          <div class="text-zinc-200">
            {{ formatGB(cat.detail.value.size_bytes) }}
          </div>
        </div>
        <div>
          <div class="text-zinc-500 mb-0.5">Downloads</div>
          <div class="text-zinc-200">
            {{ cat.detail.value.download_count.toLocaleString("en-US") }}
          </div>
        </div>
        <div v-if="cat.detail.value.rating !== null">
          <div class="text-zinc-500 mb-0.5">Rating</div>
          <div class="text-zinc-200">
            ★ {{ cat.detail.value.rating.toFixed(1) }}
          </div>
        </div>
        <div>
          <div class="text-zinc-500 mb-0.5">Modality</div>
          <div class="text-zinc-200 capitalize">
            {{ cat.detail.value.modality }}
          </div>
        </div>
        <div>
          <div class="text-zinc-500 mb-0.5">Format</div>
          <div class="text-zinc-200">{{ cat.detail.value.file_format }}</div>
        </div>
        <div v-if="cat.detail.value.license">
          <div class="text-zinc-500 mb-0.5">License</div>
          <div class="text-zinc-200 truncate">
            {{ cat.detail.value.license }}
          </div>
        </div>
      </div>

      <!-- Description -->
      <p
        v-if="cat.detail.value.description"
        class="text-xs text-zinc-400 leading-relaxed"
      >
        {{ cat.detail.value.description }}
      </p>

      <!-- Tags -->
      <div v-if="cat.detail.value.tags.length > 0" class="flex flex-wrap gap-1">
        <span
          v-for="tag in cat.detail.value.tags"
          :key="tag"
          class="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400"
        >
          {{ tag }}
        </span>
      </div>
    </div>

    <!-- Actions -->
    <div class="border-t border-zinc-800 p-4">
      <button
        data-test="download-btn"
        type="button"
        :disabled="!cat.canDownload(cat.detail.value)"
        :title="
          !cat.canDownload(cat.detail.value)
            ? `Coming in phase ${cat.detail.value.engine_phase}`
            : 'Download this model'
        "
        class="w-full rounded-lg px-4 py-2 text-sm font-medium transition"
        :class="
          cat.canDownload(cat.detail.value)
            ? 'bg-zinc-100 text-zinc-900 hover:bg-white'
            : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
        "
        @click="handleDownload"
      >
        Download
      </button>
    </div>
  </div>
</template>
