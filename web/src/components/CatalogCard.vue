<script setup lang="ts">
import type { CatalogEntryWire } from "../types";

const props = defineProps<{ entry: CatalogEntryWire }>();

function formatGB(bytes: number | null): string {
  if (!bytes) return "—";
  const gb = bytes / 1_000_000_000;
  return `${gb.toFixed(1)} GB`;
}

function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
</script>

<template>
  <article
    class="rounded-lg border border-zinc-800 bg-zinc-900 p-3 hover:border-zinc-600 transition-colors flex flex-col gap-2"
  >
    <div
      class="aspect-square bg-zinc-950 rounded overflow-hidden flex items-center justify-center"
    >
      <img
        v-if="props.entry.thumbnail_url"
        :src="props.entry.thumbnail_url"
        :alt="props.entry.name"
        loading="lazy"
        class="object-cover w-full h-full"
      />
      <span v-else class="text-zinc-600 text-xs">no thumbnail</span>
    </div>
    <div class="flex items-start justify-between gap-2">
      <h3 class="text-sm font-medium text-zinc-100 truncate">
        {{ props.entry.name }}
      </h3>
      <span
        v-if="props.entry.engine_phase >= 3"
        class="text-[10px] uppercase tracking-wide px-1.5 py-0.5 bg-amber-700/30 text-amber-200 rounded"
        :title="`Coming in phase ${props.entry.engine_phase}`"
      >
        phase {{ props.entry.engine_phase }}
      </span>
    </div>
    <p class="text-xs text-zinc-500 truncate">
      {{ props.entry.author ?? "unknown" }}
    </p>
    <div class="flex items-center justify-between text-[11px] text-zinc-400">
      <span>{{ formatGB(props.entry.size_bytes) }}</span>
      <span>{{ formatCount(props.entry.download_count) }} dl</span>
      <span v-if="props.entry.rating !== null"
        >★ {{ props.entry.rating.toFixed(1) }}</span
      >
    </div>
  </article>
</template>
