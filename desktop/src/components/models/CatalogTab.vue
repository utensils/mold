<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useDownloadsStore } from "../../stores/downloads";
import { useToastStore } from "../../stores/toasts";
import { ApiError } from "../../lib/api/client";
import { fetchCatalogFamilies, searchCatalog, startCatalogDownload } from "../../lib/api/catalog";
import {
  catalogFetchCaption,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
} from "../../lib/catalog";
import type { CatalogEntry } from "../../lib/api/types";

const props = defineProps<{ query: string }>();

const downloads = useDownloadsStore();
const toasts = useToastStore();

const PAGE_SIZE = 24;
type Source = "all" | "hf" | "civitai";

const source = ref<Source>("all");
const family = ref("");
const includeNsfw = ref(false);
const families = ref<string[]>([]);

const entries = ref<CatalogEntry[]>([]);
const page = ref(1);
const hasMore = ref(false);
const loading = ref(false);
const error = ref<string | null>(null);
const pulling = ref<Set<string>>(new Set());

let debounce: ReturnType<typeof setTimeout> | null = null;

async function runSearch(reset: boolean) {
  if (reset) {
    page.value = 1;
    entries.value = [];
  }
  loading.value = true;
  error.value = null;
  try {
    const res = await searchCatalog({
      q: props.query || undefined,
      family: family.value || undefined,
      source: source.value === "all" ? undefined : source.value,
      include_nsfw: includeNsfw.value,
      page: page.value,
      page_size: PAGE_SIZE,
    });
    entries.value = reset ? res.entries : [...entries.value, ...res.entries];
    hasMore.value = res.entries.length === PAGE_SIZE;
  } catch (err) {
    error.value = String(err);
    hasMore.value = false;
  } finally {
    loading.value = false;
  }
}

function scheduleSearch() {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(() => void runSearch(true), 400);
}

function loadMore() {
  page.value += 1;
  void runSearch(false);
}

async function pull(entry: CatalogEntry) {
  pulling.value.add(entry.id);
  try {
    await startCatalogDownload(entry.id);
    toasts.push(`Pulling ${entry.name}`);
    downloads.subscribe();
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      toasts.push(`${entry.name} is already queued.`);
    } else {
      toasts.push(String(err), "error");
    }
  } finally {
    pulling.value.delete(entry.id);
  }
}

watch([() => props.query, source, family, includeNsfw], scheduleSearch);

onMounted(async () => {
  try {
    families.value = await fetchCatalogFamilies();
  } catch {
    /* families are a nicety; search still works without them */
  }
  void runSearch(true);
});
</script>

<template>
  <div class="flex flex-col gap-3 p-4">
    <!-- Filter chips -->
    <div class="flex flex-wrap items-center gap-2">
      <div class="flex items-center gap-1">
        <button
          v-for="s in ['all', 'hf', 'civitai'] as const"
          :key="s"
          type="button"
          class="border-edge h-7 rounded-full border px-2.5 text-caption"
          :class="source === s ? 'bg-safelight text-[#141110]' : 'text-ink-2 hover:text-ink'"
          @click="source = s"
        >
          {{ s === "all" ? "All" : s === "hf" ? "HuggingFace" : "Civitai" }}
        </button>
      </div>

      <select
        v-model="family"
        class="border-edge h-7 rounded-control border bg-bath px-1.5 text-caption text-ink"
      >
        <option value="">All families</option>
        <option v-for="f in families" :key="f" :value="f">{{ f }}</option>
      </select>

      <label class="flex items-center gap-1 text-caption text-ink-2">
        <input v-model="includeNsfw" type="checkbox" class="accent-[var(--safelight)]" />
        NSFW
      </label>
    </div>

    <p v-if="error" class="text-caption text-stop">{{ error }}</p>

    <!-- Empty state -->
    <div v-else-if="!loading && entries.length === 0" class="p-8 text-center text-body text-ink-2">
      <template v-if="query">Nothing on the shelf for "{{ query }}".</template>
      <template v-else>Search the catalog to find models.</template>
    </div>

    <!-- Result cards -->
    <div v-else class="grid grid-cols-2 gap-2">
      <div
        v-for="entry in entries"
        :key="entry.id"
        class="border-edge flex flex-col gap-1.5 rounded-chrome border bg-bath p-3 transition-colors duration-100 hover:bg-bench"
      >
        <div class="flex items-start justify-between gap-2">
          <span class="truncate text-body text-ink" :title="entry.name">{{ entry.name }}</span>
          <span
            class="border-edge data-mono shrink-0 rounded-full border px-1.5 text-caption text-ink-2"
          >
            {{ entry.family }}
          </span>
        </div>
        <span v-if="entry.author" class="truncate text-caption text-ink-3">{{ entry.author }}</span>

        <div class="data-mono text-caption text-ink-2">
          {{ catalogSizeLabel(catalogSizeInfo(entry)) }}
        </div>
        <div v-if="catalogFetchCaption(catalogSizeInfo(entry))" class="text-caption text-ink-3">
          {{ catalogFetchCaption(catalogSizeInfo(entry)) }}
        </div>

        <div class="mt-1 flex justify-end">
          <span v-if="entry.installed" class="data-mono text-caption text-halide">
            ● installed
          </span>
          <button
            v-else
            type="button"
            class="border-edge h-7 rounded-control border px-2.5 text-caption text-safelight transition-colors duration-100 hover:border-safelight active:translate-y-px disabled:opacity-50"
            :disabled="pulling.has(entry.id)"
            @click="pull(entry)"
          >
            {{ pulling.has(entry.id) ? "Pulling…" : catalogPullLabel(catalogSizeInfo(entry)) }}
          </button>
        </div>
      </div>
    </div>

    <button
      v-if="hasMore"
      type="button"
      class="border-edge mx-auto h-8 rounded-control border px-4 text-body text-ink-2 hover:text-ink disabled:opacity-50"
      :disabled="loading"
      @click="loadMore"
    >
      {{ loading ? "Loading…" : "Load more" }}
    </button>
  </div>
</template>
