<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useDownloadsStore } from "../../stores/downloads";
import { useToastStore } from "../../stores/toasts";
import { ApiError } from "../../lib/api/client";
import { fetchCatalogFamilies, searchCatalog, startCatalogDownload } from "../../lib/api/catalog";
import { sortInstalledFirst } from "../../lib/catalog";
import CatalogCard from "./CatalogCard.vue";
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

// What you already have surfaces first; the divider marks where "available"
// begins so installed models are visible at a glance.
const displayEntries = computed(() => sortInstalledFirst(entries.value));
const installedCount = computed(() => entries.value.filter((e) => e.installed).length);

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

    <!-- Result cards, installed first -->
    <div v-else class="grid grid-cols-2 gap-2">
      <template v-for="(entry, index) in displayEntries" :key="entry.id">
        <div
          v-if="installedCount > 0 && index === 0"
          class="col-span-2 flex items-center gap-2"
          data-test="installed-divider"
        >
          <span class="edge-code">Installed</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <div
          v-if="
            installedCount > 0 && installedCount < displayEntries.length && index === installedCount
          "
          class="col-span-2 flex items-center gap-2"
          data-test="available-divider"
        >
          <span class="edge-code">Available</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <CatalogCard :entry="entry" :pulling="pulling.has(entry.id)" @pull="pull" />
      </template>
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
