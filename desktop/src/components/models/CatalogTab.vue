<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useDownloadsStore } from "../../stores/downloads";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { isGenerationModel, useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";
import { ApiError, type ApiTarget } from "../../lib/api/client";
import { fetchCatalogFamilies, searchCatalog, startCatalogDownload } from "../../lib/api/catalog";
import { isVideoFamily } from "../../lib/capabilities";
import { sortInstalledFirst } from "../../lib/catalog";
import { type MediaType } from "../../lib/modelAvailability";
import CatalogCard from "./CatalogCard.vue";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";
import DownloadTargetDialog from "./DownloadTargetDialog.vue";
import type { CatalogEntry } from "../../lib/api/types";

const props = defineProps<{
  query: string;
  layout: "grid" | "table";
  excludeInstalled?: boolean;
  installedIds?: string[];
  mediaType?: MediaType;
}>();

const emit = defineEmits<{ (e: "clear-media-filter"): void }>();

const downloads = useDownloadsStore();
const toasts = useToastStore();
const hosts = useHostsStore();
const models = useModelStore();

const PAGE_SIZE = 24;
/**
 * The media chips filter client-side on `entry.family`, but the server can
 * only constrain one family per query — a single page routinely holds zero
 * video entries. Keep auto-fetching pages (bounded) until the filtered view
 * has content or the results run out.
 */
const MAX_AUTO_PAGES = 5;
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
const pendingEntry = ref<CatalogEntry | null>(null);
/** Entry whose in-app detail drawer is open. */
const detailEntry = ref<CatalogEntry | null>(null);

let debounce: ReturnType<typeof setTimeout> | null = null;

/** True when `entry` passes the active media-type chip. */
function matchesMediaType(entry: CatalogEntry): boolean {
  const type = props.mediaType ?? "all";
  return type === "all" || isVideoFamily(entry.family) === (type === "video");
}

/**
 * Safe built-in pull targets. Live HF search can return a repository that
 * contains many checkpoints (notably Lightricks/LTX-Video) as one aggregate
 * recipe; the manifest registry already describes the actual per-model files,
 * so those variants must win over a hundreds-of-GB whole-repo pull.
 */
const manifestEntries = computed<CatalogEntry[]>(() => {
  const installed = new Set(props.installedIds ?? []);
  const q = props.query.trim().toLowerCase();
  if (source.value === "civitai") return [];
  return models.all
    .filter((model) => !model.downloaded && isGenerationModel(model))
    .filter((model) => !installed.has(model.name))
    .filter((model) => !q || model.name.toLowerCase().includes(q))
    .filter((model) => !family.value || model.family === family.value)
    .map((model) => {
      const weights = Math.round(model.size_gb * 1_000_000_000);
      const fetch = model.remaining_download_bytes ?? weights;
      const shared = Math.max(0, fetch - weights);
      return {
        id: model.name,
        source: "hf",
        source_id: model.hf_repo || null,
        name: model.name,
        family: model.family,
        kind: "checkpoint",
        nsfw: false,
        installed: false,
        size_bytes: weights,
        thumbnail_url: null,
        page_url: model.hf_repo ? `https://huggingface.co/${model.hf_repo}` : null,
        companion_details:
          shared > 0 ? [{ name: "shared runtime components", size_bytes: shared }] : [],
      };
    });
});

const combinedEntries = computed(() => {
  const knownRepos = new Set(
    models.all.map((model) => model.hf_repo).filter((repo): repo is string => Boolean(repo)),
  );
  const safeLive = entries.value.filter(
    (entry) =>
      !(
        entry.source === "hf" &&
        entry.kind === "checkpoint" &&
        (entry.bundling === "separated" ||
          Boolean(entry.source_id && knownRepos.has(entry.source_id)))
      ),
  );
  const byId = new Map<string, CatalogEntry>();
  for (const entry of [...manifestEntries.value, ...safeLive]) {
    if (!byId.has(entry.id)) byId.set(entry.id, entry);
  }
  return [...byId.values()];
});

// What you already have surfaces first; the divider marks where "available"
// begins so installed models are visible at a glance. The media-type filter
// is client-side on `entry.family` — the server query stays unchanged.
const displayEntries = computed(() =>
  sortInstalledFirst(combinedEntries.value).filter(
    (entry) =>
      !(
        props.excludeInstalled &&
        (entry.installed || (props.installedIds ?? []).includes(entry.name))
      ) && matchesMediaType(entry),
  ),
);

/** Why the grid is empty while entries exist — names the active filter. */
const filteredEmptyMessage = computed(() => {
  const type = props.mediaType ?? "all";
  if (type !== "all" && !combinedEntries.value.some(matchesMediaType)) {
    const noun = type === "video" ? "video" : "image";
    return hasMore.value
      ? `No ${noun} models in these results yet — load more or show all media types.`
      : `No ${noun} models in these results.`;
  }
  return "Everything here is already installed.";
});

const readyHosts = computed(() =>
  hosts.all.filter((host) => host.status === "ready" && host.baseUrl),
);

/**
 * Where catalog calls go: the local primary when it's ready (it reads its
 * own credentials), else the first ready host — with credentials forwarded
 * for remote hosts — so browsing survives a dead built-in engine.
 */
function catalogTarget(): { target: ApiTarget | undefined; forward: boolean } {
  const primary = hosts.all.find((host) => host.id === "local");
  if (primary?.status === "ready") return { target: undefined, forward: false };
  const fallback = readyHosts.value[0];
  if (fallback?.baseUrl) {
    return {
      target: { baseUrl: fallback.baseUrl, apiKey: fallback.apiKey },
      forward: fallback.kind === "remote",
    };
  }
  return { target: undefined, forward: false };
}

/** Invalidates in-flight page loops when a newer search supersedes them. */
let searchEpoch = 0;

async function runSearch(reset: boolean) {
  const epoch = ++searchEpoch;
  if (reset) {
    page.value = 1;
    entries.value = [];
  }
  loading.value = true;
  error.value = null;
  try {
    for (let fetched = 0; ;) {
      const { target, forward } = catalogTarget();
      const res = await searchCatalog(
        {
          q: props.query || undefined,
          family: family.value || undefined,
          source: source.value === "all" ? undefined : source.value,
          include_nsfw: includeNsfw.value,
          page: page.value,
          page_size: PAGE_SIZE,
        },
        forward,
        target,
      );
      if (epoch !== searchEpoch) return;
      entries.value = [...entries.value, ...res.entries];
      hasMore.value = res.entries.length === PAGE_SIZE;
      fetched += 1;
      // Under a media chip, keep paging (bounded) until something survives
      // the filter — otherwise the chip renders a blank, message-less grid.
      const filterActive = (props.mediaType ?? "all") !== "all";
      if (!filterActive || !hasMore.value || fetched >= MAX_AUTO_PAGES) break;
      if (combinedEntries.value.some(matchesMediaType)) break;
      page.value += 1;
    }
  } catch (err) {
    if (epoch !== searchEpoch) return;
    error.value = String(err);
    hasMore.value = false;
  } finally {
    if (epoch === searchEpoch) loading.value = false;
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

async function pullTo(entry: CatalogEntry, host: HostView | null) {
  pulling.value.add(entry.id);
  try {
    const target = host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : undefined;
    // Attach the snapshot-first stream before enqueueing so a cached,
    // near-instant pull still produces a visible terminal event and refresh.
    await downloads.subscribe(host ?? undefined);
    await startCatalogDownload(entry.id, target, host ? host.kind === "remote" : false);
    toasts.push(`Pulling ${entry.name}${host ? ` on ${host.label}` : ""}`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      toasts.push(`${entry.name} is already queued.`);
    } else {
      toasts.push(String(err), "error");
    }
  } finally {
    pulling.value.delete(entry.id);
    pendingEntry.value = null;
  }
}

function pull(entry: CatalogEntry) {
  if (readyHosts.value.length > 1) {
    pendingEntry.value = entry;
    return;
  }
  void pullTo(entry, readyHosts.value[0] ?? null);
}

/** The detail drawer fetches on the same host the catalog list came from. */
const detailTarget = computed(() => catalogTarget());

/** Pull (or Repair — same endpoint, missing files only) from the drawer. */
function pullFromDrawer(entry: CatalogEntry) {
  detailEntry.value = null;
  pull(entry);
}

watch([() => props.query, source, family, includeNsfw], scheduleSearch);

// Flipping to a media chip with no matching entries loaded yet continues the
// existing pagination instead of leaving a blank grid behind the chip.
watch(
  () => props.mediaType,
  () => {
    if ((props.mediaType ?? "all") === "all" || loading.value) return;
    if (!combinedEntries.value.some(matchesMediaType) && hasMore.value) loadMore();
  },
);

onMounted(async () => {
  try {
    const { target, forward } = catalogTarget();
    families.value = await fetchCatalogFamilies(forward, target);
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
          :class="source === s ? 'bg-safelight text-on-accent' : 'text-ink-2 hover:text-ink'"
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

    <!-- Empty state — keyed on the FILTERED list so an all-image page under
         the Video chip explains itself instead of rendering a blank grid. -->
    <div
      v-else-if="!loading && displayEntries.length === 0"
      class="p-8 text-center text-body text-ink-2"
      data-test="catalog-empty"
    >
      <template v-if="combinedEntries.length === 0">
        <template v-if="query">Nothing on the shelf for "{{ query }}".</template>
        <template v-else>Search the catalog to find models.</template>
      </template>
      <template v-else>
        <p>{{ filteredEmptyMessage }}</p>
        <button
          v-if="(mediaType ?? 'all') !== 'all'"
          type="button"
          data-test="clear-media-filter"
          class="border-edge mt-3 h-7 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink"
          @click="emit('clear-media-filter')"
        >
          Show all media types
        </button>
      </template>
    </div>

    <!-- Result cards, installed first -->
    <div
      v-else
      :class="
        layout === 'grid'
          ? 'grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-2'
          : 'flex flex-col gap-1'
      "
    >
      <template v-for="entry in displayEntries" :key="entry.id">
        <CatalogCard
          :entry="entry"
          :pulling="pulling.has(entry.id)"
          :layout="layout"
          @pull="pull"
          @open="detailEntry = $event"
        />
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

    <DownloadTargetDialog
      v-if="pendingEntry"
      :model-name="pendingEntry.name"
      :hosts="readyHosts"
      @close="pendingEntry = null"
      @select="(host) => pendingEntry && void pullTo(pendingEntry, host)"
    />

    <CatalogDetailDrawer
      v-if="detailEntry"
      :entry="detailEntry"
      :pulling="pulling.has(detailEntry.id)"
      :target="detailTarget.target"
      :forward-credentials="detailTarget.forward"
      @close="detailEntry = null"
      @pull="pullFromDrawer"
    />
  </div>
</template>
