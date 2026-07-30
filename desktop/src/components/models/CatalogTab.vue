<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import CatalogLayoutToggle, {
  type CatalogLayoutChoice,
} from "@ui/components/CatalogLayoutToggle.vue";
import { useDownloadsStore } from "../../stores/downloads";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { isGenerationModel, useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { ApiError, type ApiTarget } from "../../lib/api/client";
import { fetchCatalogFamilies, searchCatalog, startCatalogDownload } from "../../lib/api/catalog";
import { isVideoFamily } from "../../lib/capabilities";
import { catalogIdentityKey, sortInstalledFirst } from "../../lib/catalog";
import {
  CATALOG_KIND_OPTIONS,
  CATALOG_SORT_OPTIONS,
  type CatalogKindFilter,
  type CatalogSortOption,
} from "../../lib/catalogFilters";
import { isCatalogModelId, modelDisplayName } from "../../lib/models";
import { type MediaType } from "../../lib/modelAvailability";
import CatalogCard from "./CatalogCard.vue";
import CatalogTableRow from "./CatalogTableRow.vue";
import CatalogDetailDrawer, { type DrawerVariant } from "./CatalogDetailDrawer.vue";
import DownloadTargetDialog from "./DownloadTargetDialog.vue";
import { installedModelToEntry } from "../../lib/catalogDetail";
import type { CatalogEntry, ModelEntry } from "../../lib/api/types";

type LibraryModelEntry = ModelEntry & { hostIds?: string[] };

const props = defineProps<{
  query: string;
  /** Optional layout override; otherwise the session-persisted ui store wins. */
  layout?: "grid" | "table";
  /** Installed models (all hosts, with hostIds) merged into the unified list
   *  as host-tagged rows — the Discover feed still surfaces what you have. */
  installedEntries?: LibraryModelEntry[];
  mediaType?: MediaType;
}>();

const emit = defineEmits<{ (e: "clear-media-filter"): void }>();

const downloads = useDownloadsStore();
const toasts = useToastStore();
const hosts = useHostsStore();
const models = useModelStore();
const ui = useUiStore();

/** Grid or table — the layout toggle lives here now (Discover's secondary
 *  control); an explicit `layout` prop still overrides for embeddings/tests. */
const effectiveLayout = computed(() => props.layout ?? ui.catalogLayout);

function setCatalogLayout(layout: CatalogLayoutChoice) {
  if (layout !== "list") ui.setCatalogLayout(layout);
}

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
const kind = ref<CatalogKindFilter | "">("");
const sort = ref<CatalogSortOption>("downloads");
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
const installedNames = computed(() => new Set((props.installedEntries ?? []).map((m) => m.name)));

/** Installed models as catalog-shaped rows for the unified list — the host
 *  chips are the visual "you have this" indicator. */
const installedCatalogEntries = computed<(CatalogEntry & { hostIds?: string[] })[]>(() => {
  const q = props.query.trim().toLowerCase();
  return (props.installedEntries ?? [])
    .filter(
      (m) =>
        !q || m.name.toLowerCase().includes(q) || modelDisplayName(m).toLowerCase().includes(q),
    )
    .filter((m) => !family.value || m.family === family.value)
    .map((m) => ({ ...installedModelToEntry(m), hostIds: m.hostIds ?? [] }))
    .filter(
      (entry) =>
        source.value === "all" ||
        (source.value === "hf" && entry.source === "hf") ||
        (source.value === "civitai" && entry.source === "civitai"),
    );
});

/** Preserve installed identity/host state while filling older `/api/models`
 * rows with richer metadata from the matching live catalog result. */
function enrichInstalledEntry(
  installed: CatalogEntry & { hostIds?: string[] },
  live: CatalogEntry | undefined,
): CatalogEntry & { hostIds?: string[] } {
  if (!live) return installed;
  const installedDescription = installed.description?.trim();
  const modality = installed.modality ?? live.modality;
  return {
    ...live,
    ...installed,
    author: installed.author ?? live.author ?? null,
    display_name: installed.display_name ?? live.display_name ?? null,
    kind: live.kind || installed.kind,
    ...(modality ? { modality } : {}),
    nsfw: installed.nsfw || live.nsfw,
    description: installedDescription || live.description || null,
    thumbnail_url: installed.thumbnail_url || live.thumbnail_url || null,
    page_url: installed.page_url || live.page_url || null,
    installed: true,
  };
}

/** Host chip labels for an installed row (host ids → display labels). */
function hostLabelsFor(entry: CatalogEntry & { hostIds?: string[] }): string[] {
  return (entry.hostIds ?? []).map(
    (id) =>
      hosts.all.find((host) => host.id === id)?.label ?? (id === "local" ? "This device" : id),
  );
}

const manifestEntries = computed<CatalogEntry[]>(() => {
  const installed = installedNames.value;
  const q = props.query.trim().toLowerCase();
  if (source.value === "civitai") return [];
  // Manifest rows are all checkpoints — a non-checkpoint kind hides them.
  if (kind.value && kind.value !== "checkpoint") return [];
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
  const byId = new Map<string, CatalogEntry & { hostIds?: string[] }>();
  const liveById = new Map(safeLive.map((entry) => [entry.id, entry]));
  const liveByIdentity = new Map(
    safeLive.flatMap((entry) => {
      const key = catalogIdentityKey(entry);
      return key ? [[key, entry] as const] : [];
    }),
  );
  const enrichedInstalled = installedCatalogEntries.value.map((entry) =>
    enrichInstalledEntry(
      entry,
      liveById.get(entry.id) ??
        (catalogIdentityKey(entry) ? liveByIdentity.get(catalogIdentityKey(entry)!) : undefined),
    ),
  );
  // Installed rows win the dedup — a live-catalog copy of an installed
  // model must not appear untagged next to it. Exact ids win; legacy rows
  // may additionally match by source + non-empty upstream repo/version id.
  // Human titles are deliberately never identity.
  const installedById = new Set(enrichedInstalled.map((entry) => entry.id));
  const installedByIdentity = new Set(
    enrichedInstalled.map(catalogIdentityKey).filter((key): key is string => key != null),
  );
  for (const entry of [
    ...enrichedInstalled,
    ...manifestEntries.value,
    ...safeLive.filter((entry) => {
      const key = catalogIdentityKey(entry);
      return !installedById.has(entry.id) && !(key && installedByIdentity.has(key));
    }),
  ]) {
    if (!byId.has(entry.id)) byId.set(entry.id, entry);
  }
  return [...byId.values()].filter((entry) => !kind.value || entry.kind === kind.value);
});

// What you already have surfaces first (host-tagged); the divider marks
// where "available" begins. The media-type filter is client-side on
// `entry.family` — the server query stays unchanged.
const displayEntries = computed(() =>
  sortInstalledFirst(combinedEntries.value).filter(matchesMediaType),
);

/** Why the grid is empty while entries exist — names the active filter. */
const filteredEmptyMessage = computed(() => {
  const type = props.mediaType ?? "all";
  if (type !== "all" && !combinedEntries.value.some(matchesMediaType)) {
    const noun = type === "video" ? "video" : "image";
    return hasMore.value
      ? `No ${noun} models in these results yet — keep scrolling or show all media types.`
      : `No ${noun} models in these results.`;
  }
  return "Everything here is already installed.";
});

const readyHosts = computed(() =>
  hosts.all.filter((host) => host.status === "ready" && host.baseUrl),
);

function actionHosts(entry: CatalogEntry & { hostIds?: string[] }): HostView[] {
  if (!entry.installed) return readyHosts.value;
  const ownerIds = new Set(entry.hostIds ?? []);
  return readyHosts.value.filter((host) => ownerIds.has(host.id));
}

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
          kind: kind.value || undefined,
          source: source.value === "all" ? undefined : source.value,
          include_nsfw: includeNsfw.value,
          sort: sort.value === "downloads" ? undefined : sort.value,
          page: page.value,
          page_size: PAGE_SIZE,
        },
        forward,
        target,
      );
      if (epoch !== searchEpoch) return;
      entries.value = [...entries.value, ...res.entries];
      // Exhaustion comes from the wire `total`, not page fullness: under
      // source=All the server splits the page budget across sources, so a
      // merged page is legitimately short whenever one source has no rows
      // (e.g. ControlNet, which HF never carries). Older servers without a
      // numeric total fall back to the full-page heuristic.
      hasMore.value =
        typeof res.total === "number"
          ? entries.value.length < res.total
          : res.entries.length ===
            (typeof res.page_size === "number" && res.page_size > 0 ? res.page_size : PAGE_SIZE);
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
  if (loading.value || !hasMore.value) return;
  page.value += 1;
  void runSearch(false);
}

/** End-of-list sentinel: scrolling near it fetches the next page — the
 *  old Load more button read as a dead end and (worse) looked broken
 *  whenever upstream paging returned duplicate rows the dedup swallowed. */
const sentinel = ref<HTMLElement | null>(null);
const sentinelVisible = ref(false);
/** Fetches chained since the last real intersection event — bounded like
 *  the media-filter auto-fetch so a dedup-swallowed page can't loop. */
let chainedFetches = 0;
let observer: IntersectionObserver | null = null;

watch(sentinel, (el) => {
  observer?.disconnect();
  observer = null;
  sentinelVisible.value = false;
  if (!el) return;
  observer = new IntersectionObserver(
    (hits) => {
      for (const hit of hits) sentinelVisible.value = hit.isIntersecting;
      if (sentinelVisible.value) {
        chainedFetches = 0;
        loadMore();
      }
    },
    { rootMargin: "600px 0px" },
  );
  observer.observe(el);
});

// A page that lands without pushing the sentinel out of view (dedup or the
// media filter swallowed its rows) emits no new intersection event, so the
// observer alone would stall. Chain the next fetch while the sentinel is
// still visible, up to the auto-fetch bound.
watch(loading, (isLoading) => {
  if (isLoading || !sentinelVisible.value || !hasMore.value) return;
  if (chainedFetches >= MAX_AUTO_PAGES) return;
  chainedFetches += 1;
  loadMore();
});

onBeforeUnmount(() => {
  observer?.disconnect();
});

async function pullTo(entry: CatalogEntry, host: HostView | null) {
  pulling.value.add(entry.id);
  try {
    const target = host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : undefined;
    // Attach the snapshot-first stream before enqueueing so a cached,
    // near-instant pull still produces a visible terminal event and refresh.
    await downloads.subscribe(host ?? undefined);
    await startCatalogDownload(entry.id, target, host ? host.kind === "remote" : false);
    toasts.push(`Pulling ${entry.display_name ?? entry.name}${host ? ` on ${host.label}` : ""}`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      toasts.push(`${entry.display_name ?? entry.name} is already queued.`);
    } else {
      toasts.push(String(err), "error");
    }
  } finally {
    pulling.value.delete(entry.id);
    pendingEntry.value = null;
  }
}

function pull(entry: CatalogEntry) {
  const candidates = actionHosts(entry);
  if (entry.installed && candidates.length === 0) {
    toasts.push("No online owning host is available to repair this model.", "error");
    return;
  }
  if (candidates.length > 1) {
    pendingEntry.value = entry;
    return;
  }
  void pullTo(entry, candidates[0] ?? null);
}

/** The detail drawer fetches on the same host the catalog list came from. */
const detailTarget = computed(() => catalogTarget());

/**
 * Quantization variants for a manifest model (`base:tag`) — the sibling rows
 * the manifest already describes, so the drawer's variant chips repoint a Pull
 * without changing which runnable model it targets. Live HF/Civitai rows carry
 * no colon base and get no chips (their precedence is decided in the list).
 */
function variantsFor(entry: CatalogEntry): DrawerVariant[] | undefined {
  // Catalog ids (`cv:252914`) contain a colon that is part of the
  // identifier, not a `base:tag` quant split — they have no variants.
  if (isCatalogModelId(entry.id)) return undefined;
  const base = entry.name.split(":")[0]!;
  if (base === entry.name) return undefined;
  const siblings = models.all.filter((model) => model.name.split(":")[0] === base);
  if (siblings.length < 2) return undefined;
  return siblings.map((model) => ({
    id: model.name,
    label: model.name.slice(base.length + 1) || model.name,
    sizeBytes: model.size_gb > 0 ? Math.round(model.size_gb * 1_000_000_000) : null,
  }));
}

const detailVariants = computed(() =>
  detailEntry.value ? variantsFor(detailEntry.value) : undefined,
);

/** Pull (or Repair — same endpoint, missing files only) from the drawer. */
function pullFromDrawer(entry: CatalogEntry) {
  detailEntry.value = null;
  pull(entry);
}

watch([() => props.query, source, family, kind, sort, includeNsfw], () => {
  scheduleSearch();
});

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
      <div class="flex items-center gap-1" data-test="catalog-source-chips">
        <button
          v-for="s in ['all', 'hf', 'civitai'] as const"
          :key="s"
          type="button"
          class="border-edge h-7 rounded-full border px-2.5 text-caption"
          :class="source === s ? 'bg-safelight text-on-accent' : 'text-ink-2 hover:text-ink'"
          :aria-pressed="source === s"
          @click="source = s"
        >
          {{ s === "all" ? "All" : s === "hf" ? "HuggingFace" : "Civitai" }}
        </button>
      </div>

      <div class="flex items-center gap-1" data-test="catalog-kind-chips" aria-label="Model kind">
        <button
          type="button"
          class="border-edge h-7 rounded-full border px-2.5 text-caption"
          :class="kind === '' ? 'bg-safelight text-on-accent' : 'text-ink-2 hover:text-ink'"
          :aria-pressed="kind === ''"
          @click="kind = ''"
        >
          All
        </button>
        <button
          v-for="opt in CATALOG_KIND_OPTIONS"
          :key="opt.value"
          type="button"
          class="border-edge h-7 rounded-full border px-2.5 text-caption"
          :class="kind === opt.value ? 'bg-safelight text-on-accent' : 'text-ink-2 hover:text-ink'"
          :aria-pressed="kind === opt.value"
          @click="kind = opt.value"
        >
          {{ opt.label }}
        </button>
      </div>

      <select
        v-model="family"
        class="border-edge h-7 rounded-control border bg-bath px-1.5 text-caption text-ink"
      >
        <option value="">All families</option>
        <option v-for="f in families" :key="f" :value="f">{{ f }}</option>
      </select>

      <select
        v-model="sort"
        data-test="catalog-sort"
        aria-label="Sort by"
        class="border-edge h-7 rounded-control border bg-bath px-1.5 text-caption text-ink"
      >
        <option v-for="opt in CATALOG_SORT_OPTIONS" :key="opt.value" :value="opt.value">
          {{ opt.label }}
        </option>
      </select>

      <label class="flex items-center gap-1 text-caption text-ink-2">
        <input v-model="includeNsfw" type="checkbox" class="accent-[var(--safelight)]" />
        Include NSFW
      </label>

      <!-- Grid/table toggle — Discover's secondary control (session-persisted
           in the ui store; table is the default). -->
      <CatalogLayoutToggle
        class="ml-auto"
        :model-value="ui.catalogLayout"
        list-value="table"
        list-label="Table"
        @update:model-value="setCatalogLayout"
      />
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

    <!-- Results, installed first. Grid keeps preview cards; the table layout
         is the app-wide model-row shape — clean info, no thumbnails. -->
    <div
      v-else
      :class="
        effectiveLayout === 'grid'
          ? 'grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-2'
          : 'border-edge divide-edge flex flex-col divide-y overflow-hidden rounded-control border bg-bench'
      "
    >
      <template v-for="entry in displayEntries" :key="entry.id">
        <CatalogCard
          v-if="effectiveLayout === 'grid'"
          :entry="entry"
          :pulling="pulling.has(entry.id)"
          :hosts="hostLabelsFor(entry)"
          @pull="pull"
          @open="detailEntry = $event"
        />
        <CatalogTableRow
          v-else
          :entry="entry"
          :pulling="pulling.has(entry.id)"
          :hosts="hostLabelsFor(entry)"
          class="px-3 py-2"
          @pull="pull"
          @open="detailEntry = $event"
        />
      </template>
    </div>

    <div
      v-if="hasMore"
      ref="sentinel"
      data-test="catalog-scroll-sentinel"
      class="flex h-8 items-center justify-center text-caption text-ink-2"
      aria-hidden="true"
    >
      {{ loading ? "Loading…" : "" }}
    </div>

    <DownloadTargetDialog
      v-if="pendingEntry"
      :model-name="pendingEntry.display_name ?? pendingEntry.name"
      :hosts="actionHosts(pendingEntry)"
      :action="pendingEntry.installed ? 'repair' : 'download'"
      @close="pendingEntry = null"
      @select="(host) => pendingEntry && void pullTo(pendingEntry, host)"
    />

    <CatalogDetailDrawer
      v-if="detailEntry"
      :entry="detailEntry"
      :pulling="pulling.has(detailEntry.id)"
      :target="detailTarget.target"
      :forward-credentials="detailTarget.forward"
      :variants="detailVariants"
      @close="detailEntry = null"
      @pull="pullFromDrawer"
    />
  </div>
</template>
