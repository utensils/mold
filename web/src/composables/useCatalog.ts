import { computed, ref, watch } from "vue";
import {
  fetchCatalogEntry,
  fetchCatalogFamilies,
  fetchCatalogSearch,
  postCatalogDownload,
} from "../api";
import { useDownloads } from "./useDownloads";
import type {
  CatalogEntryWire,
  CatalogFamilyCount,
  CatalogListParams,
} from "../types";

const DEBOUNCE_MS = 250;
const PAGE_SIZE = 48;

// User-facing filter shape. Pagination (`page`, `page_size`) is internal —
// keeping it out of the watched ref means `loadMore`'s page bumps don't
// retrigger the deep watcher and stomp on the appended entries.
type CatalogFilter = Omit<CatalogListParams, "page" | "page_size">;

let singleton: ReturnType<typeof build> | null = null;

function build() {
  const filter = ref<CatalogFilter>({ sort: "downloads" });
  const page = ref(1);
  const entries = ref<CatalogEntryWire[]>([]);
  const total = ref<number | null>(null);
  const families = ref<CatalogFamilyCount[]>([]);
  const loading = ref(false);
  const loadingMore = ref(false);
  const errorMsg = ref<string | null>(null);
  const detail = ref<CatalogEntryWire | null>(null);

  // hasMore: server-reported total wins; if absent (older server), fall
  // back to "the last page came back full" so we keep fetching until a
  // short page signals end-of-stream.
  const hasMore = computed(() => {
    if (total.value !== null) return entries.value.length < total.value;
    if (entries.value.length === 0) return false;
    return entries.value.length % PAGE_SIZE === 0;
  });

  let debounceHandle: ReturnType<typeof setTimeout> | null = null;

  async function refresh() {
    loading.value = true;
    errorMsg.value = null;
    try {
      const [list, fams] = await Promise.all([
        fetchCatalogSearch({ ...filter.value, page: 1, page_size: PAGE_SIZE }),
        fetchCatalogFamilies(),
      ]);
      entries.value = list.entries;
      total.value = typeof list.total === "number" ? list.total : null;
      page.value = 1;
      families.value = fams.families;
    } catch (e: unknown) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  async function loadMore() {
    if (loading.value || loadingMore.value) return;
    if (!hasMore.value) return;
    loadingMore.value = true;
    try {
      const next = page.value + 1;
      const list = await fetchCatalogSearch({
        ...filter.value,
        page: next,
        page_size: PAGE_SIZE,
      });
      entries.value = [...entries.value, ...list.entries];
      page.value = next;
      if (typeof list.total === "number") total.value = list.total;
    } catch (e: unknown) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    } finally {
      loadingMore.value = false;
    }
  }

  function setFilter(patch: Partial<CatalogFilter>) {
    filter.value = { ...filter.value, ...patch };
  }

  watch(
    filter,
    () => {
      if (debounceHandle) clearTimeout(debounceHandle);
      debounceHandle = setTimeout(() => {
        void refresh();
      }, DEBOUNCE_MS);
    },
    { deep: true },
  );

  async function openDetail(id: string) {
    // Live-search rows aren't in the DB-backed `/api/catalog/:id` endpoint,
    // so re-fetching would 404 and a `void cat.openDetail(...)` caller would
    // silently leave the drawer unmounted. The list response already carries
    // every field the drawer needs (name, family, engine_phase, installed,
    // download_recipe, …), so prefer the in-memory entry and only fall back
    // to the API when the user opens an id that isn't in the current page —
    // e.g. a future deep-link path.
    const cached = entries.value.find((e) => e.id === id);
    if (cached) {
      detail.value = cached;
      return;
    }
    try {
      detail.value = await fetchCatalogEntry(id);
    } catch {
      // 404 (live row not in DB, deep link to a stale id) or transient
      // network — keep `detail` null so the drawer stays closed instead of
      // wedging on stale data.
      detail.value = null;
    }
  }

  function closeDetail() {
    detail.value = null;
  }

  function canDownload(entry: Pick<CatalogEntryWire, "engine_phase">): boolean {
    return entry.engine_phase <= 5;
  }

  async function startDownload(id: string) {
    const result = await postCatalogDownload(id);
    // The catalog endpoint enqueues 1–N download jobs (primary + companions).
    // Force the downloads drawer to repaint immediately rather than waiting
    // for the SSE `enqueued` events, which can lag when the page is in a
    // background tab or right after an SSE reconnect.
    void useDownloads().refresh();
    return result;
  }

  return {
    filter,
    page,
    entries,
    total,
    hasMore,
    families,
    loading,
    loadingMore,
    errorMsg,
    detail,
    refresh,
    loadMore,
    setFilter,
    openDetail,
    closeDetail,
    canDownload,
    startDownload,
  };
}

export function useCatalog() {
  if (!singleton) singleton = build();
  return singleton;
}

/**
 * Test-only: drop the module-level singleton so each test starts with a
 * fresh state machine. Production code never touches this.
 */
export function __resetCatalogSingletonForTests() {
  singleton = null;
}
