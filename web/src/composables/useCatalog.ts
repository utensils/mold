import { ref, watch } from "vue";
import {
  fetchActiveCatalogRefresh,
  fetchCatalog,
  fetchCatalogEntry,
  fetchCatalogFamilies,
  postCatalogDownload,
  postCatalogRefresh,
  fetchCatalogRefresh,
} from "../api";
import type {
  CatalogEntryWire,
  CatalogFamilyCount,
  CatalogListParams,
  CatalogRefreshStatus,
} from "../types";

const DEBOUNCE_MS = 250;

let singleton: ReturnType<typeof build> | null = null;

function build() {
  const filter = ref<CatalogListParams>({
    page: 1,
    page_size: 48,
    sort: "downloads",
  });
  const entries = ref<CatalogEntryWire[]>([]);
  const families = ref<CatalogFamilyCount[]>([]);
  const loading = ref(false);
  const errorMsg = ref<string | null>(null);
  const detail = ref<CatalogEntryWire | null>(null);
  const refreshStatus = ref<CatalogRefreshStatus | null>(null);

  let debounceHandle: ReturnType<typeof setTimeout> | null = null;

  async function refresh() {
    loading.value = true;
    errorMsg.value = null;
    try {
      const [list, fams] = await Promise.all([
        fetchCatalog(filter.value),
        fetchCatalogFamilies(),
      ]);
      entries.value = list.entries;
      families.value = fams.families;
    } catch (e: unknown) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function setFilter(patch: Partial<CatalogListParams>) {
    filter.value = { ...filter.value, ...patch, page: 1 };
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
    detail.value = await fetchCatalogEntry(id);
  }

  function closeDetail() {
    detail.value = null;
  }

  function canDownload(entry: Pick<CatalogEntryWire, "engine_phase">): boolean {
    return entry.engine_phase <= 2;
  }

  async function startDownload(id: string) {
    return await postCatalogDownload(id);
  }

  async function startRefresh(family?: string) {
    refreshStatus.value = { state: "pending" };
    let id: string;
    try {
      ({ id } = await postCatalogRefresh(family ? { family } : {}));
    } catch (e: unknown) {
      // Most common case here is the server returning 409 with body
      // "a catalog refresh is already in progress" — bubble that into
      // refreshStatus so the TopBar's failed-state chip shows it
      // instead of swallowing the error in console.error.
      const message = e instanceof Error ? e.message : String(e);
      refreshStatus.value = { state: "failed", message };
      throw e;
    }
    pollRefresh(id);
  }

  function pollRefresh(id: string) {
    const tick = async () => {
      try {
        const status = await fetchCatalogRefresh(id);
        refreshStatus.value = status;
        if (status.state === "done" || status.state === "failed") {
          await refresh();
          return;
        }
      } catch {
        // swallow poll errors; retry on next tick
      }
      setTimeout(tick, 1500);
    };
    void tick();
  }

  /// Discover an in-flight scan started by another tab/CLI and attach to
  /// it so the refresh button reflects cross-client state. Safe to call
  /// repeatedly — it short-circuits when we already have a status.
  async function discoverActiveRefresh() {
    const s = refreshStatus.value;
    if (s && (s.state === "pending" || s.state === "running")) return;
    try {
      const active = await fetchActiveCatalogRefresh();
      if (active) {
        refreshStatus.value = active.status;
        if (
          active.status.state === "pending" ||
          active.status.state === "running"
        ) {
          pollRefresh(active.id);
        }
      }
    } catch {
      // Older servers (without GET /api/catalog/refresh) just leave the
      // chip in its default idle state — graceful degrade.
    }
  }

  return {
    filter,
    entries,
    families,
    loading,
    errorMsg,
    detail,
    refreshStatus,
    refresh,
    setFilter,
    openDetail,
    closeDetail,
    canDownload,
    startDownload,
    startRefresh,
    discoverActiveRefresh,
  };
}

export function useCatalog() {
  if (!singleton) singleton = build();
  return singleton;
}
