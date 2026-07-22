import { computed, ref, watch } from "vue";
import {
  deleteModel,
  fetchCatalogEntry,
  fetchCatalogFamilies,
  fetchCatalogSearch,
  fetchModelComponents,
  fetchModels,
  loadModel,
  postCatalogDownload,
  unloadModel,
} from "../api";
import { useDownloads } from "./useDownloads";
import { isStandaloneGenerationModel } from "../lib/modelFilters";
import type {
  CatalogEntryWire,
  CatalogFamilyCount,
  CatalogListParams,
  ModelComponentStatus,
  ModelInfoExtended,
} from "../types";

/** One selectable pull target inside the model detail drawer. Catalog rows
 * resolve to a single variant today; the drawer supports N so future
 * quant/format variant sets slot straight in. */
export interface ModelVariant {
  id: string;
  label: string;
}

/** Unified detail payload the model drawer renders. A live catalog row
 * (Discover) carries its wire entry + pull variants; an installed model
 * (Installed) carries its `/api/models` row + fetched components. */
export type ModelDetail =
  | { kind: "catalog"; entry: CatalogEntryWire; variants: ModelVariant[] }
  | {
      kind: "installed";
      model: ModelInfoExtended;
      components: ModelComponentStatus[];
    };

export type ModelsTab = "installed" | "discover";

/** Single variant a catalog row resolves to. Label leans on the file format
 * since the live search wire exposes one downloadable form per row; a row
 * without one simply offers no variant chips rather than an empty button. */
function catalogVariants(entry: CatalogEntryWire): ModelVariant[] {
  if (!entry?.file_format) return [];
  return [{ id: entry.id, label: entry.file_format }];
}

/** The minimum a row needs to back a detail view. Anything less can only
 * paint an empty panel, so the drawer shows its retryable error instead —
 * that dead panel is exactly what users reported. */
function isRenderableEntry(entry: unknown): entry is CatalogEntryWire {
  if (typeof entry !== "object" || entry === null) return false;
  const e = entry as Partial<CatalogEntryWire>;
  return (
    typeof e.id === "string" &&
    typeof e.name === "string" &&
    typeof e.family === "string"
  );
}

const DETAIL_ERROR_MESSAGE = "Couldn't load this model's details.";

const DEBOUNCE_MS = 250;
const PAGE_SIZE = 48;

// User-facing filter shape. Pagination (`page`, `page_size`) is internal —
// keeping it out of the watched ref means `loadMore`'s page bumps don't
// retrigger the deep watcher and stomp on the appended entries.
type CatalogFilter = Omit<CatalogListParams, "page" | "page_size"> & {
  /** Local-only because the server has no modality query parameter. */
  modality?: "image" | "video";
};

let singleton: ReturnType<typeof build> | null = null;

function build() {
  // `include_nsfw` is spelled out because the server defaults a missing one
  // to true — leaving it unset put the topbar's unchecked box over an
  // NSFW-inclusive result set.
  const filter = ref<CatalogFilter>({ sort: "downloads", include_nsfw: false });
  const page = ref(1);
  const entries = ref<CatalogEntryWire[]>([]);
  const total = ref<number | null>(null);
  const families = ref<CatalogFamilyCount[]>([]);
  const loading = ref(false);
  const loadingMore = ref(false);
  const errorMsg = ref<string | null>(null);
  const detail = ref<ModelDetail | null>(null);
  // Detail drawer non-content states (G4: the drawer is never blank). While a
  // non-cached row's `/api/catalog/:id` fetch is in flight the drawer shows a
  // loading state keyed on that id; a failure shows a retryable error instead
  // of silently closing.
  const detailLoadingId = ref<string | null>(null);
  const detailError = ref<{ id: string; message: string } | null>(null);

  // ── Installed models (the Installed tab) ──────────────────────────────
  const tab = ref<ModelsTab>("installed");
  const installed = ref<ModelInfoExtended[]>([]);
  const installedLoading = ref(false);
  const installedError = ref<string | null>(null);
  // The default tab follows whether anything is installed — but only until
  // the user picks a tab themselves, after which their choice is sticky.
  let tabTouched = false;

  const hasMore = computed(
    () => total.value !== null && entries.value.length < total.value,
  );

  // `modality` is the one UI filter the server has no parameter for. Every
  // entry carries its own derived modality, so it is applied locally and
  // deliberately stripped from requests by `searchParams`.
  const visibleEntries = computed(() => {
    const m = filter.value.modality;
    if (!m) return entries.value;
    return entries.value.filter((e) => e.modality === m);
  });

  /** Rows the grid actually shows. The server total describes the unfiltered
   * feed, so it overstates the grid whenever a client-side filter is on. */
  const resultCount = computed(() => {
    if (filter.value.modality) return visibleEntries.value.length;
    if (total.value !== null) return total.value;
    return entries.value.length;
  });

  // Bounds `autoFill` so a feed that never yields a matching row can't spin
  // through every page — the grid settles on its empty state instead.
  const AUTO_FILL_MAX_PAGES = 10;

  let debounceHandle: ReturnType<typeof setTimeout> | null = null;

  function searchParams(requestedPage: number): CatalogListParams {
    const { modality: _localModality, ...serverFilters } = filter.value;
    return { ...serverFilters, page: requestedPage, page_size: PAGE_SIZE };
  }

  /** Fetch the page after the current one. Returns false when there was
   * nothing left to fetch, so callers can tell progress from exhaustion. */
  async function fetchNextPage(): Promise<boolean> {
    if (!hasMore.value) return false;
    const next = page.value + 1;
    const list = await fetchCatalogSearch(searchParams(next));
    entries.value = [...entries.value, ...list.entries];
    page.value = next;
    if (typeof list.total === "number") total.value = list.total;
    return list.entries.length > 0;
  }

  /** Client-side filtering can hide a whole page. Keep pulling pages while
   * the visible set is empty so the user never sees "No models found." over
   * a feed the server still has rows for. Runs with `loading`/`loadingMore`
   * still set, so the grid shows its loading state rather than flashing an
   * empty one. */
  async function autoFill() {
    if (!filter.value.modality) return;
    for (let i = 0; i < AUTO_FILL_MAX_PAGES; i += 1) {
      if (visibleEntries.value.length > 0 || !hasMore.value) return;
      if (!(await fetchNextPage())) return;
    }
  }

  async function refresh() {
    loading.value = true;
    errorMsg.value = null;
    try {
      const [list, fams] = await Promise.all([
        fetchCatalogSearch(searchParams(1)),
        fetchCatalogFamilies(),
      ]);
      entries.value = list.entries;
      total.value = typeof list.total === "number" ? list.total : null;
      page.value = 1;
      families.value = fams.families;
      await autoFill();
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
      await fetchNextPage();
      await autoFill();
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
    detailError.value = null;
    const cached = entries.value.find((e) => e?.id === id);
    if (cached) {
      // Everything from here to `detail.value = …` runs on wire data. A throw
      // would escape the async function (callers use `void openDetail(id)`)
      // and leave the drawer open on nothing — surface a retryable error.
      try {
        if (!isRenderableEntry(cached)) throw new Error("malformed entry");
        const variants = catalogVariants(cached);
        detailLoadingId.value = null;
        detail.value = { kind: "catalog", entry: cached, variants };
      } catch {
        detail.value = null;
        detailLoadingId.value = null;
        detailError.value = { id, message: DETAIL_ERROR_MESSAGE };
      }
      return;
    }
    // Not on the current page — open the drawer in a loading state while the
    // DB-backed `/api/catalog/:id` fetch resolves, rather than opening blank.
    detail.value = null;
    detailLoadingId.value = id;
    try {
      const entry = await fetchCatalogEntry(id);
      if (detailLoadingId.value !== id) return; // superseded by another open
      if (!isRenderableEntry(entry)) throw new Error("malformed entry");
      const variants = catalogVariants(entry);
      detailLoadingId.value = null;
      detail.value = { kind: "catalog", entry, variants };
    } catch {
      if (detailLoadingId.value !== id) return;
      detail.value = null;
      detailLoadingId.value = null;
      // 404 (live row not in DB) / 502 (upstream) / transient network /
      // unusable payload — show a retryable error in the drawer instead of a
      // blank panel or a silent close the user can't tell apart from a
      // mis-click.
      detailError.value = { id, message: DETAIL_ERROR_MESSAGE };
    }
  }

  /** Retry the last errored detail open. */
  function retryDetail(id: string) {
    void openDetail(id);
  }

  /** Open the drawer for an installed model. Components load asynchronously
   * from `/api/models/:model/components`; the drawer renders immediately and
   * fills the Components chips once the fetch resolves. */
  async function openInstalledDetail(model: ModelInfoExtended) {
    detail.value = { kind: "installed", model, components: [] };
    try {
      const resp = await fetchModelComponents(model.name);
      if (
        detail.value?.kind === "installed" &&
        detail.value.model.name === model.name
      ) {
        detail.value = { ...detail.value, components: resp.components };
      }
    } catch {
      // Leave components empty — the tiles + name still render.
    }
  }

  function closeDetail() {
    detail.value = null;
    detailLoadingId.value = null;
    detailError.value = null;
  }

  function setTab(next: ModelsTab) {
    tabTouched = true;
    tab.value = next;
  }

  async function refreshInstalled() {
    installedLoading.value = true;
    installedError.value = null;
    try {
      // Only models actually on disk are "installed". /api/models returns
      // the whole manifest (97 rows here, 2 downloaded), so an unfiltered
      // assignment renders the entire catalog as if it were installed — and
      // the tab heuristic below then always lands on a shelf of models the
      // user does not have. Host detail already filters this way.
      installed.value = (await fetchModels()).filter(
        (m) => m.downloaded && isStandaloneGenerationModel(m),
      );
      // First load with no explicit user choice: land on Installed when the
      // user has models, otherwise open straight into Discover.
      if (!tabTouched) {
        tab.value = installed.value.length > 0 ? "installed" : "discover";
      }
    } catch (e: unknown) {
      installedError.value = e instanceof Error ? e.message : String(e);
    } finally {
      installedLoading.value = false;
    }
  }

  // Re-sync the open installed detail from the freshly fetched list so the
  // drawer's loaded state / footprint reflect the action just taken.
  function syncInstalledDetail(name: string) {
    const updated = installed.value.find((m) => m.name === name);
    if (
      updated &&
      detail.value?.kind === "installed" &&
      detail.value.model.name === name
    ) {
      detail.value = { ...detail.value, model: updated };
    }
  }

  async function loadInstalled(name: string) {
    await loadModel(name);
    await refreshInstalled();
    syncInstalledDetail(name);
  }

  async function unloadInstalled(name: string) {
    await unloadModel(name);
    await refreshInstalled();
    syncInstalledDetail(name);
  }

  async function deleteInstalled(name: string) {
    const result = await deleteModel(name);
    // The model's files are gone — drop the drawer and refresh the shelf.
    if (
      detail.value?.kind === "installed" &&
      detail.value.model.name === name
    ) {
      detail.value = null;
    }
    await refreshInstalled();
    return result;
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
    visibleEntries,
    resultCount,
    total,
    hasMore,
    families,
    loading,
    loadingMore,
    errorMsg,
    detail,
    detailLoadingId,
    detailError,
    refresh,
    loadMore,
    setFilter,
    openDetail,
    retryDetail,
    closeDetail,
    canDownload,
    startDownload,
    // Installed tab + unified detail actions
    tab,
    setTab,
    installed,
    installedLoading,
    installedError,
    refreshInstalled,
    openInstalledDetail,
    loadInstalled,
    unloadInstalled,
    deleteInstalled,
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
