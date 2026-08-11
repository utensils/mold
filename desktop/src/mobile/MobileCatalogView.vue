<script setup lang="ts">
import {
  computed,
  nextTick,
  onActivated,
  onBeforeUnmount,
  onDeactivated,
  onMounted,
  ref,
  watch,
} from "vue";
import { apiFetchTo, ApiError, type ApiTarget } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import { fetchCatalogDetail, fetchCatalogFamilies, searchCatalog } from "../lib/api/catalog";
import {
  catalogFetchCaption,
  catalogIdentityKey,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  sortInstalledFirst,
} from "../lib/catalog";
import {
  buildDownloadContents,
  canDownloadEntry,
  downloadContentsTotalBytes,
  installedModelToEntry,
  mergeCatalogSummaryDetail,
} from "../lib/catalogDetail";
import {
  CATALOG_KIND_OPTIONS,
  CATALOG_SORT_OPTIONS,
  type CatalogKindFilter,
  type CatalogSortOption,
} from "../lib/catalogFilters";
import { familyLabel } from "@studio/lib/modelFamily";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import MinimaxH3InventoryPanel from "@studio/components/MinimaxH3InventoryPanel.vue";
import { modelKindLabel, modelKindValue, modelWeightsLabel } from "@studio/lib/modelMetadata";
import {
  planModelInstall,
  type ModelInstallAction,
  type ModelInstallPlan,
} from "@studio/lib/modelInstallTargets";
import { catalogThumbnailUrl } from "../lib/catalogThumbnails";
import { isVideoFamily } from "../lib/capabilities";
import { useInfiniteScrollSentinel } from "../lib/useInfiniteScrollSentinel";
import { formatCount, formatGB, percent } from "../lib/format";
import {
  isCatalogModelId,
  isUtilityModel,
  mergeModelPresentationMetadata,
  modelDisplayNameForId,
} from "../lib/models";
import { fetchModelComponents, loadModel, removeModel, unloadModel } from "../lib/api/models";
import type {
  CatalogEntry,
  DownloadJob,
  ModelComponentStatus,
  ModelEntry,
  ServerCapabilities,
} from "../lib/api/types";
import { mobileHostTarget, type MobileHost } from "./hosts";
import {
  useMobileDownloadsStore,
  type MobileDownloadEventContext,
  type MobilePullStatus,
} from "./mobileDownloads";

type MediaType = "all" | "image" | "video";
type CatalogSource = "all" | "hf" | "civitai" | "installed";

type MobileCatalogEntry = CatalogEntry & {
  hostIds?: string[];
  hostLabels?: string[];
};

interface DownloadRow {
  host: MobileHost;
  job: DownloadJob;
}

/** A deep link from another surface (today: Create's sequence empty state).
 *  `token` re-fires the intent even when the KeepAlive-cached view is asked
 *  for the same filters twice. */
export interface CatalogFilterIntent {
  mediaType: MediaType;
  kind: CatalogKindFilter | "";
  token: number;
}

const props = withDefaults(
  defineProps<{
    hosts: MobileHost[];
    selectedHostId: string;
    filterIntent?: CatalogFilterIntent | null;
  }>(),
  { filterIntent: null },
);

const emit = defineEmits<{
  (event: "select-host", hostId: string): void;
  (event: "models-changed", hostId: string): void;
}>();

const PAGE_SIZE = 24;
const MAX_AUTO_PAGES = 5;
const DOWNLOAD_CONSUMER_ID = "mobile-catalog";
const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

const query = ref("");
const mediaType = ref<MediaType>("all");
const source = ref<CatalogSource>("all");
// Remembers the last Discover sub-source so toggling Installed → Discover
// restores the chosen HuggingFace/Civitai filter instead of resetting to All.
const lastDiscoverSource = ref<CatalogSource>("all");
const family = ref("");
const kind = ref<CatalogKindFilter | "">("");
const sort = ref<CatalogSortOption>("downloads");
const includeNsfw = ref(false);
const families = ref<string[]>([]);
/** Families seen in any catalog page or host inventory this session. Backs the
 *  Family filter whenever the taxonomy endpoint could not be read (an offline
 *  host, an API key still coming out of the Keychain). Deriving the list from
 *  the rows currently on screen would collapse it to the family already
 *  filtered on, stranding the picker on that one choice. */
const observedFamilies = ref<string[]>([]);
const entries = ref<CatalogEntry[]>([]);
const page = ref(1);
const hasMore = ref(false);
const loading = ref(false);
const error = ref("");
const announcement = ref("");
const announcementIsError = ref(false);
const modelsByHost = ref<Record<string, ModelEntry[]>>({});
const capabilitiesByHost = ref<Record<string, ServerCapabilities>>({});
const mobileDownloads = useMobileDownloadsStore();

const detailEntry = ref<MobileCatalogEntry | null>(null);
const detail = ref<CatalogEntry | null>(null);
const detailLoading = ref(false);
const detailComponents = ref<ModelComponentStatus[]>([]);
const detailComponentsLoading = ref(false);
const detailBusy = ref("");
const detailRemoveConfirm = ref(false);
const catalogRoot = ref<HTMLElement | null>(null);
const detailDialog = ref<HTMLElement | null>(null);
const detailCloseButton = ref<HTMLButtonElement | null>(null);
const targetEntry = ref<MobileCatalogEntry | null>(null);
const targetDialog = ref<HTMLElement | null>(null);
const targetCloseButton = ref<HTMLButtonElement | null>(null);
let restoreFocus: HTMLElement | null = null;
let targetRestoreFocus: HTMLElement | null = null;
let inertBackground: HTMLElement | null = null;

let mounted = false;
let interactionListenersActive = false;
let searchEpoch = 0;
let familyEpoch = 0;
let modelsEpoch = 0;
let detailEpoch = 0;
let debounce: ReturnType<typeof setTimeout> | null = null;

const selectedHost = computed(
  () => props.hosts.find((host) => host.id === props.selectedHostId) ?? null,
);

const selectedTarget = computed<ApiTarget | null>(() => {
  const host = selectedHost.value;
  return host ? mobileHostTarget(host) : null;
});

const h3Hosts = computed(() =>
  props.hosts.map((host) => ({
    id: host.id,
    label: host.name,
    capabilities: capabilitiesByHost.value[host.id],
  })),
);

/** The selected host remains usable while its latest reachability probe is pending. */
const downloadHosts = computed(() =>
  props.hosts.filter((host) => host.online || host.id === props.selectedHostId),
);

/** Machines known to hold this row on disk. A merged installed row carries
 * every owner in `hostIds`; a live catalog row's `installed` flag is only the
 * browsed host's answer, so that host stands in as the sole known owner. */
function ownerHostIds(entry: MobileCatalogEntry): string[] {
  const ids = entry.hostIds ?? [];
  if (ids.length > 0) return ids;
  return entry.installed ? [props.selectedHostId] : [];
}

/** A row is a merge of every connected machine, so "installed" means installed
 * SOMEWHERE. An install therefore stays on offer until every reachable machine
 * owns the model, at which point the only remaining action is a repair. */
function installPlan(entry: MobileCatalogEntry): ModelInstallPlan<MobileHost> {
  return planModelInstall(downloadHosts.value, ownerHostIds(entry));
}

const targetPlan = computed<ModelInstallPlan<MobileHost> | null>(() => {
  const entry = targetEntry.value;
  return entry ? installPlan(entry) : null;
});

/** Install and repair targets in the same list — the sheet copy covers both. */
const targetPlanMixed = computed(
  () =>
    (targetPlan.value?.canInstall ?? false) &&
    (targetPlan.value?.targets.some((target) => target.action === "repair") ?? false),
);

function targetActionLabel(action: ModelInstallAction): string {
  return action === "repair" ? "Repair · already installed" : "Install";
}

const installedEntries = computed<MobileCatalogEntry[]>(() => {
  const byName = new Map<string, { model: ModelEntry; hostIds: string[]; hostLabels: string[] }>();
  for (const host of props.hosts) {
    for (const model of modelsByHost.value[host.id] ?? []) {
      if (!model.downloaded) continue;
      const found = byName.get(model.name);
      if (found) {
        found.model = mergeModelPresentationMetadata(found.model, model);
        found.hostIds.push(host.id);
        found.hostLabels.push(host.name);
      } else {
        byName.set(model.name, {
          model,
          hostIds: [host.id],
          hostLabels: [host.name],
        });
      }
    }
  }
  return [...byName.values()].map(({ model, hostIds, hostLabels }) => ({
    ...installedModelToEntry(model),
    hostIds,
    hostLabels,
  }));
});

function recordFamilies(values: Iterable<string | null | undefined>): void {
  const seen = new Set(observedFamilies.value);
  const before = seen.size;
  for (const value of values) {
    const name = value?.trim();
    if (name) seen.add(name);
  }
  if (seen.size !== before) observedFamilies.value = [...seen].sort((a, b) => a.localeCompare(b));
}

/** The host taxonomy when it could be read, the observed families otherwise.
 *  The active family always stays listed so a reload can never silently blank
 *  the control (and with it the filter it is still applying). */
const familyOptions = computed(() => {
  const options = families.value.length > 0 ? [...families.value] : [...observedFamilies.value];
  if (family.value && !options.includes(family.value)) options.push(family.value);
  return options;
});

const installedNames = computed(() => new Set(installedEntries.value.map((entry) => entry.name)));
const modelLabel = (name: string) => modelDisplayNameForId(name, installedEntries.value);

function manifestModelToEntry(model: ModelEntry): MobileCatalogEntry {
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
}

/** Curated `/api/models` variants are safe pull targets and must beat an aggregate HF repo row. */
const manifestEntries = computed<MobileCatalogEntry[]>(() => {
  // Manifest rows are all checkpoints — a non-checkpoint kind hides them.
  if (kind.value && kind.value !== "checkpoint") return [];
  const q = query.value.trim().toLowerCase();
  return (modelsByHost.value[props.selectedHostId] ?? [])
    .filter((model) => !model.downloaded && !isUtilityModel(model))
    .filter((model) => !installedNames.value.has(model.name))
    .filter((model) => !q || model.name.toLowerCase().includes(q))
    .filter((model) => !family.value || model.family === family.value)
    .map(manifestModelToEntry);
});

function sourceMatches(entry: CatalogEntry): boolean {
  if (source.value === "all" || source.value === "installed") return true;
  return entry.source === source.value;
}

function mediaMatches(entry: CatalogEntry): boolean {
  if (mediaType.value === "all") return true;
  if (isUtilityModel({ family: entry.family } as ModelEntry)) return false;
  return isVideoFamily(entry.family) === (mediaType.value === "video");
}

const safeLiveEntries = computed(() => {
  const knownRepos = new Set(
    (modelsByHost.value[props.selectedHostId] ?? [])
      .map((model) => model.hf_repo)
      .filter((repo): repo is string => Boolean(repo)),
  );
  return entries.value.filter(
    (entry) =>
      !(
        entry.source === "hf" &&
        entry.kind === "checkpoint" &&
        (entry.bundling === "separated" ||
          Boolean(entry.source_id && knownRepos.has(entry.source_id)))
      ),
  );
});

/** Preserve installed identity/host state while a current live row fills
 * presentation fields that an older `/api/models` response omitted. */
function enrichInstalledEntry(
  installed: MobileCatalogEntry,
  live: CatalogEntry | undefined,
): MobileCatalogEntry {
  if (!live) return installed;
  const installedDescription = installed.description?.trim();
  const modality = installed.modality?.trim() ? installed.modality : live.modality;
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

const enrichedInstalledEntries = computed(() => {
  const liveById = new Map(safeLiveEntries.value.map((entry) => [entry.id, entry]));
  const liveByIdentity = new Map(
    safeLiveEntries.value.flatMap((entry) => {
      const key = catalogIdentityKey(entry);
      return key ? [[key, entry] as const] : [];
    }),
  );
  return installedEntries.value.map((installed) =>
    enrichInstalledEntry(
      installed,
      liveById.get(installed.id) ??
        (catalogIdentityKey(installed)
          ? liveByIdentity.get(catalogIdentityKey(installed)!)
          : undefined),
    ),
  );
});

const filteredInstalled = computed(() => {
  const q = query.value.trim().toLowerCase();
  return enrichedInstalledEntries.value
    .filter(
      (entry) =>
        !q || entry.name.toLowerCase().includes(q) || entryTitle(entry).toLowerCase().includes(q),
    )
    .filter((entry) => !family.value || entry.family === family.value)
    .filter((entry) => !kind.value || entry.kind === kind.value)
    .filter(sourceMatches)
    .filter(mediaMatches);
});

const combinedEntries = computed<MobileCatalogEntry[]>(() => {
  if (source.value === "installed") return filteredInstalled.value;

  const safeLive = safeLiveEntries.value;
  const installedById = new Set(filteredInstalled.value.map((entry) => entry.id));
  const installedByIdentity = new Set(
    filteredInstalled.value.map(catalogIdentityKey).filter((key): key is string => key != null),
  );
  const byId = new Map<string, MobileCatalogEntry>();
  for (const entry of [
    ...filteredInstalled.value,
    ...manifestEntries.value.filter(sourceMatches).filter(mediaMatches),
    ...safeLive
      .filter((entry) => {
        const key = catalogIdentityKey(entry);
        return !installedById.has(entry.id) && !(key && installedByIdentity.has(key));
      })
      .filter(mediaMatches),
  ]) {
    if (!byId.has(entry.id)) byId.set(entry.id, entry);
  }
  return sortInstalledFirst([...byId.values()]);
});

const downloadRows = computed<DownloadRow[]>(() => {
  const seen = new Set<string>();
  return props.hosts.flatMap((host) => {
    const state = mobileDownloads.downloadsByHost[host.id];
    if (!state) return [];
    return [...state.activeJobs, ...state.queued]
      .filter((job) => {
        const key = `${host.id}:${job.id}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .map((job) => ({ host, job }));
  });
});

const mergedDetail = computed<CatalogEntry | null>(() => {
  const summary = detailEntry.value;
  if (!summary) return null;
  return detail.value ? mergeCatalogSummaryDetail(summary, detail.value) : summary;
});

const detailDownloadItems = computed(() =>
  mergedDetail.value ? buildDownloadContents(mergedDetail.value) : [],
);
const detailDownloadTotal = computed(() => downloadContentsTotalBytes(detailDownloadItems.value));
const detailSize = computed(() =>
  mergedDetail.value ? catalogSizeInfo(mergedDetail.value) : null,
);
function detailFootprintLabel(): string {
  const info = detailSize.value;
  const bytes = info?.fetchBytes ?? info?.weightsBytes ?? null;
  return bytes != null ? formatGB(bytes) : "—";
}
const detailModel = computed(() => {
  const entry = detailEntry.value;
  const host = entry ? owningHost(entry) : null;
  return host
    ? ((modelsByHost.value[host.id] ?? []).find(
        (model) => model.name === entry?.name || model.name === entry?.id,
      ) ?? null)
    : null;
});

/** Runnable manifest siblings (`base:tag`) become exact pull targets. */
const detailVariants = computed(() => {
  const entry = detailEntry.value;
  if (!entry) return [];
  // Catalog ids (`cv:252914`) contain a colon that is part of the
  // identifier, not a `base:tag` quant split — they have no variants.
  if (isCatalogModelId(entry.id)) return [];
  const base = entry.name.split(":")[0]!;
  if (base === entry.name) return [];
  const siblings = (modelsByHost.value[props.selectedHostId] ?? [])
    .filter((model) => !isUtilityModel(model) && model.name.split(":")[0] === base)
    .map(
      (model) =>
        installedEntries.value.find((installed) => installed.name === model.name) ??
        manifestModelToEntry(model),
    );
  return siblings.length > 1 ? siblings : [];
});

/** Human title for announce/toast text; ids stay `entry.name` in API calls. */
function entryTitle(entry: MobileCatalogEntry): string {
  return entry.display_name ?? entry.name;
}

function entryAccessibilityLabel(entry: MobileCatalogEntry): string {
  const labels = [
    entryTitle(entry),
    modelKindLabel(modelKindValue(entry)),
    ...(entry.nsfw ? ["18+ NSFW"] : []),
    "view details",
  ];
  return labels.join(" — ");
}

function catalogSourceLabel(source: string): string {
  switch (source.trim().toLowerCase()) {
    case "hf":
      return "Hugging Face";
    case "civitai":
      return "Civitai";
    case "local":
      return "Local";
    default:
      return source;
  }
}

function variantLabel(entry: MobileCatalogEntry): string {
  if (isCatalogModelId(entry.id)) return entryTitle(entry);
  const base = entry.name.split(":")[0]!;
  return entry.name.slice(base.length + 1) || entry.name;
}

function selectDetailVariant(entry: MobileCatalogEntry): void {
  if (entry.id === detailEntry.value?.id) return;
  void openDetail(entry);
}

function announce(message: string, isError = false): void {
  announcement.value = message;
  announcementIsError.value = isError;
}

function errorMessage(cause: unknown, hostName?: string): string {
  return describeTransportError(cause, hostName);
}

function setInert(element: HTMLElement | null, inert: boolean): void {
  if (!element) return;
  element.inert = inert;
  if (inert) element.setAttribute("inert", "");
  else element.removeAttribute("inert");
}

function updateModalInert(): void {
  const overlayOpen = Boolean(detailEntry.value || targetEntry.value);
  const nextBackground = overlayOpen
    ? (catalogRoot.value?.closest<HTMLElement>(".mobile-shell") ?? catalogRoot.value)
    : null;
  if (inertBackground && inertBackground !== nextBackground) setInert(inertBackground, false);
  if (nextBackground) setInert(nextBackground, true);
  inertBackground = nextBackground;
  setInert(detailDialog.value, Boolean(targetEntry.value));
}

function clearModalInert(): void {
  setInert(detailDialog.value, false);
  setInert(inertBackground, false);
  inertBackground = null;
}

function pullStatus(entry: MobileCatalogEntry, hostId?: string): MobilePullStatus | null {
  if (hostId) return mobileDownloads.pullStatusForHost(entry, hostId);
  const preferred = entry.installed ? owningHost(entry) : selectedHost.value;
  const orderedHosts = preferred
    ? [preferred, ...downloadHosts.value.filter((host) => host.id !== preferred.id)]
    : downloadHosts.value;
  for (const host of orderedHosts) {
    const status = mobileDownloads.pullStatusForHost(entry, host.id);
    if (status) return status;
  }
  return null;
}

/** In-flight state wins; otherwise the plan decides whether this is still a
 * pull (some machine lacks it) or has degraded to a repair. */
function pullButtonLabel(entry: MobileCatalogEntry): string {
  const status = pullStatus(entry);
  if (status) return status.label;
  return installPlan(entry).label === "Repair"
    ? "Repair"
    : catalogPullLabel(catalogSizeInfo(entry));
}

/** What sending `entry` to `host` does there, for announcements. */
function hostActionLabel(entry: MobileCatalogEntry, host: MobileHost): "Pull" | "Repair" {
  return ownerHostIds(entry).includes(host.id) ? "Repair" : "Pull";
}

function owningHost(entry: MobileCatalogEntry): MobileHost | null {
  const ids = entry.hostIds ?? [];
  const preferred = ids.includes(props.selectedHostId) ? props.selectedHostId : ids[0];
  return props.hosts.find((host) => host.id === preferred) ?? selectedHost.value;
}

function targetForEntry(entry: MobileCatalogEntry): ApiTarget | null {
  const host = entry.installed ? owningHost(entry) : selectedHost.value;
  return host ? mobileHostTarget(host) : null;
}

function scheduleSearch(): void {
  if (debounce) clearTimeout(debounce);
  if (source.value === "installed") {
    ++searchEpoch;
    loading.value = false;
    error.value = "";
    return;
  }
  debounce = setTimeout(() => void runSearch(true), 400);
}

async function runSearch(reset: boolean): Promise<void> {
  const target = selectedTarget.value;
  if (!target || source.value === "installed") return;
  const epoch = ++searchEpoch;
  if (reset) {
    page.value = 1;
    entries.value = [];
  }
  loading.value = true;
  error.value = "";
  try {
    for (let fetched = 0; ; fetched += 1) {
      const response = await searchCatalog(
        {
          q: query.value.trim() || undefined,
          family: family.value || undefined,
          kind: kind.value || undefined,
          source: source.value === "all" ? undefined : source.value,
          include_nsfw: includeNsfw.value,
          sort: sort.value === "downloads" ? undefined : sort.value,
          page: page.value,
          page_size: PAGE_SIZE,
        },
        false,
        target,
      );
      if (epoch !== searchEpoch) return;
      entries.value = [...entries.value, ...response.entries];
      recordFamilies(response.entries.map((row) => row.family));
      // Exhaustion comes from the wire `total`, not page fullness: under
      // source=All the server splits the page budget across sources, so a
      // merged page is legitimately short whenever one source has no rows
      // (e.g. ControlNet, which HF never carries). Older servers without a
      // numeric total fall back to the full-page heuristic.
      hasMore.value =
        typeof response.total === "number"
          ? entries.value.length < response.total
          : response.entries.length ===
            (typeof response.page_size === "number" && response.page_size > 0
              ? response.page_size
              : PAGE_SIZE);
      const filtered = combinedEntries.value.some(mediaMatches);
      if (mediaType.value === "all" || filtered || !hasMore.value || fetched + 1 >= MAX_AUTO_PAGES)
        break;
      page.value += 1;
    }
  } catch (cause) {
    if (epoch !== searchEpoch) return;
    error.value = errorMessage(cause, selectedHost.value?.name);
    hasMore.value = false;
  } finally {
    if (epoch === searchEpoch) loading.value = false;
  }
}

function loadMore(): void {
  if (loading.value || !hasMore.value) return;
  page.value += 1;
  void runSearch(false);
}

const sentinel = ref<HTMLElement | null>(null);
useInfiniteScrollSentinel(sentinel, loading, hasMore, loadMore, MAX_AUTO_PAGES);

async function loadFamilies(): Promise<void> {
  const target = selectedTarget.value;
  const epoch = ++familyEpoch;
  if (!target) {
    families.value = [];
    return;
  }
  try {
    // The iPhone shell has no desktop `secret_get` command. Remote catalog
    // hosts use their own configured HF/Civitai credentials; only the Mold
    // host API key from `target` crosses the wire.
    const result = await fetchCatalogFamilies(false, target);
    if (epoch === familyEpoch) families.value = result;
  } catch {
    // Keep the last taxonomy that did load — a failed reload must never empty
    // the filter. `familyOptions` covers the never-loaded case.
  }
}

async function refreshModels(): Promise<void> {
  const epoch = ++modelsEpoch;
  const next: Record<string, ModelEntry[]> = {};
  const nextCapabilities: Record<string, ServerCapabilities> = {};
  await Promise.all(
    downloadHosts.value.map(async (host) => {
      const target = mobileHostTarget(host);
      const [modelResult, capabilityResult] = await Promise.allSettled([
        apiFetchTo(target, "/api/models").then(
          (response) => response.json() as Promise<ModelEntry[]>,
        ),
        apiFetchTo(target, "/api/capabilities").then(
          (response) => response.json() as Promise<ServerCapabilities>,
        ),
      ]);
      if (epoch !== modelsEpoch) return;
      if (modelResult.status === "fulfilled") {
        next[host.id] = modelResult.value;
        recordFamilies(modelResult.value.map((model) => model.family));
      }
      if (capabilityResult.status === "fulfilled") {
        nextCapabilities[host.id] = capabilityResult.value;
      }
    }),
  );
  if (epoch === modelsEpoch) {
    modelsByHost.value = next;
    capabilitiesByHost.value = nextCapabilities;
  }
}

function handleDownloadEvent({
  host,
  event,
  hadSnapshot,
  newlySettled,
}: MobileDownloadEventContext): void {
  if (!mounted) return;
  if (event.type === "job_done") {
    announce(`${event.model} is ready on ${host.name}.`);
    emit("models-changed", host.id);
    void refreshModels();
    if (host.id === props.selectedHostId) void runSearch(true);
  } else if (event.type === "job_failed") {
    announce(`Download failed on ${host.name}: ${event.error}`, true);
  } else if (event.type === "snapshot") {
    const settled = event.listing.history.filter(
      (job) => job.status === "completed" || job.status === "failed",
    );
    // A completed pull may have happened while the iPhone was suspended, so
    // refresh model views even on the first reconciliation snapshot. Announce
    // terminal deltas only after that baseline has been established.
    if (
      (!hadSnapshot && settled.some((job) => job.status === "completed")) ||
      newlySettled.some((job) => job.status === "completed")
    ) {
      emit("models-changed", host.id);
      void refreshModels();
      if (host.id === props.selectedHostId) void runSearch(true);
    }
    if (hadSnapshot) {
      const failed = newlySettled.find((job) => job.status === "failed");
      const completed = newlySettled.find((job) => job.status === "completed");
      if (failed) {
        announce(
          `Download of ${failed.model} failed on ${host.name}${failed.error ? `: ${failed.error}` : "."}`,
          true,
        );
      } else if (completed) {
        announce(`${completed.model} is ready on ${host.name}.`);
      }
    }
  }
}

function syncDownloadStreams(): void {
  mobileDownloads.registerConsumer(DOWNLOAD_CONSUMER_ID, downloadHosts.value, {
    onEvent: handleDownloadEvent,
    onStreamError: (host, cause, phase) => {
      if (!mounted) return;
      announce(
        phase === "closed"
          ? `Download monitoring stopped on ${host.name}: ${errorMessage(cause, host.name)}`
          : `Could not monitor downloads on ${host.name}: ${errorMessage(cause, host.name)}`,
        true,
      );
    },
  });
}

async function cancelDownload(row: DownloadRow): Promise<void> {
  try {
    await mobileDownloads.cancel(row.host, row.job);
    announce(`Cancelling ${row.job.model}.`);
  } catch (cause) {
    announce(
      `Could not cancel ${row.job.model} on ${row.host.name}: ${errorMessage(cause, row.host.name)}`,
      true,
    );
  }
}

function requestPull(entry: MobileCatalogEntry): void {
  const candidates = installPlan(entry).targets;
  if (candidates.length === 0 && ownerHostIds(entry).length > 0) {
    announce("No reachable machine is available for this model right now.", true);
    return;
  }
  if (candidates.length > 1) {
    targetRestoreFocus = document.activeElement as HTMLElement | null;
    targetEntry.value = entry;
    updateModalInert();
    void nextTick(() => {
      updateModalInert();
      targetCloseButton.value?.focus();
    });
    return;
  }
  const host = candidates[0]?.host ?? selectedHost.value;
  if (host) void pullTo(entry, host);
}

async function pullTo(entry: MobileCatalogEntry, host: MobileHost): Promise<void> {
  if (mobileDownloads.pullStatusForHost(entry, host.id)) return;
  closeTargetPicker();
  announce(`Connecting to downloads on ${host.name}…`);
  try {
    const result = await mobileDownloads.startPull(entry, host);
    if (result.kind === "conflict") {
      announce(`${entryTitle(entry)} is already queued on ${host.name}.`);
      return;
    }
    if (result.kind !== "started") return;
    announce(`${hostActionLabel(entry, host)}ing ${entryTitle(entry)} on ${host.name}.`);
  } catch (cause) {
    const alreadyQueued = cause instanceof ApiError && cause.status === 409;
    announce(
      alreadyQueued
        ? `${entryTitle(entry)} is already queued on ${host.name}.`
        : `Could not ${hostActionLabel(entry, host).toLowerCase()} ${entryTitle(entry)} on ${host.name}: ${errorMessage(cause, host.name)}`,
      !alreadyQueued,
    );
  }
}

async function openDetail(entry: MobileCatalogEntry): Promise<void> {
  restoreFocus = document.activeElement as HTMLElement | null;
  announcement.value = "";
  announcementIsError.value = false;
  detailEntry.value = entry;
  detail.value = null;
  detailComponents.value = [];
  detailComponentsLoading.value = false;
  detailBusy.value = "";
  detailRemoveConfirm.value = false;
  const target = targetForEntry(entry);
  const epoch = ++detailEpoch;
  detailLoading.value = true;
  updateModalInert();
  void nextTick(() => {
    updateModalInert();
    detailCloseButton.value?.focus();
  });

  if (target) {
    void fetchCatalogDetail(entry.id, false, target)
      .then((result) => {
        if (epoch === detailEpoch) detail.value = result;
      })
      .catch(() => {
        // Summary data remains useful when an older host has no detail endpoint.
      })
      .finally(() => {
        if (epoch === detailEpoch) detailLoading.value = false;
      });
  } else {
    detailLoading.value = false;
  }

  if (entry.installed && target) {
    detailComponentsLoading.value = true;
    void fetchModelComponents(entry.id, target)
      .then((result) => {
        if (epoch === detailEpoch) detailComponents.value = result.components;
      })
      .catch(() => {
        if (epoch === detailEpoch) detailComponents.value = [];
      })
      .finally(() => {
        if (epoch === detailEpoch) detailComponentsLoading.value = false;
      });
  }
}

function closeDetail(): void {
  ++detailEpoch;
  detailEntry.value = null;
  detail.value = null;
  detailComponents.value = [];
  detailLoading.value = false;
  detailComponentsLoading.value = false;
  detailBusy.value = "";
  detailRemoveConfirm.value = false;
  updateModalInert();
  const focus = restoreFocus;
  restoreFocus = null;
  void nextTick(() => focus?.focus?.());
}

function closeTargetPicker(): void {
  if (!targetEntry.value) return;
  targetEntry.value = null;
  updateModalInert();
  const focus = targetRestoreFocus;
  targetRestoreFocus = null;
  void nextTick(() => focus?.focus?.());
}

async function changeLoadedState(load: boolean): Promise<void> {
  const entry = detailEntry.value;
  const host = entry ? owningHost(entry) : null;
  if (!entry || !host) return;
  const epoch = detailEpoch;
  detailBusy.value = load ? "load" : "unload";
  try {
    if (load) await loadModel(entry.name, mobileHostTarget(host));
    else await unloadModel(entry.name, mobileHostTarget(host));
    announce(`${entryTitle(entry)} ${load ? "loaded" : "unloaded"} on ${host.name}.`);
    emit("models-changed", host.id);
    await refreshModels();
  } catch (cause) {
    announce(
      `Could not ${load ? "load" : "unload"} ${entryTitle(entry)} on ${host.name}: ${errorMessage(cause, host.name)}`,
      true,
    );
  } finally {
    if (epoch === detailEpoch && detailEntry.value === entry) detailBusy.value = "";
  }
}

async function removeInstalled(): Promise<void> {
  const entry = detailEntry.value;
  const host = entry ? owningHost(entry) : null;
  if (!entry || !host) return;
  if (!detailRemoveConfirm.value) {
    detailRemoveConfirm.value = true;
    return;
  }
  const epoch = detailEpoch;
  detailBusy.value = "remove";
  try {
    const result = await removeModel(entry.name, mobileHostTarget(host));
    announce(
      `Removed ${entryTitle(entry)} from ${host.name}; freed ${formatGB(result.freed_bytes)}.`,
    );
    emit("models-changed", host.id);
    // The user may close this sheet and inspect another model while removal is
    // still in flight. Never let the old operation dismiss that new sheet.
    if (epoch === detailEpoch && detailEntry.value === entry) closeDetail();
    await refreshModels();
    await runSearch(true);
  } catch (cause) {
    const onGpu = cause instanceof ApiError && cause.status === 409;
    announce(
      onGpu
        ? `${entryTitle(entry)} is on the GPU. Unload it first.`
        : `Could not remove ${entryTitle(entry)} from ${host.name}: ${errorMessage(cause, host.name)}`,
      true,
    );
  } finally {
    if (epoch === detailEpoch && detailEntry.value === entry) detailBusy.value = "";
  }
}

function detailTotalLabel(): string {
  const total = detailDownloadTotal.value;
  if (total.bytes == null) return "Size unavailable";
  return `${total.complete ? "" : "At least "}${formatGB(total.bytes)}`;
}

function selectHost(event: Event): void {
  emit("select-host", (event.target as HTMLSelectElement).value);
}

function trapFocus(event: KeyboardEvent, dialog: HTMLElement): void {
  const focusable = [...dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)];
  if (focusable.length === 0) {
    event.preventDefault();
    return;
  }
  const first = focusable[0]!;
  const last = focusable[focusable.length - 1]!;
  const active = document.activeElement;
  if (event.shiftKey && (active === first || !dialog.contains(active))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (active === last || !dialog.contains(active))) {
    event.preventDefault();
    first.focus();
  }
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Tab") {
    const dialog = targetDialog.value ?? detailDialog.value;
    if (dialog) trapFocus(event, dialog);
    return;
  }
  if (event.key !== "Escape") return;
  if (targetEntry.value) {
    closeTargetPicker();
    return;
  }
  if (detailEntry.value) closeDetail();
}

function activateInteractions(): void {
  if (!interactionListenersActive) {
    window.addEventListener("keydown", onKeydown);
    interactionListenersActive = true;
  }
  void nextTick(() => {
    updateModalInert();
    if (targetEntry.value) targetCloseButton.value?.focus();
    else if (detailEntry.value) detailCloseButton.value?.focus();
  });
}

function deactivateInteractions(): void {
  if (interactionListenersActive) {
    window.removeEventListener("keydown", onKeydown);
    interactionListenersActive = false;
  }
  clearModalInert();
}

watch(
  source,
  (next) => {
    if (next === "installed") {
      family.value = "";
      kind.value = "";
    } else {
      lastDiscoverSource.value = next;
    }
  },
  { flush: "sync" },
);

function showInstalledModels(): void {
  source.value = "installed";
}

function showDiscoverModels(): void {
  if (source.value === "installed") source.value = lastDiscoverSource.value;
}

watch([query, source, family, kind, sort, includeNsfw], scheduleSearch);

/**
 * Apply a deep link's filters. Create's "Browse video models" must LAND on
 * the Video + Models shelf — it used to only switch tabs and leave the user
 * to rediscover the filters. Watching the token (not object identity) means
 * the same intent re-applies after the user has since changed a chip.
 */
watch(
  () => props.filterIntent?.token,
  (token) => {
    const intent = props.filterIntent;
    if (!intent || token === undefined) return;
    showDiscoverModels();
    mediaType.value = intent.mediaType;
    kind.value = intent.kind;
  },
  { immediate: true },
);

watch(mediaType, () => {
  if (source.value === "installed" || loading.value || mediaType.value === "all") return;
  if (!combinedEntries.value.some(mediaMatches) && hasMore.value) loadMore();
});

/** Everything about the browsed host that decides what the catalog can read:
 *  which host, where it answers, the key it answers to, and whether the last
 *  probe reached it. */
const catalogRoute = computed(() => {
  const host = selectedHost.value;
  return host ? JSON.stringify([host.id, host.baseUrl, host.apiKey, host.online]) : "";
});

let routedHostId = props.selectedHostId;

// A host that was unreachable — or whose API key was still coming out of the
// Keychain — when this view mounted answers nothing, and neither the taxonomy
// nor a failed search is retried by the debounce. Re-drive them off the route
// so the Family filter and the grid recover on their own once the host does.
watch(catalogRoute, () => {
  if (!mounted) return;
  const switched = props.selectedHostId !== routedHostId;
  routedHostId = props.selectedHostId;
  if (switched) {
    closeDetail();
    family.value = "";
    families.value = [];
  }
  void loadFamilies();
  // A same-host route change keeps results the user may be scrolled into;
  // only a run that produced nothing usable is worth restarting.
  if (switched || error.value || entries.value.length === 0) void runSearch(true);
});

watch(
  () =>
    props.hosts
      .map(
        (host) =>
          `${host.id}:${host.baseUrl}:${host.apiKey}:${host.online}:${host.name}:${host.version ?? ""}`,
      )
      .join("|"),
  () => {
    if (!mounted) return;
    syncDownloadStreams();
    void refreshModels();
  },
);

onMounted(() => {
  mounted = true;
  activateInteractions();
  syncDownloadStreams();
  void refreshModels();
  void loadFamilies();
  void runSearch(true);
});

onActivated(activateInteractions);
onDeactivated(deactivateInteractions);

onBeforeUnmount(() => {
  mounted = false;
  ++searchEpoch;
  ++familyEpoch;
  ++modelsEpoch;
  ++detailEpoch;
  if (debounce) clearTimeout(debounce);
  deactivateInteractions();
  mobileDownloads.unregisterConsumer(DOWNLOAD_CONSUMER_ID);
});
</script>

<template>
  <section ref="catalogRoot" class="mobile-catalog" aria-labelledby="mobile-catalog-title">
    <header class="mobile-catalog-header">
      <div>
        <h1 id="mobile-catalog-title" class="section-title">Catalog</h1>
        <p class="section-note">Models for your remote Mold hosts</p>
      </div>
      <label v-if="hosts.length > 1" class="mobile-catalog-host-picker">
        <span>Browse on</span>
        <select :value="selectedHostId" aria-label="Catalog host" @change="selectHost">
          <option v-for="host in hosts" :key="host.id" :value="host.id">
            {{ host.name }}{{ host.online ? "" : " · offline" }}
          </option>
        </select>
      </label>
    </header>

    <p
      v-if="announcement && !detailEntry && !targetEntry"
      class="mobile-catalog-action-status mobile-catalog-error"
      :class="{ 'error-text': announcementIsError }"
      :role="announcementIsError ? 'alert' : 'status'"
      aria-atomic="true"
      data-test="mobile-catalog-action-status"
    >
      {{ announcement }}
    </p>

    <div v-if="!selectedHost" class="mobile-catalog-empty empty-state">
      Add and select a remote host to browse its catalog.
    </div>

    <template v-else>
      <section
        v-if="downloadRows.length"
        class="mobile-catalog-downloads"
        aria-labelledby="mobile-catalog-downloads-title"
      >
        <div class="mobile-catalog-downloads-head">
          <h2 id="mobile-catalog-downloads-title">Downloads</h2>
          <span>{{ downloadRows.length }} active</span>
        </div>
        <ul>
          <li
            v-for="row in downloadRows"
            :key="`${row.host.id}:${row.job.id}`"
            class="mobile-catalog-download"
            data-test="mobile-catalog-download"
          >
            <div class="mobile-catalog-download-copy">
              <strong>{{ modelLabel(row.job.model) }}</strong>
              <span>
                {{ row.host.name }} ·
                {{
                  row.job.status === "active" && !row.job.bytes_total ? "Preparing" : row.job.status
                }}
              </span>
            </div>
            <div
              class="mobile-catalog-download-progress"
              role="progressbar"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="percent(row.job.bytes_done, row.job.bytes_total)"
              :aria-label="`Downloading ${modelLabel(row.job.model)} on ${row.host.name}`"
            >
              <span :style="{ width: `${percent(row.job.bytes_done, row.job.bytes_total)}%` }" />
            </div>
            <span class="mobile-catalog-download-size">
              <template v-if="row.job.bytes_total">
                {{ formatGB(row.job.bytes_done) }} / {{ formatGB(row.job.bytes_total) }}
              </template>
              <template v-else-if="row.job.status === 'active'">
                {{ row.job.current_file || "Preparing…" }}
              </template>
              <template v-else>Waiting…</template>
            </span>
            <button
              type="button"
              :disabled="mobileDownloads.isCancelling(row.host, row.job.id)"
              :aria-label="`Cancel download of ${modelLabel(row.job.model)} on ${row.host.name}`"
              @click="cancelDownload(row)"
            >
              {{ mobileDownloads.isCancelling(row.host, row.job.id) ? "…" : "Cancel" }}
            </button>
          </li>
        </ul>
      </section>

      <MinimaxH3InventoryPanel :hosts="h3Hosts" />

      <div class="mobile-catalog-search-row">
        <input
          v-model="query"
          type="search"
          inputmode="search"
          autocomplete="off"
          aria-label="Search catalog"
          placeholder="Search models…"
          data-test="mobile-catalog-search"
        />
      </div>

      <div class="mobile-catalog-segment" role="group" aria-label="Model shelf">
        <button
          type="button"
          :aria-pressed="source === 'installed'"
          data-test="mobile-catalog-segment-installed"
          @click="showInstalledModels"
        >
          Installed
        </button>
        <button
          type="button"
          :aria-pressed="source !== 'installed'"
          data-test="mobile-catalog-segment-discover"
          @click="showDiscoverModels"
        >
          Discover
        </button>
      </div>

      <div class="mobile-catalog-media" role="group" aria-label="Media type">
        <button
          v-for="option in ['all', 'image', 'video'] as const"
          :key="option"
          type="button"
          :aria-pressed="mediaType === option"
          @click="mediaType = option"
        >
          {{ option === "all" ? "All" : option === "image" ? "Images" : "Video" }}
        </button>
      </div>

      <div
        v-if="source !== 'installed'"
        class="mobile-catalog-sources"
        role="group"
        aria-label="Catalog source"
      >
        <button
          v-for="option in ['all', 'hf', 'civitai'] as const"
          :key="option"
          type="button"
          :aria-pressed="source === option"
          @click="source = option"
        >
          {{ option === "all" ? "All" : option === "hf" ? "HuggingFace" : "Civitai" }}
        </button>
      </div>

      <div
        v-if="source !== 'installed'"
        class="mobile-catalog-kinds"
        role="group"
        aria-label="Model kind"
        data-test="mobile-catalog-kind-chips"
      >
        <button type="button" :aria-pressed="kind === ''" @click="kind = ''">All</button>
        <button
          v-for="option in CATALOG_KIND_OPTIONS"
          :key="option.value"
          type="button"
          :aria-pressed="kind === option.value"
          @click="kind = option.value"
        >
          {{ option.label }}
        </button>
      </div>

      <div v-if="source !== 'installed'" class="mobile-catalog-filters">
        <label>
          <span>Family</span>
          <select v-model="family" data-test="mobile-catalog-family">
            <option value="">All families</option>
            <option v-for="option in familyOptions" :key="option" :value="option">
              {{ option }}
            </option>
          </select>
        </label>
        <label>
          <span>Sort</span>
          <select v-model="sort" data-test="mobile-catalog-sort">
            <option
              v-for="option in CATALOG_SORT_OPTIONS"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </option>
          </select>
        </label>
        <label class="mobile-catalog-nsfw">
          <input v-model="includeNsfw" type="checkbox" />
          <span>Include NSFW</span>
        </label>
      </div>

      <p v-if="error" class="mobile-catalog-error error-text" role="alert">{{ error }}</p>
      <div v-if="loading && combinedEntries.length === 0" class="mobile-catalog-empty empty-state">
        Loading models…
      </div>
      <div
        v-else-if="combinedEntries.length === 0"
        class="mobile-catalog-empty empty-state"
        data-test="mobile-catalog-empty"
      >
        <template v-if="source === 'installed'">No installed models match these filters.</template>
        <template v-else-if="query.trim()">Nothing found for “{{ query.trim() }}”.</template>
        <template v-else>No catalog models found.</template>
      </div>

      <ul v-else class="mobile-catalog-results" aria-label="Catalog models">
        <li
          v-for="entry in combinedEntries"
          :key="entry.id"
          class="mobile-catalog-card"
          data-test="mobile-catalog-card"
        >
          <button
            type="button"
            class="mobile-catalog-card-open"
            :aria-label="entryAccessibilityLabel(entry)"
            @click="openDetail(entry)"
          >
            <span class="mobile-catalog-card-preview">
              <img
                v-if="entry.thumbnail_url"
                :src="catalogThumbnailUrl(entry.thumbnail_url)"
                alt=""
                loading="lazy"
                decoding="async"
              />
              <span v-else aria-hidden="true">{{ entry.family.slice(0, 3).toUpperCase() }}</span>
            </span>
            <span class="mobile-catalog-card-body">
              <span class="mobile-catalog-card-title">{{ entry.display_name ?? entry.name }}</span>
              <ModelMetadataBadges
                class="mobile-catalog-card-badges"
                :kind="entry.kind"
                :family="entry.family"
                :nsfw="entry.nsfw"
                :show-modality="false"
              />
              <span class="mobile-catalog-card-meta">
                {{ entry.author ? `${entry.author} · ` : "" }}{{ familyLabel(entry.family) }}
                <template v-if="entry.download_count">
                  · ↓ {{ formatCount(entry.download_count) }}
                </template>
              </span>
              <span v-if="entry.hostLabels?.length" class="mobile-catalog-host-labels">
                <span v-for="label in entry.hostLabels" :key="label">{{ label }}</span>
              </span>
              <span v-if="entry.size_bytes != null" class="mobile-catalog-card-size">
                {{ catalogSizeLabel(catalogSizeInfo(entry)) }}
              </span>
            </span>
          </button>
          <span class="mobile-catalog-card-actions">
            <span v-if="entry.installed" class="mobile-catalog-installed">Installed</span>
            <!-- Installed here is not installed everywhere: keep the action
                 whenever a reachable machine is still missing this model. -->
            <button
              v-if="!entry.installed || installPlan(entry).canInstall"
              type="button"
              class="mobile-catalog-pull"
              :disabled="downloadHosts.length <= 1 && Boolean(pullStatus(entry))"
              :aria-busy="Boolean(pullStatus(entry))"
              :data-pull-state="pullStatus(entry)?.phase"
              :aria-label="`${pullButtonLabel(entry)} ${entry.display_name ?? entry.name}`"
              @click="requestPull(entry)"
            >
              {{ pullButtonLabel(entry) }}
            </button>
          </span>
        </li>
      </ul>

      <div
        v-if="source !== 'installed' && hasMore"
        ref="sentinel"
        data-test="mobile-catalog-sentinel"
        class="mobile-catalog-more"
        aria-hidden="true"
      >
        {{ loading ? "Loading…" : "" }}
      </div>
    </template>

    <Teleport to="body">
      <section
        v-if="detailEntry && mergedDetail"
        ref="detailDialog"
        class="mobile-catalog-detail"
        role="dialog"
        aria-modal="true"
        aria-labelledby="mobile-catalog-detail-heading mobile-catalog-detail-title"
        data-test="mobile-catalog-detail"
      >
        <header class="mobile-catalog-detail-header">
          <button
            ref="detailCloseButton"
            type="button"
            aria-label="Close model details"
            @click="closeDetail"
          >
            ‹ <span>Catalog</span>
          </button>
          <strong id="mobile-catalog-detail-heading">Model details</strong>
          <span aria-hidden="true" />
        </header>

        <div class="mobile-catalog-detail-scroll">
          <div class="mobile-catalog-detail-preview">
            <img
              v-if="mergedDetail.thumbnail_url"
              :src="catalogThumbnailUrl(mergedDetail.thumbnail_url)"
              alt=""
            />
            <span v-else aria-hidden="true">{{ mergedDetail.family.toUpperCase() }}</span>
          </div>

          <article class="mobile-catalog-detail-copy">
            <ModelMetadataBadges
              class="mobile-catalog-detail-badges"
              :kind="mergedDetail.kind"
              :family="mergedDetail.family"
              :modality="mergedDetail.modality ?? null"
              :nsfw="mergedDetail.nsfw"
            />
            <div class="mobile-catalog-detail-title">
              <div>
                <h2 id="mobile-catalog-detail-title">
                  {{ mergedDetail.display_name ?? mergedDetail.name }}
                </h2>
                <p v-if="mergedDetail.author">by {{ mergedDetail.author }}</p>
              </div>
              <span v-if="mergedDetail.installed">Installed</span>
            </div>

            <section
              v-if="detailVariants.length"
              class="mobile-catalog-variants"
              data-test="mobile-catalog-variants"
              aria-label="Model variants"
            >
              <h3>Variants</h3>
              <div>
                <button
                  v-for="variant in detailVariants"
                  :key="variant.id"
                  type="button"
                  data-test="variant-chip"
                  :aria-pressed="variant.id === detailEntry.id"
                  @click="selectDetailVariant(variant)"
                >
                  {{ variantLabel(variant) }}
                  <span v-if="variant.size_bytes != null"
                    >· {{ formatGB(variant.size_bytes) }}</span
                  >
                </button>
              </div>
            </section>

            <div class="mobile-catalog-detail-tiles" data-test="mobile-catalog-detail-tiles">
              <div class="mobile-catalog-detail-tile">
                <span>{{ modelWeightsLabel(modelKindValue(mergedDetail)) }}</span>
                <strong>{{
                  detailSize?.weightsBytes != null ? formatGB(detailSize.weightsBytes) : "—"
                }}</strong>
              </div>
              <div class="mobile-catalog-detail-tile">
                <span>Footprint</span>
                <strong>{{ detailFootprintLabel() }}</strong>
              </div>
            </div>

            <dl class="mobile-catalog-detail-meta">
              <div>
                <dt>Source</dt>
                <dd>{{ catalogSourceLabel(mergedDetail.source) }}</dd>
              </div>
              <div>
                <dt>Family</dt>
                <dd>{{ familyLabel(mergedDetail.family) }}</dd>
              </div>
              <div v-if="mergedDetail.file_format">
                <dt>Format</dt>
                <dd>{{ mergedDetail.file_format }}</dd>
              </div>
              <div v-if="mergedDetail.license">
                <dt>License</dt>
                <dd>{{ mergedDetail.license }}</dd>
              </div>
              <div v-if="mergedDetail.rating != null">
                <dt>Rating</dt>
                <dd>★ {{ mergedDetail.rating.toFixed(1) }}</dd>
              </div>
              <div v-if="mergedDetail.download_count">
                <dt>Downloads</dt>
                <dd>{{ formatCount(mergedDetail.download_count) }}</dd>
              </div>
              <div v-if="mergedDetail.likes">
                <dt>Likes</dt>
                <dd>{{ formatCount(mergedDetail.likes) }}</dd>
              </div>
            </dl>

            <p v-if="mergedDetail.description" class="mobile-catalog-detail-description">
              {{ mergedDetail.description }}
            </p>
            <p v-else-if="detailLoading" class="mobile-catalog-detail-muted">Loading details…</p>

            <div v-if="mergedDetail.tags?.length" class="mobile-catalog-detail-tags">
              <span v-for="tag in mergedDetail.tags" :key="tag">{{ tag }}</span>
            </div>

            <section
              v-if="detailDownloadItems.length"
              class="mobile-catalog-detail-section"
              aria-labelledby="mobile-catalog-contents-title"
            >
              <div class="mobile-catalog-detail-section-head">
                <h3 id="mobile-catalog-contents-title">Download contents</h3>
                <span>{{ detailTotalLabel() }}</span>
              </div>
              <ul>
                <li v-for="item in detailDownloadItems" :key="item.key">
                  <span>{{ item.label }}</span>
                  <span
                    >{{ item.kind }} · {{ item.sizeBytes ? formatGB(item.sizeBytes) : "—" }}</span
                  >
                </li>
              </ul>
            </section>

            <section
              v-if="detailComponents.length || detailComponentsLoading"
              class="mobile-catalog-detail-section"
              aria-labelledby="mobile-catalog-components-title"
            >
              <div class="mobile-catalog-detail-section-head">
                <h3 id="mobile-catalog-components-title">On this host</h3>
                <span v-if="detailComponents.length">
                  {{ detailComponents.filter((item) => item.present).length }}/{{
                    detailComponents.length
                  }}
                  present
                </span>
              </div>
              <p v-if="detailComponentsLoading" class="mobile-catalog-detail-muted">
                Checking files…
              </p>
              <ul v-else>
                <li v-for="component in detailComponents" :key="component.name">
                  <span>{{ component.name }}</span>
                  <span
                    class="mobile-catalog-component-status"
                    :data-present="component.present"
                    :aria-label="`${component.present ? 'Present' : 'Missing'}: ${component.kind}`"
                  >
                    <span
                      class="mobile-catalog-component-dot"
                      :data-present="component.present"
                      aria-hidden="true"
                    />
                    {{ component.present ? "Present" : "Missing" }} · {{ component.kind }}
                  </span>
                </li>
              </ul>
            </section>

            <section
              v-if="mergedDetail.installed && detailModel"
              class="mobile-catalog-installed-actions"
              aria-label="Installed model actions"
            >
              <button
                v-if="detailModel.is_loaded"
                type="button"
                class="secondary-button"
                :disabled="Boolean(detailBusy)"
                @click="changeLoadedState(false)"
              >
                {{ detailBusy === "unload" ? "Unloading…" : "Unload from GPU" }}
              </button>
              <button
                v-else
                type="button"
                class="secondary-button"
                :disabled="Boolean(detailBusy)"
                @click="changeLoadedState(true)"
              >
                {{ detailBusy === "load" ? "Loading…" : "Load on GPU" }}
              </button>
              <button
                type="button"
                class="danger-button"
                :disabled="Boolean(detailBusy)"
                @blur="detailRemoveConfirm = false"
                @click="removeInstalled"
              >
                {{
                  detailBusy === "remove"
                    ? "Removing…"
                    : detailRemoveConfirm
                      ? "Tap again to remove"
                      : "Remove from host"
                }}
              </button>
            </section>
          </article>
        </div>

        <footer class="mobile-catalog-detail-action">
          <p
            v-if="announcement"
            class="mobile-catalog-action-status"
            :class="{ 'error-text': announcementIsError }"
            :role="announcementIsError ? 'alert' : 'status'"
            aria-atomic="true"
            data-test="mobile-catalog-detail-action-status"
          >
            {{ announcement }}
          </p>
          <p v-if="mergedDetail.supported != null && !canDownloadEntry(mergedDetail)">
            This catalog package is not supported by a shipped Mold engine yet.
          </p>
          <p v-else-if="catalogFetchCaption(catalogSizeInfo(mergedDetail))">
            {{ catalogFetchCaption(catalogSizeInfo(mergedDetail)) }}
          </p>
          <button
            type="button"
            class="primary-button"
            :disabled="
              !canDownloadEntry(mergedDetail) ||
              (downloadHosts.length <= 1 && Boolean(pullStatus(detailEntry)))
            "
            :aria-busy="Boolean(pullStatus(detailEntry))"
            :data-pull-state="pullStatus(detailEntry)?.phase"
            @click="requestPull(detailEntry)"
          >
            {{ pullStatus(detailEntry)?.label ?? installPlan(detailEntry).label }}
            <template v-if="!pullStatus(detailEntry) && detailDownloadTotal.bytes != null">
              · {{ detailTotalLabel() }}</template
            >
          </button>
        </footer>
      </section>

      <section
        v-if="targetEntry"
        ref="targetDialog"
        class="mobile-catalog-target-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="mobile-catalog-target-title"
        data-test="mobile-catalog-target-sheet"
      >
        <div class="mobile-catalog-target-backdrop" @click="closeTargetPicker" />
        <div class="mobile-catalog-target-panel">
          <header>
            <div>
              <h2 id="mobile-catalog-target-title">
                {{ targetPlan?.canInstall ? "Choose where to install" : "Choose where to repair" }}
              </h2>
              <p>
                <!-- Only a mixed list needs the repair caveat; a plain first
                     download must not imply some machine already has it. -->
                <template v-if="targetPlanMixed">
                  {{ targetEntry.display_name ?? targetEntry.name }} and its required components
                  will be stored on the machine you pick; machines that already have it are repaired
                  instead.
                </template>
                <template v-else-if="targetPlan?.canInstall">
                  {{ targetEntry.display_name ?? targetEntry.name }} and its required components
                  will be stored on the machine you pick.
                </template>
                <template v-else>
                  Only missing or damaged files for
                  {{ targetEntry.display_name ?? targetEntry.name }} will be fetched.
                </template>
              </p>
            </div>
            <button
              ref="targetCloseButton"
              type="button"
              aria-label="Close host picker"
              @click="closeTargetPicker"
            >
              Close
            </button>
          </header>
          <div class="mobile-catalog-target-list">
            <button
              v-for="target in targetPlan?.targets ?? []"
              :key="target.host.id"
              type="button"
              data-test="mobile-catalog-target-option"
              :data-action="target.action"
              :disabled="Boolean(pullStatus(targetEntry, target.host.id))"
              :aria-busy="Boolean(pullStatus(targetEntry, target.host.id))"
              :data-pull-state="pullStatus(targetEntry, target.host.id)?.phase"
              :aria-label="`${targetActionLabel(target.action)} on ${target.host.name}`"
              @click="pullTo(targetEntry, target.host)"
            >
              <span>
                <strong>{{ target.host.name }}</strong>
                <small>{{ target.host.baseUrl }}</small>
                <small class="mobile-catalog-target-action">{{
                  targetActionLabel(target.action)
                }}</small>
              </span>
              <span>
                {{
                  pullStatus(targetEntry, target.host.id)?.label ??
                  (target.host.id === selectedHostId ? "Current" : "")
                }}
                <template v-if="!pullStatus(targetEntry, target.host.id)"> ›</template>
              </span>
            </button>
          </div>
        </div>
      </section>
    </Teleport>
  </section>
</template>
