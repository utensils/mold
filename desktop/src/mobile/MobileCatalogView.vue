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
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  sortInstalledFirst,
} from "../lib/catalog";
import {
  buildDownloadContents,
  canDownloadEntry,
  catalogActionLabel,
  downloadContentsTotalBytes,
  installedModelToEntry,
} from "../lib/catalogDetail";
import { catalogThumbnailUrl } from "../lib/catalogThumbnails";
import { isVideoFamily } from "../lib/capabilities";
import { formatCount, formatGB, percent } from "../lib/format";
import { isCatalogModelId, isUtilityModel } from "../lib/models";
import { fetchModelComponents, loadModel, removeModel, unloadModel } from "../lib/api/models";
import type { CatalogEntry, DownloadJob, ModelComponentStatus, ModelEntry } from "../lib/api/types";
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

const props = defineProps<{
  hosts: MobileHost[];
  selectedHostId: string;
}>();

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
const includeNsfw = ref(false);
const families = ref<string[]>([]);
const entries = ref<CatalogEntry[]>([]);
const page = ref(1);
const hasMore = ref(false);
const loading = ref(false);
const error = ref("");
const announcement = ref("");
const announcementIsError = ref(false);
const modelsByHost = ref<Record<string, ModelEntry[]>>({});
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

/** The selected host remains usable while its latest reachability probe is pending. */
const downloadHosts = computed(() =>
  props.hosts.filter((host) => host.online || host.id === props.selectedHostId),
);

const installedEntries = computed<MobileCatalogEntry[]>(() => {
  const byName = new Map<string, { model: ModelEntry; hostIds: string[]; hostLabels: string[] }>();
  for (const host of props.hosts) {
    for (const model of modelsByHost.value[host.id] ?? []) {
      if (!model.downloaded) continue;
      const found = byName.get(model.name);
      if (found) {
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

const installedNames = computed(() => new Set(installedEntries.value.map((entry) => entry.name)));

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

const filteredInstalled = computed(() => {
  const q = query.value.trim().toLowerCase();
  return installedEntries.value
    .filter(
      (entry) =>
        !q || entry.name.toLowerCase().includes(q) || entryTitle(entry).toLowerCase().includes(q),
    )
    .filter((entry) => !family.value || entry.family === family.value)
    .filter(sourceMatches)
    .filter(mediaMatches);
});

const combinedEntries = computed<MobileCatalogEntry[]>(() => {
  if (source.value === "installed") return filteredInstalled.value;

  const knownRepos = new Set(
    (modelsByHost.value[props.selectedHostId] ?? [])
      .map((model) => model.hf_repo)
      .filter((repo): repo is string => Boolean(repo)),
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
  const installedByName = new Set(filteredInstalled.value.map((entry) => entry.name));
  const byId = new Map<string, MobileCatalogEntry>();
  for (const entry of [
    ...filteredInstalled.value,
    ...manifestEntries.value.filter(sourceMatches).filter(mediaMatches),
    ...safeLive.filter((entry) => !installedByName.has(entry.name)).filter(mediaMatches),
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
  return detail.value
    ? { ...summary, ...detail.value, installed: summary.installed || detail.value.installed }
    : summary;
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

function pullButtonLabel(entry: MobileCatalogEntry): string {
  return pullStatus(entry)?.label ?? catalogPullLabel(catalogSizeInfo(entry));
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
          source: source.value === "all" ? undefined : source.value,
          include_nsfw: includeNsfw.value,
          page: page.value,
          page_size: PAGE_SIZE,
        },
        false,
        target,
      );
      if (epoch !== searchEpoch) return;
      entries.value = [...entries.value, ...response.entries];
      hasMore.value = response.entries.length === PAGE_SIZE;
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

/** End-of-list sentinel: scrolling near it fetches the next page — the
 *  old Load more button read as a dead end and (worse) looked broken
 *  whenever upstream paging returned duplicate rows the dedup swallowed. */
const sentinel = ref<HTMLElement | null>(null);
let sentinelObserver: IntersectionObserver | null = null;

watch(sentinel, (el) => {
  sentinelObserver?.disconnect();
  sentinelObserver = null;
  if (!el) return;
  sentinelObserver = new IntersectionObserver(
    (hits) => {
      if (hits.some((hit) => hit.isIntersecting)) loadMore();
    },
    { rootMargin: "600px 0px" },
  );
  sentinelObserver.observe(el);
});

async function loadFamilies(): Promise<void> {
  const target = selectedTarget.value;
  const epoch = ++familyEpoch;
  families.value = [];
  if (!target) return;
  try {
    // The iPhone shell has no desktop `secret_get` command. Remote catalog
    // hosts use their own configured HF/Civitai credentials; only the Mold
    // host API key from `target` crosses the wire.
    const result = await fetchCatalogFamilies(false, target);
    if (epoch === familyEpoch) families.value = result;
  } catch {
    // Family filtering is optional; catalog search remains fully usable.
  }
}

async function refreshModels(): Promise<void> {
  const epoch = ++modelsEpoch;
  const next: Record<string, ModelEntry[]> = {};
  await Promise.all(
    downloadHosts.value.map(async (host) => {
      try {
        const response = await apiFetchTo(mobileHostTarget(host), "/api/models");
        const result = (await response.json()) as ModelEntry[];
        if (epoch === modelsEpoch) next[host.id] = result;
      } catch {
        // One unavailable host must not blank models from the other hosts.
      }
    }),
  );
  if (epoch === modelsEpoch) modelsByHost.value = next;
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
  if (entry.installed) {
    const host = owningHost(entry);
    if (host) void pullTo(entry, host);
    return;
  }
  if (downloadHosts.value.length > 1) {
    targetRestoreFocus = document.activeElement as HTMLElement | null;
    targetEntry.value = entry;
    updateModalInert();
    void nextTick(() => {
      updateModalInert();
      targetCloseButton.value?.focus();
    });
    return;
  }
  const host = downloadHosts.value[0] ?? selectedHost.value;
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
    announce(`${catalogActionLabel(entry)}ing ${entryTitle(entry)} on ${host.name}.`);
  } catch (cause) {
    const alreadyQueued = cause instanceof ApiError && cause.status === 409;
    announce(
      alreadyQueued
        ? `${entryTitle(entry)} is already queued on ${host.name}.`
        : `Could not ${catalogActionLabel(entry).toLowerCase()} ${entryTitle(entry)} on ${host.name}: ${errorMessage(cause, host.name)}`,
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
    if (next === "installed") family.value = "";
    else lastDiscoverSource.value = next;
  },
  { flush: "sync" },
);

function showInstalledModels(): void {
  source.value = "installed";
}

function showDiscoverModels(): void {
  if (source.value === "installed") source.value = lastDiscoverSource.value;
}

watch([query, source, family, includeNsfw], scheduleSearch);

watch(mediaType, () => {
  if (source.value === "installed" || loading.value || mediaType.value === "all") return;
  if (!combinedEntries.value.some(mediaMatches) && hasMore.value) loadMore();
});

watch(
  () => props.selectedHostId,
  () => {
    if (!mounted) return;
    closeDetail();
    family.value = "";
    void loadFamilies();
    void runSearch(true);
  },
);

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
  sentinelObserver?.disconnect();
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
              <strong>{{ row.job.model }}</strong>
              <span>{{ row.host.name }} · {{ row.job.status }}</span>
            </div>
            <div
              class="mobile-catalog-download-progress"
              role="progressbar"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="percent(row.job.bytes_done, row.job.bytes_total)"
              :aria-label="`Downloading ${row.job.model} on ${row.host.name}`"
            >
              <span :style="{ width: `${percent(row.job.bytes_done, row.job.bytes_total)}%` }" />
            </div>
            <span class="mobile-catalog-download-size">
              <template v-if="row.job.bytes_total">
                {{ formatGB(row.job.bytes_done) }} / {{ formatGB(row.job.bytes_total) }}
              </template>
              <template v-else>Waiting…</template>
            </span>
            <button
              type="button"
              :disabled="mobileDownloads.isCancelling(row.host, row.job.id)"
              :aria-label="`Cancel download of ${row.job.model} on ${row.host.name}`"
              @click="cancelDownload(row)"
            >
              {{ mobileDownloads.isCancelling(row.host, row.job.id) ? "…" : "Cancel" }}
            </button>
          </li>
        </ul>
      </section>

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

      <div v-if="source !== 'installed'" class="mobile-catalog-filters">
        <label>
          <span>Family</span>
          <select v-model="family">
            <option value="">All families</option>
            <option v-for="option in families" :key="option" :value="option">
              {{ option }}
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
            :aria-label="`${entry.display_name ?? entry.name} — view details`"
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
              <span class="mobile-catalog-card-meta">
                {{ entry.author ? `${entry.author} · ` : "" }}{{ entry.family }}
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
          <span v-if="entry.installed" class="mobile-catalog-installed">Installed</span>
          <button
            v-else
            type="button"
            class="mobile-catalog-pull"
            :disabled="downloadHosts.length <= 1 && Boolean(pullStatus(entry))"
            :aria-busy="Boolean(pullStatus(entry))"
            :data-pull-state="pullStatus(entry)?.phase"
            @click="requestPull(entry)"
          >
            {{ pullButtonLabel(entry) }}
          </button>
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
        :aria-labelledby="`mobile-catalog-detail-${detailEntry.id}`"
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
          <strong :id="`mobile-catalog-detail-${detailEntry.id}`">Model details</strong>
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
            <span
              v-if="mergedDetail.modality"
              class="mobile-catalog-detail-badge"
              data-test="mobile-catalog-detail-badge"
              >{{ mergedDetail.modality }}</span
            >
            <div class="mobile-catalog-detail-title">
              <div>
                <h2>{{ mergedDetail.display_name ?? mergedDetail.name }}</h2>
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
                <span>Checkpoint</span>
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
                <dt>Family</dt>
                <dd>{{ mergedDetail.family }}</dd>
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
          <p v-if="mergedDetail.engine_phase != null && !canDownloadEntry(mergedDetail)">
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
            {{ pullStatus(detailEntry)?.label ?? catalogActionLabel(mergedDetail) }}
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
              <h2 id="mobile-catalog-target-title">Choose a host</h2>
              <p>
                {{ catalogActionLabel(targetEntry) }}
                {{ targetEntry.display_name ?? targetEntry.name }}
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
              v-for="host in downloadHosts"
              :key="host.id"
              type="button"
              :disabled="Boolean(pullStatus(targetEntry, host.id))"
              :aria-busy="Boolean(pullStatus(targetEntry, host.id))"
              :data-pull-state="pullStatus(targetEntry, host.id)?.phase"
              @click="pullTo(targetEntry, host)"
            >
              <span>
                <strong>{{ host.name }}</strong>
                <small>{{ host.baseUrl }}</small>
              </span>
              <span>
                {{
                  pullStatus(targetEntry, host.id)?.label ??
                  (host.id === selectedHostId ? "Current" : "")
                }}
                <template v-if="!pullStatus(targetEntry, host.id)"> ›</template>
              </span>
            </button>
          </div>
        </div>
      </section>
    </Teleport>
  </section>
</template>
