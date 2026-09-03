<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { catalogFamily, matchesCatalogFamily } from "@studio/lib/modelFamily";
import CatalogLayoutToggle, {
  type CatalogLayoutChoice,
} from "@ui/components/CatalogLayoutToggle.vue";
import { planModelInstall } from "@studio/lib/modelInstallTargets";
import { isAlreadyQueuedError, planBatchInstallTargets } from "@studio/lib/modelBatchInstall";
import { useDownloadsStore } from "../../stores/downloads";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { useInventoryKnown } from "../../lib/modelInventory";
import { isGenerationModel, useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { modelRuntimeNoticeAcrossHosts } from "@studio/lib/modelRuntimeAvailability";
import { useToastStore } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { ApiError, currentTarget, type ApiTarget } from "../../lib/api/client";
import { runWithLicenseConsent } from "@studio/composables/useLicenseAcceptance";
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
import { useInfiniteScrollSentinel } from "../../lib/useInfiniteScrollSentinel";
import CatalogCard from "./CatalogCard.vue";
import CatalogTableRow from "./CatalogTableRow.vue";
import CatalogDetailDrawer, { type DrawerVariant } from "./CatalogDetailDrawer.vue";
import DownloadTargetDialog from "./DownloadTargetDialog.vue";
import { canDownloadEntry, installedModelToEntry } from "../../lib/catalogDetail";
import type { CatalogEntry, CatalogProviderError, ModelEntry } from "../../lib/api/types";

type LibraryModelEntry = ModelEntry & { hostIds?: string[] };
type CatalogListEntry = CatalogEntry & { hostIds?: string[] };

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
const hostModels = useHostModelsStore();
const ui = useUiStore();
const inventoryKnown = useInventoryKnown();

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
/** Families observed anywhere in this component's inventory/search lifetime.
 *  This is deliberately not derived from the currently filtered rows: doing
 *  so collapses the selector to its active family and strands the user. */
const observedFamilies = ref<string[]>([]);

const entries = ref<CatalogEntry[]>([]);
const page = ref(1);
const hasMore = ref(false);
const loading = ref(false);
const error = ref<string | null>(null);
const providerErrors = ref<CatalogProviderError[]>([]);
const pulling = ref<Set<string>>(new Set());
const selected = ref(new Map<string, CatalogListEntry>());
const selectedTargetId = ref("");
const batchStarting = ref(false);
const pendingEntry = ref<CatalogEntry | null>(null);
/** Entry whose in-app detail drawer is open. */
const detailEntry = ref<CatalogListEntry | null>(null);

let debounce: ReturnType<typeof setTimeout> | null = null;
let familyEpoch = 0;
let mounted = false;
let activeCatalogRoute = "";

function recordFamilies(values: Iterable<string | null | undefined>): void {
  const seen = new Set(observedFamilies.value);
  const before = seen.size;
  for (const value of values) {
    const name = value?.trim();
    if (name) seen.add(catalogFamily(name));
  }
  if (seen.size !== before) observedFamilies.value = [...seen].sort((a, b) => a.localeCompare(b));
}

/** Prefer the host taxonomy once available; inventory/results cover startup
 *  latency and failed taxonomy reads without ever presenting an empty picker. */
const familyOptions = computed(() => {
  const options = families.value.length > 0 ? [...families.value] : [...observedFamilies.value];
  if (family.value && !options.includes(family.value)) options.push(family.value);
  return options.sort((a, b) => a.localeCompare(b));
});

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
const installedCatalogEntries = computed<CatalogListEntry[]>(() => {
  const q = props.query.trim().toLowerCase();
  return (props.installedEntries ?? [])
    .filter(
      (m) =>
        !q || m.name.toLowerCase().includes(q) || modelDisplayName(m).toLowerCase().includes(q),
    )
    .filter((m) => matchesCatalogFamily(m.family, family.value))
    .map((m) => ({ ...installedModelToEntry(m), hostIds: m.hostIds ?? [] }))
    .filter(
      (entry) =>
        source.value === "all" ||
        (source.value === "hf" && entry.source === "hf") ||
        (source.value === "civitai" && entry.source === "civitai"),
    );
});

/** Canonical catalog shape for one not-yet-installed manifest model. Keep
 * this outside the filtered list so variant selection can resolve a sibling
 * hidden by search/source/family controls without fabricating install state
 * or dropping its shared-runtime footprint. */
function manifestCatalogEntry(model: ModelEntry): CatalogEntry {
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

/** Preserve installed identity/host state while filling older `/api/models`
 * rows with richer metadata from the matching live catalog result. */
function enrichInstalledEntry(
  installed: CatalogListEntry,
  live: CatalogEntry | undefined,
): CatalogListEntry {
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
function hostLabelsFor(entry: CatalogListEntry): string[] {
  return (entry.hostIds ?? []).map(
    (id) =>
      hosts.all.find((host) => host.id === id)?.label ?? (id === "local" ? "This device" : id),
  );
}

/** Whether the FLEET can run the model behind a Discover row, from each
 *  machine's own `/api/models` listing. Knowable before the pull — which for
 *  the affected checkpoints is 21-42 GB (#1276). Pull targets any connected
 *  machine, so one machine that can run it withdraws the warning; a live
 *  catalog id nobody lists is unknown and gets no badge rather than a guess. */
function runtimeNoticeFor(id: string) {
  return modelRuntimeNoticeAcrossHosts(id, [
    models.all,
    ...Object.values(hostModels.byHost).map((list) => list?.entries),
  ]);
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
    .filter((model) => matchesCatalogFamily(model.family, family.value))
    .map(manifestCatalogEntry);
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

/**
 * Where this model can still go. Being installed on one machine says nothing
 * about the others, so every reachable machine that lacks it stays an install
 * target and only the owners degrade to repair. Machines whose inventory has
 * not been read are left out entirely rather than assumed empty.
 */
function installPlan(entry: CatalogEntry & { hostIds?: string[] }) {
  return planModelInstall(readyHosts.value, ownerIdsFor(entry), { inventoryKnown });
}

/**
 * Machines known to hold this row. A merged installed row carries every owner
 * in `hostIds`; a live catalog row carries none, and its `installed` flag is
 * only the browsed host's answer — so that host stands in as the sole known
 * owner. Installed LoRAs and ControlNets arrive exactly this way, being
 * excluded from the merged generation-model shelf.
 */
function ownerIdsFor(entry: CatalogEntry & { hostIds?: string[] }): string[] {
  const ids = entry.hostIds ?? [];
  if (ids.length > 0) return ids;
  if (!entry.installed) return [];
  const browsing = readyHosts.value.find((host) => host.id === "local") ?? readyHosts.value[0];
  return browsing ? [browsing.id] : [];
}

function actionTargets(entry: CatalogEntry & { hostIds?: string[] }) {
  return installPlan(entry).targets;
}

function selectable(entry: CatalogListEntry): boolean {
  return canDownloadEntry(entry) && actionTargets(entry).length > 0;
}

function toggleSelection(entry: CatalogListEntry, checked: boolean): void {
  const next = new Map(selected.value);
  if (checked) next.set(entry.id, entry);
  else next.delete(entry.id);
  selected.value = next;
}

const batchTargets = computed(() =>
  planBatchInstallTargets(
    [...selected.value.values()].map((entry) => ({
      modelId: entry.id,
      targets: actionTargets(entry),
    })),
  ),
);

watch(
  batchTargets,
  (targets) => {
    if (targets.some(({ host }) => host.id === selectedTargetId.value)) return;
    selectedTargetId.value = targets.length === 1 ? targets[0]!.host.id : "";
  },
  { immediate: true },
);

const selectedBatchTarget = computed(() =>
  batchTargets.value.find(({ host }) => host.id === selectedTargetId.value),
);

function targetSummary(installCount: number, repairCount: number): string {
  const parts = [];
  if (installCount) parts.push(`${installCount} new`);
  if (repairCount) parts.push(`${repairCount} repair`);
  return parts.join(", ");
}

/** The row keeps its action while any machine can still receive the model —
 *  and an entry nobody owns always keeps it, even before hosts resolve. */
function installable(entry: CatalogEntry & { hostIds?: string[] }): boolean {
  return !entry.installed || installPlan(entry).canInstall;
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
  let nextEntries = reset ? [] : [...entries.value];
  if (reset) {
    page.value = 1;
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
      providerErrors.value = res.provider_errors ?? [];
      recordFamilies(res.entries.map((entry) => entry.family));
      nextEntries = [...nextEntries, ...res.entries];
      entries.value = nextEntries;
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

function retrySearch(): void {
  void runSearch(true);
}

const sentinel = ref<HTMLElement | null>(null);
useInfiniteScrollSentinel(sentinel, loading, hasMore, loadMore, MAX_AUTO_PAGES);

/** Returns false when the user declined the host's licence terms. */
async function queueOnHost(entry: CatalogEntry, host: HostView | null): Promise<boolean> {
  const target = host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : undefined;
  // Attach the snapshot-first stream before enqueueing so a cached,
  // near-instant pull still produces a visible terminal event and refresh.
  await downloads.subscribe(host ?? undefined);
  // A gated bundle is refused with the pinned terms attached. Take consent and
  // re-drive this same enqueue, so the job lands in the downloads tray exactly
  // as an ungated one does.
  const outcome = await runWithLicenseConsent({
    hostLabel: host?.label ?? "This device",
    target: target ?? currentTarget(),
    installModel: entry.id,
    start: () => startCatalogDownload(entry.id, target, host ? host.kind === "remote" : false),
  });
  return outcome.kind !== "declined";
}

async function pullTo(entry: CatalogEntry, host: HostView | null) {
  pulling.value.add(entry.id);
  try {
    // A decline needs no toast: the modal WAS the interaction.
    if (!(await queueOnHost(entry, host))) return;
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

async function startBatch(): Promise<void> {
  const target = selectedBatchTarget.value;
  if (!target || batchStarting.value) return;
  batchStarting.value = true;
  for (const item of target.items) pulling.value.add(item.modelId);
  const entriesById = new Map(selected.value);
  const results = await Promise.allSettled(
    target.items.map(async (item) => {
      const entry = entriesById.get(item.modelId);
      if (!entry) throw new Error(`Model ${item.modelId} is no longer selected`);
      try {
        // A declined license queued nothing. Returning the id anyway would
        // drop the model from the selection and count it in the "downloads
        // queued" toast, so carry the decision through the batch.
        if (!(await queueOnHost(entry, target.host))) return null;
      } catch (error) {
        if (!isAlreadyQueuedError(error)) throw error;
      }
      return item.modelId;
    }),
  );
  const next = new Map(selected.value);
  let succeeded = 0;
  const failures: string[] = [];
  results.forEach((result, index) => {
    const item = target.items[index]!;
    pulling.value.delete(item.modelId);
    if (result.status === "fulfilled") {
      // `null` is a decline: leave it selected and out of the count. It is
      // neither a success nor a failure — the dialog was the interaction.
      if (result.value === null) return;
      next.delete(result.value);
      succeeded += 1;
    } else {
      const entry = entriesById.get(item.modelId);
      failures.push(
        `${entry?.display_name ?? entry?.name ?? "Model"}: ${result.reason instanceof Error ? result.reason.message : String(result.reason)}`,
      );
    }
  });
  selected.value = next;
  if (succeeded) {
    toasts.push(
      `${succeeded} ${succeeded === 1 ? "download" : "downloads"} queued on ${target.host.label}`,
    );
  }
  if (failures.length) toasts.push(failures.join(" · "), "error");
  batchStarting.value = false;
}

function pull(entry: CatalogEntry) {
  const candidates = actionTargets(entry);
  if (entry.installed && candidates.length === 0) {
    toasts.push("No online machine is available for this model.", "error");
    return;
  }
  if (candidates.length > 1) {
    pendingEntry.value = entry;
    return;
  }
  void pullTo(entry, candidates[0]?.host ?? null);
}

/** The detail drawer fetches on the same host the catalog list came from. */
const detailTarget = computed(() => catalogTarget());

/**
 * Quantization variants for a manifest model (`base:tag`) — the sibling rows
 * the manifest already describes. Live HF/Civitai rows carry no colon base
 * and get no chips (their precedence is decided in the list).
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

/** A variant chip selects the sibling as the drawer's real entry, not merely
 * as a hidden download override. That keeps weights, footprint, installed
 * state, repair components, and the eventual Pull target on one authority. */
function selectDrawerVariant(id: string): void {
  const listed = combinedEntries.value.find((entry) => entry.id === id);
  if (listed) {
    detailEntry.value = listed;
    return;
  }
  const installed = (props.installedEntries ?? []).find((model) => model.name === id);
  if (installed) {
    detailEntry.value = {
      ...installedModelToEntry(installed),
      hostIds: [...(installed.hostIds ?? [])],
    };
    return;
  }
  const inventory = models.all.find((model) => model.name === id);
  if (!inventory) return;
  detailEntry.value = inventory.downloaded
    ? installedModelToEntry(inventory)
    : manifestCatalogEntry(inventory);
}

/** Pull vs Repair in the drawer follows the fleet, not this one row's flag. */
const detailAction = computed(() =>
  detailEntry.value ? installPlan(detailEntry.value).label : undefined,
);

/** Pull (or Repair — same endpoint, missing files only) from the drawer. */
function pullFromDrawer(entry: CatalogEntry) {
  detailEntry.value = null;
  pull(entry);
}

watch([() => props.query, source, family, kind, sort, includeNsfw], () => {
  scheduleSearch();
});

watch(
  () => [
    ...(props.installedEntries ?? []).map((model) => model.family),
    ...models.all.map((model) => model.family),
  ],
  (values) => recordFamilies(values),
  { immediate: true },
);

async function loadFamilies(): Promise<void> {
  const epoch = ++familyEpoch;
  try {
    const { target, forward } = catalogTarget();
    const result = await fetchCatalogFamilies(forward, target);
    if (epoch === familyEpoch) families.value = result;
  } catch {
    // Retain the last good taxonomy. `familyOptions` covers first-load errors.
  }
}

/** URL/key/reachability decide which catalog authority answers. Re-drive a
 * failed/empty view when that route changes so a late connection self-heals. */
const catalogRoute = computed(() => {
  const { target, forward } = catalogTarget();
  return JSON.stringify([
    target?.baseUrl ?? "primary",
    target?.apiKey ?? null,
    forward,
    hosts.all.map((host) => [host.id, host.status, host.baseUrl]),
  ]);
});

watch(
  catalogRoute,
  (route) => {
    if (!mounted) return;
    if (route === activeCatalogRoute) return;
    activeCatalogRoute = route;
    void loadFamilies();
    if (error.value || entries.value.length === 0) void runSearch(true);
  },
  { flush: "sync" },
);

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
  const startingRoute = catalogRoute.value;
  activeCatalogRoute = startingRoute;
  await Promise.allSettled([loadFamilies(), runSearch(true)]);
  mounted = true;
  const currentRoute = catalogRoute.value;
  if (currentRoute === startingRoute) return;
  activeCatalogRoute = currentRoute;
  void loadFamilies();
  if (error.value || entries.value.length === 0) void runSearch(true);
});

onUnmounted(() => {
  mounted = false;
  familyEpoch += 1;
  searchEpoch += 1;
  if (debounce) clearTimeout(debounce);
  debounce = null;
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
          class="border-border h-7 rounded-control border px-2.5 text-micro"
          :class="source === s ? 'bg-accent text-on-accent' : 'text-fg-2 hover:text-fg'"
          :aria-pressed="source === s"
          @click="source = s"
        >
          {{ s === "all" ? "All" : s === "hf" ? "HuggingFace" : "Civitai" }}
        </button>
      </div>

      <div class="flex items-center gap-1" data-test="catalog-kind-chips" aria-label="Model kind">
        <button
          type="button"
          class="border-border h-7 rounded-control border px-2.5 text-micro"
          :class="kind === '' ? 'bg-accent text-on-accent' : 'text-fg-2 hover:text-fg'"
          :aria-pressed="kind === ''"
          @click="kind = ''"
        >
          All
        </button>
        <button
          v-for="opt in CATALOG_KIND_OPTIONS"
          :key="opt.value"
          type="button"
          class="border-border h-7 rounded-control border px-2.5 text-micro"
          :class="kind === opt.value ? 'bg-accent text-on-accent' : 'text-fg-2 hover:text-fg'"
          :aria-pressed="kind === opt.value"
          @click="kind = opt.value"
        >
          {{ opt.label }}
        </button>
      </div>

      <select
        v-model="family"
        aria-label="Model family"
        class="border-border h-7 rounded-control border bg-bg-deep px-1.5 text-micro text-fg"
      >
        <option value="">All families</option>
        <option v-for="f in familyOptions" :key="f" :value="f">{{ f }}</option>
      </select>

      <select
        v-model="sort"
        data-test="catalog-sort"
        aria-label="Sort by"
        class="border-border h-7 rounded-control border bg-bg-deep px-1.5 text-micro text-fg"
      >
        <option v-for="opt in CATALOG_SORT_OPTIONS" :key="opt.value" :value="opt.value">
          {{ opt.label }}
        </option>
      </select>

      <label class="flex items-center gap-1 text-micro text-fg-2">
        <input v-model="includeNsfw" type="checkbox" class="accent-accent" />
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

    <div
      v-if="error || providerErrors.length"
      data-test="catalog-provider-warning"
      class="flex items-center gap-3 rounded-control border px-3 py-2 text-micro"
      :class="
        error
          ? 'border-error/30 bg-error/5 text-error'
          : 'border-warning/30 bg-warning/10 text-warning'
      "
      role="alert"
    >
      <div class="min-w-0 flex-1">
        <p class="font-medium">
          {{ error ? "Couldn’t refresh the catalog." : "The catalog is catching up." }}
        </p>
        <p class="opacity-80">
          {{ error ?? providerErrors.map((item) => item.message).join(" ") }}
          <span v-if="providerErrors.length && displayEntries.length">
            Showing available models.</span
          >
        </p>
      </div>
      <button
        type="button"
        data-test="catalog-retry"
        class="shrink-0 rounded-control border border-current/40 px-2 py-1 font-medium hover:bg-current/10"
        :disabled="loading"
        @click="retrySearch"
      >
        {{ loading ? "Retrying…" : "Retry" }}
      </button>
    </div>

    <div
      v-if="selected.size > 0"
      class="border-border sticky top-2 z-20 flex flex-wrap items-center gap-3 rounded-control border bg-bg/95 p-2.5 shadow-md backdrop-blur"
      data-test="catalog-batch-bar"
      aria-live="polite"
    >
      <strong class="text-sm text-fg">{{ selected.size }} selected</strong>
      <template v-if="batchTargets.length">
        <label class="ml-auto flex items-center gap-2 text-micro text-fg-2">
          Target machine
          <select
            v-model="selectedTargetId"
            data-test="catalog-batch-target"
            :disabled="batchStarting"
            class="border-border h-8 rounded-control border bg-bg-deep px-2 text-micro text-fg"
          >
            <option value="" disabled>Choose a machine…</option>
            <option v-for="target in batchTargets" :key="target.host.id" :value="target.host.id">
              {{ target.host.label }} · {{ targetSummary(target.installCount, target.repairCount) }}
            </option>
          </select>
        </label>
        <button
          type="button"
          data-test="catalog-batch-download"
          class="h-8 rounded-control bg-accent px-3 text-micro font-semibold text-on-accent disabled:opacity-50"
          :disabled="!selectedBatchTarget || batchStarting"
          @click="startBatch"
        >
          {{ batchStarting ? "Starting…" : `Download ${selected.size}` }}
        </button>
      </template>
      <span v-else class="ml-auto text-micro text-error">
        No machine can receive every selected model.
      </span>
      <button
        type="button"
        class="border-border h-8 rounded-control border px-2.5 text-micro text-fg-2 hover:text-fg"
        :disabled="batchStarting"
        @click="selected = new Map()"
      >
        Clear
      </button>
    </div>

    <!-- Empty state — keyed on the FILTERED list so an all-image page under
         the Video chip explains itself instead of rendering a blank grid. -->
    <div
      v-if="!loading && displayEntries.length === 0 && !error && providerErrors.length === 0"
      class="p-8 text-center text-sm text-fg-2"
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
          class="border-border mt-3 h-7 rounded-control border px-2.5 text-micro text-fg-2 hover:text-fg"
          @click="emit('clear-media-filter')"
        >
          Show all media types
        </button>
      </template>
    </div>

    <!-- Results, installed first. Grid keeps preview cards; the table layout
         is the app-wide model-row shape — clean info, no thumbnails. -->
    <div
      v-if="loading || displayEntries.length > 0"
      :class="
        effectiveLayout === 'grid'
          ? 'grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-2'
          : 'border-border divide-border flex flex-col divide-y overflow-hidden rounded-control border bg-bg'
      "
    >
      <template v-for="entry in displayEntries" :key="entry.id">
        <CatalogCard
          v-if="effectiveLayout === 'grid'"
          :entry="entry"
          :pulling="pulling.has(entry.id)"
          :hosts="hostLabelsFor(entry)"
          :installable="installable(entry)"
          :selected="detailEntry?.id === entry.id"
          :selectable="!batchStarting && selectable(entry)"
          :checked="selected.has(entry.id)"
          :runtime-notice="runtimeNoticeFor(entry.id)"
          @pull="pull"
          @open="detailEntry = $event"
          @toggle-select="toggleSelection"
        />
        <CatalogTableRow
          v-else
          :entry="entry"
          :pulling="pulling.has(entry.id)"
          :hosts="hostLabelsFor(entry)"
          :installable="installable(entry)"
          :selected="detailEntry?.id === entry.id"
          :selectable="!batchStarting && selectable(entry)"
          :checked="selected.has(entry.id)"
          :runtime-notice="runtimeNoticeFor(entry.id)"
          class="px-3 py-2"
          @pull="pull"
          @open="detailEntry = $event"
          @toggle-select="toggleSelection"
        />
      </template>
    </div>

    <div
      v-if="hasMore"
      ref="sentinel"
      data-test="catalog-scroll-sentinel"
      class="flex h-8 items-center justify-center text-micro text-fg-2"
      aria-hidden="true"
    >
      {{ loading ? "Loading…" : "" }}
    </div>

    <DownloadTargetDialog
      v-if="pendingEntry"
      :model-name="pendingEntry.display_name ?? pendingEntry.name"
      :targets="actionTargets(pendingEntry)"
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
      :action="detailAction"
      :runtime-notice="runtimeNoticeFor(detailEntry.id)"
      @close="detailEntry = null"
      @pull="pullFromDrawer"
      @select-variant="selectDrawerVariant"
    />
  </div>
</template>
