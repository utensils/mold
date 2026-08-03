<script setup lang="ts">
/*
 * Gallery workspace (Mold Studio W5, spec §06 + prototype WEB GALLERY /
 * LIGHTBOX). Grid-first: session prints render as MediaTiles on a responsive
 * grid, with a secondary feed view kept for existing users. All / Images /
 * Video filter, a search that narrows prompt + model + filename (synced to
 * `?q=`), marquee multi-select + bulk delete, and a two-pane / full-screen
 * Lightbox with reuse / use-as-source / download / delete.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import ThumbnailSizeSlider from "@ui/components/ThumbnailSizeSlider.vue";
import {
  GALLERY_THUMBNAIL_SIZE_MAX,
  GALLERY_THUMBNAIL_SIZE_MIN,
  GALLERY_THUMBNAIL_SIZE_STEP,
  loadGalleryThumbnailSize,
  saveGalleryThumbnailSize,
} from "@studio/lib/galleryThumbnailSize";
import {
  ApiHttpError,
  deleteGalleryImage,
  fetchModels,
  getChainJob,
} from "../api";
import {
  planSequenceReuse,
  sequenceEditAvailability,
  sequenceGoneMessage,
  sequenceHostUnreachableMessage,
} from "@studio/lib/sequenceReuse";
import { useChainJobs } from "../composables/useChainJobs";
import { blobToBase64 } from "../lib/base64";
import { fetchGalleryBlob } from "../lib/galleryMedia";
import { requestConfirm, toast, undoableAction } from "../lib/toasts";
import {
  applyMetadataToForm,
  useGenerateForm,
} from "../composables/useGenerateForm";
import {
  fetchMergedGallery,
  printKey,
  type HostGalleryImage,
} from "../lib/multiHostGallery";
import {
  ORIGIN_HOST_ID,
  getHost,
  listHosts,
  originHost,
} from "../lib/hostRegistry";
import { hostDeleteGalleryImage } from "../components/machines/hostClient";
import type { GalleryImage, ModelInfoExtended } from "../types";
import { mediaKind } from "../types";
import GalleryGrid from "../components/gallery/GalleryGrid.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import HistoryDrawer from "../components/library/HistoryDrawer.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import { setSequenceHandoff } from "../composables/useSequenceHandoff";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  groupLogicalGalleryPrints,
  sameLogicalGalleryPrint,
} from "@studio/lib/galleryPrintIdentity";

type FilterKind = "all" | "images" | "video";
type ViewMode = "feed" | "grid";

// Persist the layout. The redesign is grid-first, so `grid` is the default;
// `feed` stays available as a single-column, prompt-forward stream.
const VIEW_STORAGE_KEY = "mold.gallery.view";
function loadViewMode(): ViewMode {
  try {
    const v = localStorage.getItem(VIEW_STORAGE_KEY);
    if (v === "feed" || v === "grid") return v;
  } catch {
    /* localStorage may be blocked */
  }
  return "grid";
}

const entries = ref<HostGalleryImage[]>([]);
/** Concrete device copies retained behind the deduplicated All view. */
const rawEntries = ref<HostGalleryImage[]>([]);
const models = ref<ModelInfoExtended[]>([]);
// Hosts whose /api/gallery failed this refresh (surfaced, not hidden), and
// how many non-origin hosts were attempted (drives the honest count line).
const unreachableHostIds = ref<string[]>([]);
const remoteHostCount = ref(0);
const loading = ref(true);
const errorMessage = ref<string | null>(null);
const filter = ref<FilterKind>("all");
const search = ref("");
const view = ref<ViewMode>(loadViewMode());
const hostFilter = ref("all");
const thumbnailSize = ref(loadGalleryThumbnailSize());

const form = useGenerateForm();
const draft = useSequenceDraftStore();
draft.hydrate();
const chainJobs = useChainJobs();
const route = useRoute();
const router = useRouter();

const filterOptions: SegmentOption<FilterKind>[] = [
  { value: "all", label: "All" },
  { value: "images", label: "Images" },
  { value: "video", label: "Video" },
];

// Seed the search from the global nav's `?q=` and keep the two in sync.
watch(
  () => route.query.q,
  (q) => {
    const next = typeof q === "string" ? q : "";
    if (next !== search.value) search.value = next;
  },
  { immediate: true },
);
watch(
  () => route.query.type,
  (type) => {
    if (type === "images" || type === "video" || type === "all")
      filter.value = type;
  },
  { immediate: true },
);

// History drawer state lives in the URL (?panel=history), so the Create
// activity digest can deep-link straight to its Sequences lens.
const historyOpen = computed(() => route.query.panel === "history");
function openHistory() {
  void router.push({
    query: { ...route.query, panel: "history", tab: "sequences" },
  });
}
function closeHistory() {
  const query = { ...route.query };
  delete query.panel;
  delete query.tab;
  void router.replace({ query });
}
/** Re-enter a durable sequence in Create: the job is watched (or edited)
 *  there, never in a Library drawer. */
function onOpenSequence(payload: {
  hostId: string;
  jobId: string;
  edit: boolean;
}) {
  closeHistory();
  setSequenceHandoff({
    kind: payload.edit ? "edit" : "inspect",
    hostId: payload.hostId,
    jobId: payload.jobId,
  });
  void router.push({ path: "/create", query: { output: "sequence" } });
}

function syncSearchToUrl(value: string) {
  const q = value.trim();
  const current = typeof route.query.q === "string" ? route.query.q : "";
  if (q === current) return;
  const query = { ...route.query };
  if (q) query.q = q;
  else delete query.q;
  void router.replace({ query });
}
function onSearchInput(value: string) {
  search.value = value;
  syncSearchToUrl(value);
}
function clearSearch() {
  search.value = "";
  syncSearchToUrl("");
}
function setFilter(next: FilterKind) {
  filter.value = next;
}

/*
 * Audio preference. Browsers forbid unmuted autoplay without a gesture, so we
 * default muted; the first toggle click doubles as that gesture. Persisted.
 */
const MUTED_STORAGE_KEY = "mold.gallery.muted";
function loadMuted(): boolean {
  try {
    const v = localStorage.getItem(MUTED_STORAGE_KEY);
    if (v === "false") return false;
    if (v === "true") return true;
  } catch {
    /* localStorage may be blocked */
  }
  return true;
}
const muted = ref<boolean>(loadMuted());
function setMuted(next: boolean) {
  muted.value = next;
  try {
    localStorage.setItem(MUTED_STORAGE_KEY, String(next));
  } catch {
    /* ignore */
  }
}

/*
 * Print identity is the (host, filename) pair, never the filename alone: mold
 * names outputs model+seed+timestamp, so two connected machines routinely hold
 * the same filename. Keying on the filename made those one print — selecting,
 * deleting or opening either one hit both, and the DELETE could land on the
 * wrong host. Everything below keys on `printKey(entry)`.
 */
const keyOf = (entry: GalleryImage) =>
  printKey(entry as { hostId?: string; filename: string });

function entryForKey(key: string): HostGalleryImage | null {
  return rawEntries.value.find((e) => keyOf(e) === key) ?? null;
}

/*
 * The host that owns a print. An entry tagged for a host the registry no longer
 * knows resolves to nothing rather than quietly falling back to the origin —
 * that fallback is how a remote print's delete or source-fetch lands on a
 * same-named local file.
 */
function hostForEntry(entry: GalleryImage) {
  const id = (entry as { hostId?: string }).hostId ?? ORIGIN_HOST_ID;
  if (id === ORIGIN_HOST_ID) return originHost();
  return getHost(id);
}

function missingHostError(entry: GalleryImage): Error {
  const label =
    (entry as { hostLabel?: string }).hostLabel ??
    (entry as { hostId?: string }).hostId ??
    "That host";
  return new Error(`${label} isn't connected anymore.`);
}

// ── NEW badge tracking ──────────────────────────────────────────────────────
// Prints present on first load are "seen" (never badged). Anything that
// arrives on a later refresh is fresh until the next reload.
const seen = new Set<string>();
const fresh = ref<Set<string>>(new Set());
let firstLoadDone = false;
function reconcileFresh(list: GalleryImage[]) {
  if (!firstLoadDone) {
    for (const e of list) seen.add(keyOf(e));
    firstLoadDone = true;
    return;
  }
  let changed = false;
  const next = new Set(fresh.value);
  for (const e of list) {
    const key = keyOf(e);
    if (!seen.has(key)) {
      seen.add(key);
      next.add(key);
      changed = true;
    }
  }
  if (changed) fresh.value = next;
}

// ── Multi-select ────────────────────────────────────────────────────────────
const selectMode = ref(false);
/** Selected print keys (`hostId|filename`). */
const selection = ref<Set<string>>(new Set());
const selectionAnchor = ref<string | null>(null);

function setSelectMode(next: boolean) {
  selectMode.value = next;
  if (!next) {
    selection.value = new Set();
    selectionAnchor.value = null;
  }
}

function toggleSelect(payload: {
  item: GalleryImage;
  shift: boolean;
  meta: boolean;
}) {
  const { item, shift, meta } = payload;
  const key = keyOf(item);
  if (shift && selectionAnchor.value) {
    const list = filtered.value;
    const a = list.findIndex((e) => keyOf(e) === selectionAnchor.value);
    const b = list.findIndex((e) => keyOf(e) === key);
    if (a === -1 || b === -1) {
      toggleOne(key, meta);
      return;
    }
    const [lo, hi] = a < b ? [a, b] : [b, a];
    const next = new Set(selection.value);
    for (let i = lo; i <= hi; i++) {
      const entry = list[i];
      if (entry) next.add(keyOf(entry));
    }
    selection.value = next;
    return;
  }
  toggleOne(key, meta);
  selectionAnchor.value = key;
}

function toggleOne(key: string, _meta: boolean) {
  const next = new Set(selection.value);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  selection.value = next;
}

function onDragSelect(payload: { keys: string[] }) {
  selection.value = new Set(payload.keys);
}

function selectAllVisible() {
  const next = new Set<string>();
  for (const e of filtered.value) next.add(keyOf(e));
  selection.value = next;
}

function clearSelection() {
  selection.value = new Set();
  selectionAnchor.value = null;
}

// A concrete copy still routes to its owning host. Callers expand one logical
// print to every matching device copy before invoking this primitive.
function deleteRouted(entry: GalleryImage): Promise<void> {
  const host = hostForEntry(entry);
  if (!host) return Promise.reject(missingHostError(entry));
  if (host.id === ORIGIN_HOST_ID) return deleteGalleryImage(entry.filename);
  return hostDeleteGalleryImage(host, entry.filename);
}

function copiesOf(entry: GalleryImage): HostGalleryImage[] {
  return rawEntries.value.filter((candidate) =>
    sameLogicalGalleryPrint(entry, candidate),
  );
}

function syncLogicalEntries(): void {
  rawEntries.value.sort((a, b) => b.timestamp - a.timestamp);
  entries.value = groupLogicalGalleryPrints(rawEntries.value).map(
    (group) => group.representative,
  );
}

async function handleDeleteMany(keys: string[]): Promise<number> {
  const selectedTargets = keys
    .map((key) => ({ key, entry: entryForKey(key) }))
    .filter((t): t is { key: string; entry: HostGalleryImage } => !!t.entry);
  const groups = selectedTargets.map((target) => copiesOf(target.entry));
  const targetsByKey = new Map<string, HostGalleryImage>();
  for (const group of groups) {
    for (const entry of group) targetsByKey.set(keyOf(entry), entry);
  }
  const targets = [...targetsByKey.entries()].map(([key, entry]) => ({
    key,
    entry,
  }));
  const results = await Promise.allSettled(
    targets.map((t) => deleteRouted(t.entry)),
  );
  const deleted = new Set<string>();
  let failed = 0;
  targets.forEach((t, i) => {
    if (results[i]?.status === "fulfilled") deleted.add(t.key);
    else failed++;
  });
  rawEntries.value = rawEntries.value.filter((e) => !deleted.has(keyOf(e)));
  syncLogicalEntries();
  if (deleted.size > 0) {
    const next = new Set(selection.value);
    for (const key of deleted) next.delete(key);
    selection.value = next;
  }
  const failedKeys = new Set(
    targets
      .filter((_, index) => results[index]?.status === "rejected")
      .map((target) => target.key),
  );
  const failedPrints = groups.filter((group) =>
    group.some((entry) => failedKeys.has(keyOf(entry))),
  ).length;
  const deletedPrints = selectedTargets.length - failedPrints;
  if (failed > 0) {
    toast(
      "error",
      `Deleted ${deletedPrints} of ${selectedTargets.length} prints everywhere. ${failedPrints} still have a copy on an unavailable device.`,
    );
  } else if (selectedTargets.length > 0) {
    toast(
      "success",
      selectedTargets.length === 1
        ? "Deleted print everywhere"
        : `Deleted ${selectedTargets.length} prints everywhere`,
    );
  }
  return deletedPrints;
}

async function deleteSelected() {
  const keys = Array.from(selection.value);
  if (keys.length === 0) return;
  const accepted = await requestConfirm({
    title:
      keys.length === 1 ? "Delete print?" : `Delete ${keys.length} prints?`,
    body: "Every matching copy on your connected devices will be deleted. This can't be undone.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!accepted) return;
  await handleDeleteMany(keys);
}

async function deleteAllFiltered() {
  const list = filtered.value;
  if (list.length === 0) return;
  const everything = list.length === entries.value.length;
  const accepted = await requestConfirm({
    title: everything
      ? `Delete all ${list.length} prints?`
      : `Delete ${list.length} filtered prints?`,
    body: "Every matching copy on your connected devices will be deleted. This can't be undone.",
    confirmLabel: "Delete",
    danger: true,
    typedPhrase: "delete",
  });
  if (!accepted) return;
  await handleDeleteMany(list.map((e) => keyOf(e)));
}

// ── Filtering ────────────────────────────────────────────────────────────────
const hostOptions = computed(() => {
  const options = new Map<string, string>();
  for (const entry of rawEntries.value) {
    const id = entry.hostId ?? ORIGIN_HOST_ID;
    options.set(id, entry.hostLabel ?? getHost(id)?.name ?? id);
  }
  return Array.from(options, ([id, label]) => ({ id, label }));
});

const hostFiltered = computed(() =>
  hostFilter.value === "all"
    ? entries.value
    : rawEntries.value.filter(
        (entry) => (entry.hostId ?? ORIGIN_HOST_ID) === hostFilter.value,
      ),
);

const kindFiltered = computed(() => {
  if (filter.value === "all") return hostFiltered.value;
  return hostFiltered.value.filter((e) => {
    const k = mediaKind(e.format, e.filename);
    if (filter.value === "video") return k === "video" || k === "animated";
    return k === "image";
  });
});

watch(hostOptions, (options) => {
  if (
    hostFilter.value !== "all" &&
    !options.some((option) => option.id === hostFilter.value)
  ) {
    hostFilter.value = "all";
  }
});

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return kindFiltered.value;
  return kindFiltered.value.filter((e) => {
    if (e.filename.toLowerCase().includes(q)) return true;
    const m = e.metadata;
    if (m.model.toLowerCase().includes(q)) return true;
    if (m.prompt && m.prompt.toLowerCase().includes(q)) return true;
    return false;
  });
});

const total = computed(() => entries.value.length);
const searchActive = computed(() => search.value.trim().length > 0);

// The empty-state variant to show when nothing is visible and we're not
// mid-load. `null` means render the grid/feed (skeletons handle first load).
const emptyKind = computed<null | "none" | "search" | "video" | "images">(
  () => {
    if (loading.value || filtered.value.length > 0) return null;
    if (searchActive.value) return "search";
    if (filter.value === "video") return "video";
    if (filter.value === "images") return "images";
    return "none";
  },
);

let refreshInFlight: Promise<void> | null = null;
async function performRefresh() {
  loading.value = true;
  errorMessage.value = null;
  try {
    const merged = await fetchMergedGallery(listHosts());
    rawEntries.value = merged.rawEntries;
    syncLogicalEntries();
    unreachableHostIds.value = merged.unreachableHostIds;
    remoteHostCount.value = merged.remoteHostCount;
    reconcileFresh(entries.value);
    // Only a total wipe-out (no host answered) is an error; one box down just
    // shows an "unreachable" note while the rest render.
    if (
      merged.reachableHostIds.length === 0 &&
      merged.unreachableHostIds.length > 0
    ) {
      errorMessage.value = "Couldn't reach any host's gallery.";
    }
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

function refresh(): Promise<void> {
  if (refreshInFlight) return refreshInFlight;
  const operation = performRefresh();
  refreshInFlight = operation;
  void operation.finally(() => {
    if (refreshInFlight === operation) refreshInFlight = null;
  });
  return operation;
}

// Honest count line: "all hosts" only when remotes are actually connected,
// otherwise "this server". Names the unreachable hosts rather than hiding them.
const scopeLabel = computed(() =>
  remoteHostCount.value > 0 ? "all hosts" : "this server",
);
const unreachableLabel = computed(() => {
  const names = unreachableHostIds.value
    .map((id) => getHost(id)?.name ?? id)
    .filter((n) => n !== originHost().name);
  return names.length ? `${names.join(", ")} unreachable` : "";
});

// ── Lightbox ─────────────────────────────────────────────────────────────────
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);

function openItem(item: GalleryImage) {
  const key = keyOf(item);
  selectedIndex.value = filtered.value.findIndex((e) => keyOf(e) === key);
  selected.value = item;
}
function closeLightbox() {
  selected.value = null;
  selectedIndex.value = -1;
}
function stepLightbox(delta: number) {
  if (selectedIndex.value < 0) return;
  const list = filtered.value;
  const next = selectedIndex.value + delta;
  if (next < 0 || next >= list.length) return;
  selectedIndex.value = next;
  selected.value = list[next] ?? null;
}

// ── Sequence prints ─────────────────────────────────────────────────────────
// A print stitched from a sequence carries per-clip provenance
// (`metadata.chain`) and, when a durable job produced it, that job's id.
// Reuse settings follows a still. A sequence's primary action CONTINUES the
// original durable job with its cached clips; Duplicate as new is the explicit
// fresh-draft path.

const isSequencePrint = (item: GalleryImage | null) =>
  item !== null && planSequenceReuse(item.metadata) !== null;

/** Render-time gate — never probes. See `sequenceEditAvailability`. */
function canEditSequence(item: GalleryImage | null): boolean {
  if (!item || !isSequencePrint(item)) return false;
  const host = hostForEntry(item);
  return (
    sequenceEditAvailability({
      chainJobId: item.metadata.chain_job_id,
      hostId: host?.id ?? null,
      knownJobIds: host
        ? (chainJobs.state.byHost[host.id]?.jobs.map((job) => job.id) ?? null)
        : null,
    }) === "available"
  );
}

function reuseSequence(item: GalleryImage) {
  // Clips are clamped against the LIVE model's motion tail in Create, so the
  // metadata travels and Create decides.
  setSequenceHandoff({ kind: "reuse", metadata: item.metadata });
  closeLightbox();
  void router.push({ path: "/create", query: { output: "sequence" } });
}

function onReuse(item: GalleryImage) {
  if (isSequencePrint(item)) {
    reuseSequence(item);
    return;
  }
  // A rendered non-sequence print is always a One shot. Switch before
  // restoring its metadata so Create's persisted Sequence mode and
  // sequence-only model guard cannot replace the recorded settings.
  draft.setOutput(
    "single",
    {
      getPrompt: () => form.state.value.prompt,
      setPrompt: (prompt) => (form.state.value.prompt = prompt),
    },
    25,
  );
  draft.stopEditing();
  draft.lastSingleModel = null;
  form.state.value = applyMetadataToForm(form.state.value, item.metadata, {
    format: item.format,
    models: models.value,
  });
  closeLightbox();
  void router.push({ name: "create" });
}

/**
 * Check once, on click. A 404 means the job was deleted or GC'd, so fall back
 * to the reuse path rather than leaving an enabled control as a dead end; any
 * other failure keeps the cached clips by refusing to downgrade.
 */
async function onEditSequence(item: GalleryImage) {
  const host = hostForEntry(item);
  const jobId = item.metadata.chain_job_id;
  if (!host || !jobId) return;
  try {
    await getChainJob(jobId, {
      baseUrl: host.url,
      ...(host.apiKey ? { apiKey: host.apiKey } : {}),
    });
  } catch (error) {
    if (error instanceof ApiHttpError && error.status === 404) {
      toast("info", sequenceGoneMessage(host.name));
      reuseSequence(item);
      return;
    }
    toast("error", sequenceHostUnreachableMessage(host.name));
    return;
  }
  setSequenceHandoff({ kind: "edit", hostId: host.id, jobId });
  closeLightbox();
  void router.push({ path: "/create", query: { output: "sequence" } });
}

async function setAsSource(item: GalleryImage): Promise<boolean> {
  try {
    // Bytes come from the host that owns the print, with that host's key —
    // fetching the origin would 404 or, worse, grab a same-named local file.
    const host = hostForEntry(item);
    if (!host) throw missingHostError(item);
    const blob = await fetchGalleryBlob(host, item.filename);
    const base64 = await blobToBase64(blob);
    form.state.value.imageAttachments = [
      { kind: "gallery", filename: item.filename, base64 },
    ];
    return true;
  } catch (err) {
    toast("error", err instanceof Error ? err.message : String(err));
    return false;
  }
}

async function onUseAsSource(item: GalleryImage) {
  if (!(await setAsSource(item))) return;
  closeLightbox();
  void router.push({ name: "create" });
}

async function onUpscale(item: GalleryImage) {
  if (!(await setAsSource(item))) return;
  closeLightbox();
  toast("info", "Added as source — pick an upscaler in Controls.");
  void router.push({ name: "create" });
}

async function onLightboxDelete(item: GalleryImage) {
  const accepted = await requestConfirm({
    title: "Delete print?",
    body: `${item.filename} will be deleted from every connected device. You can undo for a few seconds.`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!accepted) return;
  const key = keyOf(item);
  const entryIdx = rawEntries.value.findIndex((e) => keyOf(e) === key);
  if (entryIdx === -1) return;
  const removed = copiesOf(item);
  const removedKeys = new Set(removed.map((entry) => keyOf(entry)));

  // Optimistic removal; commit the DELETE only once the undo window elapses.
  rawEntries.value = rawEntries.value.filter((e) => !removedKeys.has(keyOf(e)));
  syncLogicalEntries();
  if (filtered.value.length === 0) {
    closeLightbox();
  } else {
    selectedIndex.value = Math.min(
      selectedIndex.value,
      filtered.value.length - 1,
    );
    selected.value = filtered.value[selectedIndex.value] ?? null;
    if (!selected.value) closeLightbox();
  }

  undoableAction({
    text: "Print deleted everywhere",
    undo: () => {
      rawEntries.value = [...rawEntries.value, ...removed];
      syncLogicalEntries();
    },
    commit: async () => {
      const results = await Promise.allSettled(
        removed.map((entry) => deleteRouted(entry)),
      );
      const failed = removed.filter(
        (_, index) => results[index]?.status === "rejected",
      );
      if (failed.length > 0) {
        rawEntries.value = [...rawEntries.value, ...failed];
        syncLogicalEntries();
        toast(
          "error",
          `${failed.length} device ${failed.length === 1 ? "copy remains" : "copies remain"} because a delete failed.`,
        );
      }
    },
  });
}

function setView(next: ViewMode) {
  view.value = next;
  try {
    localStorage.setItem(VIEW_STORAGE_KEY, next);
  } catch {
    /* ignore */
  }
}

function setThumbnailSize(next: number) {
  thumbnailSize.value = next;
  saveGalleryThumbnailSize(next);
}

// ── Tile context menu ───────────────────────────────────────────────────────
const contextMenu = ref<{
  item: GalleryImage;
  x: number;
  y: number;
} | null>(null);
function openContextMenu(payload: {
  item: GalleryImage;
  x: number;
  y: number;
}) {
  contextMenu.value = payload;
}
function closeContextMenu() {
  contextMenu.value = null;
}
function contextReuse() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) onReuse(item);
}
async function contextEditSequence() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) await onEditSequence(item);
}
async function contextSource() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) await onUseAsSource(item);
}
async function contextDelete() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) await onLightboxDelete(item);
}
function onDocumentPointerDown(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (!target?.closest("[data-test='gallery-context-menu']"))
    closeContextMenu();
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") closeContextMenu();
}

// ── Back-to-top FAB ──────────────────────────────────────────────────────────
const showBackToTop = ref(false);
function onScroll() {
  showBackToTop.value = window.scrollY > window.innerHeight;
}
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}
onMounted(() => {
  window.addEventListener("scroll", onScroll, { passive: true });
  document.addEventListener("pointerdown", onDocumentPointerDown);
  document.addEventListener("keydown", onDocumentKeydown);
});
onBeforeUnmount(() => {
  window.removeEventListener("scroll", onScroll);
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  document.removeEventListener("keydown", onDocumentKeydown);
});

let refreshTimer: ReturnType<typeof setInterval> | null = null;
let disposed = false;
onMounted(async () => {
  const [, listing] = await Promise.all([
    refresh(),
    fetchModels().catch(() => [] as ModelInfoExtended[]),
  ]);
  if (disposed) return;
  models.value = listing;
  refreshTimer = setInterval(() => {
    if (!document.hidden) void refresh();
  }, 10_000);
});
onBeforeUnmount(() => {
  disposed = true;
  if (refreshTimer) clearInterval(refreshTimer);
});
</script>

<template>
  <div class="gal">
    <header class="gal__head">
      <h1 class="gal__title">Gallery</h1>
      <span class="gal__count" data-test="gallery-count"
        >{{ total }} prints · {{ scopeLabel
        }}<span v-if="unreachableLabel" class="gal__unreachable">
          · {{ unreachableLabel }}</span
        ></span
      >
      <span class="gal__flex"></span>

      <ThumbnailSizeSlider
        v-if="view === 'grid'"
        class="gal__thumbnail-size"
        :model-value="thumbnailSize"
        :min="GALLERY_THUMBNAIL_SIZE_MIN"
        :max="GALLERY_THUMBNAIL_SIZE_MAX"
        :step="GALLERY_THUMBNAIL_SIZE_STEP"
        @update:model-value="setThumbnailSize"
      />

      <SegmentedControl
        class="gal__filter"
        data-test="gallery-filter"
        :model-value="filter"
        :options="filterOptions"
        label="Filter prints"
        @update:model-value="setFilter"
      />

      <div
        v-if="hostOptions.length > 1"
        class="gal__hosts"
        role="group"
        aria-label="Filter by machine"
      >
        <button
          v-for="option in [
            { id: 'all', label: 'All machines' },
            ...hostOptions,
          ]"
          :key="option.id"
          type="button"
          class="gal__host-chip"
          :data-on="hostFilter === option.id ? 'true' : undefined"
          data-test="gallery-host-filter"
          @click="hostFilter = option.id"
        >
          {{ option.label }}
        </button>
      </div>

      <label class="gal__search">
        <Icon name="search" :size="15" />
        <input
          :value="search"
          type="search"
          placeholder="Search prompts, models…"
          aria-label="Search gallery"
          data-test="gallery-search"
          @input="onSearchInput(($event.target as HTMLInputElement).value)"
        />
      </label>

      <div class="gal__tools">
        <button
          type="button"
          class="gal__icon"
          data-test="open-history"
          aria-label="History"
          title="History"
          :aria-pressed="historyOpen"
          @click="openHistory"
        >
          <Icon name="history" :size="16" />
        </button>

        <div class="gal__viewtoggle" role="group" aria-label="View mode">
          <button
            type="button"
            class="gal__vbtn"
            :data-on="view === 'grid' ? 'true' : undefined"
            :aria-pressed="view === 'grid'"
            aria-label="Grid view"
            @click="setView('grid')"
          >
            <Icon name="library" :size="16" />
          </button>
          <button
            type="button"
            class="gal__vbtn"
            :data-on="view === 'feed' ? 'true' : undefined"
            :aria-pressed="view === 'feed'"
            aria-label="Feed view"
            @click="setView('feed')"
          >
            <Icon name="menu" :size="16" />
          </button>
        </div>

        <button
          type="button"
          class="gal__icon"
          :aria-pressed="!muted"
          :aria-label="muted ? 'Unmute videos' : 'Mute videos'"
          :title="muted ? 'Unmute videos' : 'Mute videos'"
          @click="setMuted(!muted)"
        >
          <svg
            v-if="muted"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="m22 9-6 6" />
            <path d="m16 9 6 6" />
          </svg>
          <svg
            v-else
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="M15.5 8.5a5 5 0 0 1 0 7" />
            <path d="M18.5 5.5a9 9 0 0 1 0 13" />
          </svg>
        </button>

        <button
          type="button"
          class="gal__select"
          :class="{ 'gal__select--on': selectMode }"
          :aria-pressed="selectMode"
          data-test="gallery-select"
          @click="setSelectMode(!selectMode)"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"
            />
          </svg>
          <span class="gal__select-label">
            {{ selectMode ? `${selection.size} selected` : "Select" }}
          </span>
        </button>

        <button
          type="button"
          class="gal__icon"
          :disabled="loading"
          :aria-busy="loading"
          aria-label="Refresh gallery"
          @click="refresh"
        >
          <Icon name="refresh" :size="16" :class="{ gal__spin: loading }" />
        </button>
      </div>
    </header>

    <main class="gal__main">
      <div v-if="errorMessage" class="gal__error" role="alert">
        <p class="gal__error-title">Couldn't load the gallery.</p>
        <p class="gal__error-body">{{ errorMessage }}</p>
      </div>

      <template v-else>
        <EmptyStateBlock
          v-if="emptyKind === 'search'"
          icon="search"
          headline="No prints match"
          guidance="Nothing matches your search. Clear it to browse the full library."
        >
          <template #action>
            <button
              type="button"
              class="gal__emptybtn"
              data-test="clear-search"
              @click="clearSearch"
            >
              Clear search
            </button>
          </template>
        </EmptyStateBlock>

        <EmptyStateBlock
          v-else-if="emptyKind === 'video'"
          icon="video"
          headline="No video clips yet"
          guidance="Generate with an LTX Video model to see clips here."
        />

        <EmptyStateBlock
          v-else-if="emptyKind === 'images'"
          icon="image"
          headline="No images yet"
          guidance="Generate an image and it will appear in your library."
        />

        <EmptyStateBlock
          v-else-if="emptyKind === 'none'"
          icon="image"
          headline="No prints yet"
          guidance="Head to Create — every image and clip you make lands here."
        />

        <GalleryGrid
          v-else-if="view === 'grid'"
          :entries="filtered"
          :models="models"
          :loading="loading"
          :thumbnail-size="thumbnailSize"
          :select-mode="selectMode"
          :selection="selection"
          :fresh="fresh"
          @open="openItem"
          @toggle-select="toggleSelect"
          @drag-select="onDragSelect"
          @context-menu="openContextMenu"
        />

        <GalleryFeed
          v-else
          :entries="filtered"
          :models="models"
          :loading="loading"
          :view="'feed'"
          :muted="muted"
          :select-mode="selectMode"
          :selection="selection"
          @open="openItem"
          @toggle-select="toggleSelect"
          @drag-select="onDragSelect"
        />
      </template>
    </main>

    <div
      v-if="contextMenu"
      class="gal__context"
      data-test="gallery-context-menu"
      role="menu"
      :style="{ left: `${contextMenu.x}px`, top: `${contextMenu.y}px` }"
    >
      <button
        type="button"
        role="menuitem"
        @click="
          openItem(contextMenu.item);
          closeContextMenu();
        "
      >
        Open
      </button>
      <button
        v-if="canEditSequence(contextMenu.item)"
        type="button"
        role="menuitem"
        data-test="context-edit-sequence"
        @click="contextEditSequence"
      >
        Edit sequence
      </button>
      <button type="button" role="menuitem" @click="contextReuse">
        {{
          isSequencePrint(contextMenu.item)
            ? "Duplicate as new"
            : "Reuse settings"
        }}
      </button>
      <button type="button" role="menuitem" @click="contextSource">
        Use as source
      </button>
      <button
        type="button"
        role="menuitem"
        class="gal__context-danger"
        @click="contextDelete"
      >
        Delete
      </button>
    </div>

    <!-- Selection action bar (bulk delete). -->
    <Transition name="fade">
      <div
        v-if="selectMode"
        class="gal__bar-wrap"
        :style="{
          bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
        }"
      >
        <div class="gal__bar" role="toolbar" aria-label="Selection actions">
          <span class="gal__bar-count">
            {{ selection.size }}
            <span class="gal__bar-of">/ {{ filtered.length }} selected</span>
          </span>
          <button
            type="button"
            class="gal__bar-btn"
            :disabled="filtered.length === 0"
            @click="selectAllVisible"
          >
            Select all
          </button>
          <button
            type="button"
            class="gal__bar-btn"
            :disabled="selection.size === 0"
            @click="clearSelection"
          >
            Clear
          </button>
          <button
            type="button"
            class="gal__bar-danger"
            :disabled="selection.size === 0"
            @click="deleteSelected"
          >
            Delete selected
          </button>
          <button
            type="button"
            class="gal__bar-danger gal__bar-danger--soft"
            :disabled="filtered.length === 0"
            @click="deleteAllFiltered"
          >
            Delete all
          </button>
          <button
            type="button"
            class="gal__bar-x"
            aria-label="Exit select mode"
            @click="setSelectMode(false)"
          >
            <Icon name="close" :size="15" />
          </button>
        </div>
      </div>
    </Transition>

    <Transition name="fade">
      <button
        v-if="showBackToTop"
        type="button"
        aria-label="Scroll to top"
        class="gal__fab"
        :style="{
          bottom: 'max(0.75rem, env(safe-area-inset-bottom))',
        }"
        @click="scrollToTop"
      >
        <Icon name="chevron-up" :size="20" :stroke-width="2.2" />
      </button>
    </Transition>

    <Lightbox
      :item="selected"
      :models="models"
      :index="selectedIndex"
      :total="filtered.length"
      :has-prev="selectedIndex > 0"
      :has-next="selectedIndex >= 0 && selectedIndex < filtered.length - 1"
      :muted="muted"
      :is-sequence="isSequencePrint(selected)"
      :can-edit-sequence="canEditSequence(selected)"
      @close="closeLightbox"
      @prev="stepLightbox(-1)"
      @next="stepLightbox(1)"
      @reuse="onReuse"
      @use-source="onUseAsSource"
      @upscale="onUpscale"
      @delete="onLightboxDelete"
      @edit-sequence="onEditSequence"
    />

    <HistoryDrawer
      :open="historyOpen"
      @close="closeHistory"
      @open-sequence="onOpenSequence"
    />
  </div>
</template>

<style scoped>
.gal {
  position: relative;
  width: 100%;
  max-width: 1800px;
  margin: 0 auto;
  padding: 22px 20px 160px;
  box-sizing: border-box;
}

.gal__head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 14px;
  margin-bottom: 18px;
}
.gal__title {
  margin: 0;
  font-family: var(--f-display);
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}
.gal__count {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.gal__flex {
  flex: 1;
  min-width: 0;
}

.gal__filter {
  flex: 0 0 auto;
}
.gal__thumbnail-size {
  flex: 0 0 136px;
}
.gal__hosts {
  display: flex;
  gap: 6px;
  overflow-x: auto;
}
.gal__host-chip {
  min-height: 34px;
  padding: 0 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 10px;
  white-space: nowrap;
  cursor: pointer;
}
.gal__host-chip[data-on="true"] {
  border-color: var(--safelight);
  color: var(--rebate);
  background: var(--sel-bg);
}

.gal__context {
  position: fixed;
  z-index: 70;
  display: grid;
  min-width: 170px;
  padding: 6px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: var(--bench);
  box-shadow: var(--shadow-popover);
}
.gal__context button {
  min-height: 40px;
  padding: 0 10px;
  border: 0;
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  text-align: left;
  cursor: pointer;
}
.gal__context button:hover {
  background: var(--sel-bg);
}
.gal__context-danger {
  color: var(--stop) !important;
}

.gal__search {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-3);
}
.gal__search input {
  width: 180px;
  max-width: 42vw;
  border: 0;
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13px;
  outline: none;
}
.gal__search input::placeholder {
  color: var(--ink-3);
}
.gal__search:focus-within {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

.gal__tools {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 639px) {
  .gal__thumbnail-size {
    display: none;
  }
}

.gal__viewtoggle {
  display: inline-flex;
  gap: 3px;
  padding: 3px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
}
.gal__vbtn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 28px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.gal__vbtn[data-on="true"] {
  background: var(--sel-bg);
  color: var(--sel-ink);
}
.gal__vbtn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.gal__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}
.gal__icon svg {
  width: 16px;
  height: 16px;
}
.gal__icon:hover:not(:disabled) {
  color: var(--rebate);
}
.gal__icon:disabled {
  opacity: 0.6;
  cursor: default;
}
.gal__icon:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.gal__spin {
  animation: gal-spin 0.9s linear infinite;
}
@keyframes gal-spin {
  to {
    transform: rotate(360deg);
  }
}

.gal__select {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 13px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}
.gal__select svg {
  width: 16px;
  height: 16px;
}
.gal__select:hover {
  color: var(--rebate);
}
.gal__select--on {
  background: var(--sel-bg);
  color: var(--sel-ink);
  border-color: var(--sel-border);
}
.gal__select:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.gal__main {
  margin-top: 4px;
}

.gal__error {
  background: color-mix(in srgb, var(--stop) 12%, var(--bench));
  border: 1px solid color-mix(in srgb, var(--stop) 40%, transparent);
  border-radius: var(--radius-card);
  padding: 14px 16px;
}
.gal__error-title {
  margin: 0;
  font-weight: 600;
  color: var(--rebate);
}
.gal__error-body {
  margin: 4px 0 0;
  font-size: 13px;
  color: var(--ink-2);
}

.gal__emptybtn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 9px 16px;
  border-radius: var(--radius-control);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}
.gal__emptybtn:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}
.gal__emptybtn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* Selection action bar. */
.gal__bar-wrap {
  position: fixed;
  inset-inline: 0;
  z-index: 40;
  display: flex;
  justify-content: center;
  padding: 0 16px;
  pointer-events: none;
}
.gal__bar {
  pointer-events: auto;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  max-width: 100%;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-pill);
  box-shadow: var(--shadow-raised);
  padding: 8px 12px;
  font-size: 13px;
  color: var(--rebate);
}
.gal__bar-count {
  padding: 0 6px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.gal__bar-of {
  color: var(--ink-3);
}
.gal__bar-btn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 6px 12px;
  border-radius: var(--radius-pill);
  font-weight: 600;
  cursor: pointer;
}
.gal__bar-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.gal__bar-danger {
  border: 0;
  background: var(--stop);
  color: #fff;
  padding: 6px 12px;
  border-radius: var(--radius-pill);
  font-weight: 700;
  cursor: pointer;
}
.gal__bar-danger--soft {
  background: color-mix(in srgb, var(--stop) 22%, transparent);
  color: var(--stop);
}
.gal__bar-danger:disabled {
  opacity: 0.5;
  cursor: default;
}
.gal__bar-x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
}
.gal__bar-x:hover {
  background: color-mix(in srgb, var(--rebate) 8%, transparent);
  color: var(--rebate);
}

.gal__fab {
  position: fixed;
  right: 20px;
  z-index: 20;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 46px;
  height: 46px;
  border: 0;
  border-radius: 50%;
  background: var(--safelight);
  color: var(--on-accent);
  box-shadow: var(--shadow-raised);
  cursor: pointer;
}
.gal__fab:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
