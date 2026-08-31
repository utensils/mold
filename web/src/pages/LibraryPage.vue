<script setup lang="ts">
/*
 * Library workspace (Mold Studio W5, spec §06 + V3 "Shelf"). One workspace,
 * three scopes in its header — **Prints | Collections | Trash** — synced to
 * `?scope=` (+ `?c=<slug>` for a collection drill-in, `?tag=a,b`, `?fav=1`).
 *
 *   Prints       today's grid/feed plus a filter-chip row (♥ Favorites, tag
 *                chips, host chips), All / Images / Video / Audio, search over
 *                prompt + model + filename + title + tags (`?q=`), marquee
 *                multi-select with a bulk bar (Add to collection · Tag ·
 *                ♥ Favorite · Trash), and the two-pane / full-screen Lightbox
 *                whose aside is the one editing surface.
 *   Collections  a shelf of cover-mosaic cards merged across hosts by name,
 *                a dashed "New collection" card, and a breadcrumb drill-in
 *                with Edit (rename / set cover / remove prints / delete).
 *   Trash        the same grid with a retention banner, per-tile purge
 *                countdown, Restore / Delete forever, and header Empty trash.
 *
 * Organization lives per host and fans out to every copy of a logical print
 * (`lib/libraryOrganization`); a host without `gallery.organize` /
 * `gallery.trash` contributes nothing and keeps the hard-delete wording.
 * Destructive copy is deliberately plain: Empty trash / Delete forever use
 * `requestConfirm` with a danger button and never a typed phrase.
 */
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import ThumbnailSizeSlider from "@ui/components/ThumbnailSizeSlider.vue";
import Popover from "@ui/components/Popover.vue";
import UpscaleDialog from "@ui/components/UpscaleDialog.vue";
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
import {
  tagKey,
  trashRetentionSummary,
  visibleTagCounts,
  rememberSessionScroll,
  sessionScrollPosition,
  type MergedCollection,
  type OrganizationMutation,
} from "@studio/lib/libraryOrganization";
import { defaultSourceFitPolicy } from "@studio/lib/sourceFit";
import { useChainJobs } from "../composables/useChainJobs";
import { blobToBase64 } from "../lib/base64";
import { fetchGalleryBlob } from "../lib/galleryMedia";
import {
  requestConfirm,
  requestText,
  toast,
  undoableAction,
} from "../lib/toasts";
import {
  applyMetadataToForm,
  useGenerateForm,
} from "../composables/useGenerateForm";
import {
  decorateEntries,
  fetchMergedGallery,
  mergeLogicalEntries,
  printKey,
  type HostGalleryImage,
} from "../lib/multiHostGallery";
import {
  anyHostOrganizes,
  anyHostTrashes,
  applyOrganizationMutation,
  collectionCards,
  collectionResolver,
  createCollectionOn,
  deleteCollectionEverywhere,
  emptyTrashEverywhere,
  entryMatchesSearch,
  fetchOrganization,
  filterByOrganization,
  hostOrganizes,
  hostTrashes,
  mergedCollections,
  mergedTags,
  renameCollectionEverywhere,
  setCollectionHiddenEverywhere,
  retentionHosts,
  setCollectionCover,
  type FanoutResult,
  type HostOrganizationSnapshot,
} from "../lib/libraryOrganization";
import {
  ORIGIN_HOST_ID,
  getHost,
  listHosts,
  originHost,
} from "../lib/hostRegistry";
import {
  hostCapabilities,
  hostDeleteGalleryImage,
} from "../components/machines/hostClient";
import {
  createFramewiseUpscale,
  findRecoverableFramewiseUpscale,
  getFramewiseUpscale,
  transitionFramewiseUpscale,
  upscaleLibraryImage,
  type VideoUpscaleJob,
} from "@studio/api/videoUpscale";
import {
  defaultUpscaler,
  framewiseProgress,
  framewiseStatus,
  libraryUpscaleLabel,
  shouldPollFramewiseJob,
} from "@studio/lib/upscale";
import { selectUpscalers } from "../components/create/advanced/upscalers";
import type { GalleryImage, ModelInfoExtended } from "../types";
import { mediaKind } from "../types";
import GalleryGrid from "../components/gallery/GalleryGrid.vue";
import GalleryFeed from "../components/GalleryFeed.vue";
import HistoryDrawer from "../components/library/HistoryDrawer.vue";
import LibraryChipRow from "../components/library/LibraryChipRow.vue";
import CollectionsShelf from "../components/library/CollectionsShelf.vue";
import CollectionPicker, {
  type CollectionPickerRow,
} from "../components/library/CollectionPicker.vue";
import TagEditor from "../components/library/TagEditor.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import { setSequenceHandoff } from "../composables/useSequenceHandoff";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { groupLogicalGalleryPrints } from "@studio/lib/galleryPrintIdentity";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { restoreGenerationSourceMedia } from "@studio/lib/generationSourceMedia";
import {
  retainedSourceMediaBlob,
  retainedSourceMediaDisclosure,
  retainedSourceMediaInventory,
  type RetainedSourceMediaAvailability,
} from "@studio/api/gallerySourceMedia";
import {
  beginRetainedSourceReuseIntent,
  retainedSourceReuseIsCurrent,
  setRetainedSourceReuseIntentIfCurrent,
} from "../lib/retainedSourceReuse";
import {
  appendMinimaxH3GalleryImageReference,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  setMinimaxH3GalleryImageFirstFrame,
} from "@studio/lib/minimaxH3Authoring";

const WEB_LIBRARY_SCROLL_KEY = "web:library";

type FilterKind = "all" | "images" | "video" | "audio";
type ViewMode = "feed" | "grid";
type Scope = "prints" | "collections" | "trash";

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
/** Per-host organization: capabilities, collections, tags, trash listing. */
const snapshots = ref<HostOrganizationSnapshot[]>([]);
const galleryImageUpscaleByHost = ref<Record<string, boolean>>({});
/** Copies moved to the trash locally (undo window elapsed) until a
 * SUCCESSFUL server listing confirms them (or they age out). */
const pendingTrashed = ref<HostGalleryImage[]>([]);
/** A shadow with no server confirmation is dropped after this bound so a
 * host that never lists it can't pin a phantom row forever. */
const PENDING_TRASH_MAX_AGE_SECS = 15 * 60;
/** Print keys optimistically removed while their delete/trash commit waits
 * out the undo window. The 10 s poll masks these so a refresh can't
 * resurrect still-live rows mid-window (codex review). */
const pendingRemovalKeys = ref<Set<string>>(new Set());
function addPendingRemovals(keys: Iterable<string>) {
  const next = new Set(pendingRemovalKeys.value);
  for (const key of keys) next.add(key);
  pendingRemovalKeys.value = next;
}
function clearPendingRemovals(keys: Iterable<string>) {
  const next = new Set(pendingRemovalKeys.value);
  for (const key of keys) next.delete(key);
  pendingRemovalKeys.value = next;
}
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
const scope = ref<Scope>("prints");
const collectionSlug = ref<string | null>(null);
const favoritesOnly = ref(false);
const tagFilter = ref<string[]>([]);

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
  { value: "audio", label: "Audio" },
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
// Scope, drill-in, tag, and favorite filters live in the URL too so the
// palette and History can deep-link straight into a collection or the trash.
watch(
  () => route.query.scope,
  (value) => {
    if (value === "prints" || value === "collections" || value === "trash") {
      if (scope.value !== value) scope.value = value;
    } else if (value === undefined && scope.value !== "prints") {
      scope.value = "prints";
    }
  },
  { immediate: true },
);
watch(
  () => route.query.c,
  (value) => {
    const next = typeof value === "string" && value ? value : null;
    if (next !== collectionSlug.value) collectionSlug.value = next;
  },
  { immediate: true },
);
watch(
  () => route.query.tag,
  (value) => {
    const next =
      typeof value === "string" && value
        ? value
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean)
        : [];
    if (next.join(" ") !== tagFilter.value.join(" ")) tagFilter.value = next;
  },
  { immediate: true },
);
watch(
  () => route.query.fav,
  (value) => {
    const next = value === "1" || value === "true";
    if (next !== favoritesOnly.value) favoritesOnly.value = next;
  },
  { immediate: true },
);

function syncOrganizationToUrl() {
  const query = { ...route.query };
  if (scope.value === "prints") delete query.scope;
  else query.scope = scope.value;
  if (scope.value === "collections" && collectionSlug.value)
    query.c = collectionSlug.value;
  else delete query.c;
  if (tagFilter.value.length > 0) query.tag = tagFilter.value.join(",");
  else delete query.tag;
  if (favoritesOnly.value) query.fav = "1";
  else delete query.fav;
  void router.replace({ query });
}

function setScope(next: Scope) {
  if (scope.value === next) return;
  scope.value = next;
  if (next !== "collections") collectionSlug.value = null;
  clearSelection();
  closeLightbox();
  syncOrganizationToUrl();
}
function openCollection(slug: string | null) {
  scope.value = "collections";
  collectionSlug.value = slug;
  clearSelection();
  closeLightbox();
  syncOrganizationToUrl();
}
function toggleFavoritesOnly() {
  favoritesOnly.value = !favoritesOnly.value;
  syncOrganizationToUrl();
}
function toggleTag(tag: string) {
  const key = tagKey(tag);
  const next = tagFilter.value.filter((t) => tagKey(t) !== key);
  if (next.length === tagFilter.value.length) next.push(tag);
  tagFilter.value = next;
  syncOrganizationToUrl();
}
function clearOrganizationFilters() {
  favoritesOnly.value = false;
  tagFilter.value = [];
  syncOrganizationToUrl();
}

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

/** Every physical copy the current scope can route to. */
const scopeRaw = computed<HostGalleryImage[]>(() =>
  scope.value === "trash" ? trashRaw.value : rawEntries.value,
);

function entryForKey(key: string): HostGalleryImage | null {
  return scopeRaw.value.find((e) => keyOf(e) === key) ?? null;
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
const hostById = (id: string) =>
  id === ORIGIN_HOST_ID ? originHost() : getHost(id);

function missingHostError(entry: GalleryImage): Error {
  const label =
    (entry as { hostLabel?: string }).hostLabel ??
    (entry as { hostId?: string }).hostId ??
    "That host";
  return new Error(`${label} isn't connected anymore.`);
}

// ── Organization state (merged across hosts) ───────────────────────────────
const resolver = computed(() => collectionResolver(snapshots.value));
const collections = computed(() => mergedCollections(snapshots.value));
const hiddenCollectionSlugs = computed(
  () =>
    new Set(
      collections.value
        .filter((collection) => collection.hidden)
        .map((collection) => collection.slug),
    ),
);
const tags = computed(() => mergedTags(snapshots.value));
const filterChipTags = computed(() => {
  const visible = filterByOrganization(entries.value, {
    excludeCollectionSlugs:
      scope.value === "prints" ? hiddenCollectionSlugs.value : undefined,
  });
  const visibleKeys = new Set(visible.map((entry) => keyOf(entry)));
  const excluded =
    scope.value === "prints"
      ? entries.value.filter((entry) => !visibleKeys.has(keyOf(entry)))
      : [];
  return visibleTagCounts(tags.value, visible, excluded);
});
const canOrganize = computed(() => anyHostOrganizes(snapshots.value));
const canTrash = computed(() => anyHostTrashes(snapshots.value));
const organizingHostCount = computed(
  () => snapshots.value.filter((s) => s.organize).length,
);
const trashRaw = computed<HostGalleryImage[]>(() => {
  const listed = snapshots.value.flatMap((s) => s.trashed);
  const listedKeys = new Set(listed.map((e) => keyOf(e)));
  return [
    ...listed,
    ...pendingTrashed.value.filter((e) => !listedKeys.has(keyOf(e))),
  ];
});
const trashEntries = computed(() =>
  mergeLogicalEntries(trashRaw.value, {
    resolveCollectionSlug: resolver.value,
  }),
);
const cards = computed(() =>
  collectionCards(
    collections.value,
    entries.value,
    snapshots.value,
    rawEntries.value,
  ),
);
const currentCollection = computed<MergedCollection | null>(() =>
  scope.value === "collections" && collectionSlug.value
    ? (collections.value.find((c) => c.slug === collectionSlug.value) ?? null)
    : null,
);
const currentCard = computed(() =>
  currentCollection.value
    ? (cards.value.find((c) => c.slug === currentCollection.value!.slug) ??
      null)
    : null,
);
const favoriteCount = computed(
  () =>
    filterByOrganization(entries.value, {
      excludeCollectionSlugs:
        scope.value === "prints" ? hiddenCollectionSlugs.value : undefined,
    }).filter((entry) => entry.favorite).length,
);
const retentionSummary = computed(() =>
  trashRetentionSummary(retentionHosts(snapshots.value)),
);

/** Some copy of this logical print can be titled / tagged / favorited. */
function canOrganizeEntry(entry: GalleryImage | null): boolean {
  if (!entry) return false;
  return copiesOf(entry).some((copy) =>
    hostOrganizes(snapshots.value, copy.hostId),
  );
}
/** Every copy goes to a trash on delete (else the old hard delete). */
function allCopiesTrash(entry: GalleryImage | null): boolean {
  if (!entry) return false;
  const copies = copiesOf(entry);
  return (
    copies.length > 0 &&
    copies.every((copy) => hostTrashes(snapshots.value, copy.hostId))
  );
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
// print to every matching device copy before invoking this primitive. On a
// trash-capable host the same DELETE moves the print to the trash; on an older
// host it is permanent.
function deleteRouted(entry: GalleryImage): Promise<void> {
  const host = hostForEntry(entry);
  if (!host) return Promise.reject(missingHostError(entry));
  if (host.id === ORIGIN_HOST_ID) return deleteGalleryImage(entry.filename);
  return hostDeleteGalleryImage(host, entry.filename);
}

function copiesOf(entry: GalleryImage): HostGalleryImage[] {
  const key = keyOf(entry);
  return (
    groupLogicalGalleryPrints(scopeRaw.value).find((group) =>
      group.copies.some((copy) => keyOf(copy) === key),
    )?.copies ?? []
  );
}

/** Expand many logical keys to every distinct physical copy. */
function copiesOfKeys(keys: readonly string[]): HostGalleryImage[] {
  const byKey = new Map<string, HostGalleryImage>();
  for (const key of keys) {
    const entry = entryForKey(key);
    if (!entry) continue;
    for (const copy of copiesOf(entry)) byKey.set(keyOf(copy), copy);
  }
  return [...byKey.values()];
}

function syncLogicalEntries(): void {
  rawEntries.value.sort((a, b) => b.timestamp - a.timestamp);
  entries.value = mergeLogicalEntries(rawEntries.value, {
    resolveCollectionSlug: resolver.value,
  });
  reselectCurrent();
}
watch(resolver, () => {
  if (rawEntries.value.length > 0) syncLogicalEntries();
});

/** Re-add optimistically removed copies, deduped by key — a refresh that
 * raced the undo window may already have re-added the live rows. */
function restoreRemovedEntries(removed: readonly HostGalleryImage[]) {
  const present = new Set(rawEntries.value.map((e) => keyOf(e)));
  const missing = removed.filter((e) => !present.has(keyOf(e)));
  if (missing.length > 0) rawEntries.value = [...rawEntries.value, ...missing];
  syncLogicalEntries();
}

/** Keep the Lightbox pointed at the freshest object for its print. */
function reselectCurrent() {
  if (!selected.value) return;
  const key = keyOf(selected.value);
  const next = filtered.value.find((e) => keyOf(e) === key);
  if (next) selected.value = next;
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
  const trashedNow = targets
    .filter((t) => deleted.has(t.key))
    .map((t) => t.entry)
    .filter((entry) => hostTrashes(snapshots.value, entry.hostId));
  rawEntries.value = rawEntries.value.filter((e) => !deleted.has(keyOf(e)));
  syncLogicalEntries();
  markTrashedLocally(trashedNow);
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
  // Copies that hard-deleted on a host without a trash. "Trashed" wording is
  // truthful only when EVERY successful copy landed in a trash — a mixed
  // fleet names its permanent deletions instead (codex review).
  const hardDeletedNow = targets
    .filter((t) => deleted.has(t.key))
    .map((t) => t.entry)
    .filter((entry) => !hostTrashes(snapshots.value, entry.hostId));
  if (failed > 0) {
    toast(
      "error",
      `Deleted ${deletedPrints} of ${selectedTargets.length} prints everywhere. ${failedPrints} still have a copy on an unavailable device.`,
    );
  } else if (selectedTargets.length > 0) {
    if (trashedNow.length > 0 && hardDeletedNow.length > 0) {
      const copies =
        hardDeletedNow.length === 1
          ? "1 copy on an older machine was"
          : `${hardDeletedNow.length} copies on older machines were`;
      toast(
        "success",
        `Trashed ${deletedPrints} ${deletedPrints === 1 ? "print" : "prints"}; ${copies} deleted permanently.`,
      );
    } else {
      const verb = trashedNow.length > 0 ? "Trashed" : "Deleted";
      toast(
        "success",
        selectedTargets.length === 1
          ? `${verb} print everywhere`
          : `${verb} ${selectedTargets.length} prints everywhere`,
      );
    }
  }
  return deletedPrints;
}

/** Copies whose DELETE succeeded on a trash-capable host show up in the Trash
 * scope immediately, stamped with that host's retention. */
function markTrashedLocally(copies: readonly HostGalleryImage[]) {
  if (copies.length === 0) return;
  const now = Math.floor(Date.now() / 1000);
  const stamped = copies.map((copy) => {
    const retention =
      snapshots.value.find((s) => s.hostId === copy.hostId)?.trash
        ?.retentionDays ?? 0;
    const next: HostGalleryImage = { ...copy, trashed_at: now };
    if (retention > 0) next.purge_at = now + retention * 86_400;
    return next;
  });
  pendingTrashed.value = [...pendingTrashed.value, ...stamped];
}

/** Trash (or delete, on an older host) every copy of the selected prints. */
async function deleteSelected() {
  const keys = Array.from(selection.value);
  if (keys.length === 0) return;
  const copies = copiesOfKeys(keys);
  const reversible =
    copies.length > 0 &&
    copies.every((copy) => hostTrashes(snapshots.value, copy.hostId));
  if (reversible) {
    // Reversible: no confirm — the prints wait in the Trash, and the toast
    // offers the 6 s undo before the request even fires. The keys stay
    // masked from refresh until undo or commit settles (codex review).
    const removedKeys = new Set(copies.map((copy) => keyOf(copy)));
    const removed = rawEntries.value.filter((e) => removedKeys.has(keyOf(e)));
    addPendingRemovals(removedKeys);
    rawEntries.value = rawEntries.value.filter(
      (e) => !removedKeys.has(keyOf(e)),
    );
    syncLogicalEntries();
    clearSelection();
    undoableAction({
      text:
        keys.length === 1
          ? "Moved to trash"
          : `Moved ${keys.length} prints to the trash`,
      undo: () => {
        clearPendingRemovals(removedKeys);
        restoreRemovedEntries(removed);
      },
      commit: async () => {
        try {
          const result = await applyOrganizationMutation(
            removed,
            { kind: "trash" },
            mutationContext(),
          );
          const failedHosts = new Set(result.failed.map((f) => f.hostId));
          const failed = removed.filter((e) => failedHosts.has(e.hostId));
          if (failed.length > 0) restoreRemovedEntries(failed);
          markTrashedLocally(removed.filter((e) => !failedHosts.has(e.hostId)));
          reportFanout(result, "move to the trash");
        } finally {
          clearPendingRemovals(removedKeys);
        }
      },
    });
    return;
  }
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
  const copies = copiesOfKeys(list.map((e) => keyOf(e)));
  const reversible =
    copies.length > 0 &&
    copies.every((copy) => hostTrashes(snapshots.value, copy.hostId));
  const accepted = await requestConfirm({
    title: reversible
      ? everything
        ? `Move all ${list.length} prints to the trash?`
        : `Move ${list.length} filtered prints to the trash?`
      : everything
        ? `Delete all ${list.length} prints?`
        : `Delete ${list.length} filtered prints?`,
    body: reversible
      ? "Every matching copy on your connected devices moves to that device's trash. You can restore them until they're purged."
      : "Every matching copy on your connected devices will be deleted. This can't be undone.",
    confirmLabel: reversible ? "Move to trash" : "Delete",
    danger: true,
  });
  if (!accepted) return;
  await handleDeleteMany(list.map((e) => keyOf(e)));
}

// ── Organization mutations ──────────────────────────────────────────────────
function mutationContext() {
  return { hostById, snapshots: snapshots.value };
}

function reportFanout(result: FanoutResult, what: string) {
  if (result.failed.length === 0) return;
  const names = result.failed
    .map((f) => hostById(f.hostId)?.name ?? f.hostId)
    .join(", ");
  toast("error", `Couldn't ${what} on ${names}: ${result.failed[0]!.error}`);
}

/** Optimistically patch every copy of a logical print in place. */
function patchCopies(
  copies: readonly HostGalleryImage[],
  patch: (copy: HostGalleryImage) => HostGalleryImage,
) {
  const keys = new Set(copies.map((copy) => keyOf(copy)));
  const apply = (list: HostGalleryImage[]) =>
    list.map((entry) => (keys.has(keyOf(entry)) ? patch(entry) : entry));
  rawEntries.value = apply(rawEntries.value);
  pendingTrashed.value = apply(pendingTrashed.value);
  snapshots.value = snapshots.value.map((s) => ({
    ...s,
    trashed: apply(s.trashed),
  }));
  syncLogicalEntries();
}

async function mutateEntries(
  copies: readonly HostGalleryImage[],
  mutation: OrganizationMutation,
  what: string,
  optimistic?: (copy: HostGalleryImage) => HostGalleryImage,
  refreshAfter = false,
) {
  if (copies.length === 0) return;
  if (optimistic) patchCopies(copies, optimistic);
  const result = await applyOrganizationMutation(
    copies,
    mutation,
    mutationContext(),
  );
  reportFanout(result, what);
  if (refreshAfter || result.failed.length > 0) void refresh();
}

function onRename(item: GalleryImage, title: string | null) {
  void mutateEntries(
    copiesOf(item),
    { kind: "setTitle", title },
    "rename the print",
    (copy) => ({ ...copy, title }),
  );
}
function onFavorite(item: GalleryImage, favorite: boolean) {
  void mutateEntries(
    copiesOf(item),
    { kind: "setFavorite", favorite },
    favorite ? "favorite the print" : "unfavorite the print",
    (copy) => ({ ...copy, favorite }),
  );
}
function onAddTag(item: GalleryImage, tag: string) {
  void mutateEntries(
    copiesOf(item),
    { kind: "addTags", tags: [tag] },
    "add the tag",
    (copy) => ({
      ...copy,
      tags: (copy.tags ?? []).some((t) => tagKey(t) === tagKey(tag))
        ? copy.tags
        : [...(copy.tags ?? []), tag],
    }),
    true,
  );
}
function onRemoveTag(item: GalleryImage, tag: string) {
  void mutateEntries(
    copiesOf(item),
    { kind: "removeTags", tags: [tag] },
    "remove the tag",
    (copy) => ({
      ...copy,
      tags: (copy.tags ?? []).filter((t) => tagKey(t) !== tagKey(tag)),
    }),
    true,
  );
}
function onSetCollection(item: GalleryImage, slug: string, member: boolean) {
  const collection = collections.value.find((c) => c.slug === slug);
  if (!collection) return;
  void mutateEntries(
    copiesOf(item),
    member
      ? { kind: "addToCollection", name: collection.name, slug }
      : { kind: "removeFromCollection", slug },
    member ? `add to ${collection.name}` : `remove from ${collection.name}`,
    undefined,
    true,
  );
}

/** The host a brand-new collection is created on: the primary when it
 * organizes, else the first host that does. */
function collectionHomeHost() {
  const capable = snapshots.value.filter((s) => s.organize);
  const origin = capable.find((s) => s.hostId === ORIGIN_HOST_ID);
  const pick = origin ?? capable[0];
  return pick ? hostById(pick.hostId) : null;
}

/** Name → create on the home host → returns the merged slug, or null. */
async function createCollectionFlow(
  prefill = "",
): Promise<{ slug: string; name: string } | null> {
  const host = collectionHomeHost();
  if (!host) {
    toast("error", "No connected host supports collections yet.");
    return null;
  }
  const name = (
    await requestText({
      title: "New collection",
      label: "Name",
      initial: prefill,
      confirmLabel: "Create",
    })
  )?.trim();
  if (!name) return null;
  try {
    const created = await createCollectionOn(host, name);
    await refresh();
    return { slug: created.slug, name: created.name };
  } catch (error) {
    toast(
      "error",
      `Couldn't create the collection: ${error instanceof Error ? error.message : String(error)}`,
    );
    return null;
  }
}

async function onNewCollection(item?: GalleryImage) {
  const created = await createCollectionFlow();
  if (!created) return;
  if (item) {
    await mutateEntries(
      copiesOf(item),
      { kind: "addToCollection", name: created.name, slug: created.slug },
      `add to ${created.name}`,
      undefined,
      true,
    );
  }
}

async function renameCollection(slug: string) {
  const collection = collections.value.find((c) => c.slug === slug);
  if (!collection) return;
  const name = (
    await requestText({
      title: "Rename collection",
      label: "Name",
      initial: collection.name,
      confirmLabel: "Rename",
    })
  )?.trim();
  if (!name || name === collection.name) return;
  const result = await renameCollectionEverywhere(collection, name, hostById);
  reportFanout(result, "rename the collection");
  await refresh();
  if (collectionSlug.value === slug) {
    const renamed = collections.value.find((c) => c.name === name);
    if (renamed) openCollection(renamed.slug);
  }
}

async function setCollectionHidden(slug: string, hidden: boolean) {
  const collection = collections.value.find(
    (candidate) => candidate.slug === slug,
  );
  if (!collection) return;
  const result = await setCollectionHiddenEverywhere(
    collection,
    hidden,
    hostById,
  );
  reportFanout(result, hidden ? "hide the collection" : "show the collection");
  await refresh();
}

async function deleteCollection(slug: string) {
  const collection = collections.value.find((c) => c.slug === slug);
  if (!collection) return;
  const accepted = await requestConfirm({
    title: `Delete collection “${collection.name}”?`,
    body: "Its prints stay in the Library.",
    confirmLabel: "Delete collection",
    danger: true,
  });
  if (!accepted) return;
  const result = await deleteCollectionEverywhere(collection, hostById);
  reportFanout(result, "delete the collection");
  if (collectionSlug.value === slug) openCollection(null);
  await refresh();
}

const collectionEditOpen = ref(false);
async function setCoverFromSelection() {
  collectionEditOpen.value = false;
  const collection = currentCollection.value;
  const key = [...selection.value][0];
  const entry = key ? entryForKey(key) : null;
  if (!collection || !entry || selection.value.size !== 1) return;
  // The cover is a filename on one host: pick the copy that lives on a host
  // holding the collection.
  const copy =
    copiesOf(entry).find((c) =>
      collection.hosts.some((h) => h.hostId === c.hostId),
    ) ?? null;
  if (!copy) {
    toast("error", "That print has no copy on a host holding this collection.");
    return;
  }
  const result = await setCollectionCover(
    collection,
    { hostId: copy.hostId, filename: copy.filename },
    hostById,
  );
  reportFanout(result, "set the cover");
  await refresh();
}
async function removeSelectedFromCollection() {
  collectionEditOpen.value = false;
  const collection = currentCollection.value;
  if (!collection || selection.value.size === 0) return;
  const copies = copiesOfKeys([...selection.value]);
  clearSelection();
  await mutateEntries(
    copies,
    { kind: "removeFromCollection", slug: collection.slug },
    `remove from ${collection.name}`,
    undefined,
    true,
  );
}

// ── Bulk bar popovers ───────────────────────────────────────────────────────
const bulkCollectionsOpen = ref(false);
const bulkTagsOpen = ref(false);
const bulkTagEditor = ref<InstanceType<typeof TagEditor> | null>(null);

const selectedEntries = computed(() =>
  [...selection.value]
    .map(
      (key) => filtered.value.find((e) => keyOf(e) === key) ?? entryForKey(key),
    )
    .filter((e): e is HostGalleryImage => !!e),
);
const selectionCopies = computed(() => copiesOfKeys([...selection.value]));
const selectionOrganizes = computed(() =>
  selectionCopies.value.some((copy) =>
    hostOrganizes(snapshots.value, copy.hostId),
  ),
);
const selectionTrashes = computed(
  () =>
    selectionCopies.value.length > 0 &&
    selectionCopies.value.every((copy) =>
      hostTrashes(snapshots.value, copy.hostId),
    ),
);
const selectionAllFavorite = computed(
  () =>
    selectedEntries.value.length > 0 &&
    selectedEntries.value.every((e) => e.favorite),
);
/** Tags on every selected print, and tags on only some. */
const selectionTags = computed(() => {
  const all = new Map<string, string>();
  const common = new Map<string, string>();
  selectedEntries.value.forEach((entry, index) => {
    const own = new Map((entry.tags ?? []).map((t) => [tagKey(t), t]));
    for (const [k, v] of own) if (!all.has(k)) all.set(k, v);
    if (index === 0) for (const [k, v] of own) common.set(k, v);
    else for (const k of [...common.keys()]) if (!own.has(k)) common.delete(k);
  });
  const shared = [...common.values()];
  const mixed = [...all.values()].filter((t) => !common.has(tagKey(t)));
  return { shared, mixed, all: [...shared, ...mixed] };
});
function collectionRowsFor(
  items: readonly HostGalleryImage[],
): CollectionPickerRow[] {
  return collections.value.map((collection) => {
    const inIt = items.filter((item) =>
      (item.organization?.collections ?? []).includes(collection.slug),
    ).length;
    const partial =
      collection.hosts.length < organizingHostCount.value
        ? collection.hosts
            .map((h) => hostById(h.hostId)?.name ?? h.hostId)
            .join(" · ") + " only"
        : "";
    const row: CollectionPickerRow = {
      slug: collection.slug,
      name: collection.name,
      checked: items.length > 0 && inIt === items.length,
      mixed: inIt > 0 && inIt < items.length,
    };
    if (partial) row.note = partial;
    return row;
  });
}
const selectionCollectionRows = computed(() =>
  collectionRowsFor(selectedEntries.value),
);
const lightboxCollectionRows = computed(() =>
  selected.value ? collectionRowsFor([selected.value as HostGalleryImage]) : [],
);
const selectionHostsLabel = computed(() =>
  [...new Set(selectionCopies.value.map((c) => c.hostLabel))].join(" · "),
);

function bulkFavorite() {
  const favorite = !selectionAllFavorite.value;
  void mutateEntries(
    selectionCopies.value,
    { kind: "setFavorite", favorite },
    favorite ? "favorite the prints" : "unfavorite the prints",
    (copy) => ({ ...copy, favorite }),
  );
}
function bulkAddTag(tag: string) {
  void mutateEntries(
    selectionCopies.value,
    { kind: "addTags", tags: [tag] },
    "add the tag",
    (copy) => ({
      ...copy,
      tags: (copy.tags ?? []).some((t) => tagKey(t) === tagKey(tag))
        ? copy.tags
        : [...(copy.tags ?? []), tag],
    }),
    true,
  );
}
function bulkRemoveTag(tag: string) {
  void mutateEntries(
    selectionCopies.value,
    { kind: "removeTags", tags: [tag] },
    "remove the tag",
    (copy) => ({
      ...copy,
      tags: (copy.tags ?? []).filter((t) => tagKey(t) !== tagKey(tag)),
    }),
    true,
  );
}
function bulkSetCollection(slug: string, member: boolean) {
  const collection = collections.value.find((c) => c.slug === slug);
  if (!collection) return;
  void mutateEntries(
    selectionCopies.value,
    member
      ? { kind: "addToCollection", name: collection.name, slug }
      : { kind: "removeFromCollection", slug },
    member ? `add to ${collection.name}` : `remove from ${collection.name}`,
    undefined,
    true,
  );
}
async function bulkNewCollection() {
  bulkCollectionsOpen.value = false;
  const copies = selectionCopies.value;
  const created = await createCollectionFlow();
  if (!created) return;
  await mutateEntries(
    copies,
    { kind: "addToCollection", name: created.name, slug: created.slug },
    `add to ${created.name}`,
    undefined,
    true,
  );
}

// ── Trash scope actions ─────────────────────────────────────────────────────
async function restoreCopies(copies: readonly HostGalleryImage[]) {
  if (copies.length === 0) return;
  const result = await applyOrganizationMutation(
    copies,
    { kind: "restore" },
    mutationContext(),
  );
  const conflicts = result.failed.filter((f) => /409|conflict/i.test(f.error));
  if (conflicts.length > 0) {
    const names = conflicts
      .map((f) => hostById(f.hostId)?.name ?? f.hostId)
      .join(", ");
    toast(
      "error",
      `Couldn't restore on ${names}: a print with that name is back in the Library there.`,
    );
  } else {
    reportFanout(result, "restore");
  }
  if (result.ok.length > 0) {
    const okHosts = new Set(result.ok);
    const keys = new Set(
      copies.filter((c) => okHosts.has(c.hostId)).map((c) => keyOf(c)),
    );
    pendingTrashed.value = pendingTrashed.value.filter(
      (e) => !keys.has(keyOf(e)),
    );
    snapshots.value = snapshots.value.map((s) => ({
      ...s,
      trashed: s.trashed.filter((e) => !keys.has(keyOf(e))),
    }));
    toast(
      "success",
      copies.length === 1
        ? "Restored print"
        : `Restored ${copies.length} prints`,
    );
  }
  closeLightbox();
  clearSelection();
  await refresh();
}
function restoreOne(item: GalleryImage) {
  void restoreCopies(copiesOf(item));
}
function restoreSelected() {
  void restoreCopies(selectionCopies.value);
}

async function deleteForeverCopies(
  copies: readonly HostGalleryImage[],
  count: number,
) {
  if (copies.length === 0) return;
  const hosts = [...new Set(copies.map((c) => c.hostLabel))].join(" · ");
  const accepted = await requestConfirm({
    title:
      count === 1 ? "Delete print forever?" : `Delete ${count} prints forever?`,
    body: `${count === 1 ? "This print" : "These prints"} on ${hosts} will be deleted forever. This can't be undone.`,
    confirmLabel: "Delete forever",
    danger: true,
  });
  if (!accepted) return;
  const result = await applyOrganizationMutation(
    copies,
    { kind: "deleteForever" },
    mutationContext(),
  );
  reportFanout(result, "delete forever");
  const okHosts = new Set(result.ok);
  const keys = new Set(
    copies.filter((c) => okHosts.has(c.hostId)).map((c) => keyOf(c)),
  );
  pendingTrashed.value = pendingTrashed.value.filter(
    (e) => !keys.has(keyOf(e)),
  );
  snapshots.value = snapshots.value.map((s) => ({
    ...s,
    trashed: s.trashed.filter((e) => !keys.has(keyOf(e))),
  }));
  rawEntries.value = rawEntries.value.filter((e) => !keys.has(keyOf(e)));
  syncLogicalEntries();
  closeLightbox();
  clearSelection();
}
function deleteForeverOne(item: GalleryImage) {
  void deleteForeverCopies(copiesOf(item), 1);
}
function deleteForeverSelected() {
  void deleteForeverCopies(selectionCopies.value, selection.value.size);
}

async function emptyTrash() {
  const count = trashEntries.value.length;
  if (count === 0) return;
  const hosts = snapshots.value
    .filter((s) => s.trash?.enabled && s.trashed.length > 0)
    .map((s) => s.hostLabel)
    .join(" · ");
  const accepted = await requestConfirm({
    title: "Empty trash?",
    body: `Delete ${count} ${count === 1 ? "print" : "prints"} in the trash on ${hosts} forever? This can't be undone.`,
    confirmLabel: "Delete forever",
    danger: true,
  });
  if (!accepted) return;
  const result = await emptyTrashEverywhere(snapshots.value, hostById);
  reportFanout(result, "empty the trash");
  const okHosts = new Set(result.ok);
  pendingTrashed.value = pendingTrashed.value.filter(
    (e) => !okHosts.has(e.hostId),
  );
  snapshots.value = snapshots.value.map((s) =>
    okHosts.has(s.hostId) ? { ...s, trashed: [] } : s,
  );
  closeLightbox();
  clearSelection();
  void refresh();
}

// ── Filtering ────────────────────────────────────────────────────────────────
const hostOptions = computed(() => {
  const options = new Map<string, string>();
  for (const entry of scopeRaw.value) {
    const id = entry.hostId ?? ORIGIN_HOST_ID;
    options.set(id, entry.hostLabel ?? getHost(id)?.name ?? id);
  }
  return Array.from(options, ([id, label]) => ({ id, label }));
});

const scopeEntries = computed<HostGalleryImage[]>(() =>
  scope.value === "trash" ? trashEntries.value : entries.value,
);

const hostFiltered = computed(() =>
  hostFilter.value === "all"
    ? scopeEntries.value
    : decorateEntries(
        scopeRaw.value.filter(
          (entry) => (entry.hostId ?? ORIGIN_HOST_ID) === hostFilter.value,
        ),
        { resolveCollectionSlug: resolver.value },
      ),
);

const kindFiltered = computed(() => {
  if (filter.value === "all") return hostFiltered.value;
  return hostFiltered.value.filter((e) => {
    const k = mediaKind(e.format, e.filename);
    if (filter.value === "video") return k === "video" || k === "animated";
    if (filter.value === "audio") return k === "audio";
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

const organizationFiltered = computed(() => {
  if (scope.value === "trash") return kindFiltered.value;
  return filterByOrganization(kindFiltered.value, {
    favoritesOnly: favoritesOnly.value,
    tags: tagFilter.value,
    collectionSlug: scope.value === "collections" ? collectionSlug.value : null,
    excludeCollectionSlugs:
      scope.value === "prints" ? hiddenCollectionSlugs.value : undefined,
  });
});

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return organizationFiltered.value;
  return organizationFiltered.value.filter((e) => entryMatchesSearch(e, q));
});

const total = computed(
  () =>
    filterByOrganization(entries.value, {
      excludeCollectionSlugs: hiddenCollectionSlugs.value,
    }).length,
);
const searchActive = computed(() => search.value.trim().length > 0);
const organizationFilterActive = computed(
  () => favoritesOnly.value || tagFilter.value.length > 0,
);
const filteredCards = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return cards.value;
  return cards.value.filter((card) => card.name.toLowerCase().includes(q));
});

// The empty-state variant to show when nothing is visible and we're not
// mid-load. `null` means render the grid/feed (skeletons handle first load).
const emptyKind = computed<
  | null
  | "none"
  | "search"
  | "video"
  | "images"
  | "organization"
  | "trash"
  | "collection"
>(() => {
  if (loading.value || filtered.value.length > 0) return null;
  if (searchActive.value) return "search";
  if (scope.value === "trash") return "trash";
  if (scope.value === "collections") return "collection";
  if (organizationFilterActive.value) return "organization";
  if (filter.value === "video") return "video";
  if (filter.value === "images") return "images";
  return "none";
});

let refreshInFlight: Promise<void> | null = null;
async function performRefresh() {
  loading.value = true;
  errorMessage.value = null;
  try {
    const hosts = listHosts();
    const [merged, organization, upscaleCapabilities] = await Promise.all([
      fetchMergedGallery(hosts),
      fetchOrganization(hosts).catch(() => null),
      Promise.all(
        hosts.map(
          async (host) =>
            [
              host.id,
              (await hostCapabilities(host)).video_upscale?.gallery_image === true,
            ] as const,
        ),
      ),
    ]);
    galleryImageUpscaleByHost.value = Object.fromEntries(upscaleCapabilities);
    if (organization) {
      snapshots.value = organization;
      // Drop a shadow copy only once its host's trash listing SUCCEEDED and
      // actually lists it — a failed listing degrades to an empty list and
      // is no evidence the trash move was lost (codex review). A generous
      // age bound keeps a host that never confirms from pinning a phantom.
      const now = Math.floor(Date.now() / 1000);
      pendingTrashed.value = pendingTrashed.value.filter((copy) => {
        const snap = organization.find((s) => s.hostId === copy.hostId);
        const key = keyOf(copy);
        const confirmed =
          snap?.trashListingOk === true &&
          snap.trashed.some((row) => keyOf(row) === key);
        if (confirmed) return false;
        return now - (copy.trashed_at ?? now) <= PENDING_TRASH_MAX_AGE_SECS;
      });
    }
    // Never resurrect prints whose optimistic removal still waits out its
    // undo window — the server lists them live until the commit fires.
    rawEntries.value =
      pendingRemovalKeys.value.size > 0
        ? merged.rawEntries.filter(
            (e) => !pendingRemovalKeys.value.has(keyOf(e)),
          )
        : merged.rawEntries;
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
const countLabel = computed(() => {
  if (scope.value === "collections")
    return `${collections.value.length} ${collections.value.length === 1 ? "collection" : "collections"} · ${total.value} prints`;
  if (scope.value === "trash")
    return `${trashEntries.value.length} ${trashEntries.value.length === 1 ? "print" : "prints"} in trash`;
  return `${total.value} prints · ${scopeLabel.value}`;
});
/** Each scope is offered on its OWN capability: Collections needs a host
 * with `gallery.organize`, Trash a host with `gallery.trash.enabled` — a
 * DB-backed host with its trash disabled must not offer Trash just because
 * it organizes (codex review). */
const scopeOptions = computed<SegmentOption<Scope>[]>(() => {
  const options: SegmentOption<Scope>[] = [
    { value: "prints", label: `Prints · ${total.value}` },
  ];
  if (canOrganize.value)
    options.push({
      value: "collections",
      label: `Collections · ${collections.value.length}`,
    });
  if (canTrash.value)
    options.push({
      value: "trash",
      label: `Trash · ${trashEntries.value.length}`,
    });
  return options;
});
/** Scopes only exist once some host can organize or trash. */
const showScopes = computed(() => scopeOptions.value.length > 1);
// Clamp a deep-linked `?scope=` no host offers back to Prints. Reactive, not
// mount-only: capabilities arrive asynchronously, so the verdict lands after
// the URL did. A fleet with any UNKNOWN capability probe (`organize: null`)
// never clamps — unknown is not incapable.
watch([scopeOptions, snapshots], () => {
  if (scope.value === "prints") return;
  if (snapshots.value.length === 0) return;
  if (snapshots.value.some((s) => s.organize === null)) return;
  if (!scopeOptions.value.some((option) => option.value === scope.value))
    setScope("prints");
});

// ── Lightbox ─────────────────────────────────────────────────────────────────
const selected = ref<GalleryImage | null>(null);
const selectedIndex = ref<number>(-1);
const lightbox = ref<InstanceType<typeof Lightbox> | null>(null);

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

let reuseEpoch = 0;

async function restoreLibrarySource(
  item: GalleryImage,
  epoch: number,
  retainedVersion: number,
  expected: {
    model: string;
    width: number;
    height: number;
    sourceFit: string;
  },
): Promise<void> {
  const stillOwnsEmptySource = () =>
    epoch === reuseEpoch &&
    retainedSourceReuseIsCurrent(retainedVersion) &&
    form.state.value.model === expected.model &&
    form.state.value.width === expected.width &&
    form.state.value.height === expected.height &&
    JSON.stringify(form.state.value.sourceFitPolicy) === expected.sourceFit &&
    form.state.value.imageAttachments.length === 0;
  const restoreCanvas = async (base64: string) => {
    await nextTick();
    if (
      epoch !== reuseEpoch ||
      !retainedSourceReuseIsCurrent(retainedVersion) ||
      form.state.value.model !== expected.model ||
      form.state.value.imageAttachments[0]?.base64 !== base64
    )
      return;
    form.state.value.width =
      item.metadata.generation_width ?? item.metadata.width;
    form.state.value.height =
      item.metadata.generation_height ?? item.metadata.height;
  };
  const owner = hostForEntry(item);
  const retainedTarget = owner
    ? { baseUrl: owner.url, apiKey: owner.apiKey ?? null }
    : null;
  let retainedUnavailable: RetainedSourceMediaAvailability | null = null;
  const retainedRead = retainedTarget
    ? retainedSourceMediaInventory(item.filename, retainedTarget).catch(
        () => null,
      )
    : Promise.resolve(null);
  const acceptRetainedInventory = (
    inventory: Awaited<ReturnType<typeof retainedSourceMediaInventory>> | null,
  ) => {
    if (!inventory || !retainedTarget || epoch !== reuseEpoch) return;
    retainedUnavailable = inventory.availability;
    setRetainedSourceReuseIntentIfCurrent(retainedVersion, {
      filename: item.filename,
      origin: retainedTarget,
      inventory,
    });
  };
  const sha256 = item.metadata.source_image_sha256;
  const stored = await restoreGenerationSourceMedia(sha256).catch(() => null);
  if (stored && stillOwnsEmptySource()) {
    form.state.value.imageAttachments = [
      {
        kind: stored.kind ?? "upload",
        filename: stored.filename,
        base64: stored.base64,
        width: stored.width ?? undefined,
        height: stored.height ?? undefined,
        mime: stored.mime ?? undefined,
      },
    ];
    await restoreCanvas(stored.base64);
    void retainedRead.then((inventory) => {
      acceptRetainedInventory(inventory);
      const disclosure = inventory
        ? retainedSourceMediaDisclosure(inventory.availability)
        : null;
      if (disclosure) toast("error", disclosure);
    });
    return;
  }

  const resolvedRetainedInventory = await retainedRead;
  acceptRetainedInventory(resolvedRetainedInventory);

  if (retainedTarget && resolvedRetainedInventory) {
    try {
      const retained = resolvedRetainedInventory.members.find(
        (member) => member.role === "source_image",
      );
      if (resolvedRetainedInventory.availability === "available" && retained) {
        const blob = await retainedSourceMediaBlob(
          item.filename,
          retained.member_id,
          retainedTarget,
        );
        const base64 = await blobToBase64(blob);
        if (!stillOwnsEmptySource()) return;
        const dimensions = imageDimensionsFromBase64(base64);
        form.state.value.imageAttachments = [
          {
            kind: "gallery",
            filename: retained.display_name,
            base64,
            width: dimensions?.width,
            height: dimensions?.height,
            mime: blob.type || undefined,
          },
        ];
        await restoreCanvas(base64);
        return;
      }
    } catch {
      // Preserve the established same-name gallery fallback below.
    }
  }

  const filename = item.metadata.source_image_name;
  if (!filename) {
    const disclosure = retainedUnavailable
      ? retainedSourceMediaDisclosure(retainedUnavailable)
      : null;
    if (disclosure) toast("error", disclosure);
    return;
  }
  const candidates = [owner, ...listHosts()].filter(
    (host, index, hosts) =>
      host && hosts.findIndex((other) => other?.id === host.id) === index,
  );
  for (const host of candidates) {
    if (!host) continue;
    try {
      const blob = await fetchGalleryBlob(host, filename);
      const base64 = await blobToBase64(blob);
      if (!stillOwnsEmptySource()) return;
      const dimensions = imageDimensionsFromBase64(base64);
      form.state.value.imageAttachments = [
        {
          kind: "gallery",
          filename,
          base64,
          width: dimensions?.width,
          height: dimensions?.height,
          mime: blob.type || undefined,
        },
      ];
      await restoreCanvas(base64);
      return;
    } catch {
      // A source picked on another machine may not exist on the print owner.
    }
  }
  toast(
    "error",
    (retainedUnavailable &&
      retainedSourceMediaDisclosure(retainedUnavailable)) ||
      "The original source image is unavailable. Reattach it before generating.",
  );
}

async function onReuse(item: GalleryImage) {
  const epoch = ++reuseEpoch;
  const retainedVersion = beginRetainedSourceReuseIntent();
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
  // The gallery row's editable title wins over the creation-time one.
  form.state.value.title = item.title ?? item.metadata.title ?? null;
  const expected = {
    model: form.state.value.model,
    width: form.state.value.width,
    height: form.state.value.height,
    sourceFit: JSON.stringify(form.state.value.sourceFitPolicy),
  };
  closeLightbox();
  await router.push({ name: "create" });
  if (epoch !== reuseEpoch) return;
  await restoreLibrarySource(item, epoch, retainedVersion, expected);
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
    const state = form.state.value;
    const h3Task = minimaxH3TaskForModel(state.model);
    if (h3Task) {
      const dimensions = imageDimensionsFromBase64(base64) ?? {
        width: item.metadata.width,
        height: item.metadata.height,
      };
      const image = {
        filename: item.filename,
        mimeType: galleryImageMimeType(item, blob.type),
        width: dimensions.width,
        height: dimensions.height,
        data: base64,
      };
      const result =
        h3Task === "ref2va"
          ? await appendMinimaxH3GalleryImageReference(state.h3Authoring, image)
          : setMinimaxH3GalleryImageFirstFrame(state.h3Authoring, image);
      if (!result.ok) throw new Error(result.error);
      state.h3Authoring = result.state;
    } else if (isMinimaxH3Identity(state.modelFamily, state.model)) {
      throw new Error(
        "Choose an explicit MiniMax H3 FL2VA or Ref2VA model before adding a source.",
      );
    } else {
      state.imageAttachments = [
        { kind: "gallery", filename: item.filename, base64 },
      ];
      state.sourceFitPolicy = defaultSourceFitPolicy();
    }
    return true;
  } catch (err) {
    toast("error", err instanceof Error ? err.message : String(err));
    return false;
  }
}

function galleryImageMimeType(item: GalleryImage, declared: string): string {
  const mime = declared.split(";", 1)[0]!.trim().toLowerCase();
  if (mime.startsWith("image/")) return mime;
  const format = (item.format ?? item.filename.split(".").pop() ?? "")
    .toLowerCase()
    .replace("jpg", "jpeg");
  return format ? `image/${format}` : "application/octet-stream";
}

async function onUseAsSource(item: GalleryImage) {
  if (!(await setAsSource(item))) return;
  closeLightbox();
  void router.push({ name: "create" });
}

const upscaleItem = ref<GalleryImage | null>(null);
const upscaleModel = ref("");
const upscaleBusy = ref(false);
const upscaleJob = ref<VideoUpscaleJob | null>(null);
let upscalePoll: ReturnType<typeof setTimeout> | null = null;
let upscaleEpoch = 0;
const upscalers = computed(() => selectUpscalers(models.value));
const upscaleKind = computed(() =>
  upscaleItem.value && mediaKind(upscaleItem.value.format, upscaleItem.value.filename) === "video"
    ? "video"
    : "image",
);

function stopUpscalePoll() {
  if (upscalePoll) clearTimeout(upscalePoll);
  upscalePoll = null;
}
function closeUpscaleDialog() {
  upscaleEpoch += 1;
  stopUpscalePoll();
  upscaleItem.value = null;
  upscaleJob.value = null;
}
async function onUpscale(item: GalleryImage) {
  if (mediaKind(item.format, item.filename) === "audio") return;
  stopUpscalePoll();
  const epoch = ++upscaleEpoch;
  upscaleItem.value = item;
  upscaleJob.value = null;
  upscaleModel.value = defaultUpscaler(upscalers.value);
  if (mediaKind(item.format, item.filename) === "video") {
    try {
      const recovered = await findRecoverableFramewiseUpscale(
        upscaleTarget(item),
        item.filename,
      );
      if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
      upscaleJob.value = recovered;
      if (recovered) upscaleModel.value = recovered.model;
      if (shouldPollFramewiseJob(upscaleJob.value)) void pollUpscaleJob();
    } catch {
      // Older hosts do not expose durable Framewise history.
    }
  }
}
function upscaleTarget(item: GalleryImage) {
  const host = hostForEntry(item);
  if (!host) throw missingHostError(item);
  return { baseUrl: host.url, apiKey: host.apiKey ?? null };
}
async function pollUpscaleJob() {
  const item = upscaleItem.value;
  const job = upscaleJob.value;
  if (
    !item ||
    !job ||
    !shouldPollFramewiseJob(job)
  )
    return;
  const epoch = upscaleEpoch;
  try {
    const next = await getFramewiseUpscale(upscaleTarget(item), job.id);
    if (
      epoch !== upscaleEpoch ||
      upscaleItem.value !== item ||
      upscaleJob.value?.id !== job.id
    )
      return;
    upscaleJob.value = next;
    if (next.state === "completed") {
      toast("success", `Framewise upscale complete — ${next.output_filename}`);
      await refresh();
      return;
    }
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
    toast("error", error instanceof Error ? error.message : String(error));
    return;
  }
  if (shouldPollFramewiseJob(upscaleJob.value))
    upscalePoll = setTimeout(() => void pollUpscaleJob(), 750);
}
async function startUpscale() {
  const item = upscaleItem.value;
  if (!item || upscaleBusy.value) return;
  const epoch = ++upscaleEpoch;
  stopUpscalePoll();
  upscaleBusy.value = true;
  try {
    if (upscaleKind.value === "video") {
      const created = await createFramewiseUpscale(
        upscaleTarget(item),
        item.filename,
        upscaleModel.value,
      );
      if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
      upscaleJob.value = created;
      toast("info", `Framewise upscale queued (${created.id}).`);
      void pollUpscaleJob();
    } else {
      const host = hostForEntry(item);
      if (host && galleryImageUpscaleByHost.value[host.id] === true) {
        const result = await upscaleLibraryImage(
          upscaleTarget(item),
          item.filename,
          upscaleModel.value,
        );
        toast("success", `Upscaled — ${result.filename}`);
        await refresh();
        closeUpscaleDialog();
      } else if (await setAsSource(item)) {
        closeUpscaleDialog();
        closeLightbox();
        toast("info", "Added as source — pick an upscaler in Controls.");
        void router.push({ name: "create" });
      }
    }
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
    toast("error", error instanceof Error ? error.message : String(error));
  } finally {
    upscaleBusy.value = false;
  }
}
async function transitionUpscale(action: "pause" | "resume" | "cancel") {
  const item = upscaleItem.value;
  const job = upscaleJob.value;
  if (!item || !job) return;
  stopUpscalePoll();
  const epoch = ++upscaleEpoch;
  try {
    const transitioned = await transitionFramewiseUpscale(
      upscaleTarget(item),
      job.id,
      action,
    );
    if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
    upscaleJob.value = transitioned;
    if (action === "resume") void pollUpscaleJob();
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleItem.value !== item) return;
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

/**
 * Single-print delete from the Lightbox / context menu. Optimistic with a
 * 6 s undo window either way; the commit is the same DELETE, which a
 * trash-capable host turns into a trash move (so no confirm is asked there)
 * and an older host executes permanently (so the confirm stays).
 */
async function onLightboxDelete(item: GalleryImage) {
  const reversible = allCopiesTrash(item);
  if (!reversible) {
    const accepted = await requestConfirm({
      title: "Delete print?",
      body: `${item.filename} will be deleted from every connected device. You can undo for a few seconds.`,
      confirmLabel: "Delete",
      danger: true,
    });
    if (!accepted) return;
  }
  const key = keyOf(item);
  const entryIdx = rawEntries.value.findIndex((e) => keyOf(e) === key);
  if (entryIdx === -1) return;
  const removed = copiesOf(item);
  const removedKeys = new Set(removed.map((entry) => keyOf(entry)));

  // Optimistic removal; commit the DELETE only once the undo window elapses.
  // Masked from refresh until then so a poll can't resurrect the rows.
  addPendingRemovals(removedKeys);
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
    text: reversible ? "Moved to trash" : "Print deleted everywhere",
    undo: () => {
      clearPendingRemovals(removedKeys);
      restoreRemovedEntries(removed);
    },
    commit: async () => {
      try {
        const results = await Promise.allSettled(
          removed.map((entry) => deleteRouted(entry)),
        );
        const failed = removed.filter(
          (_, index) => results[index]?.status === "rejected",
        );
        const succeeded = removed.filter(
          (_, index) => results[index]?.status === "fulfilled",
        );
        markTrashedLocally(
          succeeded.filter((entry) =>
            hostTrashes(snapshots.value, entry.hostId),
          ),
        );
        if (failed.length > 0) {
          restoreRemovedEntries(failed);
          toast(
            "error",
            `${failed.length} device ${failed.length === 1 ? "copy remains" : "copies remain"} because a delete failed.`,
          );
        }
      } finally {
        clearPendingRemovals(removedKeys);
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
function contextUpscale() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) onUpscale(item);
}
async function contextDelete() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) await onLightboxDelete(item);
}
function contextFavorite() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) onFavorite(item, !item.favorite);
}
async function contextRename() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (!item) return;
  const next = await requestText({
    title: "Rename print",
    label: "Title",
    initial: item.title ?? "",
    confirmLabel: "Rename",
  });
  if (next === null) return;
  onRename(item, next.trim() || null);
}
function contextRestore() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) restoreOne(item);
}
function contextDeleteForever() {
  const item = contextMenu.value?.item;
  closeContextMenu();
  if (item) deleteForeverOne(item);
}
function onDocumentPointerDown(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (!target?.closest("[data-test='gallery-context-menu']"))
    closeContextMenu();
}

// ── Keyboard ─────────────────────────────────────────────────────────────────
// F favorite · T tag · ⌘⇧N new collection · ⌫ trash · ⌘⌫ delete forever.
// Never while typing, and the Lightbox's own ←/→/Esc stay untouched.
function typingTarget(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null;
  if (!target) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}
function focusedEntries(): HostGalleryImage[] {
  if (selected.value) return [selected.value as HostGalleryImage];
  if (selectMode.value) return selectedEntries.value;
  return [];
}
function onDocumentKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") {
    closeContextMenu();
    return;
  }
  if (typingTarget(event) || event.altKey) return;
  const meta = event.metaKey || event.ctrlKey;
  const key = event.key.toLowerCase();
  if (meta && event.shiftKey && key === "n") {
    if (!canOrganize.value) return;
    event.preventDefault();
    void onNewCollection(selected.value ?? undefined);
    return;
  }
  if (
    meta &&
    !event.shiftKey &&
    (event.key === "Backspace" || event.key === "Delete")
  ) {
    const items = focusedEntries();
    if (items.length === 0 || !canTrash.value) return;
    event.preventDefault();
    if (selected.value) deleteForeverOne(selected.value);
    else void deleteForeverCopies(selectionCopies.value, selection.value.size);
    return;
  }
  if (meta || event.shiftKey) return;
  if (event.key === "Backspace" || event.key === "Delete") {
    if (scope.value === "trash") return;
    if (selected.value) {
      event.preventDefault();
      void onLightboxDelete(selected.value);
    } else if (selectMode.value && selection.value.size > 0) {
      event.preventDefault();
      void deleteSelected();
    }
    return;
  }
  if (key === "f") {
    if (!canOrganize.value) return;
    if (selected.value) {
      event.preventDefault();
      onFavorite(selected.value, !selected.value.favorite);
    } else if (selectMode.value && selection.value.size > 0) {
      event.preventDefault();
      bulkFavorite();
    }
    return;
  }
  if (key === "t") {
    if (!canOrganize.value) return;
    if (selected.value) {
      event.preventDefault();
      lightbox.value?.focusTags();
    } else if (selectMode.value && selection.value.size > 0) {
      event.preventDefault();
      bulkTagsOpen.value = true;
      void nextTick(() => bulkTagEditor.value?.focus());
    }
  }
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
  rememberSessionScroll(WEB_LIBRARY_SCROLL_KEY, {
    top: window.scrollY,
    left: window.scrollX,
  });
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
  await nextTick();
  const position = sessionScrollPosition(WEB_LIBRARY_SCROLL_KEY);
  window.scrollTo({ top: position.top, left: position.left });
  refreshTimer = setInterval(() => {
    if (!document.hidden) void refresh();
  }, 10_000);
});
onBeforeUnmount(() => {
  disposed = true;
  stopUpscalePoll();
  if (refreshTimer) clearInterval(refreshTimer);
});
</script>

<template>
  <div class="gal" :data-scope="scope">
    <header class="gal__head">
      <h1 class="gal__title">Library</h1>
      <span class="gal__count" data-test="gallery-count"
        >{{ countLabel
        }}<span v-if="unreachableLabel" class="gal__unreachable">
          · {{ unreachableLabel }}</span
        ></span
      >

      <SegmentedControl
        v-if="showScopes"
        class="gal__scope"
        data-test="library-scope"
        :model-value="scope"
        :options="scopeOptions"
        label="Library scope"
        compact
        @update:model-value="setScope"
      />

      <span class="gal__flex"></span>

      <template v-if="scope === 'collections'">
        <label class="gal__search">
          <Icon name="search" :size="15" />
          <input
            :value="search"
            type="search"
            :placeholder="
              currentCollection ? 'Search prints…' : 'Search collections…'
            "
            aria-label="Search gallery"
            data-test="gallery-search"
            @input="onSearchInput(($event.target as HTMLInputElement).value)"
          />
        </label>
        <button
          v-if="canOrganize"
          type="button"
          class="gal__primary"
          data-test="new-collection"
          @click="onNewCollection()"
        >
          <Icon name="plus" :size="14" /> New collection
        </button>
        <button
          v-if="currentCollection"
          type="button"
          class="gal__select"
          :class="{ 'gal__select--on': selectMode }"
          :aria-pressed="selectMode"
          data-test="gallery-select"
          @click="setSelectMode(!selectMode)"
        >
          <Icon name="check" :size="15" />
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
      </template>

      <template v-else>
        <ThumbnailSizeSlider
          v-if="view === 'grid' || scope === 'trash'"
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

        <label class="gal__search">
          <Icon name="search" :size="15" />
          <input
            :value="search"
            type="search"
            placeholder="Search prompts, titles, tags…"
            aria-label="Search gallery"
            data-test="gallery-search"
            @input="onSearchInput(($event.target as HTMLInputElement).value)"
          />
        </label>

        <div class="gal__tools">
          <button
            v-if="scope === 'prints'"
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

          <div
            v-if="scope === 'prints'"
            class="gal__viewtoggle"
            role="group"
            aria-label="View mode"
          >
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
            v-if="scope === 'prints'"
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
            v-if="scope === 'trash'"
            type="button"
            class="gal__danger-outline"
            :disabled="trashEntries.length === 0"
            data-test="empty-trash"
            @click="emptyTrash"
          >
            <Icon name="trash" :size="14" /> Empty trash
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
      </template>
    </header>

    <!-- Filter chips: ♥ Favorites · tags · hosts (Prints + drill-in). -->
    <LibraryChipRow
      v-if="
        scope === 'prints' || (scope === 'collections' && currentCollection)
      "
      :organize="canOrganize"
      :favorites-only="favoritesOnly"
      :favorite-count="favoriteCount"
      :tags="filterChipTags"
      :active-tags="tagFilter"
      :host-options="hostOptions"
      :host-filter="hostFilter"
      @toggle-favorites="toggleFavoritesOnly"
      @toggle-tag="toggleTag"
      @set-host="hostFilter = $event"
    />
    <LibraryChipRow
      v-else-if="scope === 'trash'"
      :organize="false"
      :favorites-only="false"
      :favorite-count="0"
      :tags="[]"
      :active-tags="[]"
      :host-options="hostOptions"
      :host-filter="hostFilter"
      @set-host="hostFilter = $event"
    />

    <!-- Trash retention banner. -->
    <div
      v-if="scope === 'trash' && retentionSummary.segments.length > 0"
      class="gal__banner"
      data-test="trash-banner"
    >
      <span class="gal__banner-dot" aria-hidden="true">•</span>
      <span class="gal__banner-text">
        <template v-for="(segment, i) in retentionSummary.segments" :key="i">
          <b v-if="segment.mono" class="gal__mono">{{ segment.text }}</b>
          <template v-else>{{ segment.text }}</template>
        </template>
      </span>
      <span class="gal__flex"></span>
      <router-link class="gal__banner-link" to="/settings"
        >Change retention · Settings</router-link
      >
      <router-link
        v-if="remoteHostCount > 0"
        class="gal__banner-link"
        to="/machines"
        >Machines</router-link
      >
    </div>

    <!-- Collection drill-in breadcrumb + Edit. -->
    <div
      v-if="scope === 'collections' && collectionSlug"
      class="gal__crumbs"
      data-test="collection-crumbs"
    >
      <button
        type="button"
        class="gal__crumb"
        data-test="crumb-collections"
        @click="openCollection(null)"
      >
        <Icon name="chevron-left" :size="14" /> Collections
      </button>
      <span class="gal__crumb-sep" aria-hidden="true">›</span>
      <span class="gal__crumb-here" data-test="crumb-here">{{
        currentCollection?.name ?? collectionSlug
      }}</span>
      <span v-if="currentCard" class="gal__crumb-meta">
        {{ currentCard.count }}
        {{ currentCard.count === 1 ? "print" : "prints" }} ·
        {{ currentCard.hostLabels.join(" · ") }}
      </span>
      <span class="gal__flex"></span>
      <Popover
        v-if="currentCollection && canOrganize"
        :open="collectionEditOpen"
        placement="bottom-end"
        label="Edit collection"
        @update:open="collectionEditOpen = $event"
      >
        <template #trigger>
          <button
            type="button"
            class="gal__bar-btn"
            :aria-expanded="collectionEditOpen"
            data-test="collection-edit"
            @click="collectionEditOpen = !collectionEditOpen"
          >
            Edit <Icon name="chevron-down" :size="13" />
          </button>
        </template>
        <div class="gal__menu" role="menu" data-test="collection-edit-menu">
          <button
            type="button"
            role="menuitem"
            data-test="collection-edit-rename"
            @click="
              collectionEditOpen = false;
              renameCollection(currentCollection.slug);
            "
          >
            Rename…
          </button>
          <button
            type="button"
            role="menuitem"
            :disabled="selection.size !== 1"
            data-test="collection-edit-cover"
            @click="setCoverFromSelection"
          >
            Set cover from selection
          </button>
          <button
            type="button"
            role="menuitem"
            :disabled="selection.size === 0"
            data-test="collection-edit-remove"
            @click="removeSelectedFromCollection"
          >
            Remove selected prints
          </button>
          <button
            type="button"
            role="menuitem"
            class="gal__context-danger"
            data-test="collection-edit-delete"
            @click="
              collectionEditOpen = false;
              deleteCollection(currentCollection.slug);
            "
          >
            Delete collection…
          </button>
        </div>
      </Popover>
    </div>

    <main class="gal__main">
      <div v-if="errorMessage" class="gal__error" role="alert">
        <p class="gal__error-title">Couldn't load the gallery.</p>
        <p class="gal__error-body">{{ errorMessage }}</p>
      </div>

      <!-- Collections shelf -->
      <template v-else-if="scope === 'collections' && !collectionSlug">
        <EmptyStateBlock
          v-if="!loading && cards.length === 0"
          icon="collection"
          headline="No collections yet"
          guidance="Name one, then add prints from the grid or a selection."
        >
          <template #action>
            <button
              v-if="canOrganize"
              type="button"
              class="gal__emptybtn"
              data-test="empty-new-collection"
              @click="onNewCollection()"
            >
              Create a collection
            </button>
          </template>
        </EmptyStateBlock>
        <CollectionsShelf
          v-else
          :cards="filteredCards"
          :can-create="canOrganize"
          @open="openCollection"
          @new="onNewCollection()"
          @rename="renameCollection"
          @hidden="setCollectionHidden"
          @delete="deleteCollection"
        />
      </template>

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
          v-else-if="emptyKind === 'trash'"
          icon="trash"
          headline="No prints in the trash"
          guidance="Prints you trash wait here until they're restored or purged."
        />

        <EmptyStateBlock
          v-else-if="emptyKind === 'collection'"
          icon="collection"
          headline="This collection is empty"
          guidance="Select prints in the Library and use Add to collection."
        >
          <template #action>
            <button
              type="button"
              class="gal__emptybtn"
              data-test="browse-prints"
              @click="setScope('prints')"
            >
              Browse prints
            </button>
          </template>
        </EmptyStateBlock>

        <EmptyStateBlock
          v-else-if="emptyKind === 'organization'"
          icon="tag"
          headline="No prints match"
          guidance="Nothing carries every selected chip. Clear the filters to see everything."
        >
          <template #action>
            <button
              type="button"
              class="gal__emptybtn"
              data-test="clear-filters"
              @click="clearOrganizationFilters"
            >
              Clear filters
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
          v-else-if="view === 'grid' || scope !== 'prints'"
          :entries="filtered"
          :models="models"
          :loading="loading"
          :thumbnail-size="thumbnailSize"
          :select-mode="selectMode"
          :selection="selection"
          :fresh="fresh"
          :trash="scope === 'trash'"
          @open="openItem"
          @toggle-select="toggleSelect"
          @drag-select="onDragSelect"
          @context-menu="openContextMenu"
          @restore="restoreOne"
          @delete-forever="deleteForeverOne"
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
      <template v-if="scope === 'trash'">
        <button
          type="button"
          role="menuitem"
          data-test="context-restore"
          @click="contextRestore"
        >
          Restore
        </button>
        <button
          type="button"
          role="menuitem"
          class="gal__context-danger"
          data-test="context-delete-forever"
          @click="contextDeleteForever"
        >
          Delete forever
        </button>
      </template>
      <template v-else>
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
          v-if="mediaKind(contextMenu.item.format, contextMenu.item.filename) !== 'audio'"
          type="button"
          role="menuitem"
          data-test="context-upscale"
          @click="contextUpscale"
        >
          {{
            libraryUpscaleLabel(
              mediaKind(contextMenu.item.format, contextMenu.item.filename) === "video"
                ? "video"
                : "image",
            )
          }}
        </button>
        <template v-if="canOrganizeEntry(contextMenu.item)">
          <button
            type="button"
            role="menuitem"
            data-test="context-favorite"
            @click="contextFavorite"
          >
            {{ contextMenu.item.favorite ? "Unfavorite" : "Favorite" }}
          </button>
          <button
            type="button"
            role="menuitem"
            data-test="context-rename"
            @click="contextRename"
          >
            Rename…
          </button>
        </template>
        <button
          type="button"
          role="menuitem"
          class="gal__context-danger"
          data-test="context-delete"
          @click="contextDelete"
        >
          {{ allCopiesTrash(contextMenu.item) ? "Trash" : "Delete" }}
        </button>
      </template>
    </div>

    <!-- Selection action bar. -->
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

          <template v-if="scope === 'trash'">
            <button
              type="button"
              class="gal__bar-btn"
              :disabled="selection.size === 0"
              data-test="bulk-restore"
              @click="restoreSelected"
            >
              Restore
            </button>
            <button
              type="button"
              class="gal__bar-danger"
              :disabled="selection.size === 0"
              data-test="bulk-delete-forever"
              @click="deleteForeverSelected"
            >
              Delete forever
            </button>
          </template>

          <template v-else>
            <template v-if="canOrganize">
              <Popover
                :open="bulkCollectionsOpen"
                placement="top-start"
                label="Add to collection"
                @update:open="bulkCollectionsOpen = $event"
              >
                <template #trigger>
                  <button
                    type="button"
                    class="gal__bar-btn"
                    :disabled="selection.size === 0 || !selectionOrganizes"
                    :aria-expanded="bulkCollectionsOpen"
                    data-test="bulk-collections"
                    @click="bulkCollectionsOpen = !bulkCollectionsOpen"
                  >
                    <Icon name="collection" :size="14" /> Add to collection
                  </button>
                </template>
                <div class="gal__pop" data-test="bulk-collections-panel">
                  <p class="gal__pop-k">
                    Add {{ selection.size }}
                    {{ selection.size === 1 ? "print" : "prints" }} to
                  </p>
                  <CollectionPicker
                    :rows="selectionCollectionRows"
                    :footer="
                      selectionHostsLabel
                        ? `fans out to ${selectionHostsLabel}`
                        : ''
                    "
                    @toggle="bulkSetCollection"
                    @new="bulkNewCollection"
                  />
                </div>
              </Popover>
              <Popover
                :open="bulkTagsOpen"
                placement="top-start"
                label="Tag"
                @update:open="bulkTagsOpen = $event"
              >
                <template #trigger>
                  <button
                    type="button"
                    class="gal__bar-btn"
                    :disabled="selection.size === 0 || !selectionOrganizes"
                    :aria-expanded="bulkTagsOpen"
                    data-test="bulk-tags"
                    @click="bulkTagsOpen = !bulkTagsOpen"
                  >
                    <Icon name="tag" :size="14" /> Tag
                  </button>
                </template>
                <div class="gal__pop" data-test="bulk-tags-panel">
                  <p class="gal__pop-k">
                    Tags on {{ selection.size }}
                    {{ selection.size === 1 ? "print" : "prints" }}
                  </p>
                  <TagEditor
                    ref="bulkTagEditor"
                    :tags="selectionTags.all"
                    :mixed="selectionTags.mixed"
                    :suggestions="tags"
                    @add="bulkAddTag"
                    @remove="bulkRemoveTag"
                  />
                </div>
              </Popover>
              <button
                type="button"
                class="gal__bar-btn gal__bar-fav"
                :data-on="selectionAllFavorite ? 'true' : undefined"
                :disabled="selection.size === 0 || !selectionOrganizes"
                data-test="bulk-favorite"
                @click="bulkFavorite"
              >
                <Icon name="heart" :size="14" :stroke-width="2" />
                {{ selectionAllFavorite ? "Unfavorite" : "Favorite" }}
              </button>
            </template>
            <button
              type="button"
              class="gal__bar-danger"
              :disabled="selection.size === 0"
              data-test="bulk-delete"
              @click="deleteSelected"
            >
              {{ selectionTrashes ? "Trash" : "Delete selected" }}
            </button>
            <button
              type="button"
              class="gal__bar-danger gal__bar-danger--soft"
              :disabled="filtered.length === 0"
              data-test="bulk-delete-all"
              @click="deleteAllFiltered"
            >
              {{ canTrash ? "Trash all" : "Delete all" }}
            </button>
          </template>

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
      ref="lightbox"
      :item="selected"
      :models="models"
      :index="selectedIndex"
      :total="filtered.length"
      :has-prev="selectedIndex > 0"
      :has-next="selectedIndex >= 0 && selectedIndex < filtered.length - 1"
      :muted="muted"
      :is-sequence="isSequencePrint(selected)"
      :can-edit-sequence="canEditSequence(selected)"
      :can-organize="canOrganizeEntry(selected)"
      :can-trash="allCopiesTrash(selected)"
      :in-trash="scope === 'trash'"
      :collections="lightboxCollectionRows"
      :tag-suggestions="tags"
      @close="closeLightbox"
      @prev="stepLightbox(-1)"
      @next="stepLightbox(1)"
      @reuse="onReuse"
      @use-source="onUseAsSource"
      @upscale="onUpscale"
      @delete="onLightboxDelete"
      @edit-sequence="onEditSequence"
      @rename="onRename"
      @favorite="onFavorite"
      @add-tag="onAddTag"
      @remove-tag="onRemoveTag"
      @set-collection="onSetCollection"
      @new-collection="onNewCollection"
      @restore="restoreOne"
      @delete-forever="deleteForeverOne"
      @context-menu="openContextMenu"
    />

    <UpscaleDialog
      :open="!!upscaleItem"
      :kind="upscaleKind"
      :source-name="upscaleItem?.filename ?? ''"
      :models="upscalers"
      v-model="upscaleModel"
      :busy="upscaleBusy"
      :job-state="upscaleJob?.state ?? null"
      :status="upscaleJob ? framewiseStatus(upscaleJob) : null"
      :progress="upscaleJob ? framewiseProgress(upscaleJob) : null"
      @confirm="startUpscale"
      @close="closeUpscaleDialog"
      @pause="transitionUpscale('pause')"
      @resume="transitionUpscale('resume')"
      @cancel="transitionUpscale('cancel')"
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

.gal__scope {
  flex: 0 0 auto;
}
.gal__scope :deep(.ms-seg__label) {
  white-space: nowrap;
}
.gal__filter {
  flex: 0 0 auto;
}
.gal__thumbnail-size {
  flex: 0 0 136px;
}

/* Below 640px the scope control spans the row over the grid. */
@media (max-width: 639px) {
  .gal__scope {
    order: 10;
    flex: 1 0 100%;
  }
}

.gal__context,
.gal__menu {
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
.gal__menu {
  position: static;
  min-width: 220px;
  box-shadow: var(--shadow-raised);
}
.gal__context button,
.gal__menu button {
  min-height: 40px;
  padding: 0 10px;
  border: 0;
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.gal__menu button {
  min-height: 34px;
  font-size: 13px;
}
.gal__context button:hover,
.gal__menu button:hover:not(:disabled) {
  background: var(--sel-bg);
}
.gal__menu button:disabled {
  opacity: 0.5;
  cursor: default;
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

.gal__select,
.gal__primary,
.gal__danger-outline {
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
.gal__select:hover,
.gal__danger-outline:hover:not(:disabled) {
  color: var(--rebate);
}
.gal__select--on {
  background: var(--sel-bg);
  color: var(--sel-ink);
  border-color: var(--sel-border);
}
.gal__select:focus-visible,
.gal__primary:focus-visible,
.gal__danger-outline:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.gal__primary {
  border-color: transparent;
  background: var(--safelight);
  color: var(--on-accent);
}
.gal__danger-outline {
  color: var(--stop);
  border-color: color-mix(in srgb, var(--stop) 50%, transparent);
}
.gal__danger-outline:disabled {
  opacity: 0.5;
  cursor: default;
}

.gal__main {
  margin-top: 4px;
}

/* Trash retention banner */
.gal__banner {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin: 0 0 14px;
  padding: 9px 12px;
  border-radius: var(--radius-control-lg);
  background: color-mix(in srgb, var(--halide) 12%, var(--bath));
  color: var(--ink-2);
  font-size: 12.5px;
}
.gal__banner-dot {
  color: var(--halide);
}
.gal__mono {
  font-family: var(--f-mono);
  font-size: 11.5px;
  font-weight: 600;
  color: var(--rebate);
}
.gal__banner-link {
  color: var(--safelight);
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}
.gal__banner-link:hover {
  text-decoration: underline;
}

/* Collection drill-in crumb bar */
.gal__crumbs {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  margin: -6px 0 12px;
  border-bottom: 1px solid var(--edge);
}
.gal__crumb {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-family: var(--f-body);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.gal__crumb:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.gal__crumb-sep {
  color: var(--ink-3);
}
.gal__crumb-here {
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--rebate);
}
.gal__crumb-meta {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-3);
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
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 6px 12px;
  border-radius: var(--radius-pill);
  font-family: var(--f-body);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.gal__bar-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.gal__bar-fav[data-on="true"] {
  color: var(--safelight);
  border-color: var(--safelight);
}
.gal__bar-fav[data-on="true"] :deep(svg) {
  fill: currentColor;
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

/* Bulk-bar popovers */
.gal__pop {
  width: min(280px, calc(100vw - 32px));
  padding: 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: var(--bench);
  box-shadow: var(--shadow-raised);
}
.gal__pop-k {
  margin: 0 0 8px;
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
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
