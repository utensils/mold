<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  onUnmounted,
  ref,
  unref,
  watch,
} from "vue";
import { useRoute, useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import {
  galleryThumbnailScheduler,
  type ThumbnailHandle,
  type ThumbnailPriority,
} from "@studio/lib/thumbnailScheduler";
import { chunkForProbe, planPrewarm, type PrewarmCandidate } from "../lib/gallery/thumbnailPrewarm";
import Icon from "@ui/components/Icon.vue";
import UpscaleDialog from "@ui/components/UpscaleDialog.vue";
import {
  loadGalleryThumbnailSize,
  saveGalleryThumbnailSize,
} from "@studio/lib/galleryThumbnailSize";
import { blobToBase64 } from "@studio/lib/base64";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import BulkBar from "../components/library/BulkBar.vue";
import CollectionsShelf, { type ShelfCard } from "../components/library/CollectionsShelf.vue";
import type { CoverTile } from "../components/library/CollectionCard.vue";
import HistoryDrawer from "../components/library/HistoryDrawer.vue";
import LibraryChipRow from "../components/library/LibraryChipRow.vue";
import LibraryHeader from "../components/library/LibraryHeader.vue";
import TrashBanner from "../components/library/TrashBanner.vue";
import TrashTileActions from "../components/library/TrashTileActions.vue";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import RenameDialog from "../components/shell/RenameDialog.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import {
  fetchGalleryMediaBytes,
  galleryMediaPath,
  isAudioItem,
  isMeshItem,
  isVideoItem,
  prepareNativeThumbnail,
  thumbnailTier,
} from "../lib/gallery/media";
import { applySelectionClick } from "../lib/gallery/selection";
import {
  planSequenceReuse,
  sequenceEditAvailability,
  sequenceGoneMessage,
  sequenceHostUnreachableMessage,
} from "@studio/lib/sequenceReuse";
import {
  displayTitle,
  rememberSessionScroll,
  sessionScrollPosition,
  validatePrintTitle,
  type MergedCollection,
} from "@studio/lib/libraryOrganization";
import { ApiError, type ApiTarget } from "../lib/api/client";
import {
  retainedSourceMediaDisclosable,
  retainedSourceMediaDisclosure,
  retainedSourceMediaInventory,
} from "@studio/api/gallerySourceMedia";
import {
  useGalleryStore,
  type FanoutResult,
  type GalleryKindFilter,
  type GalleryLocation,
  type LibraryScope,
  type MergedPrint,
} from "../stores/gallery";
import { useModelStore } from "../stores/models";
import { useChainJobsStore } from "../stores/chainJobs";
import { useComposerStore } from "../stores/composer";
import { useGenerateFormStore } from "../stores/generateForm";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { inTauri, ipc } from "../lib/ipc";
import { copyImageBytesToClipboard } from "../lib/clipboard";
import { copyLocalOutputPath } from "../lib/localOutputPath";
import { formatBytes } from "../lib/format";
import { primaryModifierPressed } from "../lib/platform";
import { allowsNativeContextMenu, allowsNativeSelectAll, isSelectAllChord } from "../lib/shortcuts";
import { modelDisplayNameForId } from "../lib/models";
import type { GalleryImage } from "../lib/api/types";
import { isUpscaledImage } from "../lib/gallery/upscaled";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { readGalleryMediaBase64, readGalleryMediaBlob } from "../lib/gallery/sourceMedia";
import { attachPickedImage, attachPickedVideo } from "../lib/sourceAttachment";
import {
  appendMinimaxH3GalleryImageReference,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  setMinimaxH3GalleryImageFirstFrame,
} from "@studio/lib/minimaxH3Authoring";
import {
  createFramewiseUpscale,
  findRecoverableFramewiseUpscale,
  getFramewiseUpscale,
  transitionFramewiseUpscale,
  upscaleLibraryImage,
  type VideoUpscaleJob,
} from "@studio/api/videoUpscale";
import { useHostsStore } from "../stores/hosts";
import {
  defaultUpscaler,
  framewiseProgress,
  framewiseStatus,
  libraryUpscaleLabel,
  shouldPollFramewiseJob,
} from "@studio/lib/upscale";

const GAP = 8;
const PAD = 16;
/** Extra hosts have no SSE — their buckets poll while the view is open. */
const EXTRA_POLL_MS = 15_000;
/** Tags offered in the tile menu's Tags ▸ submenu. */
const MENU_TAG_LIMIT = 12;
/** Pointer sweep selection auto-scrolls near the top/bottom of the grid. */
const DRAG_SCROLL_EDGE = 72;
const DRAG_SCROLL_MAX = 18;
const DESKTOP_LIBRARY_SCROLL_KEY = "desktop:library";

const router = useRouter();
const route = useRoute();
const gallery = useGalleryStore();
const hosts = useHostsStore();
const models = useModelStore();
const chains = useChainJobsStore();
const composer = useComposerStore();
const generateForm = useGenerateFormStore();
const contextMenu = useContextMenuStore();
const toasts = useToastStore();
/** `modelDisplayNameForId` scans the installed list; a grid of 2 000 tiles
 *  would scan it 2 000 times per render. Memoize per name, dropping the memo
 *  whenever the installed inventory changes. */
const modelLabelMemo = computed(() => {
  const inventory = models.all;
  const cache = new Map<string, string>();
  return (name: string) => {
    let label = cache.get(name);
    if (label === undefined) {
      label = modelDisplayNameForId(name, inventory);
      cache.set(name, label);
    }
    return label;
  };
});
const modelLabel = (name: string) => modelLabelMemo.value(name);

const targetFor = (entry: MergedPrint): ApiTarget | null => gallery.targetOf(entry.sourceKey);

// ── NEW badges ──────────────────────────────────────────────────────────────
// A print not seen as of the last Library visit wears a NEW badge. Snapshot the
// pre-visit "seen" set (and whether we've ever visited) at setup so this
// visit's badges survive `markLibrarySeen` below; the very first visit only
// establishes the baseline and shows nothing new.
const freshBaseline = new Set(gallery.seenFilenames);
const hadVisited = gallery.libraryVisited;
const isFresh = (entry: MergedPrint) => hadVisited && !freshBaseline.has(entry.item.filename);
// Mark the current prints seen once they've loaded, so re-opening clears NEW.
watch(
  () => gallery.loaded,
  (loaded) => {
    if (loaded) gallery.markLibrarySeen();
  },
  { immediate: true },
);

// ── Scopes (V3 "Shelf": Prints | Collections | Trash) ────────────────────────
// Capability-gated: Collections needs an organize-capable host, Trash a
// trash-capable one (or this device's offline `.trash/` listing). With only
// Prints the control disappears and every organization affordance hides, so
// old servers keep today's Library and its hard-delete wording.
const organizeAvailable = computed(() => gallery.anyOrganizeCapable);
const offlineLocalTrash = computed(() => inTauri() && !gallery.hostFor("local"));
const trashAvailable = computed(() => gallery.anyTrashCapable || offlineLocalTrash.value);
const scopes = computed<LibraryScope[]>(() => [
  "prints",
  ...(organizeAvailable.value ? (["collections"] as const) : []),
  ...(trashAvailable.value ? (["trash"] as const) : []),
]);
const scope = computed<LibraryScope>(() =>
  scopes.value.includes(gallery.scope) ? gallery.scope : "prints",
);
const inTrash = computed(() => scope.value === "trash");
const inCollections = computed(() => scope.value === "collections");
/** The open collection (drill-in), resolved from the merged shelf. */
const openCollection = computed<MergedCollection | null>(() =>
  inCollections.value && gallery.collectionSlug
    ? (gallery.mergedCollections.find((c) => c.slug === gallery.collectionSlug) ?? null)
    : null,
);
const drillInName = computed(() =>
  inCollections.value && gallery.collectionSlug
    ? (openCollection.value?.name ?? gallery.collectionSlug)
    : null,
);
/** The shelf (cards) shows only in Collections with no collection open. */
const showShelf = computed(() => inCollections.value && !gallery.collectionSlug);

function setScope(next: LibraryScope) {
  if (next === gallery.scope) return;
  gallery.scope = next;
  setSelectMode(false);
  selected.value = null;
  lightboxOpen.value = false;
  if (next === "trash") void gallery.fetchTrash();
  if (next === "collections") void gallery.fetchCollections();
}

/** What the grid renders right now: the trash, or the live filtered set
 *  (which already applies ♥ / tags / the open collection). */
const entries = computed<MergedPrint[]>(() =>
  inTrash.value ? gallery.trashFiltered : gallery.filtered,
);

const orgOf = (entry: MergedPrint) => gallery.organizationOf(entry);
const isFavorite = (entry: MergedPrint) => orgOf(entry).favorite;
const tileTitle = (entry: MergedPrint) =>
  displayTitle({ ...entry.item, title: orgOf(entry).title });

/** Some copy of this print sits on a host that can organize. */
const canOrganizeEntry = (entry: MergedPrint) =>
  gallery.allLocationsOf(entry).some((l) => gallery.organizeCapable(l.sourceKey));
/** Every copy of this print deletes into a trash (else the old hard delete). */
const entryTrashCapable = (entry: MergedPrint) => {
  const locations = gallery.locationsOf(entry);
  if (locations.length === 0) return gallery.trashCapable(entry.sourceKey);
  return locations.every(
    (l) =>
      gallery.trashCapable(l.sourceKey) || (l.sourceKey === "local" && offlineLocalTrash.value),
  );
};

// ── Header labels ────────────────────────────────────────────────────────────
const hostScopeLabel = computed(() =>
  gallery.filter === "all"
    ? null
    : (gallery.sources.find((s) => s.key === gallery.filter)?.label ?? null),
);
const scopeCounts = computed(() => {
  const hidden = new Set(
    gallery.mergedCollections
      .filter((collection) => collection.hidden)
      .map((collection) => collection.slug),
  );
  const prints = gallery.merged.filter(
    (entry) => !gallery.organizationOf(entry).collections.some((slug) => hidden.has(slug)),
  ).length;
  return {
    prints,
    collections: gallery.mergedCollections.length,
    trash: gallery.trashCount,
  };
});
const countLabel = computed(() => {
  if (showShelf.value) {
    const n = shelfCards.value.length;
    return `${n} ${n === 1 ? "collection" : "collections"}`;
  }
  const list = entries.value;
  const n = list.length;
  const noun = n === 1 ? "print" : "prints";
  const parts = [inTrash.value ? `${n} ${noun} in trash` : `${n} ${noun}`];
  const bytes = list.reduce((sum, e) => sum + (e.item.size_bytes ?? 0), 0);
  if (bytes > 0) parts.push(formatBytes(bytes));
  if (hostScopeLabel.value) parts.push(hostScopeLabel.value);
  return parts.join(" · ");
});
const trashBytes = computed(() =>
  gallery.trashMerged.reduce((sum, e) => sum + (e.item.size_bytes ?? 0), 0),
);
const favoritesCount = computed(
  () => gallery.basePrints.filter((entry) => isFavorite(entry)).length,
);
const sourceLabel = (key: string) => gallery.sources.find((s) => s.key === key)?.label ?? key;
/** "This Mac · plato" — every organize-capable host, for the fan-out notes. */
const organizeHostNote = computed(() => {
  const labels = gallery.sources.filter((s) => gallery.organizeCapable(s.key)).map((s) => s.label);
  return labels.length > 0 ? labels.join(" · ") : null;
});

/** Every SELECTED logical print has at least one copy on an
 *  organize-capable host — the bulk bar's Favorite / Tag / Collection
 *  controls act on nothing otherwise. Empty selections stay enabled (the
 *  buttons already disable on `none`) so the bar reads normally. */
const selectionOrganizeBlockedReason = computed<string | null>(() => {
  const blocked = selectedEntries.value.filter((e) => !canOrganizeEntry(e));
  if (blocked.length === 0) return null;
  const labels = [...new Set(blocked.flatMap((e) => e.availableOn.map((s) => s.label)))];
  const n = blocked.length;
  return `${n} selected ${n === 1 ? "print lives" : "prints live"} only on ${joinNames(labels)}, which can't organize prints.`;
});

/** Reveal works for files on this Mac: the IPC bucket, or a local-kind
 *  (built-in/external) engine whose output dir is this machine's. */
const canReveal = (entry: MergedPrint) =>
  entry.sourceKey === "local" || gallery.hostFor(entry.sourceKey)?.kind === "local";

/** Copyable to this Mac: a remote-origin tile with no local copy yet (by
 *  filename or byte identity). The menu item stays visible and grays out
 *  once a local copy exists. */
const canSaveLocally = (entry: MergedPrint) =>
  inTauri() && gallery.hostFor(entry.sourceKey)?.kind === "remote" && !gallery.existsLocally(entry);

/** Authed source bytes for a host-gallery item (origin-aware). */
async function fetchItemBlob(entry: MergedPrint): Promise<Blob> {
  return readGalleryMediaBlob(entry, gallery);
}

async function fetchItemBase64(entry: MergedPrint): Promise<string> {
  return readGalleryMediaBase64(entry, gallery);
}

async function saveToThisMac(entry: MergedPrint) {
  try {
    // The origin row's metadata rides along so the local DB row matches the
    // origin exactly — videos embed nothing in the file itself.
    const saved = await ipc.saveOutputBytes(
      entry.item.filename,
      await fetchItemBase64(entry),
      entry.item.metadata,
    );
    toasts.push(`Saved locally — ${saved}`);
    void gallery.refreshHost("local");
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

// ── Upscale ────────────────────────────────────────────────────────────────
const upscalingFilename = ref<string | null>(null);
const upscaleEntry = ref<MergedPrint | null>(null);
const upscaleModel = ref(defaultUpscaler(models.upscalers));
const upscaleJob = ref<VideoUpscaleJob | null>(null);
const upscaleError = ref("");
let upscalePoll: ReturnType<typeof setTimeout> | null = null;
let upscaleEpoch = 0;
const upscaleKind = computed(() =>
  upscaleEntry.value && isVideo(upscaleEntry.value.item) ? "video" : "image",
);
type UpscaleAuthority = {
  sourceKey: string;
  label: string;
  filename: string;
  target: ApiTarget;
};
const upscaleAuthority = ref<UpscaleAuthority | null>(null);

/**
 * A saved-local print is represented by the local row, but its merged print
 * still carries the original host copy. Prefer the displayed copy when it can
 * run the operation; otherwise fall back to a capable copy of the same print.
 */
function upscaleAuthoritiesFor(entry: MergedPrint | null): UpscaleAuthority[] {
  if (!entry || isAudio(entry.item) || isMesh(entry.item)) return [];
  // Host-chip projections intentionally carry only their displayed bucket.
  // Recover the logical print here so upscale can still see alternate copies
  // without widening delete/organize actions on the host-filtered tile.
  const logical = entry.copies?.length
    ? entry
    : (gallery.mergedIndex.get(entry.item.filename) ?? entry);
  const copies = logical.copies?.length
    ? [...logical.copies]
    : [{ sourceKey: entry.sourceKey, item: entry.item }];
  copies.sort((left, right) => {
    const leftPreferred = left.sourceKey === entry.sourceKey ? 0 : 1;
    const rightPreferred = right.sourceKey === entry.sourceKey ? 0 : 1;
    return leftPreferred - rightPreferred;
  });
  const authorities: UpscaleAuthority[] = [];
  const seen = new Set<string>();
  for (const copy of copies) {
    if (seen.has(copy.sourceKey)) continue;
    seen.add(copy.sourceKey);
    const target = gallery.targetOf(copy.sourceKey);
    if (!target) continue;
    if (
      isVideo(copy.item) &&
      hosts.capabilities[copy.sourceKey]?.video_upscale?.available !== true
    ) {
      continue;
    }
    authorities.push({
      sourceKey: copy.sourceKey,
      label:
        logical.availableOn.find((source) => source.key === copy.sourceKey)?.label ??
        copy.sourceKey,
      filename: copy.item.filename,
      target,
    });
  }
  return authorities;
}

const upscaleAuthorityFor = (entry: MergedPrint | null) => upscaleAuthoritiesFor(entry)[0] ?? null;
const upscaleHostChoices = computed(() =>
  upscaleAuthoritiesFor(upscaleEntry.value).map(({ sourceKey, label }) => ({
    key: sourceKey,
    label,
  })),
);

function stopUpscalePoll() {
  if (upscalePoll) clearTimeout(upscalePoll);
  upscalePoll = null;
}

function closeUpscaleDialog() {
  upscaleEpoch += 1;
  stopUpscalePoll();
  upscaleEntry.value = null;
  upscaleAuthority.value = null;
  upscaleJob.value = null;
  upscaleError.value = "";
}

async function openUpscaleDialog(entry: MergedPrint) {
  const authority = upscaleAuthorityFor(entry);
  if (!authority) return;
  stopUpscalePoll();
  const epoch = ++upscaleEpoch;
  upscaleEntry.value = entry;
  upscaleAuthority.value = authority;
  upscaleJob.value = null;
  upscaleError.value = "";
  upscaleModel.value = defaultUpscaler(models.upscalers);
  if (isVideo(entry.item)) {
    try {
      const recovered = await findRecoverableFramewiseUpscale(authority.target, authority.filename);
      if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
      upscaleJob.value = recovered;
      if (recovered) upscaleModel.value = recovered.model;
      if (shouldPollFramewiseJob(upscaleJob.value)) void pollUpscaleJob();
    } catch {
      // Old hosts do not expose durable video-upscale history.
    }
  }
}

async function selectUpscaleAuthority(sourceKey: string) {
  const entry = upscaleEntry.value;
  const authority = upscaleAuthoritiesFor(entry).find((choice) => choice.sourceKey === sourceKey);
  if (!entry || !authority || authority.sourceKey === upscaleAuthority.value?.sourceKey) return;
  stopUpscalePoll();
  const epoch = ++upscaleEpoch;
  upscaleAuthority.value = authority;
  upscaleJob.value = null;
  upscaleError.value = "";
  if (!isVideo(entry.item)) return;
  try {
    const recovered = await findRecoverableFramewiseUpscale(authority.target, authority.filename);
    if (
      epoch !== upscaleEpoch ||
      upscaleEntry.value !== entry ||
      upscaleAuthority.value !== authority
    ) {
      return;
    }
    upscaleJob.value = recovered;
    if (recovered) upscaleModel.value = recovered.model;
    if (shouldPollFramewiseJob(recovered)) void pollUpscaleJob();
  } catch {
    // Old hosts do not expose durable video-upscale history.
  }
}

function canUpscaleEntry(entry: MergedPrint | null): boolean {
  return upscaleAuthorityFor(entry) !== null;
}

async function pollUpscaleJob() {
  const entry = upscaleEntry.value;
  const authority = upscaleAuthority.value;
  const job = upscaleJob.value;
  if (!entry || !authority || !job || !shouldPollFramewiseJob(job)) {
    return;
  }
  const epoch = upscaleEpoch;
  try {
    const next = await getFramewiseUpscale(authority.target, job.id);
    if (epoch !== upscaleEpoch || upscaleEntry.value !== entry || upscaleJob.value?.id !== job.id)
      return;
    upscaleJob.value = next;
    if (next.state === "completed") {
      toasts.push(`Framewise upscale complete — ${next.output_filename}`);
      void gallery.refreshHost(authority.sourceKey);
    }
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
    upscaleError.value = error instanceof Error ? error.message : String(error);
    toasts.push(upscaleError.value, "error");
    return;
  }
  if (shouldPollFramewiseJob(upscaleJob.value))
    upscalePoll = setTimeout(() => void pollUpscaleJob(), 750);
}

async function legacyStreamUpscale(entry: MergedPrint): Promise<string> {
  const { sseStream } = await import("../lib/api/sse");
  const authority = upscaleAuthority.value;
  if (!authority) throw new Error("The print's source host is unavailable.");
  const image = await fetchItemBase64(entry);
  return new Promise<string>((resolve, reject) => {
    const abort = new AbortController();
    let settled = false;
    void sseStream("/api/upscale/stream", {
      method: "POST",
      target: authority.target,
      body: {
        model: upscaleModel.value,
        image,
        output_format: "png",
      },
      signal: abort.signal,
      retry: false,
      onEvent: (event, data) => {
        try {
          if (event === "complete") {
            settled = true;
            resolve((JSON.parse(data) as { image: string }).image);
          } else if (event === "error") {
            settled = true;
            const parsed = JSON.parse(data) as { message?: string; error?: string };
            reject(new Error(parsed.message ?? parsed.error ?? data));
          }
        } catch (error) {
          settled = true;
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      },
      onClose: (error) => {
        if (!settled) reject(error ?? new Error("The upscale stream ended without a result."));
      },
    });
  });
}

async function startUpscale() {
  const entry = upscaleEntry.value;
  const authority = upscaleAuthority.value;
  if (!entry || !authority || upscalingFilename.value) return;
  const epoch = ++upscaleEpoch;
  stopUpscalePoll();
  upscaleError.value = "";
  upscalingFilename.value = entry.item.filename;
  try {
    if (isVideo(entry.item)) {
      const created = await createFramewiseUpscale(
        authority.target,
        authority.filename,
        upscaleModel.value,
      );
      if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
      upscaleJob.value = created;
      toasts.push(`Framewise upscale queued (${created.id}).`);
      void pollUpscaleJob();
    } else {
      if (hosts.capabilities[authority.sourceKey]?.video_upscale?.gallery_image === true) {
        const result = await upscaleLibraryImage(
          authority.target,
          authority.filename,
          upscaleModel.value,
        );
        toasts.push(`Upscaled — ${result.filename}`);
        void gallery.refreshHost(authority.sourceKey);
      } else {
        const upscaled = await legacyStreamUpscale(entry);
        const stem = entry.item.filename.replace(/\.[^.]+$/, "");
        const saved = await ipc.saveOutputBytes(`${stem}-upscaled.png`, upscaled);
        toasts.push(`Upscaled — saved locally as ${saved}`);
        void gallery.refreshHost("local");
      }
      closeUpscaleDialog();
    }
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
    upscaleError.value = error instanceof Error ? error.message : String(error);
    toasts.push(upscaleError.value, "error");
  } finally {
    upscalingFilename.value = null;
  }
}

async function transitionUpscale(action: "pause" | "resume" | "cancel") {
  const entry = upscaleEntry.value;
  const authority = upscaleAuthority.value;
  const job = upscaleJob.value;
  if (!entry || !authority || !job) return;
  stopUpscalePoll();
  const epoch = ++upscaleEpoch;
  upscaleError.value = "";
  try {
    const transitioned = await transitionFramewiseUpscale(authority.target, job.id, action);
    if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
    upscaleJob.value = transitioned;
    if (action === "resume") void pollUpscaleJob();
  } catch (error) {
    if (epoch !== upscaleEpoch || upscaleEntry.value !== entry) return;
    upscaleError.value = error instanceof Error ? error.message : String(error);
    toasts.push(upscaleError.value, "error");
  }
}

// ── Sequence prints ─────────────────────────────────────────────────────────
// A print stitched from a sequence carries per-clip provenance
// (`metadata.chain`) and, when a durable job produced it, that job's id. Reuse
// settings follows the print: one shot for a still, a fresh clip rail for a
// sequence. A sequence's primary action CONTINUES the original durable job
// with its cached clips; Duplicate as new is the explicit fresh-draft path.

const isSequencePrint = (entry: MergedPrint) => planSequenceReuse(entry.item.metadata) !== null;

/**
 * The producing host — resolved ONLY from the entry's own origin bucket. A
 * merged print may live on three hosts; the other two hold auto-saved copies,
 * and a job-id hit there would edit an unrelated sequence.
 */
const originHostId = (entry: MergedPrint) => gallery.hostFor(entry.sourceKey)?.id ?? null;
const originHostLabel = (entry: MergedPrint) =>
  gallery.hostFor(entry.sourceKey)?.label ?? entry.hostLabel;

/** Render-time gate — never probes. See `sequenceEditAvailability`. */
function canEditSequence(entry: MergedPrint): boolean {
  if (!isSequencePrint(entry)) return false;
  const hostId = originHostId(entry);
  return (
    sequenceEditAvailability({
      chainJobId: entry.item.metadata.chain_job_id,
      hostId,
      knownJobIds: hostId ? (chains.byHost[hostId]?.jobs.map((job) => job.id) ?? null) : null,
    }) === "available"
  );
}

/** Load the recorded clips into Create as a NEW sequence draft. */
function reuseSequence(entry: MergedPrint) {
  composer.setSequence({ kind: "reuse", metadata: entry.item.metadata });
  lightboxOpen.value = false;
  void router.push({ path: "/create", query: { output: "sequence" } });
}

function reuseSettings(entry: MergedPrint) {
  if (isSequencePrint(entry)) {
    reuseSequence(entry);
    return;
  }
  // Full metadata → full-fidelity restore (negative prompt, LoRAs,
  // scheduler, video params, …) via `applyPrefillToForm`.
  const retainedVersion = composer.beginRetainedSourceReuse({ metadata: entry.item.metadata });
  const target = gallery.targetOf(entry.sourceKey);
  // Always ask — the host is the only authority on what it retained, and the
  // metadata under-reports inline video/audio/mask bytes. But a text-to-image
  // print's archive entry resolves with no pins, which the server can only
  // report as `unavailable_legacy`, so an UNAVAILABLE answer is toasted only
  // when the print's own metadata says conditioning bytes were shipped.
  if (target) {
    void retainedSourceMediaInventory(entry.item.filename, target)
      .then((inventory) => {
        if (
          !composer.setRetainedSourceIfCurrent(retainedVersion, {
            filename: entry.item.filename,
            origin: target,
            inventory,
          })
        ) {
          return;
        }
        const disclosure = retainedSourceMediaDisclosable(entry.item.metadata)
          ? retainedSourceMediaDisclosure(inventory.availability)
          : null;
        if (disclosure) toasts.push(disclosure, "error");
      })
      .catch(() => {
        // The established local stash/gallery-name restore stays live. A
        // transport failure inspecting the additive endpoint must not turn a
        // previously working Reuse settings action into a dead end.
      });
  }
  lightboxOpen.value = false;
  void router.push("/create");
}

/**
 * Check once, on click. A 404 means the job was deleted or GC'd, so fall back
 * to the reuse path rather than leaving an enabled control as a dead end; any
 * other failure keeps the cached clips by refusing to downgrade.
 */
async function editSequence(entry: MergedPrint) {
  const hostId = originHostId(entry);
  const jobId = entry.item.metadata.chain_job_id;
  if (!hostId || !jobId) return;
  try {
    await chains.fetchDetail(hostId, jobId);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      toasts.push(sequenceGoneMessage(originHostLabel(entry)));
      reuseSequence(entry);
      return;
    }
    toasts.push(sequenceHostUnreachableMessage(originHostLabel(entry)), "error");
    return;
  }
  composer.setSequence({ kind: "edit", hostId, jobId });
  lightboxOpen.value = false;
  void router.push({ path: "/create", query: { output: "sequence" } });
}

// ── Organization actions (fan-out; the store reaches every copy) ────────────
/** Toast a fan-out outcome: silent on success, the first error otherwise. */
function reportFanout(result: FanoutResult, okMessage?: string) {
  if (result.failed > 0) {
    const hosts = result.failedHosts.map(sourceLabel).join(", ");
    toasts.push(result.error ? `${result.error} (${hosts})` : `Failed on ${hosts}.`, "error");
    return false;
  }
  if (okMessage) toasts.push(okMessage);
  return true;
}

/** The prints an action applies to: the bulk selection when the entry is
 *  part of it (and there is more than one), else the entry alone. */
function actionTargets(entry: MergedPrint): MergedPrint[] {
  if (
    selectMode.value &&
    bulkSelection.value.size > 1 &&
    bulkSelection.value.has(entry.item.filename)
  ) {
    return selectedEntries.value;
  }
  return [entry];
}

async function setFavorite(targets: MergedPrint[], value: boolean) {
  if (targets.length === 0) return;
  reportFanout(await gallery.setFavorite(targets, value));
}

function toggleFavorite(entry: MergedPrint) {
  const targets = actionTargets(entry);
  // Any unfavorited → favorite all; else unfavorite all.
  void setFavorite(
    targets,
    targets.some((e) => !isFavorite(e)),
  );
}

async function applyTags(targets: MergedPrint[], change: { add: string[]; remove: string[] }) {
  if (targets.length === 0) return;
  if (change.add.length > 0) reportFanout(await gallery.addTags(targets, change.add));
  if (change.remove.length > 0) reportFanout(await gallery.removeTags(targets, change.remove));
}

async function toggleCollection(
  targets: MergedPrint[],
  change: { slug?: string; name?: string; checked: boolean },
) {
  if (targets.length === 0) return;
  // Zero addable copies would "add" nothing (and create-on-demand would
  // leave an empty collection) — refuse honestly instead.
  if (change.checked && gallery.organizeTargetsFor(targets).length === 0) {
    toasts.push("None of the selected prints are on a machine that can organize.", "error");
    return;
  }
  if (!change.checked) {
    if (!change.slug) return;
    reportFanout(await gallery.removeFromCollection(targets, change.slug));
    return;
  }
  const name =
    change.name ??
    gallery.mergedCollections.find((c) => c.slug === change.slug)?.name ??
    change.slug ??
    "";
  if (!name) return;
  const result = await gallery.addToCollection(
    targets,
    change.slug ? { slug: change.slug, name } : { name },
  );
  const n = targets.length;
  reportFanout(result, `Added ${n} ${n === 1 ? "print" : "prints"} to “${name}”`);
}

async function renamePrint(entry: MergedPrint, raw: string | null) {
  const check = validatePrintTitle(raw ?? "");
  if (!check.ok) {
    toasts.push(check.reason, "error");
    return;
  }
  reportFanout(await gallery.setTitle(entry, check.value));
}

// ── Dialogs ─────────────────────────────────────────────────────────────────
const renameTarget = ref<MergedPrint | null>(null);
const newTagTargets = ref<MergedPrint[] | null>(null);
/** New collection dialog: the prints to add once created (may be empty). */
const newCollectionTargets = ref<MergedPrint[] | null>(null);
const collectionRenameSlug = ref<string | null>(null);
const collectionDeleteSlug = ref<string | null>(null);
const emptyTrashOpen = ref(false);
const deleteForeverTargets = ref<MergedPrint[] | null>(null);
const organizeBusy = ref(false);

const collectionNamed = (slug: string | null) =>
  slug ? (gallery.mergedCollections.find((c) => c.slug === slug)?.name ?? slug) : "";

function openRename(entry: MergedPrint) {
  renameTarget.value = entry;
}
async function onRenameSave(name: string) {
  const entry = renameTarget.value;
  renameTarget.value = null;
  if (entry) await renamePrint(entry, name);
}

function openNewTag(targets: MergedPrint[]) {
  newTagTargets.value = targets;
}
async function onNewTagSave(name: string) {
  const targets = newTagTargets.value ?? [];
  newTagTargets.value = null;
  await applyTags(targets, { add: [name], remove: [] });
}

function openNewCollection(targets: MergedPrint[] = []) {
  newCollectionTargets.value = targets;
}
async function createCollection(name: string, targets: MergedPrint[] = []) {
  // Creating for a selection must not leave an empty collection behind when
  // no selected copy sits on an organize-capable host — refuse honestly
  // instead of creating everywhere and "adding" nothing.
  if (targets.length > 0 && gallery.organizeTargetsFor(targets).length === 0) {
    const n = targets.length;
    toasts.push(
      `None of the ${n === 1 ? "selected print's copies live" : `${n} selected prints' copies live`} on a machine that can organize — no collection was created.`,
      "error",
    );
    return;
  }
  organizeBusy.value = true;
  try {
    const result = await gallery.createCollection(name);
    if (!reportFanout(result)) return;
    if (targets.length > 0) {
      const added = await gallery.addToCollection(targets, { slug: result.slug, name });
      const n = targets.length;
      reportFanout(added, `Added ${n} ${n === 1 ? "print" : "prints"} to “${name}”`);
    } else {
      toasts.push(`Created collection “${name}”`);
    }
  } finally {
    organizeBusy.value = false;
  }
}
async function onNewCollectionSave(name: string) {
  const targets = newCollectionTargets.value ?? [];
  newCollectionTargets.value = null;
  await createCollection(name, targets);
}

async function onCollectionRenameSave(name: string) {
  const slug = collectionRenameSlug.value;
  collectionRenameSlug.value = null;
  if (!slug) return;
  reportFanout(await gallery.renameCollection(slug, name), `Renamed to “${name}”`);
}

async function setCollectionHidden(slug: string, hidden: boolean) {
  const name = collectionNamed(slug);
  reportFanout(
    await gallery.setCollectionHidden(slug, hidden),
    hidden ? `Hidden collection “${name}”` : `Showing collection “${name}”`,
  );
}

async function confirmDeleteCollection() {
  const slug = collectionDeleteSlug.value;
  collectionDeleteSlug.value = null;
  if (!slug) return;
  const name = collectionNamed(slug);
  organizeBusy.value = true;
  try {
    if (reportFanout(await gallery.deleteCollection(slug), `Deleted collection “${name}”`)) {
      if (gallery.collectionSlug === slug) gallery.collectionSlug = null;
    }
  } finally {
    organizeBusy.value = false;
  }
}

async function useAsCover(entry: MergedPrint) {
  const slug = gallery.collectionSlug;
  if (!slug) return;
  reportFanout(await gallery.setCollectionCover(slug, entry), "Cover set");
}

async function removeFromOpenCollection(targets: MergedPrint[]) {
  const slug = gallery.collectionSlug;
  if (!slug || targets.length === 0) return;
  const n = targets.length;
  reportFanout(
    await gallery.removeFromCollection(targets, slug),
    `Removed ${n} ${n === 1 ? "print" : "prints"} from “${collectionNamed(slug)}”`,
  );
  pruneSelection();
}

// ── Trash actions ───────────────────────────────────────────────────────────
async function restorePrints(targets: MergedPrint[]) {
  if (targets.length === 0) return;
  organizeBusy.value = true;
  try {
    const result = await gallery.restore(targets);
    const n = result.restored;
    reportFanout(result, `Restored ${n} ${n === 1 ? "print" : "prints"}`);
  } finally {
    organizeBusy.value = false;
  }
  pruneSelection();
}

function askDeleteForever(targets: MergedPrint[]) {
  if (targets.length === 0) return;
  deleteForeverTargets.value = targets;
}
const deleteForeverTitle = computed(() => {
  const targets = deleteForeverTargets.value ?? [];
  if (targets.length === 1) return `Delete “${tileTitle(targets[0]!)}” forever?`;
  return `Delete ${targets.length} prints forever?`;
});
async function confirmDeleteForever() {
  const targets = deleteForeverTargets.value ?? [];
  deleteForeverTargets.value = null;
  if (targets.length === 0) return;
  organizeBusy.value = true;
  try {
    const result = await gallery.deleteForever(targets);
    if (result.failedPrints > 0) {
      toasts.push(
        result.error ??
          `${result.failedPrints} ${result.failedPrints === 1 ? "print" : "prints"} could not be deleted.`,
        "error",
      );
    } else {
      const n = result.deletedPrints;
      toasts.push(`Deleted ${n} ${n === 1 ? "print" : "prints"} forever`);
    }
  } finally {
    organizeBusy.value = false;
  }
  pruneSelection();
}

const trashHostLabels = computed(() => {
  const labels = gallery.retentionByHost.map((h) => h.label);
  if (labels.length === 0 && offlineLocalTrash.value) labels.push(sourceLabel("local"));
  return labels;
});
function joinNames(names: string[]): string {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
const emptyTrashMessage = computed(() => {
  const n = gallery.trashCount;
  const where = joinNames(trashHostLabels.value);
  return `Delete ${n} ${n === 1 ? "print" : "prints"} in the trash${where ? ` on ${where}` : ""} forever? This can't be undone.`;
});
async function confirmEmptyTrash() {
  emptyTrashOpen.value = false;
  organizeBusy.value = true;
  try {
    const result = await gallery.emptyTrash();
    reportFanout(
      result,
      `Deleted ${result.purged} ${result.purged === 1 ? "print" : "prints"} forever`,
    );
  } finally {
    organizeBusy.value = false;
  }
  selected.value = null;
  lightboxOpen.value = false;
  clearBulkSelection();
}

/** The retention link: this device's value lives in Settings ▸ Library, a
 *  remote's in Machines ▸ host. */
const retentionLinkLabel = computed(() =>
  gallery.retentionByHost.some((h) => h.key === "local") || gallery.retentionByHost.length === 0
    ? "Change retention · Settings"
    : "Change retention · Machines",
);
function changeRetention() {
  const hostsWithTrash = gallery.retentionByHost;
  if (hostsWithTrash.some((h) => h.key === "local") || hostsWithTrash.length === 0) {
    void router.push({ path: "/settings", query: { section: "library" } });
  } else if (hostsWithTrash.length === 1) {
    void router.push(`/machines/${encodeURIComponent(hostsWithTrash[0]!.key)}`);
  } else {
    void router.push("/machines");
  }
}

// ── Collections shelf ───────────────────────────────────────────────────────
/** Collection slug → its logical prints, newest first (one pass). */
const membersBySlug = computed(() => {
  const map = new Map<string, MergedPrint[]>();
  for (const entry of gallery.merged) {
    for (const slug of orgOf(entry).collections) {
      const list = map.get(slug) ?? [];
      list.push(entry);
      map.set(slug, list);
    }
  }
  return map;
});

function coverTile(entry: MergedPrint): CoverTile {
  const source = gallery.mediaSourceOf(entry.sourceKey);
  return {
    path: galleryMediaPath(entry.item.filename, source, true),
    target: targetFor(entry),
    cacheKey: entry.sourceKey,
    mediaVersion:
      entry.item.media_version ?? `${entry.item.timestamp}:${entry.item.size_bytes ?? "unknown"}`,
    // A cover is a thumbnail now even for this device offline, so it never
    // mounts a <video>.
    video: false,
    alt: tileTitle(entry),
  };
}

function coversFor(collection: MergedCollection): CoverTile[] {
  const members = membersBySlug.value.get(collection.slug) ?? [];
  let ordered = members;
  const cover = collection.cover;
  if (cover) {
    const at = members.findIndex(
      (e) =>
        e.item.filename === cover.filename ||
        (e.copies ?? []).some(
          (copy) => copy.sourceKey === cover.hostId && copy.item.filename === cover.filename,
        ),
    );
    if (at > 0) ordered = [members[at]!, ...members.slice(0, at), ...members.slice(at + 1)];
  }
  return ordered.slice(0, 4).map(coverTile);
}

function collectionUpdatedAt(collection: MergedCollection): number | null {
  let latest: number | null = null;
  for (const host of collection.hosts) {
    const row = gallery.collectionsByHost[host.hostId]?.items.find((c) => c.id === host.id);
    if (row?.updated_at != null && (latest === null || row.updated_at > latest)) {
      latest = row.updated_at;
    }
  }
  return latest;
}

const shelfCards = computed<ShelfCard[]>(() => {
  const q = gallery.query.trim().toLowerCase();
  return gallery.mergedCollections
    .filter((c) => !q || c.name.toLowerCase().includes(q))
    .map((c) => ({
      slug: c.slug,
      name: c.name,
      count: gallery.collectionCounts(c.slug),
      hostLabels: c.hosts.map((h) => sourceLabel(h.hostId)),
      updatedAt: collectionUpdatedAt(c),
      covers: coversFor(c),
      hidden: c.hidden === true,
    }));
});

const shelf = ref<InstanceType<typeof CollectionsShelf> | null>(null);

function openCollectionSlug(slug: string) {
  gallery.collectionSlug = slug;
  setSelectMode(false);
  selected.value = null;
}
function exitCollection() {
  gallery.collectionSlug = null;
  setSelectMode(false);
  selected.value = null;
  lightboxOpen.value = false;
}

function collectionMenu(slug: string): MenuEntry[] {
  const hidden =
    gallery.mergedCollections.find((collection) => collection.slug === slug)?.hidden === true;
  return [
    { label: "Open", action: () => openCollectionSlug(slug) },
    {
      label: hidden ? "Show in Library" : "Hide from Library",
      action: () => void setCollectionHidden(slug, !hidden),
    },
    { label: "Rename…", action: () => (collectionRenameSlug.value = slug) },
    { separator: true },
    {
      label: "Delete collection…",
      danger: true,
      action: () => (collectionDeleteSlug.value = slug),
    },
  ];
}

/** The drill-in's Edit menu: rename / remove prints / delete. */
function openEditMenu(event: MouseEvent) {
  const slug = gallery.collectionSlug;
  if (!slug) return;
  contextMenu.open(event, [
    { label: "Rename…", action: () => (collectionRenameSlug.value = slug) },
    {
      label: "Remove prints…",
      action: () => setSelectMode(true),
    },
    { separator: true },
    {
      label: "Delete collection…",
      danger: true,
      action: () => (collectionDeleteSlug.value = slug),
    },
  ]);
}

// ── Tile context menu ───────────────────────────────────────────────────────
function tagSubmenu(entry: MergedPrint): MenuEntry[] {
  const targets = actionTargets(entry);
  const orgs = targets.map(orgOf);
  const hasTag = (name: string) =>
    orgs.every((o) => o.tags.some((t) => t.toLowerCase() === name.toLowerCase()));
  const items: MenuEntry[] = gallery.mergedTags.slice(0, MENU_TAG_LIMIT).map((tag) => {
    const checked = hasTag(tag.name);
    return {
      label: tag.name,
      checked,
      action: () =>
        void applyTags(
          targets,
          checked ? { add: [], remove: [tag.name] } : { add: [tag.name], remove: [] },
        ),
    };
  });
  if (items.length > 0) items.push({ separator: true });
  items.push({ label: "New tag…", action: () => openNewTag(targets) });
  return items;
}

function collectionSubmenu(entry: MergedPrint): MenuEntry[] {
  const targets = actionTargets(entry);
  const orgs = targets.map(orgOf);
  const items: MenuEntry[] = gallery.mergedCollections.map((collection) => {
    const checked = orgs.every((o) => o.collections.includes(collection.slug));
    return {
      label: collection.name,
      checked,
      action: () =>
        void toggleCollection(targets, {
          slug: collection.slug,
          name: collection.name,
          checked: !checked,
        }),
    };
  });
  if (items.length > 0) items.push({ separator: true });
  items.push({ label: "New collection…", action: () => openNewCollection(targets) });
  return items;
}

function tileMenu(entry: MergedPrint): MenuEntry[] {
  const item = entry.item;
  const m = item.metadata;
  const bulkCount = bulkSelection.value.size;
  const selectedForBulk =
    selectMode.value && bulkSelection.value.has(item.filename) && bulkCount > 1;
  const copyItems: MenuEntry[] = [
    {
      label: "Copy prompt",
      action: () => {
        void navigator.clipboard.writeText(m.prompt).then(() => toasts.push("Copied"));
      },
    },
    {
      label: "Copy seed",
      action: () => {
        void navigator.clipboard.writeText(String(m.seed)).then(() => toasts.push("Copied"));
      },
    },
  ];
  if (inTrash.value) {
    const targets = actionTargets(entry);
    return [
      {
        label: selectedForBulk ? `Restore ${bulkCount} selected` : "Restore",
        action: () => void restorePrints(targets),
      },
      ...copyItems,
      { separator: true },
      {
        label: selectedForBulk ? `Delete ${bulkCount} selected forever` : "Delete forever",
        danger: true,
        action: () => askDeleteForever(targets),
      },
    ];
  }
  const organize = canOrganizeEntry(entry);
  const favorite = isFavorite(entry);
  const trashable = entryTrashCapable(entry);
  if (selectedForBulk) {
    const targets = selectedEntries.value;
    const allFavorite = targets.every(isFavorite);
    const allOrganizable = targets.every(canOrganizeEntry);
    const allTrashable = targets.every(entryTrashCapable);
    return [
      ...(allOrganizable
        ? [
            {
              label: allFavorite
                ? `Unfavorite ${bulkCount} selected`
                : `Favorite ${bulkCount} selected`,
              checked: allFavorite,
              action: () => void setFavorite(targets, !allFavorite),
            },
            { label: "Tags", children: tagSubmenu(entry) },
            { label: "Add to collection", children: collectionSubmenu(entry) },
            ...(gallery.collectionSlug && inCollections.value
              ? [
                  {
                    label: `Remove ${bulkCount} selected from collection`,
                    action: () => void removeFromOpenCollection(targets),
                  },
                ]
              : []),
            { separator: true } as MenuEntry,
          ]
        : []),
      {
        label: allTrashable
          ? `Move ${bulkCount} selected to trash`
          : `Delete ${bulkCount} selected`,
        danger: true,
        action: () => void deleteSelectedPrints(),
      },
    ];
  }
  return [
    ...(isSequencePrint(entry)
      ? [
          ...(canEditSequence(entry)
            ? [{ label: "Edit sequence", action: () => void editSequence(entry) }]
            : []),
          { label: "Duplicate as new", action: () => reuseSequence(entry) },
        ]
      : [{ label: "Reuse settings", action: () => reuseSettings(entry) }]),
    ...(organize
      ? [
          { separator: true } as MenuEntry,
          {
            label: favorite ? "Unfavorite" : "Favorite",
            checked: favorite,
            action: () => toggleFavorite(entry),
          },
          { label: "Rename…", action: () => openRename(entry) },
          { label: "Tags", children: tagSubmenu(entry) },
          { label: "Add to collection", children: collectionSubmenu(entry) },
          ...(gallery.collectionSlug && inCollections.value
            ? [
                { label: "Use as cover", action: () => void useAsCover(entry) },
                {
                  label: selectedForBulk
                    ? `Remove ${bulkCount} selected from collection`
                    : "Remove from collection",
                  action: () => void removeFromOpenCollection(actionTargets(entry)),
                },
              ]
            : []),
          { separator: true } as MenuEntry,
        ]
      : []),
    ...copyItems,
    {
      label: "Copy image",
      disabled: isVideo(item),
      action: () => void copyImage(entry),
    },
    {
      label: "Use as source",
      disabled: isAudio(item),
      action: () => void useAsSource(entry),
    },
    {
      label: "Copy file path",
      action: () =>
        void copyLocalOutputPath(item.filename)
          .then(() => toasts.push("File path copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          ),
    },
    { separator: true },
    {
      label:
        upscalingFilename.value === item.filename
          ? "Upscaling…"
          : libraryUpscaleLabel(isVideo(item) ? "video" : "image").replace("…", ""),
      disabled: upscalingFilename.value !== null || !canUpscaleEntry(entry),
      action: () => openUpscaleDialog(entry),
    },
    {
      label: "Save locally",
      disabled: !canSaveLocally(entry),
      action: () => void saveToThisMac(entry),
    },
    {
      label: "Reveal in file manager",
      disabled: !canReveal(entry),
      action: () =>
        void ipc.revealOutputFile(item.filename).catch((e) => {
          toasts.push(e instanceof Error ? e.message : String(e), "error");
        }),
    },
    { separator: true },
    {
      label: trashable
        ? selectedForBulk
          ? `Move ${bulkCount} selected to trash`
          : "Move to trash"
        : selectedForBulk
          ? `Delete ${bulkCount} selected`
          : "Delete",
      danger: true,
      action: () => {
        if (selectedForBulk) void deleteSelectedPrints();
        else deletePrint(entry);
      },
    },
  ];
}

const scrollEl = ref<HTMLElement | null>(null);
const containerWidth = ref(0);
const selected = ref<{ sourceKey: string; filename: string } | null>(null);
const lightboxOpen = ref(false);
const rowHeight = ref(loadGalleryThumbnailSize());
// A slider drag delivers one `input` event per pointer move; persisting on
// each was a synchronous localStorage write per frame. Settle it instead.
const SLIDER_PERSIST_MS = 250;
let sliderPersistTimer: ReturnType<typeof setTimeout> | null = null;
watch(rowHeight, (value) => {
  if (sliderPersistTimer) clearTimeout(sliderPersistTimer);
  sliderPersistTimer = setTimeout(() => {
    sliderPersistTimer = null;
    saveGalleryThumbnailSize(value);
  }, SLIDER_PERSIST_MS);
});

// ── Delete: optimistic + undoable (§08 G12) ──────────────────────────────────
// The tile leaves the grid instantly; a 6 s undo toast holds the print in limbo
// (no server call). Undo restores it; letting the toast expire commits the real
// DELETE — which on a trash-capable host moves the print to the trash. Several
// deletes can be pending at once, so timers are keyed per print (or per bulk
// batch).
const UNDO_WINDOW_MS = 6000;
const pendingDeletes = new Map<
  string,
  {
    locations: GalleryLocation[];
    timer: ReturnType<typeof setTimeout>;
  }
>();
const deleteKey = (sourceKey: string, filename: string) => `${sourceKey}::${filename}`;
let bulkDeleteSeq = 0;

// ── History drawer ──────────────────────────────────────────────────────────
// Open state lives in the URL (?panel=history) so the retired /history route
// and the command palette can deep-link straight into it; the header button
// toggles the same param, and closing clears it.
const historyOpen = computed(() => route.query.panel === "history");
function openHistory() {
  void router.push({ path: "/library", query: { ...route.query, panel: "history" } });
}
function closeHistory() {
  const query = { ...route.query };
  delete query.panel;
  void router.replace({ path: "/library", query });
}

// ── Search + media-kind chips ──────────────────────────────────────────────
const SEARCH_DEBOUNCE_MS = 200;
const searchInput = ref(gallery.query);
const header = ref<InstanceType<typeof LibraryHeader> | null>(null);
let searchTimer: ReturnType<typeof setTimeout> | null = null;
watch(searchInput, (value) => {
  if (searchTimer) clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    searchTimer = null;
    gallery.query = value;
  }, SEARCH_DEBOUNCE_MS);
});

const kindOptions = computed(() => [
  { value: "all" as GalleryKindFilter, label: "All" },
  { value: "image" as GalleryKindFilter, label: "Images" },
  { value: "video" as GalleryKindFilter, label: "Video" },
  { value: "audio" as GalleryKindFilter, label: "Audio" },
  { value: "mesh" as GalleryKindFilter, label: "3D" },
]);
const setKind = (value: GalleryKindFilter) => (gallery.mediaKind = value);

// ── Filter chips (♥ / tags / hosts) ─────────────────────────────────────────
const chipRow = ref<InstanceType<typeof LibraryChipRow> | null>(null);
const showChipRow = computed(
  () =>
    scope.value !== "trash" &&
    !showShelf.value &&
    (organizeAvailable.value || gallery.chipCounts.length > 1),
);
function toggleTagFilter(name: string) {
  const key = name.toLowerCase();
  const have = gallery.tagFilter.some((t) => t.toLowerCase() === key);
  gallery.tagFilter = have
    ? gallery.tagFilter.filter((t) => t.toLowerCase() !== key)
    : [...gallery.tagFilter, name];
}
function clearFilters() {
  gallery.favoritesOnly = false;
  gallery.tagFilter = [];
  gallery.filter = "all";
  if (inCollections.value) exitCollection();
}

// ── Bulk select mode ───────────────────────────────────────────────────────
// Selection is keyed by print identity (filename — the merged grid's
// cross-host identity), never row index: the virtualized grid re-flows.
// Pointer sweep selection samples the path across visible virtual rows and
// edge-scrolls, matching the iPhone Library gesture without relying on indexes.
const selectMode = ref(false);
const bulkSelection = ref<Set<string>>(new Set());
const bulkAnchor = ref<string | null>(null);
const confirmingBulkDelete = ref(false);
const bulkDeleting = ref(false);
const bulkBar = ref<InstanceType<typeof BulkBar> | null>(null);
let dragPointerId: number | null = null;
let dragSelect = true;
let dragClientX = 0;
let dragClientY = 0;
let dragFrame: number | null = null;
let dragPendingClicks = 0;
const dragVisited = new Set<string>();

const selectedEntries = computed(() =>
  entries.value.filter((e) => bulkSelection.value.has(e.item.filename)),
);
/** Organization state over the bulk selection, for the bar's popovers. */
const selectionOrganization = computed(() => {
  const orgs = selectedEntries.value.map(orgOf);
  if (orgs.length === 0) {
    return { collectionsAll: [], collectionsSome: [], tags: [], allFavorite: false };
  }
  const slugs = new Set(orgs.flatMap((o) => o.collections));
  const collectionsAll: string[] = [];
  const collectionsSome: string[] = [];
  for (const slug of slugs) {
    if (orgs.every((o) => o.collections.includes(slug))) collectionsAll.push(slug);
    else collectionsSome.push(slug);
  }
  const first = orgs[0]!;
  const tags = first.tags.filter((tag) =>
    orgs.every((o) => o.tags.some((t) => t.toLowerCase() === tag.toLowerCase())),
  );
  return {
    collectionsAll,
    collectionsSome,
    tags,
    allFavorite: orgs.every((o) => o.favorite),
  };
});
const selectionTrashCapable = computed(
  () => selectedEntries.value.length > 0 && selectedEntries.value.every(entryTrashCapable),
);

function setSelectMode(next: boolean) {
  if (!next) finishSelectionDrag();
  selectMode.value = next;
  if (!next) {
    bulkSelection.value = new Set();
    bulkAnchor.value = null;
    confirmingBulkDelete.value = false;
    bulkBar.value?.closePopovers();
  }
}

function onTileClick(entry: MergedPrint, e: MouseEvent) {
  if (selectMode.value && dragPendingClicks > 0 && e.detail !== 0) {
    dragPendingClicks -= 1;
    return;
  }
  if (!selectMode.value) {
    const extend = e.shiftKey;
    // Accept both platform conventions here: Command-click on macOS and
    // Control-click on Windows/Linux, including browser-based desktop QA.
    const toggle = e.metaKey || e.ctrlKey;
    if (extend || toggle) {
      const current = selectedEntry.value;
      const initial = new Set(current ? [current.item.filename] : []);
      const anchor = current?.item.filename ?? null;
      selectMode.value = true;
      const next = applySelectionClick(
        initial,
        anchor,
        entries.value.map((x) => x.item.filename),
        entry.item.filename,
        { shift: extend, meta: toggle },
      );
      bulkSelection.value = next.selection;
      bulkAnchor.value = next.anchor;
      return;
    }
    select(entry);
    return;
  }
  const next = applySelectionClick(
    bulkSelection.value,
    bulkAnchor.value,
    entries.value.map((x) => x.item.filename),
    entry.item.filename,
    { shift: e.shiftKey, meta: e.metaKey || e.ctrlKey },
  );
  bulkSelection.value = next.selection;
  bulkAnchor.value = next.anchor;
}

function entryAtPoint(x: number, y: number): MergedPrint | null {
  // The floating bulk bar may overlap the grid during edge scrolling, so use
  // the complete hit stack and find the first tile underneath it.
  const elements = document.elementsFromPoint?.(x, y) ?? [document.elementFromPoint(x, y)];
  const tile = elements
    .map((element) => element?.closest<HTMLElement>("[data-filename]") ?? null)
    .find((element) => element !== null);
  const filename = tile?.dataset.filename;
  return filename
    ? (entries.value.find((entry) => entry.item.filename === filename) ?? null)
    : null;
}

function applyDragSelection(entry: MergedPrint): void {
  const filename = entry.item.filename;
  if (dragVisited.has(filename)) return;
  dragVisited.add(filename);
  const next = new Set(bulkSelection.value);
  if (dragSelect) next.add(filename);
  else next.delete(filename);
  bulkSelection.value = next;
  bulkAnchor.value = filename;
  confirmingBulkDelete.value = false;
}

function applyDragAtPoint(): void {
  const entry = entryAtPoint(dragClientX, dragClientY);
  if (entry) applyDragSelection(entry);
}

function applyDragSegment(fromX: number, fromY: number, toX: number, toY: number): void {
  const distance = Math.hypot(toX - fromX, toY - fromY);
  // Sampling prevents a quick mouse sweep from jumping over narrow tiles.
  const steps = Math.max(1, Math.ceil(distance / 12));
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps;
    const entry = entryAtPoint(fromX + (toX - fromX) * progress, fromY + (toY - fromY) * progress);
    if (entry) applyDragSelection(entry);
  }
}

function runDragFrame(): void {
  dragFrame = null;
  if (dragPointerId === null || !selectMode.value) return;
  const scroller = scrollEl.value;
  if (scroller) {
    const bounds = scroller.getBoundingClientRect();
    const topDepth = Math.max(0, bounds.top + DRAG_SCROLL_EDGE - dragClientY);
    const bottomDepth = Math.max(0, dragClientY - (bounds.bottom - DRAG_SCROLL_EDGE));
    const direction = bottomDepth > 0 ? 1 : topDepth > 0 ? -1 : 0;
    const depth = Math.max(topDepth, bottomDepth);
    if (direction && depth) {
      const speed = Math.min(
        DRAG_SCROLL_MAX,
        Math.max(2, (depth / DRAG_SCROLL_EDGE) * DRAG_SCROLL_MAX),
      );
      scroller.scrollTop += direction * speed;
      applyDragAtPoint();
    }
  }
  dragFrame = requestAnimationFrame(runDragFrame);
}

function beginSelectionDrag(event: PointerEvent, entry: MergedPrint): void {
  if (
    !selectMode.value ||
    event.isPrimary === false ||
    event.shiftKey ||
    // macOS synthesizes a context menu from Control-click. Let the click /
    // contextmenu path handle it; Windows/Linux Ctrl-click still toggles via
    // onTileClick, while no platform needs Ctrl-modified sweep selection.
    (event.ctrlKey && !event.metaKey) ||
    (event.pointerType === "mouse" && event.button !== 0)
  ) {
    return;
  }
  event.preventDefault();
  dragPointerId = event.pointerId;
  dragSelect = !bulkSelection.value.has(entry.item.filename);
  dragClientX = event.clientX;
  dragClientY = event.clientY;
  dragVisited.clear();
  applyDragSelection(entry);
  (event.currentTarget as HTMLElement | null)?.setPointerCapture?.(event.pointerId);
  if (dragFrame === null) dragFrame = requestAnimationFrame(runDragFrame);
}

function moveSelectionDrag(event: PointerEvent): void {
  if (event.pointerId !== dragPointerId) return;
  event.preventDefault();
  const points = [...(event.getCoalescedEvents?.() ?? []), event];
  for (const point of points) {
    applyDragSegment(dragClientX, dragClientY, point.clientX, point.clientY);
    dragClientX = point.clientX;
    dragClientY = point.clientY;
  }
}

function finishSelectionDrag(event?: PointerEvent): void {
  if (event && event.pointerId !== dragPointerId) return;
  if (event?.type === "pointerup") dragPendingClicks += 1;
  dragPointerId = null;
  dragVisited.clear();
  if (dragFrame !== null) cancelAnimationFrame(dragFrame);
  dragFrame = null;
}

function onTileContextMenu(entry: MergedPrint, event: MouseEvent): void {
  if (selectMode.value && !bulkSelection.value.has(entry.item.filename)) {
    bulkSelection.value = new Set([entry.item.filename]);
    bulkAnchor.value = entry.item.filename;
  }
  select(entry);
  contextMenu.open(event, tileMenu(entry));
}

function onTileDblclick(entry: MergedPrint) {
  if (selectMode.value) return;
  select(entry);
  lightboxOpen.value = true;
}

function selectAllInFilter() {
  bulkSelection.value = new Set(entries.value.map((e) => e.item.filename));
}

function selectAllFromShortcut() {
  if (entries.value.length === 0) return;
  selectMode.value = true;
  selectAllInFilter();
  bulkAnchor.value = entries.value[0]!.item.filename;
}

function clearBulkSelection() {
  bulkSelection.value = new Set();
  bulkAnchor.value = null;
}

/** Drop selection / lightbox state for prints that left the grid. */
function pruneSelection() {
  const remaining = new Set(entries.value.map((e) => e.item.filename));
  bulkSelection.value = new Set([...bulkSelection.value].filter((f) => remaining.has(f)));
  if (bulkAnchor.value && !remaining.has(bulkAnchor.value)) bulkAnchor.value = null;
  if (selected.value && !remaining.has(selected.value.filename)) {
    selected.value = null;
    lightboxOpen.value = false;
  }
}

/**
 * Bulk delete. On trash-capable hosts: optimistic, one 6 s undo toast for
 * the batch, then the real (trash-aware) DELETE per copy. Elsewhere: the
 * two-press arming button, then the immediate hard delete.
 */
async function deleteSelectedPrints() {
  const targets = selectedEntries.value;
  if (targets.length === 0) return;
  if (targets.every(entryTrashCapable)) {
    trashPrints(targets);
    return;
  }
  if (!confirmingBulkDelete.value) {
    confirmingBulkDelete.value = true;
    return;
  }
  confirmingBulkDelete.value = false;
  if (bulkDeleting.value) return;
  bulkDeleting.value = true;
  try {
    const { deletedPrints, failedPrints, deletedCopies } =
      await gallery.removeEntriesEverywhere(targets);
    if (failedPrints > 0) {
      toasts.push(
        `Deleted ${deletedPrints} of ${targets.length} prints everywhere. ${failedPrints} still have a copy on an unavailable device.`,
        "error",
      );
    } else {
      const copyNote = deletedCopies > deletedPrints ? ` (${deletedCopies} device copies)` : "";
      toasts.push(
        deletedPrints === 1
          ? `Deleted 1 print everywhere${copyNote}`
          : `Deleted ${deletedPrints} prints everywhere${copyNote}`,
      );
    }
  } finally {
    bulkDeleting.value = false;
  }
  pruneSelection();
}

let resizeObserver: ResizeObserver | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;

function remeasureLibraryAfterResume() {
  if (document.visibilityState === "hidden") return;
  void nextTick(() => {
    if (scrollEl.value) containerWidth.value = scrollEl.value.clientWidth;
    virtualizer.value?.measure?.();
  });
}

/**
 * Everything a tile renders, resolved ONCE per data change rather than per
 * template expression per render: the old template called the store five or
 * six times per tile (title, ♥ ×4, purge time, organize capability), each of
 * which walked the whole gallery. Layout (`rows`) reads these by index and
 * never touches the store, so a slider drag or a resize is pure arithmetic.
 */
interface TileModel {
  entry: MergedPrint;
  item: GalleryImage;
  key: string;
  title: string;
  modelLabel: string;
  favorite: boolean;
  canOrganize: boolean;
  availability: string;
  purgeAt: number | null;
  video: boolean;
  audio: boolean;
  upscaled: boolean;
  /** Media bytes are addressed differently for a host bucket and this Mac. */
  mediaPath: string;
  localVideo: boolean;
  target: ApiTarget | null;
  mediaVersion: string;
  fresh: boolean;
}

const tileModels = computed<TileModel[]>(() => {
  const list = entries.value;
  const organizationOf = gallery.organizationOf;
  const organizeCapable = gallery.organizeCapable;
  const trash = inTrash.value;
  const models: TileModel[] = new Array(list.length);
  for (let i = 0; i < list.length; i++) {
    const entry = list[i]!;
    const item = entry.item;
    const org = organizationOf(entry);
    const source = gallery.mediaSourceOf(entry.sourceKey);
    // Imported predicates, not the view's local aliases: `useVirtualizer`
    // evaluates `rows` (and so this computed) synchronously during setup,
    // before the aliases declared further down are initialized.
    const video = isVideoItem(item);
    models[i] = {
      entry,
      item,
      key: `${entry.sourceKey}::${item.filename}`,
      title: displayTitle({ ...item, title: org.title }),
      modelLabel: modelLabel(item.metadata.model),
      favorite: org.favorite,
      canOrganize:
        !trash && gallery.allLocationsOf(entry).some((l) => organizeCapable(l.sourceKey)),
      availability: entry.availableOn.map((s) => s.label).join(" · "),
      purgeAt: org.purgeAt,
      video,
      audio: isAudioItem(item),
      upscaled: isUpscaledImage(item),
      mediaPath: galleryMediaPath(item.filename, source, true, item.trashed_at != null),
      // The tile is always a still thumbnail now; a local clip's poster comes
      // from the native cache rather than a <video> element per tile.
      localVideo: false,
      target: targetFor(entry),
      mediaVersion: item.media_version ?? `${item.timestamp}:${item.size_bytes ?? "unknown"}`,
      fresh: isFresh(entry),
    };
  }
  return models;
});

interface LaidTile {
  model: TileModel;
  x: number;
  width: number;
  height: number;
}

/** Justified layout over the visible set. Rows are the virtualizer's unit;
 *  each laid tile carries its x offset so the tile layer below can place it
 *  absolutely without a per-row wrapper. */
const rows = computed(() => {
  const list = tileModels.value;
  const laidRows = layoutJustifiedRows(
    list.map((m) => m.item),
    Math.max(0, containerWidth.value - PAD * 2),
    rowHeight.value,
    GAP,
  );
  let cursor = 0;
  return laidRows.map((r) => {
    let x = PAD;
    const items: LaidTile[] = r.items.map((laid) => {
      const tile = { model: list[cursor++]!, x, width: laid.width, height: laid.height };
      x += laid.width + GAP;
      return tile;
    });
    return { height: r.height, items };
  });
});

const virtualizer = useVirtualizer(
  computed(() => ({
    count: rows.value.length,
    getScrollElement: () => scrollEl.value,
    estimateSize: (i: number) => (rows.value[i]?.height ?? rowHeight.value) + GAP,
    overscan: 2,
  })),
);

/**
 * The tiles inside the virtual window as ONE flat list keyed by print. The
 * old markup nested tiles under row elements keyed by row index, so a
 * re-flow that moved a print across a row boundary destroyed and re-created
 * its `AuthedMedia` — cancelling the thumbnail in flight and refetching it —
 * on every slider tick and resize pixel. A flat keyed list lets Vue MOVE the
 * element instead, so the decoded image survives the reflow.
 */
const visibleTiles = computed(() => {
  const laid = rows.value;
  const instance = unref(virtualizer);
  // `range` is the on-screen row span without overscan; rows outside it are
  // fetched at `near` priority so a fast scroll never starves what is shown.
  const range = instance.range;
  const out: Array<LaidTile & { y: number; priority: ThumbnailPriority }> = [];
  for (const vrow of instance.getVirtualItems()) {
    const row = laid[vrow.index];
    if (!row) continue;
    const priority: ThumbnailPriority =
      range && (vrow.index < range.startIndex || vrow.index > range.endIndex) ? "near" : "visible";
    for (const tile of row.items) out.push({ ...tile, y: vrow.start, priority });
  }
  return out;
});

// TanStack Virtual caches per-index measurements; a new estimateSize closure
// alone does NOT invalidate that cache. When the justified layout re-flows
// (container resize, entries arriving, row-height change) the cached offsets
// go stale and rows render overlapping — re-measure on any re-flow.
watch(rows, () => virtualizer.value?.measure?.());

// ── Thumbnail pre-warm (desktop only) ────────────────────────────────────────
// Once a listing has laid out, quietly prepare the tiles below and above the
// viewport into the persistent native cache (`thumbnailPrewarm.ts` plans;
// this runs). Requests share the scheduler key `AuthedMedia` uses, so a tile
// that scrolls into view dedupes onto its in-flight prewarm and is promoted
// rather than fetched twice. Everything is cancelled on unmount or when the
// listing changes underneath.
const PREWARM_SETTLE_MS = 500;
let prewarmHandles: ThumbnailHandle<unknown>[] = [];
let prewarmTimer: ReturnType<typeof setTimeout> | null = null;
let prewarmEpoch = 0;

function cancelPrewarm() {
  prewarmEpoch += 1;
  if (prewarmTimer) clearTimeout(prewarmTimer);
  prewarmTimer = null;
  for (const handle of prewarmHandles) handle.cancel();
  prewarmHandles = [];
}

async function runPrewarm() {
  if (!inTauri() || !scrollEl.value) return;
  const epoch = ++prewarmEpoch;
  const laid = rows.value;
  const instance = unref(virtualizer);
  const range = instance.range ?? { startIndex: 0, endIndex: 0 };
  const rowsPerViewport = Math.max(
    1,
    Math.round(scrollEl.value.clientHeight / (rowHeight.value + GAP)),
  );
  const candidates: Array<PrewarmCandidate & { model: TileModel }> = [];
  laid.forEach((row, rowIndex) => {
    for (const tile of row.items) {
      candidates.push({
        sourceKey: tile.model.entry.sourceKey,
        filename: tile.model.item.filename,
        mediaVersion: tile.model.mediaVersion,
        rowIndex,
        model: tile.model,
      });
    }
  });
  const plan = planPrewarm(candidates, {
    startRow: range.startIndex,
    endRow: range.endIndex,
    rowsPerViewport,
  });
  const byHost = new Map<string, typeof plan>();
  for (const entry of plan) {
    const list = byHost.get(entry.candidate.sourceKey) ?? [];
    list.push(entry);
    byHost.set(entry.candidate.sourceKey, list);
  }
  const tier = thumbnailTier();
  for (const [sourceKey, entries] of byHost) {
    const target = gallery.targetOf(sourceKey);
    for (const batch of chunkForProbe(entries)) {
      let cached: boolean[];
      try {
        cached = await ipc.probeGalleryThumbnails(
          sourceKey,
          target?.baseUrl ?? null,
          tier,
          batch.map((e) => ({
            filename: e.candidate.filename,
            mediaVersion: e.candidate.mediaVersion,
          })),
        );
      } catch {
        return;
      }
      if (epoch !== prewarmEpoch) return;
      batch.forEach((entry, i) => {
        if (cached[i]) return;
        const model = (entry.candidate as PrewarmCandidate & { model: TileModel }).model;
        const handle = galleryThumbnailScheduler.schedule({
          key: `${sourceKey}|${model.mediaPath}|${model.mediaVersion}|${target?.baseUrl ?? "primary"}|${target?.apiKey ?? ""}`,
          hostKey: sourceKey,
          priority: entry.priority,
          // A visible tile that arrives while this prewarm is queued dedupes
          // onto this promise, so a native refusal must REJECT rather than
          // resolve null: the tile's own retry then runs its fallback-capable
          // load instead of settling on an empty source.
          run: async (signal) => {
            const url = await prepareNativeThumbnail({
              path: model.mediaPath,
              target,
              cacheKey: sourceKey,
              mediaVersion: model.mediaVersion,
              signal,
            });
            if (url === null) throw new Error("Native thumbnail unavailable; tile will fall back.");
            return url;
          },
        });
        void handle.promise.catch(() => {});
        prewarmHandles.push(handle);
      });
    }
  }
}

function schedulePrewarm() {
  cancelPrewarm();
  prewarmTimer = setTimeout(() => {
    prewarmTimer = null;
    void runPrewarm();
  }, PREWARM_SETTLE_MS);
}

// A new listing (or a scope switch) re-plans; a slider drag does not.
watch(entries, schedulePrewarm);
watch(
  () => gallery.loaded && containerWidth.value > 0,
  (ready) => {
    if (ready) schedulePrewarm();
  },
  { immediate: true },
);

const showBadges = computed(
  () => !inTrash.value && gallery.filter === "all" && gallery.chipCounts.length > 1,
);
const availabilityLabel = (entry: MergedPrint) =>
  entry.availableOn.map((source) => source.label).join(" · ");

const isSelected = (entry: MergedPrint) =>
  selected.value !== null &&
  selected.value.sourceKey === entry.sourceKey &&
  selected.value.filename === entry.item.filename;

function select(entry: MergedPrint) {
  selected.value = { sourceKey: entry.sourceKey, filename: entry.item.filename };
}

const selectedIndex = computed(() => entries.value.findIndex((e) => isSelected(e)));
const selectedEntry = computed<MergedPrint | null>(
  () => entries.value[selectedIndex.value] ?? null,
);

/**
 * What the host that HOLDS the open print can transcode a stored GLB into.
 * GLB is the only stored form, so the export menu is this host's advertised
 * list and nothing else — a host that adds a container adds an entry with no
 * client release, and a host that predates 3-D offers none.
 */
const meshExportFormats = computed<string[]>(() => {
  const entry = selectedEntry.value;
  if (!entry) return [];
  return hosts.capabilities[entry.sourceKey]?.mesh?.export_formats ?? [];
});

/** Shared with the store's kind filter so badge and chips never disagree. */
const isVideo = (i: GalleryImage) => isVideoItem(i);
const isAudio = (i: GalleryImage) => isAudioItem(i);
const isMesh = (i: GalleryImage) => isMeshItem(i);

async function copyImage(entry: MergedPrint) {
  try {
    const path = galleryMediaPath(
      entry.item.filename,
      gallery.mediaSourceOf(entry.sourceKey),
      false,
      entry.item.trashed_at != null,
    );
    const target = targetFor(entry);
    await copyImageBytesToClipboard(
      path,
      target ? { fetchImage: (p) => fetchGalleryMediaBytes(p, target) } : undefined,
    );
    toasts.push("Image copied");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

function moveSelection(delta: number) {
  const list = entries.value;
  if (list.length === 0) return;
  const next = Math.min(
    list.length - 1,
    Math.max(0, (selectedIndex.value === -1 ? 0 : selectedIndex.value) + delta),
  );
  select(list[next]!);
}

/** The prints a keyboard action targets: the bulk selection, else the
 *  selected tile. */
function keyboardTargets(): MergedPrint[] {
  if (selectMode.value && bulkSelection.value.size > 0) return selectedEntries.value;
  return selectedEntry.value ? [selectedEntry.value] : [];
}

/** Close the topmost transient surface; true when one was open. */
function closeTransient(): boolean {
  if (bulkBar.value?.closePopovers()) return true;
  if (chipRow.value?.isOpen()) {
    chipRow.value.closeMore();
    return true;
  }
  return false;
}

function onKeydown(e: KeyboardEvent) {
  // ⌘F focuses the view's own search — the screen-level filter shortcut.
  if (e.key === "f" && primaryModifierPressed(e) && !e.altKey) {
    e.preventDefault();
    header.value?.focusSearch();
    return;
  }
  // The history drawer owns the keyboard while it's open.
  if (historyOpen.value) return;
  // ⌘A selects exactly the current filtered result set, never hidden prints.
  if (isSelectAllChord(e) && !allowsNativeSelectAll(document.activeElement)) {
    e.preventDefault();
    selectAllFromShortcut();
    return;
  }
  // ⌘⇧N — new collection (organize-capable hosts only).
  if ((e.key === "n" || e.key === "N") && e.shiftKey && primaryModifierPressed(e)) {
    if (!organizeAvailable.value || allowsNativeContextMenu(e.target as Element | null)) return;
    e.preventDefault();
    if (showShelf.value) void shelf.value?.startCreate();
    else openNewCollection(keyboardTargets());
    return;
  }
  // ⌘⌫ — delete forever (confirm), in any scope.
  if ((e.key === "Delete" || e.key === "Backspace") && primaryModifierPressed(e)) {
    if (allowsNativeContextMenu(e.target as Element | null)) return;
    e.preventDefault();
    if (e.repeat) return;
    if (trashAvailable.value) askDeleteForever(keyboardTargets());
    else if (selectMode.value && bulkSelection.value.size > 0) void deleteSelectedPrints();
    else if (selectedEntry.value) deletePrint(selectedEntry.value);
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    // WebKit treats an unhandled Backspace/Delete as history-back. Keep the
    // native editing behavior in text fields, but always claim it elsewhere
    // in Library so deleting a print can never navigate to Create.
    if (allowsNativeContextMenu(e.target as Element | null)) return;
    e.preventDefault();
    if (e.repeat) return;
    if (inTrash.value) {
      askDeleteForever(keyboardTargets());
    } else if (selectMode.value && bulkSelection.value.size > 0) {
      void deleteSelectedPrints();
    } else if (selectedEntry.value) {
      deletePrint(selectedEntry.value);
    }
  } else if (
    (e.key === "f" || e.key === "F") &&
    !allowsNativeContextMenu(e.target as Element | null)
  ) {
    if (!organizeAvailable.value || inTrash.value) return;
    const targets = keyboardTargets();
    if (targets.length === 0) return;
    e.preventDefault();
    void setFavorite(
      targets,
      targets.some((t) => !isFavorite(t)),
    );
  } else if (
    (e.key === "t" || e.key === "T") &&
    !allowsNativeContextMenu(e.target as Element | null)
  ) {
    if (!organizeAvailable.value || inTrash.value) return;
    const targets = keyboardTargets();
    if (targets.length === 0) return;
    e.preventDefault();
    // The Tag popover lives on the bulk bar: enter select mode around the
    // current print when needed, then open it.
    if (!selectMode.value) {
      selectMode.value = true;
      bulkSelection.value = new Set(targets.map((t) => t.item.filename));
      bulkAnchor.value = targets[0]!.item.filename;
    }
    void Promise.resolve().then(() => bulkBar.value?.openTags());
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    moveSelection(1);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    moveSelection(-1);
  } else if (e.key === " ") {
    if (allowsNativeContextMenu(e.target as Element | null)) return;
    e.preventDefault();
    if (selected.value) lightboxOpen.value = !lightboxOpen.value;
  } else if (e.key === "Escape") {
    if (closeTransient()) return;
    if (lightboxOpen.value) lightboxOpen.value = false;
    else if (selectMode.value) setSelectMode(false);
  }
}

/** Fire the real DELETE for a pending batch, surfacing any failure (which
 *  restores the print via the store's finally). */
async function commitDelete(key: string) {
  const pending = pendingDeletes.get(key);
  if (pending) clearTimeout(pending.timer);
  pendingDeletes.delete(key);
  if (!pending) return;
  const result = await gallery.commitDeleteEverywhere(pending.locations);
  if (result.failed > 0) {
    toasts.push(
      result.error ??
        `${result.failed} device ${result.failed === 1 ? "copy remains" : "copies remain"} because a delete failed.`,
      "error",
    );
  }
}

/** Undo a still-pending delete — the print returns to the grid, no server call. */
function undoDelete(key: string) {
  const pending = pendingDeletes.get(key);
  if (pending) clearTimeout(pending.timer);
  pendingDeletes.delete(key);
  if (pending) gallery.cancelDeleteEverywhere(pending.locations);
}

/** Hold a set of locations in limbo behind one undo toast. */
function scheduleDelete(key: string, locations: GalleryLocation[], message: string) {
  const timer = setTimeout(() => void commitDelete(key), UNDO_WINDOW_MS);
  pendingDeletes.set(key, { locations, timer });
  toasts.push(message, "info", {
    durationMs: UNDO_WINDOW_MS,
    action: { label: "Undo", run: () => undoDelete(key) },
  });
}

/**
 * Delete a single print the reversible way: hide it from the grid immediately,
 * advance selection off the vanishing tile, and open a 6 s undo toast. The real
 * DELETE fires only when that window lapses.
 */
function deletePrint(entry: MergedPrint) {
  const { sourceKey } = entry;
  const filename = entry.item.filename;
  const key = deleteKey(sourceKey, filename);
  if (pendingDeletes.has(key)) return;

  const index = entries.value.findIndex(
    (candidate) => candidate.sourceKey === sourceKey && candidate.item.filename === filename,
  );
  const wasSelected = isSelected(entry);
  const trashable = entryTrashCapable(entry);
  const title = tileTitle(entry);
  const locations = gallery.beginDeleteEverywhere(entry);
  if (wasSelected) {
    const remaining = entries.value;
    if (remaining.length === 0) {
      lightboxOpen.value = false;
      selected.value = null;
    } else {
      select(remaining[Math.min(Math.max(index, 0), remaining.length - 1)]!);
    }
  }
  scheduleDelete(
    key,
    locations,
    trashable ? `Moved “${title}” to trash` : `Deleted ${filename} everywhere`,
  );
}

/** Bulk move to trash: every selected print into limbo behind ONE undo toast. */
function trashPrints(targets: MergedPrint[]) {
  const fresh = targets.filter((e) => !pendingDeletes.has(deleteKey(e.sourceKey, e.item.filename)));
  if (fresh.length === 0) return;
  const locations: GalleryLocation[] = [];
  for (const entry of fresh) locations.push(...gallery.beginDeleteEverywhere(entry));
  const n = fresh.length;
  scheduleDelete(
    `bulk-${++bulkDeleteSeq}`,
    locations,
    `Moved ${n} ${n === 1 ? "print" : "prints"} to trash`,
  );
  pruneSelection();
}

function removeSelected() {
  const entry = selectedEntry.value;
  if (entry) deletePrint(entry);
}

/**
 * "Use as source" from the gallery — load this print's bytes into the Create
 * composer as the img2img source (raw base64, the form's contract) and open
 * Create. Deliberately does NOT touch `composer.prefill`, so GenerateView's
 * prefill watcher can't clobber the source we just attached.
 */
async function useAsSource(entry: MergedPrint) {
  try {
    const blob = await fetchItemBlob(entry);
    const base64 = await blobToBase64(blob);
    const form = generateForm.form;
    if (isVideo(entry.item)) {
      attachPickedVideo(form, { filename: entry.item.filename, base64 });
      lightboxOpen.value = false;
      toasts.push("Loaded as source video");
      void router.push("/create");
      return;
    }
    const h3Task = minimaxH3TaskForModel(form.model);
    if (h3Task) {
      const dimensions = imageDimensionsFromBase64(base64) ?? {
        width: entry.item.metadata.width,
        height: entry.item.metadata.height,
      };
      const image = {
        filename: entry.item.filename,
        mimeType: galleryImageMimeType(entry.item, blob.type),
        width: dimensions.width,
        height: dimensions.height,
        data: base64,
      };
      const result =
        h3Task === "ref2va"
          ? await appendMinimaxH3GalleryImageReference(form.h3Authoring, image)
          : setMinimaxH3GalleryImageFirstFrame(form.h3Authoring, image);
      if (!result.ok) throw new Error(result.error);
      form.h3Authoring = result.state;
    } else if (isMinimaxH3Identity(form.family, form.model)) {
      throw new Error(
        "Choose an explicit MiniMax H3 FL2VA or Ref2VA model before adding a source.",
      );
    } else {
      attachPickedImage(form, { filename: entry.item.filename, base64 });
    }
    lightboxOpen.value = false;
    toasts.push(h3Task === "ref2va" ? "Added as ordered reference" : "Loaded as source");
    void router.push("/create");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
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

async function useSelectedAsSource() {
  const entry = selectedEntry.value;
  if (entry) await useAsSource(entry);
}

// ── URL sync (?scope, ?c, ?tag, ?fav — plus the pre-existing ?host, ?print,
//    ?panel, ?tab) ───────────────────────────────────────────────────────────
// Route → store when the params are present (a deep link or a back/forward
// hop); store → route (replace) so the address bar always names the view.
// Plain /library keeps the session state, like ?host always has.
const VALID_SCOPES: readonly LibraryScope[] = ["prints", "collections", "trash"];
const asString = (value: unknown): string | null =>
  typeof value === "string" && value.length > 0 ? value : null;

let syncingFromRoute = false;
let openingPrintDeepLink = false;
watch(
  () => [route.query.scope, route.query.c, route.query.tag, route.query.fav] as const,
  ([scopeParam, c, tag, fav]) => {
    syncingFromRoute = true;
    try {
      const wantScope = asString(scopeParam);
      if (wantScope && (VALID_SCOPES as readonly string[]).includes(wantScope)) {
        gallery.scope = wantScope as LibraryScope;
      }
      if (c !== undefined) gallery.collectionSlug = asString(c);
      if (tag !== undefined) {
        gallery.tagFilter = (asString(tag) ?? "")
          .split(",")
          .map((t) => t.trim())
          .filter((t) => t.length > 0);
      }
      if (fav !== undefined) gallery.favoritesOnly = asString(fav) === "1";
    } finally {
      syncingFromRoute = false;
    }
  },
  { immediate: true },
);

watch(
  () =>
    [
      gallery.scope,
      gallery.collectionSlug,
      gallery.tagFilter.join(","),
      gallery.favoritesOnly,
    ] as const,
  ([scopeValue, slug, tags, fav]) => {
    if (syncingFromRoute || openingPrintDeepLink || route.path !== "/library") return;
    const query: Record<string, string | undefined> = {
      ...(route.query as Record<string, string | undefined>),
    };
    const set = (key: string, value: string | null) => {
      if (value === null) delete query[key];
      else query[key] = value;
    };
    set("scope", scopeValue === "prints" ? null : scopeValue);
    set("c", scopeValue === "collections" && slug ? slug : null);
    set("tag", tags.length > 0 ? tags : null);
    set("fav", fav ? "1" : null);
    const same = Object.keys({ ...route.query, ...query }).every(
      (key) => (route.query[key] ?? undefined) === (query[key] ?? undefined),
    );
    if (!same) void router.replace({ path: "/library", query });
  },
);

// Deep link: /library?host=<bucket key> pre-picks a chip ("local" = This
// Mac's key in every mode). Plain /library keeps the session filter.
watch(
  () => route.query.host,
  (host) => {
    if (host === undefined) return;
    // Verbatim: `filtered` falls back to All while the key's source doesn't
    // exist yet (e.g. the host is still connecting) and narrows on its own
    // the moment the source appears.
    gallery.filter = typeof host === "string" && host ? host : "all";
  },
  { immediate: true },
);

// Deep link: /library?print=<filename> (a ⌘K result or native notification)
// reveals that print. A print filed only in a hidden collection must open in
// that collection: the Prints scope deliberately excludes it, and selection
// is resolved from the current grid. Wait for the relevant collection listing
// before consuming the one-shot param so a fast gallery response cannot race
// the slower organization response.
watch(
  [
    () => route.query.print,
    () => {
      const print = route.query.print;
      return typeof print === "string"
        ? gallery.merged.some((entry) => entry.item.filename === print)
        : false;
    },
    () =>
      gallery.sources
        .map((source) => `${source.key}:${gallery.organizeCapable(source.key)}`)
        .join("|"),
    () =>
      Object.entries(gallery.collectionsByHost)
        .map(([key, bucket]) => `${key}:${bucket.loading}:${bucket.loaded}:${bucket.error ?? ""}`)
        .join("|"),
  ],
  ([print]) => {
    if (typeof print !== "string" || !print) return;
    const entry = gallery.merged.find((e) => e.item.filename === print);
    if (!entry) return;

    const unsettledCollectionCopy = (
      entry.copies ?? [{ sourceKey: entry.sourceKey, item: entry.item }]
    ).some((copy) => {
      // A non-empty collection id is itself evidence that this host supports
      // organization. Its capability snapshot may still be in flight after
      // the host becomes ready, so do not consume the one-shot route until
      // the listing resolves those ids to slugs and hidden state.
      if (!copy.item.collections?.length) return false;
      const bucket = gallery.collectionsByHost[copy.sourceKey];
      return !bucket || bucket.loading || (!bucket.loaded && bucket.error === null);
    });
    if (unsettledCollectionCopy) return;

    const memberships = new Set(gallery.organizationOf(entry).collections);
    const hiddenCollection = gallery.mergedCollections.find(
      (collection) => collection.hidden && memberships.has(collection.slug),
    );
    openingPrintDeepLink = true;
    gallery.scope = hiddenCollection ? "collections" : "prints";
    gallery.collectionSlug = hiddenCollection?.slug ?? null;
    gallery.favoritesOnly = false;
    gallery.tagFilter = [];
    gallery.filter = "all";
    gallery.mediaKind = "all";
    gallery.query = "";
    searchInput.value = "";
    select(entry);
    lightboxOpen.value = true;
    const query = { ...route.query };
    delete query.print;
    delete query.host;
    delete query.tag;
    delete query.fav;
    if (hiddenCollection) {
      query.scope = "collections";
      query.c = hiddenCollection.slug;
    } else {
      delete query.scope;
      delete query.c;
    }
    void router.replace({ path: "/library", query }).finally(() => (openingPrintDeepLink = false));
  },
  { immediate: true },
);

// (Re)fetch whenever the set of sources changes — covers mount, the
// connection coming up, and hosts joining or leaving mid-session.
watch(
  () => gallery.sources.map((s) => s.key).join("|"),
  () => void gallery.fetchAll(),
  { immediate: true },
);

// Collections / tags / trash ride the same trigger plus the capability
// snapshot, which can land after the buckets do.
watch(
  () =>
    [
      gallery.sources.map((s) => s.key).join("|"),
      organizeAvailable.value,
      trashAvailable.value,
    ] as const,
  ([, organize, trash]) => {
    if (organize || trash) void gallery.fetchOrganization();
  },
  { immediate: true },
);

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("mold:library-select-all", selectAllFromShortcut);
  window.addEventListener("pointermove", moveSelectionDrag, { passive: false });
  window.addEventListener("pointerup", finishSelectionDrag);
  window.addEventListener("pointercancel", finishSelectionDrag);
  window.addEventListener("pageshow", remeasureLibraryAfterResume);
  window.addEventListener("focus", remeasureLibraryAfterResume);
  document.addEventListener("visibilitychange", remeasureLibraryAfterResume);
  pollTimer = setInterval(() => void gallery.pollExtras(), EXTRA_POLL_MS);
  if (scrollEl.value) {
    containerWidth.value = scrollEl.value.clientWidth;
    resizeObserver = new ResizeObserver((observed) => {
      containerWidth.value = observed[0]?.contentRect.width ?? containerWidth.value;
    });
    resizeObserver.observe(scrollEl.value);
  }
  void nextTick(() => {
    if (!scrollEl.value) return;
    const position = sessionScrollPosition(DESKTOP_LIBRARY_SCROLL_KEY);
    scrollEl.value.scrollTop = position.top;
    scrollEl.value.scrollLeft = position.left;
  });
});
onBeforeUnmount(() => {
  if (scrollEl.value) {
    rememberSessionScroll(DESKTOP_LIBRARY_SCROLL_KEY, {
      top: scrollEl.value.scrollTop,
      left: scrollEl.value.scrollLeft,
    });
  }
});
onUnmounted(() => {
  stopUpscalePoll();
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("mold:library-select-all", selectAllFromShortcut);
  window.removeEventListener("pointermove", moveSelectionDrag);
  window.removeEventListener("pointerup", finishSelectionDrag);
  window.removeEventListener("pointercancel", finishSelectionDrag);
  window.removeEventListener("pageshow", remeasureLibraryAfterResume);
  window.removeEventListener("focus", remeasureLibraryAfterResume);
  document.removeEventListener("visibilitychange", remeasureLibraryAfterResume);
  finishSelectionDrag();
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  resizeObserver?.disconnect();
  cancelPrewarm();
  // Finalize any deletes still inside their undo window — leaving Library
  // commits them rather than stranding a hidden-but-not-deleted print.
  for (const key of [...pendingDeletes.keys()]) {
    void commitDelete(key);
  }
  // Flush a settling thumbnail-size write so the choice survives navigation.
  if (sliderPersistTimer) {
    clearTimeout(sliderPersistTimer);
    sliderPersistTimer = null;
    saveGalleryThumbnailSize(rowHeight.value);
  }
  // Flush a pending debounced search so the store matches the input.
  if (searchTimer) {
    clearTimeout(searchTimer);
    searchTimer = null;
    gallery.query = searchInput.value;
  }
});
</script>

<template>
  <div class="relative flex h-full flex-col">
    <LibraryHeader
      ref="header"
      :scope="scope"
      :scopes="scopes"
      :counts="scopeCounts"
      :count-label="countLabel"
      :error="gallery.firstError"
      :thumbnail-size="rowHeight"
      :media-kind="gallery.mediaKind"
      :kind-options="kindOptions"
      :search="searchInput"
      :select-mode="selectMode"
      :trash-count="gallery.trashCount"
      :busy="organizeBusy"
      @update:scope="setScope"
      @update:thumbnail-size="rowHeight = $event"
      @update:media-kind="setKind"
      @update:search="searchInput = $event"
      @open-history="openHistory"
      @toggle-select="setSelectMode(!selectMode)"
      @refresh="gallery.fetchAll()"
      @empty-trash="emptyTrashOpen = true"
    />

    <LibraryChipRow
      v-if="showChipRow"
      ref="chipRow"
      :organize="organizeAvailable"
      :favorites-only="gallery.favoritesOnly"
      :favorites-count="favoritesCount"
      :tags="gallery.filterChipTags"
      :active-tags="gallery.tagFilter"
      :host-chips="gallery.chipCounts"
      :host-filter="gallery.filter"
      :all-count="gallery.basePrintCount"
      :collection-name="drillInName"
      @update:favorites-only="gallery.favoritesOnly = $event"
      @toggle-tag="toggleTagFilter"
      @update:host-filter="gallery.filter = $event"
      @clear-filters="clearFilters"
      @exit-collection="exitCollection"
    />

    <!-- Collections drill-in: crumb bar with Select + Edit. -->
    <div
      v-if="inCollections && gallery.collectionSlug"
      class="flex h-11 shrink-0 items-center gap-2.5 border-b border-edge bg-[color-mix(in_srgb,var(--bench)_50%,var(--bath))] px-6"
      data-test="collection-crumbs"
    >
      <button
        type="button"
        class="flex items-center gap-0.5 rounded-control text-body text-ink-2 hover:text-ink"
        data-test="crumb-back"
        @click="exitCollection"
      >
        <Icon name="chevron-left" :size="14" />
        Collections
      </button>
      <span class="text-ink-3" aria-hidden="true">›</span>
      <span class="font-display text-[15px] font-semibold text-ink" data-test="crumb-here">
        {{ drillInName }}
      </span>
      <span v-if="openCollection" class="font-utility text-[10.5px] text-ink-3">
        {{ openCollection.hosts.map((h) => sourceLabel(h.hostId)).join(" · ") }}
      </span>
      <div class="flex-1" />
      <button
        type="button"
        class="flex h-[30px] items-center gap-1.5 rounded-chrome border border-edge bg-bench px-2.5 text-[12.5px] text-ink-2 hover:text-ink"
        data-test="collection-edit"
        aria-haspopup="menu"
        @click="openEditMenu"
      >
        Edit
        <Icon name="chevron-down" :size="12" />
      </button>
    </div>

    <TrashBanner
      v-if="inTrash"
      :hosts="gallery.retentionByHost"
      :count="gallery.trashCount"
      :bytes="trashBytes"
      :link-label="retentionLinkLabel"
      @change-retention="changeRetention"
    />

    <div ref="scrollEl" class="min-h-0 flex-1 overflow-y-auto" style="contain: strict">
      <!-- Collections shelf -->
      <template v-if="showShelf">
        <EmptyState
          v-if="gallery.mergedCollections.length === 0"
          headline="No collections yet"
          detail="Group prints however you like — a collection lives on every machine that holds a copy."
          action="New collection"
          @action="openNewCollection()"
        />
        <EmptyState
          v-else-if="shelfCards.length === 0"
          headline="No matching collections"
          detail="Nothing here matches the current search."
        />
        <CollectionsShelf
          v-else
          ref="shelf"
          :cards="shelfCards"
          :busy="organizeBusy"
          @open="openCollectionSlug"
          @create="(name) => createCollection(name)"
          @contextmenu="(slug, event) => contextMenu.open(event, collectionMenu(slug))"
        />
      </template>

      <!-- Trash empty states -->
      <EmptyState
        v-else-if="inTrash && entries.length === 0 && gallery.trashCount > 0"
        headline="No prints in the trash match"
        detail="Nothing in the trash matches the current search or filter."
      />
      <EmptyState v-else-if="inTrash && entries.length === 0" headline="Trash is empty" />

      <!-- Drill-in empty state -->
      <EmptyState
        v-else-if="inCollections && gallery.loaded && entries.length === 0"
        headline="Nothing in this collection yet"
        detail="Select prints in the Library and choose Add to collection."
        action="Go to prints"
        @action="setScope('prints')"
      />

      <!-- Prints empty states -->
      <EmptyState
        v-else-if="gallery.loaded && entries.length === 0 && gallery.hostFiltered.length > 0"
        headline="No matching prints"
        detail="Nothing here matches the current search or filters."
      />
      <EmptyState
        v-else-if="gallery.loaded && entries.length === 0"
        headline="No prints here yet"
        :detail="
          gallery.filter === 'local'
            ? 'Generations saved on this Mac will appear here.'
            : 'Generate one and it lands here.'
        "
        action="Go to Create"
        @action="router.push('/create')"
      />
      <div v-else class="ms-lib-tile-layer" :style="{ height: `${virtualizer.getTotalSize()}px` }">
        <!-- One flat, print-keyed list of the tiles inside the virtual window
             (see `visibleTiles`). A tile is a wrapper (click / context /
             dblclick) around a focusable button carrying the media, badges,
             and the rising edge code; the ♥ and trash actions are sibling
             overlays so they never nest a button inside a button. Every
             per-tile fact comes from its `TileModel` — the template calls
             nothing that walks the gallery. -->
        <div
          v-for="tile in visibleTiles"
          :key="tile.model.key"
          class="ms-lib-tile group overflow-hidden rounded-[9px] border"
          :class="
            (
              selectMode
                ? bulkSelection.has(tile.model.item.filename)
                : isSelected(tile.model.entry)
            )
              ? 'border-transparent ring-2 ring-safelight'
              : 'border-[color-mix(in_srgb,var(--rebate)_14%,transparent)]'
          "
          :style="{
            width: `${tile.width}px`,
            height: `${tile.height}px`,
            '--tile-x': `${tile.x}px`,
            '--tile-y': `${tile.y}px`,
          }"
          :data-filename="tile.model.item.filename"
          @pointerdown="beginSelectionDrag($event, tile.model.entry)"
          @click="onTileClick(tile.model.entry, $event)"
          @contextmenu="onTileContextMenu(tile.model.entry, $event)"
          @dblclick="onTileDblclick(tile.model.entry)"
        >
          <button
            type="button"
            class="absolute inset-0 block h-full w-full overflow-hidden text-left"
            :aria-label="tile.model.title"
            :aria-pressed="selectMode ? bulkSelection.has(tile.model.item.filename) : undefined"
          >
            <AuthedMedia
              :path="tile.model.mediaPath"
              :target="tile.model.target"
              :cache-key="tile.model.entry.sourceKey"
              :media-version="tile.model.mediaVersion"
              :priority="tile.priority"
              :video="tile.model.localVideo"
              :alt="tile.model.item.metadata.prompt"
            />
            <!-- NEW badge (top-left) — hidden while selecting, where the
                 checkbox owns that corner; never in the trash. -->
            <span
              v-if="!selectMode && !inTrash && tile.model.fresh"
              data-test="new-badge"
              class="ms-lib-new absolute top-2 left-2"
            >
              New
            </span>
            <span
              v-if="!selectMode && tile.model.upscaled"
              data-test="upscaled-badge"
              class="ms-lib-upscaled absolute top-2 right-2"
            >
              Upscaled
            </span>
            <span
              v-if="tile.model.video || tile.model.audio"
              class="absolute top-1.5 right-1.5 rounded-control bg-black/60 px-1 text-caption text-on-media"
            >
              {{ tile.model.audio ? "♪" : "▶" }}
            </span>
            <span
              v-if="selectMode"
              data-test="select-indicator"
              class="absolute top-1.5 left-1.5 flex h-5 w-5 items-center justify-center rounded-full text-caption"
              :class="
                bulkSelection.has(tile.model.item.filename)
                  ? 'bg-safelight font-semibold text-on-accent'
                  : 'border border-white/70 bg-black/40 text-on-media'
              "
            >
              {{ bulkSelection.has(tile.model.item.filename) ? "✓" : "" }}
            </span>
            <!-- The badge yields to the rising edge code on hover — both live
                 in the tile's bottom margin and must never overlap. -->
            <span
              v-if="showBadges"
              data-test="host-badge"
              class="edge-code absolute bottom-1.5 left-1.5 max-w-[70%] truncate rounded-control bg-black/60 px-1 !text-on-media transition-opacity duration-100 group-hover:opacity-0"
              :title="`Available on ${tile.model.availability}`"
            >
              {{ tile.model.availability }}
            </span>
            <span
              v-if="!inTrash"
              data-test="edge-strip"
              class="edge-code absolute right-0 bottom-0 left-0 translate-y-full truncate bg-black/60 py-0.5 pr-7 pl-1.5 text-left !text-on-media transition-transform duration-100 group-hover:translate-y-0"
            >
              {{ tile.model.title }} · {{ tile.model.modelLabel }} · S
              {{ tile.model.item.metadata.seed }}
            </span>
          </button>
          <!-- ♥ (bottom-right): filled when favorite, faint on hover when not;
               click toggles without opening the print. -->
          <button
            v-if="!inTrash && organizeAvailable && tile.model.canOrganize"
            type="button"
            data-test="tile-favorite"
            class="ms-lib-heart absolute right-1.5 bottom-1.5 z-10 flex h-6 w-6 items-center justify-center rounded-full text-on-media transition-opacity duration-100"
            :class="
              tile.model.favorite
                ? 'ms-lib-heart--on opacity-100'
                : 'opacity-0 group-hover:opacity-60 hover:!opacity-100'
            "
            :aria-pressed="tile.model.favorite"
            :aria-label="tile.model.favorite ? 'Unfavorite' : 'Favorite'"
            :title="tile.model.favorite ? 'Unfavorite' : 'Favorite'"
            @click.stop="toggleFavorite(tile.model.entry)"
            @dblclick.stop
          >
            <Icon name="heart" :size="14" />
          </button>
          <TrashTileActions
            v-if="inTrash"
            :purge-at="tile.model.purgeAt"
            :show-actions="!selectMode"
            :busy="organizeBusy"
            @restore="restorePrints([tile.model.entry])"
            @delete-forever="askDeleteForever([tile.model.entry])"
          />
        </div>
      </div>
    </div>

    <!-- Floating bulk-action bar while select mode is active. -->
    <BulkBar
      v-if="selectMode"
      ref="bulkBar"
      :selected-count="bulkSelection.size"
      :total="entries.length"
      :scope="scope"
      :organize="organizeAvailable"
      :organize-blocked-reason="selectionOrganizeBlockedReason"
      :trash="selectionTrashCapable"
      :confirming="confirmingBulkDelete"
      :busy="bulkDeleting || organizeBusy"
      :collections="gallery.mergedCollections"
      :collection-selected="selectionOrganization.collectionsAll"
      :collection-mixed="selectionOrganization.collectionsSome"
      :collection-counts="gallery.collectionCounts"
      :tags="selectionOrganization.tags"
      :tag-suggestions="gallery.mergedTags"
      :all-favorite="selectionOrganization.allFavorite"
      :collection-name="drillInName"
      :host-note="organizeHostNote"
      @select-all="selectAllInFilter"
      @clear="clearBulkSelection"
      @exit="setSelectMode(false)"
      @favorite="(value) => setFavorite(selectedEntries, value)"
      @trash="deleteSelectedPrints"
      @update:confirming="confirmingBulkDelete = $event"
      @delete="deleteSelectedPrints"
      @restore="restorePrints(selectedEntries)"
      @delete-forever="askDeleteForever(selectedEntries)"
      @remove-from-collection="removeFromOpenCollection(selectedEntries)"
      @toggle-collection="(slug, checked) => toggleCollection(selectedEntries, { slug, checked })"
      @create-collection="(name) => createCollection(name, selectedEntries)"
      @add-tags="(names) => applyTags(selectedEntries, { add: names, remove: [] })"
      @remove-tags="(names) => applyTags(selectedEntries, { add: [], remove: names })"
    />

    <Lightbox
      v-if="lightboxOpen && selectedEntry"
      :item="selectedEntry.item"
      :index="selectedIndex"
      :count="entries.length"
      :video="isVideo(selectedEntry.item)"
      :audio="isAudio(selectedEntry.item)"
      :mesh="isMesh(selectedEntry.item)"
      :mesh-export-formats="meshExportFormats"
      :source="gallery.mediaSourceOf(selectedEntry.sourceKey)"
      :target="targetFor(selectedEntry)"
      :cache-key="selectedEntry.sourceKey"
      :host-label="availabilityLabel(selectedEntry)"
      :can-reveal="canReveal(selectedEntry)"
      :is-sequence="isSequencePrint(selectedEntry)"
      :can-edit-sequence="canEditSequence(selectedEntry)"
      :organization="orgOf(selectedEntry)"
      :can-organize="organizeAvailable && canOrganizeEntry(selectedEntry)"
      :can-trash="entryTrashCapable(selectedEntry)"
      :trashed="inTrash"
      :upscale-enabled="canUpscaleEntry(selectedEntry)"
      :collections="gallery.mergedCollections"
      :collection-counts="gallery.collectionCounts"
      :tag-suggestions="gallery.mergedTags"
      @close="lightboxOpen = false"
      @prev="moveSelection(-1)"
      @next="moveSelection(1)"
      @delete="removeSelected"
      @use-source="useSelectedAsSource"
      @reuse="reuseSettings(selectedEntry!)"
      @reuse-sequence="reuseSequence(selectedEntry)"
      @edit-sequence="editSequence(selectedEntry)"
      @rename="(title) => renamePrint(selectedEntry!, title)"
      @favorite="(value) => setFavorite([selectedEntry!], value)"
      @tags="(change) => applyTags([selectedEntry!], change)"
      @collections="(change) => toggleCollection([selectedEntry!], change)"
      @restore="restorePrints([selectedEntry!])"
      @delete-forever="askDeleteForever([selectedEntry!])"
      @upscale="openUpscaleDialog(selectedEntry!)"
    />

    <UpscaleDialog
      :open="!!upscaleEntry"
      :kind="upscaleKind"
      :source-name="upscaleAuthority?.filename ?? upscaleEntry?.item.filename ?? ''"
      :models="models.upscalers"
      v-model="upscaleModel"
      :execution-hosts="upscaleHostChoices"
      :execution-host-value="upscaleAuthority?.sourceKey ?? ''"
      :busy="upscalingFilename !== null"
      :job-state="upscaleJob?.state ?? null"
      :status="upscaleJob ? framewiseStatus(upscaleJob) : null"
      :progress="upscaleJob ? framewiseProgress(upscaleJob) : null"
      :error="upscaleError || null"
      @update:execution-host-value="selectUpscaleAuthority"
      @confirm="startUpscale"
      @close="closeUpscaleDialog"
      @pause="transitionUpscale('pause')"
      @resume="transitionUpscale('resume')"
      @cancel="transitionUpscale('cancel')"
    />

    <HistoryDrawer :open="historyOpen" @close="closeHistory" />

    <!-- Naming dialogs (shared RenameDialog shell) -->
    <RenameDialog
      :open="renameTarget !== null"
      title="Rename print"
      :initial="renameTarget ? (orgOf(renameTarget).title ?? '') : ''"
      @save="onRenameSave"
      @cancel="renameTarget = null"
    />
    <RenameDialog
      :open="newTagTargets !== null"
      title="New tag"
      initial=""
      @save="onNewTagSave"
      @cancel="newTagTargets = null"
    />
    <RenameDialog
      :open="newCollectionTargets !== null"
      title="New collection"
      initial=""
      @save="onNewCollectionSave"
      @cancel="newCollectionTargets = null"
    />
    <RenameDialog
      :open="collectionRenameSlug !== null"
      title="Rename collection"
      :initial="collectionNamed(collectionRenameSlug)"
      @save="onCollectionRenameSave"
      @cancel="collectionRenameSlug = null"
    />

    <!-- Destructive confirms (plain shared ConfirmDialog — no typed phrase) -->
    <ConfirmDialog
      :open="collectionDeleteSlug !== null"
      :title="`Delete collection “${collectionNamed(collectionDeleteSlug)}”?`"
      message="Its prints stay in the Library."
      confirm-label="Delete"
      danger
      @confirm="confirmDeleteCollection"
      @cancel="collectionDeleteSlug = null"
    />
    <ConfirmDialog
      :open="emptyTrashOpen"
      title="Empty trash?"
      :message="emptyTrashMessage"
      confirm-label="Delete forever"
      danger
      :busy="organizeBusy"
      @confirm="confirmEmptyTrash"
      @cancel="emptyTrashOpen = false"
    />
    <ConfirmDialog
      :open="deleteForeverTargets !== null"
      :title="deleteForeverTitle"
      message="This can't be undone."
      confirm-label="Delete forever"
      danger
      :busy="organizeBusy"
      @confirm="confirmDeleteForever"
      @cancel="deleteForeverTargets = null"
    />
  </div>
</template>

<style scoped>
/* Tiles are absolutely placed by the justified layout (`translate` carries
   the row offset from the virtualizer plus the x offset within the row) and
   contained, so a hover or a badge repaint never lays out its neighbours.
   The hover lift transitions `transform` only — animating `box-shadow`
   repainted a large blurred shadow every frame of the transition. */
.ms-lib-tile {
  position: absolute;
  top: 0;
  left: 0;
  contain: layout paint style;
  transform: translate3d(var(--tile-x), var(--tile-y), 0);
  transition: transform var(--dur-quick) var(--ease);
}

.ms-lib-tile:hover {
  transform: translate3d(var(--tile-x), calc(var(--tile-y) - 2px), 0);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
}

.ms-lib-tile-layer {
  position: relative;
  will-change: transform;
}

.ms-lib-tile > button:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: -2px;
}

.ms-lib-new {
  font-family: var(--f-mono);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 2px 6px;
  border-radius: 5px;
}

.ms-lib-upscaled {
  font-family: var(--f-mono);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: color-mix(in srgb, var(--rebate) 88%, black);
  color: var(--on-accent);
  padding: 2px 6px;
  border-radius: 5px;
}

/* ♥ overlay: a drop shadow keeps the glyph legible on any print; the
   favorited state fills the outline glyph with the current color. */
.ms-lib-heart {
  filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.7));
}

.ms-lib-heart--on :deep(svg) {
  fill: currentColor;
}

.ms-lib-heart:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
  opacity: 1;
}
</style>
