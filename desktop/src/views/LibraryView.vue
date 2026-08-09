<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import Icon from "@ui/components/Icon.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import ThumbnailSizeSlider from "@ui/components/ThumbnailSizeSlider.vue";
import {
  GALLERY_THUMBNAIL_SIZE_MAX,
  GALLERY_THUMBNAIL_SIZE_MIN,
  GALLERY_THUMBNAIL_SIZE_STEP,
  loadGalleryThumbnailSize,
  saveGalleryThumbnailSize,
} from "@studio/lib/galleryThumbnailSize";
import { blobToBase64 } from "@studio/lib/base64";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import HistoryDrawer from "../components/library/HistoryDrawer.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import HostFilterChips from "../components/shell/HostFilterChips.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import { galleryMediaPath, isAudioItem, isVideoItem, mediaPath } from "../lib/gallery/media";
import { applySelectionClick } from "../lib/gallery/selection";
import {
  planSequenceReuse,
  sequenceEditAvailability,
  sequenceGoneMessage,
  sequenceHostUnreachableMessage,
} from "@studio/lib/sequenceReuse";
import { ApiError, apiFetch, apiFetchTo, type ApiTarget } from "../lib/api/client";
import {
  useGalleryStore,
  type GalleryKindFilter,
  type GalleryLocation,
  type MergedPrint,
} from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useChainJobsStore } from "../stores/chainJobs";
import { useComposerStore } from "../stores/composer";
import { useGenerateFormStore } from "../stores/generateForm";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { inTauri, ipc } from "../lib/ipc";
import { copyImageBytesToClipboard } from "../lib/clipboard";
import { copyLocalOutputPath } from "../lib/localOutputPath";
import { primaryModifierPressed } from "../lib/platform";
import { allowsNativeContextMenu } from "../lib/shortcuts";
import { modelDisplayNameForId } from "../lib/models";
import type { GalleryImage } from "../lib/api/types";
import { isUpscaledImage } from "../lib/gallery/upscaled";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import {
  appendMinimaxH3GalleryImageReference,
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  setMinimaxH3GalleryImageFirstFrame,
} from "@studio/lib/minimaxH3Authoring";

const GAP = 8;
const PAD = 16;
/** Extra hosts have no SSE — their buckets poll while the view is open. */
const EXTRA_POLL_MS = 15_000;

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
const modelLabel = (name: string) => modelDisplayNameForId(name, models.all);

const primaryId = computed(() => hosts.primaryHost?.id ?? null);

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

// ── Header labels ────────────────────────────────────────────────────────────
const scopeLabel = computed(() =>
  gallery.filter === "all"
    ? "all hosts"
    : (gallery.sources.find((s) => s.key === gallery.filter)?.label ?? "all hosts"),
);
const countLabel = computed(() => {
  const n = gallery.filtered.length;
  return `${n} ${n === 1 ? "print" : "prints"} · ${scopeLabel.value}`;
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
  const path = mediaPath(entry.item.filename);
  const target = targetFor(entry);
  return (target ? apiFetchTo(target, path) : apiFetch(path)).then((r) => r.blob());
}

async function fetchItemBase64(entry: MergedPrint): Promise<string> {
  return blobToBase64(await fetchItemBlob(entry));
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

// ── Upscale (Real-ESRGAN via the engine; result saved to this Mac) ─────────
const upscalingFilename = ref<string | null>(null);

/** First known upscaler; the server auto-pulls it on first use. */
const upscalerModel = computed(() => models.upscalers[0]?.name ?? "real-esrgan-x4plus");

/**
 * Run the upscaler over the stream endpoint: its `complete` event carries
 * the result as base64, unlike the plain endpoint whose `image` is an
 * ImageData object with a JSON byte array. Resolves with the base64 image.
 */
async function streamUpscale(image: string): Promise<string> {
  const { sseStream } = await import("../lib/api/sse");
  return new Promise<string>((resolve, reject) => {
    const abort = new AbortController();
    let settled = false;
    void sseStream("/api/upscale/stream", {
      method: "POST",
      body: { model: upscalerModel.value, image, output_format: "png" },
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
        } catch (err) {
          settled = true;
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      },
      onClose: (err) => {
        if (!settled) reject(err ?? new Error("The upscale stream ended without a result."));
      },
    });
  });
}

async function upscaleItem(entry: MergedPrint) {
  if (upscalingFilename.value) return;
  upscalingFilename.value = entry.item.filename;
  toasts.push(`Upscaling with ${upscalerModel.value}…`);
  try {
    const upscaled = await streamUpscale(await fetchItemBase64(entry));
    // The upscale endpoints don't persist server-side — the local save IS
    // the durable copy.
    const stem = entry.item.filename.replace(/\.[^.]+$/, "");
    const saved = await ipc.saveOutputBytes(`${stem}-upscaled.png`, upscaled);
    toasts.push(`Upscaled — saved locally as ${saved}`);
    // The save landed in this Mac's output dir: on a local/external primary
    // the engine bucket reads that same dir; on a remote primary it's the
    // IPC bucket. Refresh whichever of the two is loaded.
    for (const key of new Set([primaryId.value, "local"])) {
      if (key) void gallery.refreshHost(key);
    }
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  } finally {
    upscalingFilename.value = null;
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
  composer.set({ metadata: entry.item.metadata });
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

function tileMenu(entry: MergedPrint): MenuEntry[] {
  const item = entry.item;
  const m = item.metadata;
  const selectedForBulkDelete =
    selectMode.value && bulkSelection.value.has(item.filename) && bulkSelection.value.size > 1;
  return [
    ...(isSequencePrint(entry)
      ? [
          ...(canEditSequence(entry)
            ? [{ label: "Edit sequence", action: () => void editSequence(entry) }]
            : []),
          { label: "Duplicate as new", action: () => reuseSequence(entry) },
        ]
      : [{ label: "Reuse settings", action: () => reuseSettings(entry) }]),
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
    {
      label: "Copy image",
      disabled: isVideo(item),
      action: () => void copyImage(entry),
    },
    {
      label: "Use as source",
      disabled: isVideo(item),
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
      label: upscalingFilename.value === item.filename ? "Upscaling…" : "Upscale",
      // Upscale runs on the PRIMARY engine only for now — routing it to the
      // item's origin host is a follow-up.
      disabled:
        isVideo(item) || entry.sourceKey !== primaryId.value || upscalingFilename.value !== null,
      action: () => void upscaleItem(entry),
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
      label: selectedForBulkDelete ? `Delete ${bulkSelection.value.size} selected` : "Delete",
      danger: true,
      action: () => {
        if (selectedForBulkDelete) void deleteSelectedPrints();
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
watch(rowHeight, (value) => saveGalleryThumbnailSize(value));

// ── Single-print delete: optimistic + undoable (§08 G12) ─────────────────────
// The tile leaves the grid instantly; a 6 s undo toast holds the print in limbo
// (no server call). Undo restores it; letting the toast expire commits the real
// DELETE. Several deletes can be pending at once, so timers are keyed per print.
const UNDO_WINDOW_MS = 6000;
const pendingDeletes = new Map<
  string,
  {
    sourceKey: string;
    filename: string;
    locations: GalleryLocation[];
    timer: ReturnType<typeof setTimeout>;
  }
>();
const deleteKey = (sourceKey: string, filename: string) => `${sourceKey}::${filename}`;

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
const searchEl = ref<HTMLInputElement | null>(null);
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
]);
const setKind = (value: GalleryKindFilter) => (gallery.mediaKind = value);

// ── Bulk select mode ───────────────────────────────────────────────────────
// Selection is keyed by print identity (filename — the merged grid's
// cross-host identity), never row index: the virtualized grid re-flows.
// Drag-marquee selection is a deliberate follow-up — it fights the
// virtualized justified grid; click / shift-range / meta-toggle ship first.
const selectMode = ref(false);
const bulkSelection = ref<Set<string>>(new Set());
const bulkAnchor = ref<string | null>(null);
const confirmingBulkDelete = ref(false);
const bulkDeleting = ref(false);

function setSelectMode(next: boolean) {
  selectMode.value = next;
  if (!next) {
    bulkSelection.value = new Set();
    bulkAnchor.value = null;
    confirmingBulkDelete.value = false;
  }
}

function onTileClick(entry: MergedPrint, e: MouseEvent) {
  if (!selectMode.value) {
    select(entry);
    return;
  }
  const next = applySelectionClick(
    bulkSelection.value,
    bulkAnchor.value,
    gallery.filtered.map((x) => x.item.filename),
    entry.item.filename,
    { shift: e.shiftKey, meta: e.metaKey || e.ctrlKey },
  );
  bulkSelection.value = next.selection;
  bulkAnchor.value = next.anchor;
}

function onTileDblclick(entry: MergedPrint) {
  if (selectMode.value) return;
  select(entry);
  lightboxOpen.value = true;
}

function selectAllInFilter() {
  bulkSelection.value = new Set(gallery.filtered.map((e) => e.item.filename));
}

function clearBulkSelection() {
  bulkSelection.value = new Set();
  bulkAnchor.value = null;
}

async function deleteSelectedPrints() {
  if (!confirmingBulkDelete.value) {
    confirmingBulkDelete.value = true;
    return;
  }
  confirmingBulkDelete.value = false;
  if (bulkDeleting.value) return;
  const targets = gallery.filtered.filter((e) => bulkSelection.value.has(e.item.filename));
  if (targets.length === 0) return;
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
  const remaining = new Set(gallery.filtered.map((e) => e.item.filename));
  bulkSelection.value = new Set([...bulkSelection.value].filter((f) => remaining.has(f)));
  if (bulkAnchor.value && !remaining.has(bulkAnchor.value)) bulkAnchor.value = null;
  if (selected.value && !remaining.has(selected.value.filename)) {
    selected.value = null;
    lightboxOpen.value = false;
  }
}

let resizeObserver: ResizeObserver | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;

/** Justified layout over the filtered merged set; each laid tile keeps its
 *  merged entry so origin (badge, target, actions) travels with it. */
const rows = computed(() => {
  const entries = gallery.filtered;
  const laidRows = layoutJustifiedRows(
    entries.map((e) => e.item),
    Math.max(0, containerWidth.value - PAD * 2),
    rowHeight.value,
    GAP,
  );
  let cursor = 0;
  return laidRows.map((r) => ({
    height: r.height,
    items: r.items.map((laid) => ({ ...laid, entry: entries[cursor++]! })),
  }));
});

const virtualizer = useVirtualizer(
  computed(() => ({
    count: rows.value.length,
    getScrollElement: () => scrollEl.value,
    estimateSize: (i: number) => (rows.value[i]?.height ?? rowHeight.value) + GAP,
    overscan: 5,
  })),
);

// TanStack Virtual caches per-index measurements; a new estimateSize closure
// alone does NOT invalidate that cache. When the justified layout re-flows
// (container resize, entries arriving, row-height change) the cached offsets
// go stale and rows render overlapping — re-measure on any height change.
watch(
  () => rows.value.map((r) => r.height).join(","),
  () => virtualizer.value?.measure?.(),
);

const showBadges = computed(() => gallery.filter === "all" && gallery.chipCounts.length > 1);
const availabilityLabel = (entry: MergedPrint) =>
  entry.availableOn.map((source) => source.label).join(" · ");

const isSelected = (entry: MergedPrint) =>
  selected.value !== null &&
  selected.value.sourceKey === entry.sourceKey &&
  selected.value.filename === entry.item.filename;

function select(entry: MergedPrint) {
  selected.value = { sourceKey: entry.sourceKey, filename: entry.item.filename };
}

const selectedIndex = computed(() => gallery.filtered.findIndex((e) => isSelected(e)));
const selectedEntry = computed<MergedPrint | null>(
  () => gallery.filtered[selectedIndex.value] ?? null,
);

/** Shared with the store's kind filter so badge and chips never disagree. */
const isVideo = (i: GalleryImage) => isVideoItem(i);
const isAudio = (i: GalleryImage) => isAudioItem(i);

async function copyImage(entry: MergedPrint) {
  try {
    const path = galleryMediaPath(entry.item.filename, gallery.mediaSourceOf(entry.sourceKey));
    const target = targetFor(entry);
    await copyImageBytesToClipboard(
      path,
      target
        ? {
            fetchImage: async (p) =>
              new Uint8Array(await (await apiFetchTo(target, p)).arrayBuffer()),
          }
        : undefined,
    );
    toasts.push("Image copied");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

function moveSelection(delta: number) {
  const entries = gallery.filtered;
  if (entries.length === 0) return;
  const next = Math.min(
    entries.length - 1,
    Math.max(0, (selectedIndex.value === -1 ? 0 : selectedIndex.value) + delta),
  );
  select(entries[next]!);
}

function onKeydown(e: KeyboardEvent) {
  // ⌘F focuses the view's own search — the screen-level filter shortcut.
  if (e.key === "f" && primaryModifierPressed(e) && !e.altKey) {
    e.preventDefault();
    searchEl.value?.focus();
    return;
  }
  // The history drawer owns the keyboard while it's open.
  if (historyOpen.value) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    // WebKit treats an unhandled Backspace/Delete as history-back. Keep the
    // native editing behavior in text fields, but always claim it elsewhere
    // in Library so deleting a print can never navigate to Create.
    if (allowsNativeContextMenu(e.target as Element | null)) return;
    e.preventDefault();
    if (e.repeat) return;
    if (selectMode.value && bulkSelection.value.size > 0) {
      void deleteSelectedPrints();
    } else if (selectedEntry.value) {
      deletePrint(selectedEntry.value);
    }
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    moveSelection(1);
  } else if (e.key === "ArrowLeft") {
    e.preventDefault();
    moveSelection(-1);
  } else if (e.key === " ") {
    e.preventDefault();
    if (selected.value) lightboxOpen.value = !lightboxOpen.value;
  } else if (e.key === "Escape") {
    if (lightboxOpen.value) lightboxOpen.value = false;
    else if (selectMode.value) setSelectMode(false);
  }
}

/** Fire the real DELETE for a pending print, surfacing any failure (which
 *  restores the print via the store's finally). */
async function commitDelete(sourceKey: string, filename: string) {
  const key = deleteKey(sourceKey, filename);
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
function undoDelete(sourceKey: string, filename: string) {
  const key = deleteKey(sourceKey, filename);
  const pending = pendingDeletes.get(key);
  if (pending) clearTimeout(pending.timer);
  pendingDeletes.delete(key);
  if (pending) gallery.cancelDeleteEverywhere(pending.locations);
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

  const index = gallery.filtered.findIndex(
    (candidate) => candidate.sourceKey === sourceKey && candidate.item.filename === filename,
  );
  const wasSelected = isSelected(entry);
  const locations = gallery.beginDeleteEverywhere(entry);
  if (wasSelected) {
    const remaining = gallery.filtered;
    if (remaining.length === 0) {
      lightboxOpen.value = false;
      selected.value = null;
    } else {
      select(remaining[Math.min(Math.max(index, 0), remaining.length - 1)]!);
    }
  }

  const timer = setTimeout(() => void commitDelete(sourceKey, filename), UNDO_WINDOW_MS);
  pendingDeletes.set(key, { sourceKey, filename, locations, timer });
  toasts.push(`Deleted ${filename} everywhere`, "info", {
    durationMs: UNDO_WINDOW_MS,
    action: { label: "Undo", run: () => undoDelete(sourceKey, filename) },
  });
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
          ? appendMinimaxH3GalleryImageReference(form.h3Authoring, image)
          : setMinimaxH3GalleryImageFirstFrame(form.h3Authoring, image);
      if (!result.ok) throw new Error(result.error);
      form.h3Authoring = result.state;
    } else if (isMinimaxH3Identity(form.family, form.model)) {
      throw new Error(
        "Choose an explicit MiniMax H3 FL2VA or Ref2VA model before adding a source.",
      );
    } else {
      form.sourceImage = base64;
      form.sourceImageName = entry.item.filename;
      form.sourceFit = { mode: "lanczos-resize" };
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
// reveals that print — filters reset so it can't be hidden, then selection +
// lightbox open once the buckets deliver it. One-shot: the param drops after
// use so closing the lightbox doesn't re-open it.
watch(
  [() => route.query.print, () => gallery.merged.length],
  ([print]) => {
    if (typeof print !== "string" || !print) return;
    const entry = gallery.merged.find((e) => e.item.filename === print);
    if (!entry) return;
    gallery.filter = "all";
    gallery.mediaKind = "all";
    gallery.query = "";
    searchInput.value = "";
    select(entry);
    lightboxOpen.value = true;
    void router.replace({ query: { ...route.query, print: undefined } });
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

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  pollTimer = setInterval(() => void gallery.pollExtras(), EXTRA_POLL_MS);
  if (scrollEl.value) {
    containerWidth.value = scrollEl.value.clientWidth;
    resizeObserver = new ResizeObserver((entries) => {
      containerWidth.value = entries[0]?.contentRect.width ?? containerWidth.value;
    });
    resizeObserver.observe(scrollEl.value);
  }
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  resizeObserver?.disconnect();
  // Finalize any deletes still inside their undo window — leaving Library
  // commits them rather than stranding a hidden-but-not-deleted print.
  for (const { sourceKey, filename } of [...pendingDeletes.values()]) {
    void commitDelete(sourceKey, filename);
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
    <header class="flex h-[52px] shrink-0 items-center gap-3 border-b border-edge px-6">
      <span class="font-display text-[17px] font-semibold text-ink" style="font-stretch: 92%">
        Library
      </span>
      <span class="data-mono text-caption text-ink-3">{{ countLabel }}</span>
      <span v-if="gallery.firstError" class="text-caption text-stop">
        {{ gallery.firstError }}
      </span>
      <HostFilterChips
        v-if="gallery.chipCounts.length > 1"
        v-model="gallery.filter"
        class="ml-1"
        :chips="gallery.chipCounts"
        :all-count="gallery.merged.length"
      />

      <div class="flex-1" />

      <ThumbnailSizeSlider
        v-model="rowHeight"
        :min="GALLERY_THUMBNAIL_SIZE_MIN"
        :max="GALLERY_THUMBNAIL_SIZE_MAX"
        :step="GALLERY_THUMBNAIL_SIZE_STEP"
      />

      <SegmentedControl
        :model-value="gallery.mediaKind"
        :options="kindOptions"
        label="Media kind"
        @update:model-value="setKind"
      />

      <label
        class="flex h-[34px] w-[180px] items-center gap-2 rounded-chrome border border-ce bg-bench px-2.5"
      >
        <Icon name="search" :size="14" class="shrink-0 text-ink-3" />
        <input
          ref="searchEl"
          v-model="searchInput"
          data-selectable
          type="search"
          placeholder="Search prompts…"
          aria-label="Search prints"
          class="min-w-0 flex-1 bg-transparent text-body text-ink outline-none placeholder:text-ink-3"
        />
      </label>

      <button
        type="button"
        class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink"
        title="History"
        aria-label="Open history"
        @click="openHistory"
      >
        <Icon name="history" :size="17" />
      </button>
      <button
        type="button"
        class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome transition-colors duration-100"
        :class="
          selectMode
            ? 'bg-[color-mix(in_srgb,var(--safelight)_16%,transparent)] text-safelight'
            : 'text-ink-3 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink'
        "
        :aria-pressed="selectMode"
        title="Select"
        aria-label="Toggle select mode"
        @click="setSelectMode(!selectMode)"
      >
        <Icon name="check" :size="17" />
      </button>
      <button
        type="button"
        class="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-chrome text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)] hover:text-ink"
        title="Refresh"
        aria-label="Refresh library"
        @click="gallery.fetchAll()"
      >
        <Icon name="refresh" :size="17" />
      </button>
    </header>

    <div ref="scrollEl" class="min-h-0 flex-1 overflow-y-auto" style="contain: strict">
      <EmptyState
        v-if="gallery.loaded && gallery.filtered.length === 0 && gallery.hostFiltered.length > 0"
        headline="No matching prints"
        detail="Nothing here matches the current search or media filter."
      />
      <EmptyState
        v-else-if="gallery.loaded && gallery.filtered.length === 0"
        headline="No prints here yet"
        :detail="
          gallery.filter === 'local'
            ? 'Generations saved on this Mac will appear here.'
            : 'Generate one and it lands here.'
        "
        action="Go to Create"
        @action="router.push('/create')"
      />
      <div v-else :style="{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }">
        <div
          v-for="vrow in virtualizer.getVirtualItems()"
          :key="vrow.key as number"
          class="absolute right-0 left-0 flex"
          :style="{
            transform: `translateY(${vrow.start}px)`,
            gap: `${GAP}px`,
            padding: `0 ${PAD}px`,
          }"
        >
          <button
            v-for="laid in rows[vrow.index]?.items ?? []"
            :key="`${laid.entry.sourceKey}::${laid.item.filename}`"
            type="button"
            class="ms-lib-tile group relative shrink-0 overflow-hidden rounded-[9px] border"
            :class="
              (selectMode ? bulkSelection.has(laid.item.filename) : isSelected(laid.entry))
                ? 'border-transparent ring-2 ring-safelight'
                : 'border-[color-mix(in_srgb,var(--rebate)_14%,transparent)]'
            "
            :style="{ width: `${laid.width}px`, height: `${laid.height}px` }"
            @click="onTileClick(laid.entry, $event)"
            @contextmenu="
              select(laid.entry);
              contextMenu.open($event, tileMenu(laid.entry));
            "
            @dblclick="onTileDblclick(laid.entry)"
          >
            <AuthedMedia
              :path="
                galleryMediaPath(
                  laid.item.filename,
                  gallery.mediaSourceOf(laid.entry.sourceKey),
                  true,
                )
              "
              :target="targetFor(laid.entry)"
              :cache-key="laid.entry.sourceKey"
              :video="gallery.mediaSourceOf(laid.entry.sourceKey) === 'local' && isVideo(laid.item)"
              :alt="laid.item.metadata.prompt"
            />
            <!-- NEW badge (top-left) — hidden while selecting, where the
                 checkbox owns that corner. -->
            <span
              v-if="!selectMode && isFresh(laid.entry)"
              data-test="new-badge"
              class="ms-lib-new absolute top-2 left-2"
            >
              New
            </span>
            <span
              v-if="!selectMode && isUpscaledImage(laid.item)"
              data-test="upscaled-badge"
              class="ms-lib-upscaled absolute top-2 right-2"
            >
              Upscaled
            </span>
            <span
              v-if="isVideo(laid.item) || isAudio(laid.item)"
              class="absolute top-1.5 right-1.5 rounded-control bg-black/60 px-1 text-caption text-on-media"
            >
              {{ isAudio(laid.item) ? "♪" : "▶" }}
            </span>
            <span
              v-if="selectMode"
              data-test="select-indicator"
              class="absolute top-1.5 left-1.5 flex h-5 w-5 items-center justify-center rounded-full text-caption"
              :class="
                bulkSelection.has(laid.item.filename)
                  ? 'bg-safelight font-semibold text-on-accent'
                  : 'border border-white/70 bg-black/40 text-on-media'
              "
            >
              {{ bulkSelection.has(laid.item.filename) ? "✓" : "" }}
            </span>
            <!-- The badge yields to the rising edge code on hover — both live
                 in the tile's bottom margin and must never overlap. -->
            <span
              v-if="showBadges"
              data-test="host-badge"
              class="edge-code absolute bottom-1.5 left-1.5 max-w-[70%] truncate rounded-control bg-black/60 px-1 !text-on-media transition-opacity duration-100 group-hover:opacity-0"
              :title="`Available on ${availabilityLabel(laid.entry)}`"
            >
              {{ availabilityLabel(laid.entry) }}
            </span>
            <span
              class="edge-code absolute right-0 bottom-0 left-0 translate-y-full bg-black/60 px-1.5 py-0.5 text-left !text-on-media transition-transform duration-100 group-hover:translate-y-0"
            >
              {{ modelLabel(laid.item.metadata.model) }} · S {{ laid.item.metadata.seed }}
            </span>
          </button>
        </div>
      </div>
    </div>

    <!-- Floating bulk-action bar while select mode is active. -->
    <div
      v-if="selectMode"
      data-test="bulk-action-bar"
      class="border-edge absolute bottom-4 left-1/2 z-30 flex max-w-[calc(100%-2rem)] -translate-x-1/2 flex-wrap items-center gap-2 rounded-chrome border bg-bench px-3 py-2 shadow-lg"
      role="toolbar"
      aria-label="Selection actions"
    >
      <span class="data-mono px-1 text-caption text-ink">
        {{ bulkSelection.size }}
        <span class="text-ink-3">/ {{ gallery.filtered.length }} selected</span>
      </span>
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-caption text-ink-2 transition-colors duration-100 hover:text-ink disabled:opacity-50"
        :disabled="gallery.filtered.length === 0"
        @click="selectAllInFilter"
      >
        Select all
      </button>
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-caption text-ink-2 transition-colors duration-100 hover:text-ink disabled:opacity-50"
        :disabled="bulkSelection.size === 0"
        @click="clearBulkSelection"
      >
        Clear
      </button>
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-caption transition-colors duration-100 disabled:opacity-50"
        :class="
          confirmingBulkDelete
            ? 'border-stop bg-stop font-semibold text-on-accent'
            : 'text-ink-2 hover:text-stop'
        "
        :disabled="bulkSelection.size === 0 || bulkDeleting"
        @blur="confirmingBulkDelete = false"
        @click="deleteSelectedPrints"
      >
        {{
          bulkDeleting
            ? "Deleting…"
            : confirmingBulkDelete
              ? `Delete ${bulkSelection.size} ${bulkSelection.size === 1 ? "print" : "prints"}? This can't be undone.`
              : "Delete selected"
        }}
      </button>
      <button
        type="button"
        class="flex h-7 w-7 items-center justify-center rounded-control text-ink-3 transition-colors duration-100 hover:text-ink"
        aria-label="Exit select mode"
        title="Exit select mode (Esc)"
        @click="setSelectMode(false)"
      >
        ✕
      </button>
    </div>

    <Lightbox
      v-if="lightboxOpen && selectedEntry"
      :item="selectedEntry.item"
      :index="selectedIndex"
      :count="gallery.filtered.length"
      :video="isVideo(selectedEntry.item)"
      :audio="isAudio(selectedEntry.item)"
      :source="gallery.mediaSourceOf(selectedEntry.sourceKey)"
      :target="targetFor(selectedEntry)"
      :cache-key="selectedEntry.sourceKey"
      :host-label="availabilityLabel(selectedEntry)"
      :can-reveal="canReveal(selectedEntry)"
      :is-sequence="isSequencePrint(selectedEntry)"
      :can-edit-sequence="canEditSequence(selectedEntry)"
      @close="lightboxOpen = false"
      @prev="moveSelection(-1)"
      @next="moveSelection(1)"
      @delete="removeSelected"
      @use-source="useSelectedAsSource"
      @reuse-sequence="reuseSequence(selectedEntry)"
      @edit-sequence="editSequence(selectedEntry)"
    />

    <HistoryDrawer :open="historyOpen" @close="closeHistory" />
  </div>
</template>

<style scoped>
.ms-lib-tile {
  transition:
    transform var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}

.ms-lib-tile:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
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
</style>
