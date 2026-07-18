<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import HostFilterChips from "../components/shell/HostFilterChips.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import { galleryMediaPath, isVideoItem, mediaPath } from "../lib/gallery/media";
import { applySelectionClick } from "../lib/gallery/selection";
import { formatBytes } from "../lib/format";
import { apiFetch, apiFetchTo, type ApiTarget } from "../lib/api/client";
import { useGalleryStore, type GalleryKindFilter, type MergedPrint } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { inTauri, ipc } from "../lib/ipc";
import { copyImageBytesToClipboard } from "../lib/clipboard";
import { primaryModifierPressed } from "../lib/platform";
import type { GalleryImage } from "../lib/api/types";

const GAP = 8;
const PAD = 16;
/** Extra hosts have no SSE — their buckets poll while the view is open. */
const EXTRA_POLL_MS = 15_000;

const router = useRouter();
const route = useRoute();
const gallery = useGalleryStore();
const hosts = useHostsStore();
const models = useModelStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const toasts = useToastStore();

const primaryId = computed(() => hosts.primaryHost?.id ?? null);

const targetFor = (entry: MergedPrint): ApiTarget | null => gallery.targetOf(entry.sourceKey);

/** Reveal works for files on this Mac: the IPC bucket, or a local-kind
 *  (built-in/external) engine whose output dir is this machine's. */
const canReveal = (entry: MergedPrint) =>
  entry.sourceKey === "local" || gallery.hostFor(entry.sourceKey)?.kind === "local";

/** Prints on a remote host can be pulled into this Mac's gallery. */
/** Copyable to this Mac: a remote-origin tile with no local copy yet (by
 *  filename or byte identity). The menu item stays visible and grays out
 *  once a local copy exists. */
const canSaveLocally = (entry: MergedPrint) =>
  inTauri() && gallery.hostFor(entry.sourceKey)?.kind === "remote" && !gallery.existsLocally(entry);

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/** Authed source bytes for a host-gallery item, as base64 (origin-aware). */
async function fetchItemBase64(entry: MergedPrint): Promise<string> {
  const path = mediaPath(entry.item.filename);
  const target = targetFor(entry);
  const blob = await (target ? apiFetchTo(target, path) : apiFetch(path)).then((r) => r.blob());
  return blobToBase64(blob);
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

function tileMenu(entry: MergedPrint): MenuEntry[] {
  const item = entry.item;
  const m = item.metadata;
  return [
    {
      label: "Reuse settings",
      action: () => {
        // Full metadata → full-fidelity restore (negative prompt, LoRAs,
        // scheduler, video params, …) via `applyPrefillToForm`.
        composer.set({ metadata: m });
        void router.push("/generate");
      },
    },
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
      label: "Delete",
      danger: true,
      action: () => {
        select(entry);
        void removeSelected().then(() => toasts.push("Deleted"));
      },
    },
  ];
}

const scrollEl = ref<HTMLElement | null>(null);
const containerWidth = ref(0);
const selected = ref<{ sourceKey: string; filename: string } | null>(null);
const lightboxOpen = ref(false);
const rowHeight = ref(180);

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

const kindChips = computed(() => [
  { key: "image", label: "Images", count: gallery.kindCounts.image },
  { key: "video", label: "Video", count: gallery.kindCounts.video },
]);
const setKind = (value: string) => (gallery.mediaKind = value as GalleryKindFilter);

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
  // Resolve filenames to their represented origin via the current filter —
  // removeMany then mirrors the single-delete routing per item.
  const targets = gallery.filtered.filter((e) => bulkSelection.value.has(e.item.filename));
  if (targets.length === 0) return;
  bulkDeleting.value = true;
  try {
    const { deleted, failed } = await gallery.removeMany(
      targets.map((e) => ({ sourceKey: e.sourceKey, filename: e.item.filename })),
    );
    if (failed > 0) {
      toasts.push(`Deleted ${deleted} of ${targets.length}. ${failed} failed.`, "error");
    } else {
      toasts.push(deleted === 1 ? "Deleted 1 print" : `Deleted ${deleted} prints`);
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
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "ArrowRight") {
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

async function removeSelected() {
  const entry = selectedEntry.value;
  if (!entry) return;
  const index = selectedIndex.value;
  await gallery.remove(entry.sourceKey, entry.item.filename);
  const remaining = gallery.filtered;
  if (remaining.length === 0) {
    lightboxOpen.value = false;
    selected.value = null;
  } else {
    select(remaining[Math.min(index, remaining.length - 1)]!);
  }
}

// Deep link: /gallery?host=<bucket key> pre-picks a chip ("local" = This
// Mac's key in every mode). Plain /gallery keeps the session filter.
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

// Deep link: /gallery?print=<filename> (a ⌘K gallery result) reveals that
// print — filters reset so it can't be hidden, then selection + lightbox
// open once the buckets deliver it. One-shot: the param drops after use so
// closing the lightbox doesn't re-open it.
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
    <header class="border-edge flex h-11 items-center gap-3 border-b px-4">
      <span class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Gallery
      </span>
      <span class="data-mono text-caption text-ink-3">
        {{ gallery.filtered.length }} prints · {{ formatBytes(gallery.totalBytes) }}
      </span>
      <HostFilterChips
        v-if="gallery.chipCounts.length > 1"
        v-model="gallery.filter"
        class="ml-2"
        :chips="gallery.chipCounts"
        :all-count="gallery.merged.length"
      />
      <HostFilterChips
        :chips="kindChips"
        :model-value="gallery.mediaKind"
        :all-count="gallery.kindCounts.all"
        aria-label="Media kind"
        @update:model-value="setKind"
      />
      <span v-if="gallery.firstError" class="text-caption text-stop">
        {{ gallery.firstError }}
      </span>
      <input
        ref="searchEl"
        v-model="searchInput"
        data-selectable
        type="search"
        placeholder="Search prints…"
        aria-label="Search prints"
        class="border-edge ml-auto h-7 w-48 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
      />
      <button
        type="button"
        class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-caption transition-colors duration-100"
        :class="selectMode ? 'border-safelight text-safelight' : 'text-ink-2 hover:text-ink'"
        :aria-pressed="selectMode"
        @click="setSelectMode(!selectMode)"
      >
        Select
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
        action="Go to Generate"
        @action="router.push('/generate')"
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
            class="group relative shrink-0 overflow-hidden rounded-media border transition-shadow duration-100"
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
            <span
              v-if="isVideo(laid.item)"
              class="absolute top-1.5 right-1.5 rounded-control bg-black/60 px-1 text-caption text-on-media"
            >
              ▶
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
              {{ laid.item.metadata.model }} · S {{ laid.item.metadata.seed }}
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
      :source="gallery.mediaSourceOf(selectedEntry.sourceKey)"
      :target="targetFor(selectedEntry)"
      :cache-key="selectedEntry.sourceKey"
      :host-label="availabilityLabel(selectedEntry)"
      :can-reveal="canReveal(selectedEntry)"
      @close="lightboxOpen = false"
      @prev="moveSelection(-1)"
      @next="moveSelection(1)"
      @delete="removeSelected"
    />
  </div>
</template>
