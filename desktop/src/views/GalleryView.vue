<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import HostFilterChips from "../components/shell/HostFilterChips.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import { galleryMediaPath, mediaPath } from "../lib/gallery/media";
import { formatBytes } from "../lib/format";
import { apiFetch, apiFetchTo, type ApiTarget } from "../lib/api/client";
import { useGalleryStore, type MergedPrint } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { inTauri, ipc } from "../lib/ipc";
import { copyImageBytesToClipboard } from "../lib/clipboard";
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
const canSaveLocally = (entry: MergedPrint) =>
  inTauri() && gallery.hostFor(entry.sourceKey)?.kind === "remote";

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
    const saved = await ipc.saveOutputBytes(entry.item.filename, await fetchItemBase64(entry));
    toasts.push(`Saved to this Mac — ${saved}`);
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
    toasts.push(`Upscaled — saved to this Mac as ${saved}`);
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
      label: "Save to this Mac",
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

const isVideo = (i: GalleryImage) =>
  i.format === "mp4" || i.filename.endsWith(".mp4") || !!i.metadata.video_frames;

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
    lightboxOpen.value = false;
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
});
</script>

<template>
  <div class="flex h-full flex-col">
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
      <span v-if="gallery.firstError" class="ml-auto text-caption text-stop">
        {{ gallery.firstError }}
      </span>
    </header>

    <div ref="scrollEl" class="min-h-0 flex-1 overflow-y-auto" style="contain: strict">
      <EmptyState
        v-if="gallery.loaded && gallery.filtered.length === 0"
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
              isSelected(laid.entry)
                ? 'border-transparent ring-2 ring-safelight'
                : 'border-[color-mix(in_srgb,var(--rebate)_14%,transparent)]'
            "
            :style="{ width: `${laid.width}px`, height: `${laid.height}px` }"
            @click="select(laid.entry)"
            @contextmenu="
              select(laid.entry);
              contextMenu.open($event, tileMenu(laid.entry));
            "
            @dblclick="
              select(laid.entry);
              lightboxOpen = true;
            "
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
              v-if="showBadges"
              data-test="host-badge"
              class="edge-code absolute bottom-1.5 left-1.5 max-w-[70%] truncate rounded-control bg-black/60 px-1 !text-on-media"
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
