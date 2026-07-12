<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import { galleryMediaPath, type GallerySource } from "../lib/gallery/media";
import { formatBytes } from "../lib/format";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { ipc } from "../lib/ipc";
import { copyImageBytesToClipboard } from "../lib/clipboard";
import type { GalleryImage } from "../lib/api/types";

const GAP = 8;
const PAD = 16;

const router = useRouter();
const conn = useConnectionStore();
const gallery = useGalleryStore();
const models = useModelStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const toasts = useToastStore();

const canReveal = computed(
  () => gallery.source === "local" || conn.mode === "local" || conn.mode === "external",
);

/** Remote prints can be pulled into this Mac's gallery on demand. */
const canSaveLocally = computed(() => gallery.source === "engine" && conn.mode === "remote");

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/** Authed source bytes for an engine-gallery item, as base64. */
async function fetchItemBase64(item: GalleryImage): Promise<string> {
  const { apiFetch } = await import("../lib/api/client");
  const blob = await apiFetch(`/api/gallery/image/${encodeURIComponent(item.filename)}`).then((r) =>
    r.blob(),
  );
  return blobToBase64(blob);
}

async function saveToThisMac(item: GalleryImage) {
  try {
    const saved = await ipc.saveOutputBytes(item.filename, await fetchItemBase64(item));
    toasts.push(`Saved to this Mac — ${saved}`);
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

async function upscaleItem(item: GalleryImage) {
  if (upscalingFilename.value) return;
  upscalingFilename.value = item.filename;
  toasts.push(`Upscaling with ${upscalerModel.value}…`);
  try {
    const upscaled = await streamUpscale(await fetchItemBase64(item));
    // The upscale endpoints don't persist server-side — the local save IS
    // the durable copy.
    const stem = item.filename.replace(/\.[^.]+$/, "");
    const saved = await ipc.saveOutputBytes(`${stem}-upscaled.png`, upscaled);
    toasts.push(`Upscaled — saved to this Mac as ${saved}`);
    // On a local/external connection the engine gallery reads the same
    // output dir the save just wrote to — refresh whatever is on screen.
    void gallery.fetch();
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  } finally {
    upscalingFilename.value = null;
  }
}

function tileMenu(item: GalleryImage): MenuEntry[] {
  const m = item.metadata;
  return [
    {
      label: "Reuse settings",
      action: () => {
        composer.set({
          prompt: m.prompt,
          model: m.model,
          seed: m.seed,
          width: m.width,
          height: m.height,
          steps: m.steps,
          guidance: m.guidance,
        });
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
      action: () => void copyImage(item),
    },
    { separator: true },
    {
      label: upscalingFilename.value === item.filename ? "Upscaling…" : "Upscale",
      disabled: isVideo(item) || gallery.source !== "engine" || upscalingFilename.value !== null,
      action: () => void upscaleItem(item),
    },
    {
      label: "Save to this Mac",
      disabled: !canSaveLocally.value,
      action: () => void saveToThisMac(item),
    },
    {
      label: "Reveal in Finder",
      disabled: !canReveal.value,
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
        selected.value = item.filename;
        void removeSelected().then(() => toasts.push("Deleted"));
      },
    },
  ];
}

const scrollEl = ref<HTMLElement | null>(null);
const containerWidth = ref(0);
const selected = ref<string | null>(null);
const lightboxOpen = ref(false);
const rowHeight = ref(180);

let resizeObserver: ResizeObserver | null = null;

const rows = computed(() =>
  layoutJustifiedRows(
    gallery.items,
    Math.max(0, containerWidth.value - PAD * 2),
    rowHeight.value,
    GAP,
  ),
);

const virtualizer = useVirtualizer(
  computed(() => ({
    count: rows.value.length,
    getScrollElement: () => scrollEl.value,
    estimateSize: (i: number) => (rows.value[i]?.height ?? rowHeight.value) + GAP,
    overscan: 5,
  })),
);

const totalBytes = computed(() => gallery.items.reduce((sum, i) => sum + (i.size_bytes ?? 0), 0));

const selectedIndex = computed(() => gallery.items.findIndex((i) => i.filename === selected.value));
const selectedItem = computed<GalleryImage | null>(
  () => gallery.items[selectedIndex.value] ?? null,
);

const isVideo = (i: GalleryImage) =>
  i.format === "mp4" || i.filename.endsWith(".mp4") || !!i.metadata.video_frames;

async function copyImage(item: GalleryImage) {
  try {
    await copyImageBytesToClipboard(galleryMediaPath(item.filename, gallery.source));
    toasts.push("Image copied");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

function moveSelection(delta: number) {
  if (gallery.items.length === 0) return;
  const next = Math.min(
    gallery.items.length - 1,
    Math.max(0, (selectedIndex.value === -1 ? 0 : selectedIndex.value) + delta),
  );
  selected.value = gallery.items[next]!.filename;
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
  const item = selectedItem.value;
  if (!item) return;
  const index = selectedIndex.value;
  await gallery.remove(item.filename);
  if (gallery.items.length === 0) {
    lightboxOpen.value = false;
    selected.value = null;
  } else {
    selected.value = gallery.items[Math.min(index, gallery.items.length - 1)]!.filename;
  }
}

async function switchSource(source: GallerySource) {
  if (source === gallery.source) return;
  selected.value = null;
  lightboxOpen.value = false;
  await gallery.fetch(source);
}

watch(
  () => [conn.ready, conn.mode] as const,
  ([ready, mode]) => {
    if (!ready) return;
    void gallery.fetch(mode === "remote" ? gallery.source : "engine");
  },
  { immediate: true },
);

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
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
        {{ gallery.items.length }} prints · {{ formatBytes(totalBytes) }}
      </span>
      <div
        v-if="conn.mode === 'remote'"
        class="border-edge ml-2 flex rounded-control border bg-bath p-0.5"
        role="tablist"
        aria-label="Gallery location"
      >
        <button
          v-for="option in [
            ['engine', 'Remote'],
            ['local', 'This Mac'],
          ] as const"
          :key="option[0]"
          type="button"
          role="tab"
          class="rounded-control px-2.5 py-1 text-caption transition-colors"
          :class="
            gallery.source === option[0]
              ? 'bg-bench font-medium text-ink shadow-sm'
              : 'text-ink-3 hover:text-ink'
          "
          :aria-selected="gallery.source === option[0]"
          @click="switchSource(option[0])"
        >
          {{ option[1] }}
        </button>
      </div>
      <span v-if="gallery.error" class="ml-auto text-caption text-stop">{{ gallery.error }}</span>
    </header>

    <div ref="scrollEl" class="min-h-0 flex-1 overflow-y-auto" style="contain: strict">
      <EmptyState
        v-if="gallery.loaded && gallery.items.length === 0"
        headline="No prints here yet"
        :detail="
          gallery.source === 'local'
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
            :key="laid.item.filename"
            type="button"
            class="group relative shrink-0 overflow-hidden rounded-media border transition-shadow duration-100"
            :class="
              selected === laid.item.filename
                ? 'border-transparent ring-2 ring-safelight'
                : 'border-[color-mix(in_srgb,var(--rebate)_14%,transparent)]'
            "
            :style="{ width: `${laid.width}px`, height: `${laid.height}px` }"
            @click="selected = laid.item.filename"
            @contextmenu="
              selected = laid.item.filename;
              contextMenu.open($event, tileMenu(laid.item));
            "
            @dblclick="
              selected = laid.item.filename;
              lightboxOpen = true;
            "
          >
            <AuthedMedia
              :path="galleryMediaPath(laid.item.filename, gallery.source, true)"
              :video="gallery.source === 'local' && isVideo(laid.item)"
              :alt="laid.item.metadata.prompt"
            />
            <span
              v-if="isVideo(laid.item)"
              class="absolute top-1.5 right-1.5 rounded-control bg-black/55 px-1 text-caption text-rebate"
            >
              ▶
            </span>
            <span
              class="edge-code absolute right-0 bottom-0 left-0 translate-y-full bg-black/60 px-1.5 py-0.5 text-left transition-transform duration-100 group-hover:translate-y-0"
            >
              {{ laid.item.metadata.model }} · S {{ laid.item.metadata.seed }}
            </span>
          </button>
        </div>
      </div>
    </div>

    <Lightbox
      v-if="lightboxOpen && selectedItem"
      :item="selectedItem"
      :index="selectedIndex"
      :count="gallery.items.length"
      :video="isVideo(selectedItem)"
      :source="gallery.source"
      :can-reveal="canReveal"
      @close="lightboxOpen = false"
      @prev="moveSelection(-1)"
      @next="moveSelection(1)"
      @delete="removeSelected"
    />
  </div>
</template>
