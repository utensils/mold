<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useVirtualizer } from "@tanstack/vue-virtual";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import Lightbox from "../components/gallery/Lightbox.vue";
import EmptyState from "../components/shell/EmptyState.vue";
import { layoutJustifiedRows } from "../lib/gallery/layout";
import { thumbnailPath } from "../lib/gallery/media";
import { formatBytes } from "../lib/format";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useUiStore } from "../stores/ui";
import { useComposerStore } from "../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useToastStore } from "../stores/toasts";
import { ipc } from "../lib/ipc";
import type { GalleryImage } from "../lib/api/types";

const GAP = 8;
const PAD = 16;
// Zoom levels for the justified row target height (⌘0 / ⌘+ / ⌘-).
const ROW_HEIGHTS = [120, 180, 240];

const router = useRouter();
const conn = useConnectionStore();
const gallery = useGalleryStore();
const ui = useUiStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const toasts = useToastStore();

// Reveal in Finder only exists when the engine writes to this disk.
const canReveal = ref(false);
void ipc.getOutputDir().then((dir) => (canReveal.value = dir !== null));

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
    { separator: true },
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

// ⌘0 reset · ⌘+ zoom in (taller rows) · ⌘- zoom out, clamped to ROW_HEIGHTS.
watch(
  () => ui.galleryZoomTick,
  () => {
    if (ui.galleryZoomDir === "reset") {
      rowHeight.value = 180;
      return;
    }
    const i = ROW_HEIGHTS.indexOf(rowHeight.value);
    const base = i === -1 ? 1 : i;
    const next = ui.galleryZoomDir === "in" ? base + 1 : base - 1;
    rowHeight.value = ROW_HEIGHTS[Math.min(ROW_HEIGHTS.length - 1, Math.max(0, next))]!;
  },
);

const totalBytes = computed(() => gallery.items.reduce((sum, i) => sum + (i.size_bytes ?? 0), 0));

const selectedIndex = computed(() => gallery.items.findIndex((i) => i.filename === selected.value));
const selectedItem = computed<GalleryImage | null>(
  () => gallery.items[selectedIndex.value] ?? null,
);

const isVideo = (i: GalleryImage) =>
  i.format === "mp4" || i.filename.endsWith(".mp4") || !!i.metadata.video_frames;

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

watch(
  () => conn.ready,
  (ready) => {
    if (ready) void gallery.fetch();
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
  <EmptyState
    v-if="gallery.loaded && gallery.items.length === 0"
    headline="No prints yet"
    detail="Generate one and it lands here."
    action="Go to Generate"
    @action="router.push('/generate')"
  />

  <div v-else class="flex h-full flex-col">
    <header class="border-edge flex h-11 items-center gap-3 border-b px-4">
      <span class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Gallery
      </span>
      <span class="data-mono text-caption text-ink-3">
        {{ gallery.items.length }} prints · {{ formatBytes(totalBytes) }}
      </span>
      <span v-if="gallery.error" class="ml-auto text-caption text-stop">{{ gallery.error }}</span>
    </header>

    <div ref="scrollEl" class="min-h-0 flex-1 overflow-y-auto" style="contain: strict">
      <div :style="{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }">
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
              :path="thumbnailPath(laid.item.filename)"
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
      @close="lightboxOpen = false"
      @prev="moveSelection(-1)"
      @next="moveSelection(1)"
      @delete="removeSelected"
    />
  </div>
</template>
