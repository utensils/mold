<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import { galleryMediaPath } from "../../lib/gallery/media";
import { readGalleryMediaBase64 } from "../../lib/gallery/sourceMedia";
import { fileToBase64, isStillImageFile, isStillImageGalleryItem } from "../../lib/image";
import { inTauri, ipc } from "../../lib/ipc";
import type { PickedImage } from "../../lib/generateForm";
import { useGalleryStore, type MergedPrint } from "../../stores/gallery";
import { virtualGridWindow } from "@studio/lib/virtualGrid";
import type { ThumbnailPriority } from "@studio/lib/thumbnailScheduler";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title?: string;
    multiple?: boolean;
    /** Wells own drop + file picking themselves; their gallery link opens
     * this picker straight on the gallery with no redundant upload tab. */
    galleryOnly?: boolean;
  }>(),
  {
    title: "Source image",
    multiple: false,
    galleryOnly: false,
  },
);
const emit = defineEmits<{
  (e: "pick", v: PickedImage[]): void;
  (e: "close"): void;
}>();

const tab = ref<"upload" | "gallery">(props.galleryOnly ? "gallery" : "upload");
const error = ref<string | null>(null);
const dragOver = ref(false);
// Focus the close button on open and restore focus to the opener on close,
// so the dialog is keyboard-operable and doesn't strand focus (matches Lightbox).
const closeBtn = ref<HTMLButtonElement | null>(null);
const fallbackFileInput = ref<HTMLInputElement | null>(null);
const selectedGallery = ref<MergedPrint[]>([]);
const pickingGallery = ref(false);
const galleryScrollEl = ref<HTMLElement | null>(null);
let restoreFocusEl: HTMLElement | null = null;
let galleryResizeObserver: ResizeObserver | null = null;

// The gallery tab is the same unified multi-host view as the Gallery's All
// section: every connected host's bucket, merged and deduped by the store.
const gallery = useGalleryStore();

// Refetch on EVERY open (not once per mount — the modal stays mounted in its
// panel): a host that connected since the last open gets its first bucket
// fetch here, and existing buckets pick up prints from other clients. A
// previous session's pick/upload error is stale by now — clear it.
function loadGallery() {
  error.value = null;
  selectedGallery.value = [];
  if (props.galleryOnly) tab.value = "gallery";
  void gallery.fetchAll();
}

// Only PNG/JPEG are valid as source_image / mask / keyframe conditioning;
// hide video and animated outputs so a pick can't fail at generation time.
const entries = computed<MergedPrint[]>(() =>
  gallery.merged.filter((entry) => isStillImageGalleryItem(entry.item)),
);
const loading = computed(() => entries.value.length === 0 && !gallery.loaded);
/** Bucket fetch failures, shown only when there is nothing to render —
 *  partial multi-host data beats an error banner. */
const galleryError = computed(() =>
  entries.value.length === 0 && !loading.value ? gallery.firstError : null,
);
/** Per-tile origin labels only matter with more than one gallery source. */
const showHostLabels = computed(() => gallery.sources.length > 1);

// The full Library virtualizes its justified rows. This picker uses a simpler
// square grid, but it must keep the same bounded-DOM contract: mounting every
// AuthedMedia at once queues every thumbnail, decodes off-screen images, and
// makes opening/scrolling increasingly expensive as the Library grows.
const PICKER_GRID_GAP = 8;
const PICKER_GRID_FALLBACK_WIDTH = 720;
const PICKER_GRID_FALLBACK_HEIGHT = 600;
const galleryViewport = ref({
  width: PICKER_GRID_FALLBACK_WIDTH,
  height: PICKER_GRID_FALLBACK_HEIGHT,
  scrollTop: 0,
});
const focusedGalleryIndex = ref(0);

function syncGalleryViewport() {
  const element = galleryScrollEl.value;
  if (!element) return;
  galleryViewport.value = {
    width: element.clientWidth || PICKER_GRID_FALLBACK_WIDTH,
    height: element.clientHeight || PICKER_GRID_FALLBACK_HEIGHT,
    scrollTop: element.scrollTop,
  };
}

function bindGalleryViewport() {
  galleryResizeObserver?.disconnect();
  galleryResizeObserver = null;
  void nextTick(() => {
    const element = galleryScrollEl.value;
    if (!element) return;
    syncGalleryViewport();
    if (typeof ResizeObserver !== "undefined") {
      galleryResizeObserver = new ResizeObserver(syncGalleryViewport);
      galleryResizeObserver.observe(element);
    }
  });
}

const pickerLayout = computed(() =>
  virtualGridWindow({
    itemCount: entries.value.length,
    containerWidth: galleryViewport.value.width,
    minimumItemWidth: 120,
    minimumColumns: 3,
    maximumColumns: 5,
    gap: PICKER_GRID_GAP,
    viewportStart: 0,
    viewportSize: 0,
    overscanRows: 0,
  }),
);
const clampedGalleryScrollTop = computed(() =>
  Math.min(
    galleryViewport.value.scrollTop,
    Math.max(0, pickerLayout.value.totalSize - galleryViewport.value.height),
  ),
);
const pickerGrid = computed(() =>
  virtualGridWindow({
    itemCount: entries.value.length,
    containerWidth: galleryViewport.value.width,
    minimumItemWidth: 120,
    minimumColumns: 3,
    maximumColumns: 5,
    gap: PICKER_GRID_GAP,
    viewportStart: clampedGalleryScrollTop.value,
    viewportSize: galleryViewport.value.height,
    overscanRows: 2,
  }),
);
const pickerVisibleGrid = computed(() =>
  virtualGridWindow({
    itemCount: entries.value.length,
    containerWidth: galleryViewport.value.width,
    minimumItemWidth: 120,
    minimumColumns: 3,
    maximumColumns: 5,
    gap: PICKER_GRID_GAP,
    viewportStart: clampedGalleryScrollTop.value,
    viewportSize: galleryViewport.value.height,
    overscanRows: 0,
  }),
);
const galleryTabStopIndex = computed(() => {
  const window = pickerGrid.value;
  return focusedGalleryIndex.value >= window.startIndex &&
    focusedGalleryIndex.value < window.endIndex
    ? focusedGalleryIndex.value
    : window.startIndex;
});

interface PickerTile {
  entry: MergedPrint;
  index: number;
  x: number;
  y: number;
  priority: ThumbnailPriority;
  mediaVersion: string;
}

const visibleEntries = computed<PickerTile[]>(() => {
  const window = pickerGrid.value;
  const visible = pickerVisibleGrid.value;
  return entries.value.slice(window.startIndex, window.endIndex).map((entry, offset) => {
    const index = window.startIndex + offset;
    const row = Math.floor(index / window.columns);
    return {
      entry,
      index,
      x: (index % window.columns) * window.rowStep,
      y: row * window.rowStep,
      priority:
        row >= visible.startRow && row < visible.endRow ? ("visible" as const) : ("near" as const),
      mediaVersion:
        entry.item.media_version ?? `${entry.item.timestamp}:${entry.item.size_bytes ?? "unknown"}`,
    };
  });
});

async function focusGalleryIndex(index: number) {
  if (entries.value.length === 0) return;
  const target = Math.max(0, Math.min(entries.value.length - 1, index));
  focusedGalleryIndex.value = target;

  const element = galleryScrollEl.value;
  if (element) {
    const layout = pickerLayout.value;
    const row = Math.floor(target / layout.columns);
    const top = row * layout.rowStep;
    const bottom = top + layout.itemSize;
    const viewportTop = clampedGalleryScrollTop.value;
    const viewportBottom = viewportTop + galleryViewport.value.height;
    if (top < viewportTop) element.scrollTop = top;
    else if (bottom > viewportBottom) {
      element.scrollTop = Math.max(0, bottom - galleryViewport.value.height);
    }
    syncGalleryViewport();
  }

  await nextTick();
  document.querySelector<HTMLButtonElement>(`[data-picker-index="${target}"]`)?.focus();
}

function onGalleryItemKeydown(event: KeyboardEvent, index: number) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    const entry = entries.value[index];
    if (entry) void pickFromGallery(entry);
    return;
  }

  const columns = pickerLayout.value.columns;
  const pageRows = Math.max(
    1,
    Math.floor(galleryViewport.value.height / pickerLayout.value.rowStep),
  );
  const moves: Partial<Record<string, number>> = {
    ArrowLeft: index - 1,
    ArrowRight: index + 1,
    ArrowUp: index - columns,
    ArrowDown: index + columns,
    PageUp: index - pageRows * columns,
    PageDown: index + pageRows * columns,
    Home: 0,
    End: entries.value.length - 1,
  };
  const target = moves[event.key];
  if (target === undefined) return;
  event.preventDefault();
  void focusGalleryIndex(target);
}

watch(
  [
    () => entries.value.length,
    () => pickerLayout.value.totalSize,
    () => galleryViewport.value.height,
  ],
  () => {
    focusedGalleryIndex.value = Math.min(
      focusedGalleryIndex.value,
      Math.max(0, entries.value.length - 1),
    );
    void nextTick(() => {
      const element = galleryScrollEl.value;
      if (!element) return;
      const clamped = clampedGalleryScrollTop.value;
      if (element.scrollTop !== clamped) element.scrollTop = clamped;
      if (galleryViewport.value.scrollTop !== clamped) syncGalleryViewport();
    });
  },
);

// Escape must close the dialog wherever focus sits: WKWebView (Tauri on
// macOS) does not focus clicked buttons, so a keydown handler scoped to the
// overlay never fires after the user interacts with the page. Listen on the
// document while mounted and gate on `open`.
function onDocumentKeydown(event: KeyboardEvent) {
  if (props.open && event.key === "Escape") emit("close");
}
onMounted(() => {
  document.addEventListener("keydown", onDocumentKeydown);
  if (props.open) void loadGallery();
});
onBeforeUnmount(() => {
  document.removeEventListener("keydown", onDocumentKeydown);
  galleryResizeObserver?.disconnect();
});
watch(
  () => props.open,
  (open) => {
    if (open) {
      void loadGallery();
      restoreFocusEl = document.activeElement as HTMLElement | null;
      void nextTick(() => closeBtn.value?.focus());
    } else {
      restoreFocusEl?.focus?.();
      restoreFocusEl = null;
    }
  },
);
watch([() => props.open, tab], ([open, activeTab]) => {
  if (open && activeTab === "gallery") bindGalleryViewport();
});

async function ingestFiles(files: File[]) {
  // Same constraint as the gallery tab: the engine only accepts PNG/JPEG for
  // source_image / mask / keyframes — filter by MIME with a filename fallback.
  const images = files.filter(
    (f) =>
      f.type === "image/png" || f.type === "image/jpeg" || (!f.type && isStillImageFile(f.name)),
  );
  if (files.length && !images.length) {
    error.value = "Only PNG or JPEG images can be used here.";
    return;
  }
  error.value = null;
  const selected = props.multiple ? images : images.slice(0, 1);
  if (!selected.length) return;
  const picked = await Promise.all(
    selected.map(async (file) => ({ filename: file.name, base64: await fileToBase64(file) })),
  );
  emit("pick", picked);
  emit("close");
}

function onFiles(event: Event) {
  void ingestFiles(Array.from((event.target as HTMLInputElement).files ?? []));
}
function onDrop(event: DragEvent) {
  dragOver.value = false;
  void ingestFiles(Array.from(event.dataTransfer?.files ?? []));
}

async function chooseFiles() {
  try {
    const selected = await ipc.pickSourceImages(props.multiple);
    if (!selected?.length) {
      // `bun run dev` has no native dialog backend. Keep its browser-only
      // development path usable without ever invoking this input in Tauri.
      if (!inTauri()) fallbackFileInput.value?.click();
      return;
    }
    error.value = null;
    emit(
      "pick",
      selected.map(({ filename, base64 }) => ({ filename, base64 })),
    );
    emit("close");
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function pickFromGallery(entry: MergedPrint) {
  if (props.multiple) {
    const key = `${entry.sourceKey}:${entry.item.filename}`;
    const index = selectedGallery.value.findIndex(
      (selected) => `${selected.sourceKey}:${selected.item.filename}` === key,
    );
    selectedGallery.value =
      index >= 0
        ? selectedGallery.value.filter((_, item) => item !== index)
        : [...selectedGallery.value, entry];
    return;
  }
  await emitGallerySelection([entry]);
}

function galleryEntrySelected(entry: MergedPrint): boolean {
  const key = `${entry.sourceKey}:${entry.item.filename}`;
  return selectedGallery.value.some(
    (selected) => `${selected.sourceKey}:${selected.item.filename}` === key,
  );
}

async function confirmGallerySelection() {
  await emitGallerySelection(selectedGallery.value);
}

async function emitGallerySelection(entries: readonly MergedPrint[]) {
  if (!entries.length || pickingGallery.value) return;
  pickingGallery.value = true;
  error.value = null;
  try {
    const picked = await Promise.all(
      entries.map(async (entry) => ({
        filename: entry.item.filename,
        base64: await readGalleryMediaBase64(entry, gallery),
      })),
    );
    emit("pick", picked);
    emit("close");
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    pickingGallery.value = false;
  }
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"
      @click.self="emit('close')"
      @keydown.esc="emit('close')"
    >
      <div
        class="border-edge flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-chrome border bg-bench p-5"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
      >
        <div class="flex items-center justify-between">
          <h2 class="text-body-lg font-semibold text-ink">{{ title }}</h2>
          <button
            ref="closeBtn"
            type="button"
            class="text-ink-3 hover:text-ink"
            aria-label="Close"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div v-if="!galleryOnly" class="mt-4 flex gap-2">
          <button
            type="button"
            class="rounded-control px-3 py-1 text-body transition-colors duration-100"
            :class="
              tab === 'upload'
                ? 'bg-safelight font-semibold text-on-accent'
                : 'border-edge border text-ink-2 hover:text-ink'
            "
            data-test="picker-tab-upload"
            @click="tab = 'upload'"
          >
            Choose file
          </button>
          <button
            type="button"
            class="rounded-control px-3 py-1 text-body transition-colors duration-100"
            :class="
              tab === 'gallery'
                ? 'bg-safelight font-semibold text-on-accent'
                : 'border-edge border text-ink-2 hover:text-ink'
            "
            data-test="picker-tab-gallery"
            @click="tab = 'gallery'"
          >
            From gallery
          </button>
        </div>

        <div v-if="tab === 'upload' && !galleryOnly" class="mt-4 flex-1 overflow-y-auto">
          <button
            type="button"
            class="flex h-48 w-full cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors"
            :class="dragOver ? 'border-safelight text-safelight' : 'border-control-edge text-ink-3'"
            data-test="picker-native-file-button"
            @dragover.prevent="dragOver = true"
            @dragleave="dragOver = false"
            @drop.prevent="onDrop"
            @click="chooseFiles"
          >
            <span
              class="text-caption text-ink-2 underline decoration-dotted underline-offset-4 hover:text-ink"
            >
              Drop a PNG or JPEG here, or choose a file
            </span>
          </button>
          <input
            ref="fallbackFileInput"
            type="file"
            accept="image/png,image/jpeg"
            :multiple="multiple"
            class="hidden"
            data-test="picker-browser-file-input"
            @change="onFiles"
          />
          <p v-if="error" class="mt-2 text-caption text-stop" data-test="picker-upload-error">
            {{ error }}
          </p>
        </div>

        <div
          v-else
          ref="galleryScrollEl"
          class="mt-4 flex-1 overflow-y-auto"
          data-test="picker-gallery-scroll"
          @scroll.passive="syncGalleryViewport"
        >
          <p v-if="loading" class="text-caption text-ink-3">Loading…</p>
          <p v-else-if="error" class="text-caption text-stop">{{ error }}</p>
          <p
            v-else-if="galleryError"
            class="text-caption text-stop"
            data-test="picker-gallery-error"
          >
            {{ galleryError }}
          </p>
          <p v-else-if="entries.length === 0" class="text-caption text-ink-3">
            No prints in the gallery yet.
          </p>
          <ul
            v-else
            class="relative"
            :style="{ height: `${pickerGrid.totalSize}px` }"
            data-test="picker-gallery-grid"
          >
            <li
              v-for="tile in visibleEntries"
              :key="`${tile.entry.sourceKey}:${tile.entry.item.filename}`"
              class="absolute top-0 left-0"
              :style="{
                width: `${pickerGrid.itemSize}px`,
                height: `${pickerGrid.itemSize}px`,
                transform: `translate(${tile.x}px, ${tile.y}px)`,
              }"
            >
              <button
                type="button"
                class="border-edge relative aspect-square w-full overflow-hidden rounded-media border transition hover:brightness-110"
                :class="
                  galleryEntrySelected(tile.entry)
                    ? 'ring-2 ring-safelight ring-offset-2 ring-offset-bench'
                    : ''
                "
                data-test="picker-gallery-item"
                :data-picker-index="tile.index"
                :aria-label="tile.entry.item.filename"
                :disabled="pickingGallery"
                :tabindex="tile.index === galleryTabStopIndex ? 0 : -1"
                :aria-pressed="multiple ? galleryEntrySelected(tile.entry) : undefined"
                @click="pickFromGallery(tile.entry)"
                @focus="focusedGalleryIndex = tile.index"
                @keydown="onGalleryItemKeydown($event, tile.index)"
              >
                <AuthedMedia
                  :path="
                    galleryMediaPath(
                      tile.entry.item.filename,
                      gallery.mediaSourceOf(tile.entry.sourceKey),
                      true,
                    )
                  "
                  :target="gallery.targetOf(tile.entry.sourceKey)"
                  :cache-key="tile.entry.sourceKey"
                  :media-version="tile.mediaVersion"
                  :priority="tile.priority"
                  :alt="tile.entry.item.filename"
                />
                <span
                  v-if="showHostLabels"
                  class="edge-code absolute bottom-1 left-1 rounded-control bg-black/60 px-1 !text-on-media"
                  data-test="picker-item-host"
                >
                  {{ tile.entry.hostLabel }}
                </span>
                <span
                  v-if="multiple && galleryEntrySelected(tile.entry)"
                  class="absolute top-1 right-1 grid h-6 w-6 place-items-center rounded-full bg-safelight text-caption font-bold text-on-accent"
                  aria-hidden="true"
                >
                  {{
                    selectedGallery.findIndex(
                      (selected) =>
                        selected.sourceKey === tile.entry.sourceKey &&
                        selected.item.filename === tile.entry.item.filename,
                    ) + 1
                  }}
                </span>
              </button>
            </li>
          </ul>
          <div
            v-if="multiple && selectedGallery.length"
            class="border-edge sticky bottom-0 mt-3 flex items-center justify-between gap-3 border-t bg-bench pt-3"
            data-test="picker-gallery-selection"
          >
            <span class="text-caption text-ink-2">
              {{ selectedGallery.length }} selected · kept in this order
            </span>
            <button
              type="button"
              class="rounded-control bg-safelight px-3 py-2 text-body font-semibold text-on-accent"
              data-test="picker-gallery-confirm"
              :disabled="pickingGallery"
              @click="confirmGallerySelection"
            >
              Add selected
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
