<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import { apiFetch, apiJson } from "../../lib/api/client";
import { mediaPath, thumbnailPath } from "../../lib/gallery/media";
import { blobToBase64, fileToBase64, isStillImageFile } from "../../lib/image";
import type { PickedImage } from "../../lib/generateForm";
import type { GalleryImage } from "../../lib/api/types";

const props = withDefaults(defineProps<{ open: boolean; title?: string; multiple?: boolean }>(), {
  title: "Source image",
  multiple: false,
});
const emit = defineEmits<{
  (e: "pick", v: PickedImage[]): void;
  (e: "close"): void;
}>();

const tab = ref<"upload" | "gallery">("upload");
const entries = ref<GalleryImage[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const dragOver = ref(false);
const fetched = ref(false);
// Focus the close button on open and restore focus to the opener on close,
// so the dialog is keyboard-operable and doesn't strand focus (matches Lightbox).
const closeBtn = ref<HTMLButtonElement | null>(null);
let restoreFocusEl: HTMLElement | null = null;

async function loadGallery() {
  if (fetched.value) return;
  fetched.value = true;
  loading.value = true;
  error.value = null;
  try {
    const items = await apiJson<GalleryImage[]>("/api/gallery");
    // Only PNG/JPEG are valid as source_image / mask / keyframe conditioning;
    // hide video and animated outputs so a pick can't fail at generation time.
    entries.value = items
      .filter((item) => isStillImageFile(item.filename))
      .sort((a, b) => b.timestamp - a.timestamp);
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  if (props.open) void loadGallery();
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

async function pickFromGallery(item: GalleryImage) {
  try {
    const res = await apiFetch(mediaPath(item.filename));
    const blob = await res.blob();
    emit("pick", [{ filename: item.filename, base64: await blobToBase64(blob) }]);
    emit("close");
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
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

        <div class="mt-4 flex gap-2">
          <button
            type="button"
            class="rounded-control px-3 py-1 text-body transition-colors duration-100"
            :class="
              tab === 'upload'
                ? 'bg-safelight font-semibold text-[#141110]'
                : 'border-edge border text-ink-2 hover:text-ink'
            "
            data-test="picker-tab-upload"
            @click="tab = 'upload'"
          >
            Upload
          </button>
          <button
            type="button"
            class="rounded-control px-3 py-1 text-body transition-colors duration-100"
            :class="
              tab === 'gallery'
                ? 'bg-safelight font-semibold text-[#141110]'
                : 'border-edge border text-ink-2 hover:text-ink'
            "
            data-test="picker-tab-gallery"
            @click="tab = 'gallery'"
          >
            From gallery
          </button>
        </div>

        <div v-if="tab === 'upload'" class="mt-4 flex-1 overflow-y-auto">
          <label
            class="flex h-48 w-full cursor-pointer items-center justify-center rounded-media border border-dashed text-caption transition-colors"
            :class="
              dragOver
                ? 'border-safelight text-safelight'
                : 'border-[color-mix(in_srgb,var(--rebate)_25%,transparent)] text-ink-3'
            "
            @dragover.prevent="dragOver = true"
            @dragleave="dragOver = false"
            @drop.prevent="onDrop"
          >
            <span>Drop a PNG or JPEG here or click to browse</span>
            <input
              type="file"
              accept="image/png,image/jpeg"
              :multiple="multiple"
              class="hidden"
              data-test="picker-upload-input"
              @change="onFiles"
            />
          </label>
          <p v-if="error" class="mt-2 text-caption text-stop" data-test="picker-upload-error">
            {{ error }}
          </p>
        </div>

        <div v-else class="mt-4 flex-1 overflow-y-auto">
          <p v-if="loading" class="text-caption text-ink-3">Loading…</p>
          <p v-else-if="error" class="text-caption text-stop">{{ error }}</p>
          <p v-else-if="entries.length === 0" class="text-caption text-ink-3">
            No prints in the gallery yet.
          </p>
          <ul v-else class="grid grid-cols-3 gap-2 sm:grid-cols-5">
            <li v-for="item in entries" :key="item.filename">
              <button
                type="button"
                class="border-edge aspect-square w-full overflow-hidden rounded-media border transition hover:brightness-110"
                data-test="picker-gallery-item"
                :aria-label="item.filename"
                @click="pickFromGallery(item)"
              >
                <AuthedMedia :path="thumbnailPath(item.filename)" :alt="item.filename" />
              </button>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </Teleport>
</template>
