<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { ApiTarget } from "../lib/api/client";
import type { GalleryImage } from "../lib/api/types";
import {
  evictMedia,
  galleryMediaPath,
  isVideoItem,
  streamableMediaUrl,
} from "../lib/gallery/media";

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    target: ApiTarget;
    cacheKey: string;
    hostName: string;
    thumbnailUrl: string;
    reusing?: boolean;
    reuseError?: string;
    generationAnnouncement?: string;
  }>(),
  { reusing: false, reuseError: "", generationAnnouncement: "" },
);

const emit = defineEmits<{ close: []; reuse: [] }>();

const dialog = ref<HTMLDialogElement | null>(null);
const closeButton = ref<HTMLButtonElement | null>(null);
const mediaUrl = ref("");
const loading = ref(true);
const loadError = ref("");
const mediaLoadKey = ref(0);
const video = computed(() => isVideoItem(props.item));
const mediaPath = computed(() => galleryMediaPath(props.item.filename, "host"));
const canReuse = computed(
  () => !props.item.metadata_synthetic && !!props.item.metadata.prompt?.trim(),
);
const actionLabel = computed(() =>
  props.reusing ? "Loading prompt…" : canReuse.value ? "Use as prompt" : "Prompt unavailable",
);

let restoreFocusElement: HTMLElement | null = null;
let loadEpoch = 0;

async function loadMedia(): Promise<void> {
  const epoch = ++loadEpoch;
  loading.value = true;
  loadError.value = "";
  mediaUrl.value = "";
  mediaLoadKey.value += 1;
  try {
    const url = await streamableMediaUrl(mediaPath.value, {
      target: props.target,
      cacheKey: props.cacheKey,
      allowLegacyBlob: !video.value,
    });
    if (epoch !== loadEpoch) return;
    mediaUrl.value = url;
  } catch (error) {
    if (epoch !== loadEpoch) return;
    loadError.value =
      error instanceof Error && error.message.startsWith("Update this Mold host")
        ? error.message
        : "Couldn’t load the full print from this host.";
  } finally {
    if (epoch === loadEpoch && loadError.value) loading.value = false;
  }
}

function retry(): void {
  evictMedia(mediaPath.value, props.cacheKey);
  void loadMedia();
}

function mediaReady(): void {
  loading.value = false;
}

function mediaFailed(): void {
  loading.value = false;
  loadError.value = video.value
    ? "Couldn’t play this video from the host."
    : "Couldn’t load the full print from this host.";
}

function cancelViewer(): void {
  if (!props.reusing) emit("close");
}

onMounted(() => {
  restoreFocusElement = document.activeElement as HTMLElement | null;
  try {
    dialog.value?.showModal();
  } catch {
    dialog.value?.setAttribute("open", "");
  }
  closeButton.value?.focus();
  void loadMedia();
});

onBeforeUnmount(() => {
  loadEpoch += 1;
  if (dialog.value?.open && typeof dialog.value.close === "function") dialog.value.close();
  evictMedia(mediaPath.value, props.cacheKey);
  restoreFocusElement?.focus?.();
});
</script>

<template>
  <dialog
    ref="dialog"
    class="gallery-viewer"
    role="dialog"
    aria-modal="true"
    aria-labelledby="gallery-viewer-title"
    data-test="gallery-viewer"
    @cancel.prevent="cancelViewer"
  >
    <p class="sr-only" aria-live="polite" aria-atomic="true">
      {{ generationAnnouncement }}
    </p>
    <header class="gallery-viewer-header">
      <button
        ref="closeButton"
        class="gallery-viewer-close"
        type="button"
        aria-label="Close print"
        data-test="gallery-viewer-close"
        :disabled="reusing"
        @click="emit('close')"
      >
        <span aria-hidden="true">×</span>
        <span>Close</span>
      </button>
      <div class="gallery-viewer-origin">
        <h1 id="gallery-viewer-title">Print preview</h1>
        <span>{{ hostName }}</span>
      </div>
    </header>

    <div class="gallery-viewer-stage">
      <img
        v-if="!mediaUrl"
        class="gallery-viewer-placeholder"
        :src="thumbnailUrl"
        alt=""
        aria-hidden="true"
      />
      <video
        v-else-if="video"
        :key="mediaLoadKey"
        class="gallery-viewer-media"
        :src="mediaUrl"
        :poster="thumbnailUrl"
        controls
        playsinline
        preload="metadata"
        data-test="gallery-viewer-video"
        @loadedmetadata="mediaReady"
        @error="mediaFailed"
      />
      <img
        v-else
        :key="mediaLoadKey"
        class="gallery-viewer-media"
        :src="mediaUrl"
        :alt="item.metadata.prompt || item.filename"
        data-test="gallery-viewer-image"
        @load="mediaReady"
        @error="mediaFailed"
      />

      <div v-if="loading" class="gallery-viewer-loading" role="status">
        Loading full resolution…
      </div>
      <div v-else-if="loadError" class="gallery-viewer-error" role="alert">
        <span>{{ loadError }}</span>
        <button type="button" @click="retry">Try again</button>
      </div>
    </div>

    <footer class="gallery-viewer-details">
      <div class="gallery-viewer-prompt">
        <span>Prompt</span>
        <p data-selectable>{{ item.metadata.prompt || "No prompt saved with this print." }}</p>
        <p v-if="reuseError" class="gallery-viewer-reuse-error" role="alert">
          {{ reuseError }}
        </p>
      </div>
      <button
        class="primary-button gallery-viewer-reuse"
        type="button"
        data-test="gallery-viewer-reuse"
        :disabled="!canReuse || reusing"
        :aria-busy="reusing"
        @click="emit('reuse')"
      >
        {{ actionLabel }}
      </button>
    </footer>
  </dialog>
</template>
