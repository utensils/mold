<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { GalleryImage } from "../lib/api/types";
import { blobToBase64 } from "../lib/image";
import {
  evictMedia,
  galleryMediaPath,
  isAudioItem,
  isVideoItem,
  streamableMediaUrl,
} from "../lib/gallery/media";
import { isUpscaledImage } from "../lib/gallery/upscaled";
import {
  DEFAULT_VIDEO_EXPORT_CAPABILITIES,
  saveVideoExport,
  videoExportFilename,
  videoExportPath,
  type VideoExportCapabilities,
  type VideoExportOptions,
} from "@studio/lib/videoExport";

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
    position?: number;
    total?: number;
    hasPrevious?: boolean | null;
    hasNext?: boolean | null;
    canUseAsSource?: boolean;
    usingSource?: boolean;
    mediaUrlOverride?: string;
    exportEnabled?: boolean;
  }>(),
  {
    reusing: false,
    reuseError: "",
    generationAnnouncement: "",
    position: 1,
    total: 1,
    hasPrevious: null,
    hasNext: null,
    canUseAsSource: false,
    usingSource: false,
    mediaUrlOverride: "",
    exportEnabled: true,
  },
);

const emit = defineEmits<{
  close: [];
  reuse: [];
  previous: [];
  next: [];
  "use-source": [];
}>();

const dialog = ref<HTMLDialogElement | null>(null);
const closeButton = ref<HTMLButtonElement | null>(null);
const mediaUrl = ref("");
const loading = ref(true);
const loadError = ref("");
const mediaLoadKey = ref(0);
const video = computed(() => isVideoItem(props.item));
const audio = computed(() => isAudioItem(props.item));
const canExportVideo = computed(
  () => props.exportEnabled && video.value && props.item.filename.toLowerCase().endsWith(".mp4"),
);
const canSaveVideo = computed(() => props.exportEnabled && video.value);
const pipeline = computed(() => (video.value ? (props.item.metadata.pipeline ?? null) : null));
const canReuse = computed(
  () => !props.item.metadata_synthetic && !!props.item.metadata.prompt?.trim(),
);
const canUseSource = computed(() => props.canUseAsSource && !video.value && !audio.value);
const actionLabel = computed(() =>
  props.reusing ? "Loading prompt…" : canReuse.value ? "Use as prompt" : "Prompt unavailable",
);
const galleryTotal = computed(() => Math.max(1, Math.floor(props.total)));
const galleryPosition = computed(() =>
  Math.min(galleryTotal.value, Math.max(1, Math.floor(props.position))),
);
const hasPrevious = computed(
  () => !props.reusing && (props.hasPrevious ?? galleryPosition.value > 1),
);
const hasNext = computed(
  () => !props.reusing && (props.hasNext ?? galleryPosition.value < galleryTotal.value),
);
const showNavigation = computed(() => galleryTotal.value > 1);
const preparedPosition = computed(() => {
  const index = props.item.metadata.batch_index;
  const count = props.item.metadata.batch_count;
  return props.item.metadata.batch_id && index && count && index >= 1 && index <= count
    ? `Batch ${index} of ${count}`
    : "";
});
const originalPrompt = computed(() => props.item.metadata.original_prompt?.trim() ?? "");
const upscaled = computed(() => isUpscaledImage(props.item));
const actionStatus = ref("");
const actionBusy = ref<"copy" | "save" | "save-video" | null>(null);
const exportOpen = ref(false);
const exportBusy = ref(false);
const exportError = ref("");
const exportCapabilities = ref<VideoExportCapabilities>(DEFAULT_VIDEO_EXPORT_CAPABILITIES);

let restoreFocusElement: HTMLElement | null = null;
let loadEpoch = 0;
let mounted = false;
let activeMedia: MediaLoad | null = null;
let gesturePointerId: number | null = null;
let gestureStartX = 0;
let gestureStartY = 0;

const SWIPE_DISTANCE = 48;
const HORIZONTAL_INTENT_RATIO = 1.25;

interface MediaLoad {
  path: string;
  target: ApiTarget;
  cacheKey: string;
  allowLegacyBlob: boolean;
}

function currentMediaLoad(): MediaLoad {
  return {
    path: galleryMediaPath(props.item.filename, "host"),
    target: { baseUrl: props.target.baseUrl, apiKey: props.target.apiKey },
    cacheKey: props.cacheKey,
    // Audio is small and not Range-streamed, so the legacy blob path is a
    // fine fallback for it; only video must refuse to buffer whole files.
    allowLegacyBlob: !isVideoItem(props.item),
  };
}

function evictLoad(load: MediaLoad | null): void {
  if (load) evictMedia(load.path, load.cacheKey);
}

async function loadMedia(load = currentMediaLoad()): Promise<void> {
  const epoch = ++loadEpoch;
  activeMedia = load;
  loading.value = true;
  loadError.value = "";
  mediaUrl.value = "";
  mediaLoadKey.value += 1;
  if (props.mediaUrlOverride) {
    mediaUrl.value = props.mediaUrlOverride;
    loading.value = false;
    return;
  }
  try {
    const url = await streamableMediaUrl(load.path, {
      target: load.target,
      cacheKey: load.cacheKey,
      allowLegacyBlob: load.allowLegacyBlob,
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

function reloadMedia(): void {
  evictLoad(activeMedia);
  void loadMedia(currentMediaLoad());
}

function retry(): void {
  evictLoad(activeMedia);
  void loadMedia(currentMediaLoad());
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

function navigate(direction: "previous" | "next"): void {
  if (direction === "previous") {
    if (hasPrevious.value) emit("previous");
  } else if (hasNext.value) {
    emit("next");
  }
}

function isMediaControl(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    !!target.closest("video, button, input, textarea, select, a, [contenteditable='true']")
  );
}

function beginSwipe(event: PointerEvent): void {
  if (
    props.reusing ||
    isMediaControl(event.target) ||
    event.isPrimary === false ||
    (event.pointerType === "mouse" && event.button !== 0)
  ) {
    return;
  }

  gesturePointerId = event.pointerId;
  gestureStartX = event.clientX;
  gestureStartY = event.clientY;
  if (event.currentTarget instanceof Element && "setPointerCapture" in event.currentTarget) {
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // WebKit can reject pointer capture when the gesture has already ended.
    }
  }
}

function trackSwipe(event: PointerEvent): void {
  if (gesturePointerId !== event.pointerId) return;
  const deltaX = event.clientX - gestureStartX;
  const deltaY = event.clientY - gestureStartY;
  if (Math.abs(deltaX) > 12 && Math.abs(deltaX) > Math.abs(deltaY)) event.preventDefault();
}

function finishSwipe(event: PointerEvent): void {
  if (gesturePointerId !== event.pointerId) return;
  const deltaX = event.clientX - gestureStartX;
  const deltaY = event.clientY - gestureStartY;
  gesturePointerId = null;

  if (
    Math.abs(deltaX) < SWIPE_DISTANCE ||
    Math.abs(deltaX) < Math.abs(deltaY) * HORIZONTAL_INTENT_RATIO
  ) {
    return;
  }
  navigate(deltaX < 0 ? "next" : "previous");
}

function cancelSwipe(event?: PointerEvent): void {
  if (!event || gesturePointerId === event.pointerId) gesturePointerId = null;
}

function handleViewerKeydown(event: KeyboardEvent): void {
  if (isMediaControl(event.target)) return;
  if (event.key === "ArrowLeft" && hasPrevious.value) {
    event.preventDefault();
    navigate("previous");
  } else if (event.key === "ArrowRight" && hasNext.value) {
    event.preventDefault();
    navigate("next");
  }
}

watch(
  () => [
    props.item.filename,
    props.item.format,
    !!props.item.metadata.video_frames,
    props.target.baseUrl,
    props.target.apiKey,
    props.cacheKey,
    props.mediaUrlOverride,
  ],
  () => {
    if (mounted) reloadMedia();
  },
);

async function fullImageBase64(): Promise<string> {
  const response = props.mediaUrlOverride
    ? await fetch(props.mediaUrlOverride)
    : await apiFetchTo(props.target, galleryMediaPath(props.item.filename, "host"));
  return blobToBase64(await response.blob());
}

async function performImageAction(action: "copy" | "save"): Promise<void> {
  if (video.value || actionBusy.value) return;
  actionBusy.value = action;
  actionStatus.value = "";
  try {
    await invoke(action === "copy" ? "copy_image_to_clipboard" : "save_image_to_photos", {
      dataB64: await fullImageBase64(),
    });
    actionStatus.value = action === "copy" ? "Image copied" : "Sent to Photos";
  } catch (error) {
    actionStatus.value = error instanceof Error ? error.message : `Couldn’t ${action} this image.`;
  } finally {
    actionBusy.value = null;
  }
}

async function performVideoSave(): Promise<void> {
  if (!canSaveVideo.value || actionBusy.value) return;
  actionBusy.value = "save-video";
  actionStatus.value = "";
  try {
    const url = await streamableMediaUrl(galleryMediaPath(props.item.filename, "host"), {
      target: props.target,
      cacheKey: props.cacheKey,
      allowLegacyBlob: false,
    });
    await invoke("save_video_to_photos", { url });
    actionStatus.value = "Saved to Photos";
  } catch (error) {
    actionStatus.value = error instanceof Error ? error.message : "Couldn’t save this video.";
  } finally {
    actionBusy.value = null;
  }
}

async function openVideoExport(): Promise<void> {
  exportOpen.value = true;
  exportError.value = "";
  try {
    exportCapabilities.value = await apiJsonTo<VideoExportCapabilities>(
      props.target,
      "/api/gallery/export-options",
    );
  } catch (error) {
    exportCapabilities.value = DEFAULT_VIDEO_EXPORT_CAPABILITIES;
    exportError.value =
      error instanceof Error ? error.message : "Couldn’t read export options from this host.";
  }
}

async function performVideoExport(options: VideoExportOptions): Promise<void> {
  if (exportBusy.value) return;
  exportBusy.value = true;
  exportError.value = "";
  try {
    const response = await apiFetchTo(props.target, videoExportPath(props.item.filename), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(options),
    });
    const result = await saveVideoExport(
      await response.blob(),
      videoExportFilename(props.item.filename, options.format),
    );
    exportOpen.value = false;
    actionStatus.value = result === "shared" ? "Export ready to share" : "Video exported";
  } catch (error) {
    exportError.value = error instanceof Error ? error.message : String(error);
  } finally {
    exportBusy.value = false;
  }
}

onMounted(() => {
  mounted = true;
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
  mounted = false;
  loadEpoch += 1;
  if (dialog.value?.open && typeof dialog.value.close === "function") dialog.value.close();
  evictLoad(activeMedia);
  activeMedia = null;
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
    @keydown="handleViewerKeydown"
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
        <span
          v-if="showNavigation"
          class="gallery-viewer-position"
          role="status"
          aria-live="polite"
          aria-atomic="true"
          data-test="gallery-viewer-position"
        >
          {{ galleryPosition }} of {{ galleryTotal }}
        </span>
      </div>
    </header>

    <div
      class="gallery-viewer-stage"
      data-test="gallery-viewer-stage"
      @pointerdown="beginSwipe"
      @pointermove="trackSwipe"
      @pointerup="finishSwipe"
      @pointercancel="cancelSwipe"
      @lostpointercapture="cancelSwipe"
    >
      <img
        v-if="!mediaUrl"
        class="gallery-viewer-placeholder"
        :src="thumbnailUrl"
        alt=""
        aria-hidden="true"
        draggable="false"
      />
      <div v-else-if="audio" class="gallery-viewer-audio">
        <img
          class="gallery-viewer-audio-waveform"
          :src="thumbnailUrl"
          :alt="`Waveform for ${item.filename}`"
          draggable="false"
        />
        <audio
          :key="mediaLoadKey"
          class="gallery-viewer-audio-player"
          :src="mediaUrl"
          controls
          preload="metadata"
          data-test="gallery-viewer-audio"
          @loadedmetadata="mediaReady"
          @error="mediaFailed"
        />
      </div>
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
        draggable="false"
        @load="mediaReady"
        @error="mediaFailed"
      />
      <span v-if="upscaled" data-test="upscaled-badge" class="gallery-upscaled-badge">
        Upscaled
      </span>

      <div v-if="loading" class="gallery-viewer-loading" role="status">
        Loading full resolution…
      </div>
      <div v-else-if="loadError" class="gallery-viewer-error" role="alert">
        <span>{{ loadError }}</span>
        <button type="button" @click="retry">Try again</button>
      </div>

      <template v-if="showNavigation">
        <button
          class="gallery-viewer-nav gallery-viewer-nav-previous"
          type="button"
          aria-label="Previous print"
          data-test="gallery-viewer-previous"
          :disabled="!hasPrevious"
          @click="navigate('previous')"
        >
          <span aria-hidden="true">‹</span>
        </button>
        <button
          class="gallery-viewer-nav gallery-viewer-nav-next"
          type="button"
          aria-label="Next print"
          data-test="gallery-viewer-next"
          :disabled="!hasNext"
          @click="navigate('next')"
        >
          <span aria-hidden="true">›</span>
        </button>
      </template>
    </div>

    <footer class="gallery-viewer-details">
      <div class="gallery-viewer-prompt">
        <span v-if="preparedPosition" data-test="gallery-viewer-batch">{{ preparedPosition }}</span>
        <span>Prompt</span>
        <p data-selectable>{{ item.metadata.prompt || "No prompt saved with this print." }}</p>
        <p v-if="pipeline" data-test="gallery-viewer-pipeline" data-selectable>
          <span>Pipeline</span> {{ pipeline }}
        </p>
        <template v-if="originalPrompt">
          <span>Source prompt</span>
          <p data-test="gallery-viewer-original-prompt" data-selectable>{{ originalPrompt }}</p>
        </template>
        <p v-if="reuseError" class="gallery-viewer-reuse-error" role="alert">
          {{ reuseError }}
        </p>
      </div>
      <div class="gallery-viewer-actions">
        <button
          v-if="canSaveVideo"
          class="secondary-button gallery-viewer-save"
          type="button"
          data-test="gallery-viewer-save-video"
          :disabled="!!actionBusy || exportBusy"
          @click="performVideoSave"
        >
          {{ actionBusy === "save-video" ? "Saving…" : "Save video" }}
        </button>
        <button
          v-if="canExportVideo"
          class="secondary-button gallery-viewer-export"
          type="button"
          data-test="gallery-viewer-export"
          @click="openVideoExport"
        >
          Export format…
        </button>
        <template v-if="!video && !audio">
          <button
            class="secondary-button gallery-viewer-copy"
            type="button"
            data-test="gallery-viewer-copy"
            :disabled="!!actionBusy"
            @click="performImageAction('copy')"
          >
            {{ actionBusy === "copy" ? "Copying…" : "Copy image" }}
          </button>
          <button
            class="secondary-button gallery-viewer-save"
            type="button"
            data-test="gallery-viewer-save"
            :disabled="!!actionBusy"
            @click="performImageAction('save')"
          >
            {{ actionBusy === "save" ? "Saving…" : "Save photo" }}
          </button>
        </template>
        <button
          v-if="canUseSource"
          class="secondary-button gallery-viewer-source"
          type="button"
          data-test="gallery-viewer-use-source"
          :disabled="usingSource || reusing"
          :aria-busy="usingSource"
          @click="emit('use-source')"
        >
          {{ usingSource ? "Loading source…" : "Use as source" }}
        </button>
        <button
          class="primary-button gallery-viewer-reuse"
          type="button"
          data-test="gallery-viewer-reuse"
          :disabled="!canReuse || reusing || usingSource"
          :aria-busy="reusing"
          @click="emit('reuse')"
        >
          {{ actionLabel }}
        </button>
      </div>
      <p
        v-if="actionStatus"
        class="gallery-viewer-action-status"
        role="status"
        data-test="gallery-viewer-action-status"
      >
        {{ actionStatus }}
      </p>
    </footer>
    <VideoExportDialog
      :open="exportOpen"
      :filename="item.filename"
      :formats="exportCapabilities.formats"
      :busy="exportBusy"
      :error="exportError"
      @close="exportOpen = false"
      @export="performVideoExport"
    />
  </dialog>
</template>

<style scoped>
.gallery-viewer-media:not(video),
.gallery-viewer-placeholder {
  touch-action: pan-y;
  user-select: none;
  -webkit-user-drag: none;
  -webkit-touch-callout: default;
}

/* Audio has no raster to fill the stage: waveform above, transport below. */
.gallery-viewer-audio {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  width: 100%;
  padding: 1rem;
}

.gallery-viewer-audio-waveform {
  width: 100%;
  max-height: 40vh;
  object-fit: contain;
  touch-action: pan-y;
  user-select: none;
  -webkit-user-drag: none;
}

.gallery-viewer-audio-player {
  width: 100%;
  min-height: 44px;
}

.gallery-upscaled-badge {
  position: absolute;
  z-index: 2;
  top: 12px;
  right: 12px;
}

.gallery-viewer-position {
  margin-top: 2px;
  color: rgba(245, 239, 255, 0.84) !important;
}

.gallery-viewer-nav {
  position: absolute;
  z-index: 2;
  top: 50%;
  display: grid;
  width: 48px;
  height: 56px;
  place-items: center;
  transform: translateY(-50%);
  border: 1px solid rgba(245, 239, 255, 0.24);
  border-radius: 999px;
  background: rgba(8, 7, 12, 0.68);
  color: #f5efff;
  font-size: 38px;
  line-height: 1;
  -webkit-tap-highlight-color: transparent;
  backdrop-filter: blur(8px);
}

.gallery-viewer-nav:disabled {
  opacity: 0.28;
}

.gallery-viewer-nav:not(:disabled):active {
  background: rgba(245, 239, 255, 0.2);
}

.gallery-viewer-nav-previous {
  left: max(10px, env(safe-area-inset-left));
}

.gallery-viewer-nav-next {
  right: max(10px, env(safe-area-inset-right));
}

@media (hover: hover) and (pointer: fine) {
  .gallery-viewer-nav:not(:disabled):hover {
    background: rgba(245, 239, 255, 0.16);
  }
}
</style>
