<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { GalleryImage } from "../lib/api/types";
import { blobToBase64 } from "../lib/image";
import { formatBytes, formatScheduler } from "../lib/format";
import { modelDisplayNameForId } from "../lib/models";
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
  downloadVideoExport,
  videoExportFilename,
  videoExportPath,
  type VideoExportCapabilities,
  type VideoExportOptions,
} from "@studio/lib/videoExport";
import { isNativeAndroidRuntime, isNativeIOSRuntime } from "./platform";
import {
  collectionSlug,
  displayTitle,
  normalizeTagName,
  purgeCountdownFromPurgeAt,
  tagKey,
  validatePrintTitle,
  type OrganizationUnion,
} from "@studio/lib/libraryOrganization";
import type { TagCount } from "@studio/lib/api/galleryOrganization";
import { mobileIdentityProvenanceRows } from "./identity";
import { strengthSemanticsForModel } from "@studio/lib/strengthSemantics";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";
import {
  validateCollectionName,
  type MobileCollectionCard,
  type MobileGalleryImage,
} from "./libraryOrganization";

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
    /** Merged title / favorite / tags / collections across every copy. */
    organization?: OrganizationUnion | null;
    /** A connected host advertises `capabilities.gallery.organize`. */
    organizeEnabled?: boolean;
    /** The print is in the trash (info sheet swaps to Restore / Delete). */
    trashed?: boolean;
    /** An organization mutation is in flight. */
    organizing?: boolean;
    organizationError?: string;
    tagSuggestions?: TagCount[];
    collections?: MobileCollectionCard[];
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
    organization: null,
    organizeEnabled: false,
    trashed: false,
    organizing: false,
    organizationError: "",
    tagSuggestions: () => [],
    collections: () => [],
  },
);

const emit = defineEmits<{
  close: [];
  reuse: [];
  previous: [];
  next: [];
  "use-source": [];
  rename: [title: string | null];
  favorite: [favorite: boolean];
  tags: [change: { add?: string[]; remove?: string[] }];
  collection: [change: { slug: string; name: string; member: boolean }];
  restore: [];
  "delete-forever": [];
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
const canReuse = computed(() => !props.item.metadata_synthetic);
const canUseSource = computed(() => props.canUseAsSource && !video.value && !audio.value);
const actionLabel = computed(() =>
  props.reusing ? "Loading settings…" : canReuse.value ? "Reuse settings" : "Settings unavailable",
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
const originalPrompt = computed(() =>
  props.item.metadata.prompt?.trim() ? (props.item.metadata.original_prompt?.trim() ?? "") : "",
);
const upscaled = computed(() => isUpscaledImage(props.item));
const actionStatus = ref("");
const promptCopyStatus = ref("");
const actionBusy = ref<"copy" | "save" | "save-video" | null>(null);
const exportOpen = ref(false);
const exportBusy = ref(false);
const exportError = ref("");
const exportCapabilities = ref<VideoExportCapabilities>(DEFAULT_VIDEO_EXPORT_CAPABILITIES);

// ── Print info sheet (title / favorite / tags / collections / trash) ────────
const infoOpen = ref(false);
const titleDraft = ref("");
const titleError = ref("");
const infoTagDraft = ref("");
const infoCollectionDraft = ref("");
const infoCollectionError = ref("");
const deleteForeverArmed = ref(false);
/**
 * Face-identity provenance (#1224): names and digests only — saved metadata
 * never carries the photo itself, which is exactly why the digest is shown.
 */
const identityRows = computed(() => mobileIdentityProvenanceRows(props.item.metadata));
// Identity provenance is worth reading on a host that has no organization
// Metadata is useful on every print, even when its host cannot organize.
const infoAvailable = computed(() => true);
const savedTitle = computed(() => props.organization?.title ?? null);
/** Header line: the print's title, else its prompt, else the filename. */
const viewerTitle = computed(() =>
  displayTitle({
    title: savedTitle.value,
    metadata: props.item.metadata,
    filename: props.item.filename,
  }),
);
const favorite = computed(() => props.organization?.favorite ?? false);
const infoTags = computed(() => props.organization?.tags ?? []);
const infoTagSuggestions = computed(() => {
  const present = new Set(infoTags.value.map(tagKey));
  return props.tagSuggestions.filter((tag) => !present.has(tagKey(tag.name))).slice(0, 12);
});
const memberCollectionSlugs = computed(() => new Set(props.organization?.collections ?? []));
const purgeCopy = computed(() => {
  if (!props.trashed) return "";
  const purgeAt =
    props.organization?.purgeAt ?? (props.item as MobileGalleryImage).purge_at ?? null;
  return purgeCountdownFromPurgeAt(purgeAt, Date.now()).label;
});
const modelLabel = computed(() => modelDisplayNameForId(props.item.metadata.model, []));
const schedulerName = computed(() => formatScheduler(props.item.metadata.scheduler));
const strengthCaption = computed(
  () => strengthSemanticsForModel(props.item.metadata.model, null).label,
);
const frames = computed(
  () => props.item.metadata.frames ?? props.item.metadata.video_frames ?? null,
);
const fps = computed(() => props.item.metadata.fps ?? props.item.metadata.video_fps ?? null);
const fileFormat = computed(() => props.item.format ?? props.item.metadata.output_format ?? null);
const fileSize = computed(() =>
  props.item.size_bytes != null ? formatBytes(props.item.size_bytes) : null,
);
const createdAt = computed(() =>
  new Date(props.item.timestamp * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }),
);
const loraStack = computed(() => {
  const metadata = props.item.metadata;
  if (metadata.loras?.length) return metadata.loras;
  return metadata.lora ? [{ path: metadata.lora, scale: metadata.lora_scale ?? 1 }] : [];
});

function openInfo(): void {
  titleDraft.value = savedTitle.value ?? "";
  titleError.value = "";
  infoTagDraft.value = "";
  infoCollectionDraft.value = "";
  infoCollectionError.value = "";
  deleteForeverArmed.value = false;
  promptCopyStatus.value = "";
  infoOpen.value = true;
}

function closeInfo(): void {
  promptCopyStatus.value = "";
  infoOpen.value = false;
}

async function copyPrompt(): Promise<void> {
  promptCopyStatus.value = "";
  try {
    await navigator.clipboard.writeText(props.item.metadata.prompt);
    promptCopyStatus.value = "Prompt copied";
  } catch {
    promptCopyStatus.value = "Couldn’t copy prompt.";
  }
}

/** Done commits the title through PATCH; blank clears it. */
function commitTitle(): void {
  const result = validatePrintTitle(titleDraft.value);
  if (!result.ok) {
    titleError.value = result.reason;
    return;
  }
  titleError.value = "";
  if ((result.value ?? null) !== (savedTitle.value ?? null)) emit("rename", result.value);
}

function addInfoTag(raw: string): void {
  const name = normalizeTagName(raw);
  infoTagDraft.value = "";
  if (!name || infoTags.value.some((tag) => tagKey(tag) === tagKey(name))) return;
  emit("tags", { add: [name] });
}

function toggleInfoCollection(card: MobileCollectionCard): void {
  emit("collection", {
    slug: card.slug,
    name: card.name,
    member: !memberCollectionSlugs.value.has(card.slug),
  });
}

function createInfoCollection(): void {
  const validation = validateCollectionName(infoCollectionDraft.value);
  if (!validation.ok) {
    infoCollectionError.value = validation.reason ?? "";
    return;
  }
  infoCollectionError.value = "";
  infoCollectionDraft.value = "";
  emit("collection", {
    slug: collectionSlug(validation.value),
    name: validation.value,
    member: true,
  });
}

/** Two-step: first tap arms, the second deletes on every host. */
function deleteForever(): void {
  if (!deleteForeverArmed.value) {
    deleteForeverArmed.value = true;
    return;
  }
  deleteForeverArmed.value = false;
  emit("delete-forever");
}

let restoreFocusElement: HTMLElement | null = null;
let loadEpoch = 0;
let mounted = false;
let activeMedia: MediaLoad | null = null;
let gesturePointerId: number | null = null;
let gestureStartX = 0;
let gestureStartY = 0;

const SWIPE_DISTANCE = 48;
const HORIZONTAL_INTENT_RATIO = 1.25;
const VIDEO_CONTROL_STRIP_HEIGHT = 64;

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
  emit("close");
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

function isSwipeBlockingControl(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    !!target.closest("button, input, textarea, select, a, [contenteditable='true']")
  );
}

function isVideoControlStrip(event: PointerEvent): boolean {
  if (!(event.target instanceof Element)) return false;
  const videoElement = event.target.closest("video");
  if (!videoElement) return false;
  const bounds = videoElement.getBoundingClientRect();
  return bounds.height > 0 && event.clientY >= bounds.bottom - VIDEO_CONTROL_STRIP_HEIGHT;
}

function beginSwipe(event: PointerEvent): void {
  if (
    props.reusing ||
    isSwipeBlockingControl(event.target) ||
    isVideoControlStrip(event) ||
    event.isPrimary === false ||
    (event.pointerType === "mouse" && event.button !== 0)
  ) {
    return;
  }

  gesturePointerId = event.pointerId;
  gestureStartX = event.clientX;
  gestureStartY = event.clientY;
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
  [
    () => props.item.filename,
    () => props.item.format,
    () => !!props.item.metadata.video_frames,
    () => props.target.baseUrl,
    () => props.target.apiKey,
    () => props.cacheKey,
    () => props.mediaUrlOverride,
  ],
  () => {
    if (mounted) reloadMedia();
    promptCopyStatus.value = "";
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
    const path = videoExportPath(props.item.filename);
    const filename = videoExportFilename(props.item.filename, options.format);
    const native = isNativeIOSRuntime() || isNativeAndroidRuntime();
    if (native) {
      const outcome = await invoke<"shared" | "cancelled">("share_exported_animation", {
        url: `${props.target.baseUrl}${path}`,
        apiKey: props.target.apiKey,
        request: options,
        filename,
        reuseKey: `${props.target.baseUrl}\n${props.item.filename}\n${JSON.stringify(options)}`,
      });
      if (outcome === "cancelled") return;
    } else {
      const response = await apiFetchTo(props.target, path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(options),
      });
      const blob = await response.blob();
      downloadVideoExport(blob, filename);
    }
    exportOpen.value = false;
    actionStatus.value = native ? "Export ready to share" : "Video exported";
  } catch (error) {
    exportError.value = error instanceof Error ? error.message : String(error);
  } finally {
    exportBusy.value = false;
  }
}

onMounted(() => {
  mounted = true;
  // WKWebView's native video layer can stop the target/bubble phase once a
  // playback control recognizes the touch. Keep the active swipe at the
  // window capture boundary so image -> video -> image navigation cannot be
  // stranded by a media element, while the excluded control strip still gets
  // ordinary taps and scrubbing gestures.
  window.addEventListener("pointermove", trackSwipe, { capture: true, passive: false });
  window.addEventListener("pointerup", finishSwipe, true);
  window.addEventListener("pointercancel", cancelSwipe, true);
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
  window.removeEventListener("pointermove", trackSwipe, true);
  window.removeEventListener("pointerup", finishSwipe, true);
  window.removeEventListener("pointercancel", cancelSwipe, true);
  cancelSwipe();
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
        @click="emit('close')"
      >
        <span aria-hidden="true">×</span>
        <span>Close</span>
      </button>
      <div class="gallery-viewer-origin">
        <h1 id="gallery-viewer-title" data-test="gallery-viewer-title">{{ viewerTitle }}</h1>
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
      @pointerdown.capture="beginSwipe"
      @pointermove="trackSwipe"
      @pointerup="finishSwipe"
      @pointercancel="cancelSwipe"
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
        <div class="gallery-viewer-prompt-heading">
          <span>Prompt</span>
          <button
            v-if="item.metadata.prompt"
            class="gallery-viewer-copy-prompt"
            type="button"
            data-test="gallery-viewer-copy-prompt"
            aria-label="Copy prompt"
            @click="copyPrompt"
          >
            Copy
          </button>
        </div>
        <p data-selectable>{{ item.metadata.prompt || "No prompt was used for this print." }}</p>
        <p
          v-if="promptCopyStatus && !infoOpen"
          class="gallery-viewer-copy-status"
          role="status"
          data-test="gallery-viewer-copy-status"
        >
          {{ promptCopyStatus }}
        </p>
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
          v-if="infoAvailable"
          class="secondary-button gallery-viewer-info"
          type="button"
          data-test="gallery-viewer-info"
          @click="openInfo"
        >
          Info
        </button>
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
    <MobileLibrarySheet
      :open="infoOpen"
      :title="viewerTitle"
      :focus-first-control="false"
      test-id="gallery-viewer-info-sheet"
      @close="closeInfo"
    >
      <p
        v-if="organizationError"
        class="status-line error-text"
        role="alert"
        data-test="gallery-viewer-info-error"
      >
        {{ organizationError }}
      </p>
      <template v-if="organizeEnabled && !trashed">
        <form class="mobile-library-sheet-form" @submit.prevent="commitTitle">
          <label class="field">
            <span>Title</span>
            <input
              v-model="titleDraft"
              class="control"
              autocomplete="off"
              enterkeyhint="done"
              placeholder="Untitled print"
              data-test="gallery-viewer-title-input"
              @blur="commitTitle"
            />
          </label>
          <button
            class="secondary-button"
            type="submit"
            :disabled="organizing"
            data-test="gallery-viewer-title-save"
          >
            Save
          </button>
        </form>
        <p v-if="titleError" class="status-line error-text" role="alert">{{ titleError }}</p>
        <button
          class="secondary-button gallery-viewer-favorite"
          type="button"
          :aria-pressed="favorite"
          :disabled="organizing"
          data-test="gallery-viewer-favorite"
          @click="emit('favorite', !favorite)"
        >
          <span aria-hidden="true">{{ favorite ? "♥" : "♡" }}</span>
          {{ favorite ? "Favorited" : "Favorite" }}
        </button>
        <p class="mobile-library-sheet-label">Tags</p>
        <div class="mobile-library-tag-list" data-test="gallery-viewer-tags">
          <span v-if="infoTags.length === 0" class="mobile-empty-note">No tags yet.</span>
          <span v-for="tag in infoTags" :key="tag" class="mobile-library-tag">
            <span>{{ tag }}</span>
            <button
              type="button"
              :aria-label="`Remove tag ${tag}`"
              :disabled="organizing"
              data-test="gallery-viewer-tag-remove"
              @click="emit('tags', { remove: [tag] })"
            >
              ×
            </button>
          </span>
        </div>
        <form class="mobile-library-sheet-form" @submit.prevent="addInfoTag(infoTagDraft)">
          <label class="field">
            <span>Add a tag</span>
            <input
              v-model="infoTagDraft"
              class="control"
              autocomplete="off"
              autocapitalize="off"
              enterkeyhint="done"
              placeholder="smurf"
              data-test="gallery-viewer-tag-input"
            />
          </label>
          <button
            class="secondary-button"
            type="submit"
            :disabled="organizing || !infoTagDraft.trim()"
            data-test="gallery-viewer-tag-add"
          >
            Add
          </button>
        </form>
        <div
          v-if="infoTagSuggestions.length"
          class="mobile-library-tag-list"
          data-test="gallery-viewer-tag-suggestions"
        >
          <button
            v-for="tag in infoTagSuggestions"
            :key="`suggest-${tag.name}`"
            class="mobile-library-chip"
            type="button"
            :disabled="organizing"
            @click="addInfoTag(tag.name)"
          >
            {{ tag.name }}<span class="mobile-library-chip-count">{{ tag.count }}</span>
          </button>
        </div>
        <p class="mobile-library-sheet-label">In collections</p>
        <ul class="mobile-library-checklist" data-test="gallery-viewer-collections">
          <li v-for="card in collections" :key="card.slug">
            <button
              type="button"
              role="checkbox"
              :aria-checked="memberCollectionSlugs.has(card.slug)"
              :disabled="organizing"
              data-test="gallery-viewer-collection-option"
              @click="toggleInfoCollection(card)"
            >
              <span class="mobile-library-check" aria-hidden="true">{{
                memberCollectionSlugs.has(card.slug) ? "✓" : ""
              }}</span>
              <span class="mobile-collection-copy">
                <strong>{{ card.name }}</strong>
                <span
                  ><span class="mobile-collection-count">{{ card.count }}</span>
                  <template v-if="card.hostsLabel"> · {{ card.hostsLabel }}</template></span
                >
              </span>
            </button>
          </li>
          <li v-if="collections.length === 0" class="mobile-empty-note">
            No collections yet — name one below.
          </li>
        </ul>
        <form class="mobile-library-sheet-form" @submit.prevent="createInfoCollection">
          <label class="field">
            <span>New collection</span>
            <input
              v-model="infoCollectionDraft"
              class="control"
              autocomplete="off"
              enterkeyhint="done"
              placeholder="Collection name"
              data-test="gallery-viewer-collection-input"
            />
          </label>
          <button
            class="secondary-button"
            type="submit"
            :disabled="organizing || !infoCollectionDraft.trim()"
            data-test="gallery-viewer-collection-create"
          >
            New
          </button>
        </form>
        <p v-if="infoCollectionError" class="status-line error-text" role="alert">
          {{ infoCollectionError }}
        </p>
      </template>
      <template v-if="identityRows">
        <p class="mobile-library-sheet-label">Identity</p>
        <dl class="gallery-viewer-identity" data-test="gallery-viewer-identity">
          <template v-for="row in identityRows" :key="row.label">
            <dt>{{ row.label }}</dt>
            <dd :title="row.title">{{ row.value }}</dd>
          </template>
        </dl>
      </template>
      <section class="gallery-viewer-print-details" data-test="gallery-viewer-print-details">
        <p class="mobile-library-sheet-label">Print details</p>
        <p class="gallery-viewer-info-filename" :title="item.filename">{{ item.filename }}</p>
        <div class="gallery-viewer-info-prompt-row">
          <p
            class="gallery-viewer-info-prompt"
            data-selectable
            data-test="gallery-viewer-info-prompt"
            :title="item.metadata.prompt"
          >
            {{ item.metadata.prompt }}
          </p>
          <button
            v-if="item.metadata.prompt"
            class="secondary-button gallery-viewer-copy-prompt"
            type="button"
            data-test="gallery-viewer-info-copy-prompt"
            aria-label="Copy prompt"
            @click="copyPrompt"
          >
            Copy
          </button>
        </div>
        <p
          v-if="promptCopyStatus"
          class="gallery-viewer-copy-status"
          role="status"
          data-test="gallery-viewer-info-copy-status"
        >
          {{ promptCopyStatus }}
        </p>
        <p
          v-if="item.metadata.prompt.trim() && item.metadata.original_prompt"
          class="gallery-viewer-info-secondary"
          data-test="gallery-viewer-original"
        >
          <span>Original</span> {{ item.metadata.original_prompt }}
        </p>
        <p
          v-if="item.metadata.negative_prompt"
          class="gallery-viewer-info-secondary"
          data-test="gallery-viewer-negative"
        >
          <span>Negative</span> {{ item.metadata.negative_prompt }}
        </p>
        <p
          v-if="item.metadata.batch_id && item.metadata.batch_index && item.metadata.batch_count"
          class="gallery-viewer-info-secondary"
          data-test="gallery-viewer-batch"
          :title="item.metadata.batch_id"
        >
          <span>Prepared batch</span> {{ item.metadata.batch_index }} of
          {{ item.metadata.batch_count }} · {{ item.metadata.batch_id }}
        </p>
        <dl class="gallery-viewer-info-facts">
          <div>
            <dt>Model</dt>
            <dd>{{ modelLabel }}</dd>
          </div>
          <div>
            <dt>Seed</dt>
            <dd>{{ item.metadata.seed }}</dd>
          </div>
          <div>
            <dt>Dimensions</dt>
            <dd>{{ item.metadata.width }}×{{ item.metadata.height }}</dd>
          </div>
          <div>
            <dt>Steps · guidance</dt>
            <dd>{{ item.metadata.steps }} · {{ item.metadata.guidance.toFixed(1) }}</dd>
          </div>
          <div v-if="schedulerName">
            <dt>Scheduler</dt>
            <dd>{{ schedulerName }}</dd>
          </div>
          <div v-if="item.metadata.cfg_plus">
            <dt>CFG++</dt>
            <dd>on</dd>
          </div>
          <div v-if="item.metadata.strength != null">
            <dt>{{ strengthCaption }}</dt>
            <dd>{{ item.metadata.strength.toFixed(2) }}</dd>
          </div>
          <div v-if="frames">
            <dt>Frames</dt>
            <dd>
              {{ frames }}<template v-if="fps"> · {{ fps }} fps</template>
            </dd>
          </div>
          <div v-if="pipeline">
            <dt>Pipeline</dt>
            <dd>{{ pipeline }}</dd>
          </div>
          <div v-for="lora in loraStack" :key="lora.path">
            <dt>LoRA</dt>
            <dd :title="lora.path">{{ lora.path }} × {{ lora.scale.toFixed(2) }}</dd>
          </div>
          <div v-if="fileSize">
            <dt>File size</dt>
            <dd>{{ fileSize }}</dd>
          </div>
          <div v-if="fileFormat">
            <dt>Format</dt>
            <dd>{{ fileFormat.toUpperCase() }}</dd>
          </div>
          <div>
            <dt>Created</dt>
            <dd>{{ createdAt }}</dd>
          </div>
          <div>
            <dt>Host</dt>
            <dd>{{ hostName }}</dd>
          </div>
        </dl>
        <p v-if="item.metadata.version" class="gallery-viewer-info-version">
          mold {{ item.metadata.version }}
        </p>
        <span v-if="item.metadata_synthetic" class="edge-code">SYNTHETIC METADATA</span>
      </section>
      <template v-if="trashed">
        <p class="status-line" data-test="gallery-viewer-purge">{{ purgeCopy }}</p>
        <button
          class="primary-button"
          type="button"
          :disabled="organizing"
          data-test="gallery-viewer-restore"
          @click="emit('restore')"
        >
          Restore
        </button>
        <p v-if="deleteForeverArmed" class="status-line" data-test="gallery-viewer-delete-prompt">
          Delete this print forever?
        </p>
        <button
          class="danger-button"
          type="button"
          :disabled="organizing"
          data-test="gallery-viewer-delete-forever"
          @click="deleteForever"
        >
          {{ deleteForeverArmed ? "Confirm" : "Delete forever" }}
        </button>
      </template>
    </MobileLibrarySheet>
  </dialog>
</template>

<style scoped>
.gallery-viewer-identity {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 4px 12px;
  margin: 0;
}

.gallery-viewer-identity dt {
  color: var(--ink-3);
  font-size: 13px;
}

.gallery-viewer-identity dd {
  margin: 0;
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 13px;
  overflow-wrap: anywhere;
}

.gallery-viewer-print-details {
  display: grid;
  gap: 10px;
  min-width: 0;
}

.gallery-viewer-info-filename,
.gallery-viewer-info-prompt,
.gallery-viewer-info-secondary,
.gallery-viewer-info-version {
  margin: 0;
}

.gallery-viewer-info-filename,
.gallery-viewer-info-version {
  overflow: hidden;
  color: var(--ink-3);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gallery-viewer-info-prompt {
  min-width: 0;
  color: var(--rebate);
  font-size: var(--text-body);
  overflow-wrap: anywhere;
}

.gallery-viewer-info-prompt-row,
.gallery-viewer-prompt-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.gallery-viewer-copy-prompt {
  min-width: 44px;
  min-height: 44px;
  flex: 0 0 auto;
}

.gallery-viewer-prompt-heading .gallery-viewer-copy-prompt {
  border: 0;
  background: transparent;
  color: var(--safelight);
  font: inherit;
}

.gallery-viewer-prompt-heading > span {
  color: rgba(245, 239, 255, 0.62);
  font-family: var(--font-utility);
  font-size: var(--text-edge-code);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.gallery-viewer-copy-status {
  display: block;
  overflow: visible;
  margin: 0;
  color: var(--ink-2);
  font-size: 13px;
  -webkit-line-clamp: unset;
}

.gallery-viewer-info-secondary {
  color: var(--ink-2);
  font-size: 13px;
  overflow-wrap: anywhere;
}

.gallery-viewer-info-secondary span {
  color: var(--ink-3);
}

.gallery-viewer-info-facts {
  display: grid;
  gap: 8px;
  margin: 2px 0 0;
}

.gallery-viewer-info-facts > div {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.7fr);
  gap: 12px;
}

.gallery-viewer-info-facts dt {
  color: var(--ink-3);
  font-size: 13px;
}

.gallery-viewer-info-facts dd {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: var(--rebate);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 13px;
  overflow-wrap: anywhere;
  text-align: right;
}

.gallery-viewer-media,
.gallery-viewer-placeholder {
  touch-action: pan-y;
}

.gallery-viewer-media:not(video),
.gallery-viewer-placeholder {
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
