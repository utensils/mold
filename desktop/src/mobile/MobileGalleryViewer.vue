<script setup lang="ts">
import MeshViewer from "@studio/components/MeshViewer.vue";
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import VideoExportDialog, { type ExportDestination } from "@ui/components/VideoExportDialog.vue";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { GalleryImage } from "../lib/api/types";
import { blobToBase64 } from "../lib/image";
import { formatBytes, formatScheduler } from "../lib/format";
import { modelDisplayNameForId } from "../lib/models";
import {
  evictMedia,
  galleryMediaPath,
  isAudioItem,
  isMeshItem,
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
  type VideoExportFormat,
  type VideoExportOptions,
} from "@studio/lib/videoExport";
import { isAnimatedMeshExportFormat, meshExportFilename } from "./meshResult";
import {
  meshExportChoices,
  resolveSheetGesture,
  viewerKindLabel,
  viewerPeekSummary,
  type ViewerMediaKind,
} from "./galleryViewerSheet";
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
    /** The owning host can accept an upscale request for this saved print. */
    upscaleEnabled?: boolean;
    /**
     * The containers the OWNING host advertises it can transcode a stored
     * mesh into (`/api/capabilities.mesh.export_formats`). The export menu is
     * built from THIS list and never from a client constant, so a machine
     * that adds a container offers it without a client release, and one that
     * has none offers nothing.
     */
    meshExportFormats?: string[];
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
    upscaleEnabled: true,
    meshExportFormats: () => [],
  },
);

const emit = defineEmits<{
  close: [];
  reuse: [];
  previous: [];
  next: [];
  "use-source": [];
  upscale: [];
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
const mesh = computed(() => isMeshItem(props.item));
const canExportVideo = computed(
  () => props.exportEnabled && video.value && props.item.filename.toLowerCase().endsWith(".mp4"),
);
const canSaveVideo = computed(() => props.exportEnabled && video.value);
/**
 * GLB is the only container mold STORES; OBJ, STL and PLY are transcodes of
 * that stored file, produced by the same `POST /api/gallery/export/:filename`
 * route a clip's animation export uses. Geometry files export on one tap; an
 * advertised turntable carries playback/size options, so it goes through the
 * sheet the phone already has.
 */
const meshExports = computed(() =>
  props.exportEnabled && mesh.value ? props.meshExportFormats : [],
);
/**
 * On a native shell every 3-D export is a PAIR: the system share sheet, and
 * "Save to Mold folder", which writes the file into an on-device folder the
 * user can browse (Files ▸ On My iPhone ▸ Mold; Download/Mold on Android).
 * The browser build has neither, so it keeps its single download.
 */
const nativeShell = computed(() => isNativeIOSRuntime() || isNativeAndroidRuntime());
// The server lists the stored container (`glb`) first so a CLI can name it.
// A phone has no Download, so on a native shell the stored GLB leaves the
// app the same two ways a transcode does — but only when the host advertises
// it. A browser fetches that file directly, so it is not an export there.
const meshGeometryExports = computed(() =>
  meshExports.value.filter(
    (format) => (format !== "glb" || nativeShell.value) && !isAnimatedMeshExportFormat(format),
  ),
);
const meshAnimationExports = computed(
  () => meshExports.value.filter(isAnimatedMeshExportFormat) as VideoExportFormat[],
);
/** The two places a turntable can go on a phone; none elsewhere. */
const MESH_SHARE_DESTINATION: ExportDestination = { value: "share", label: "Share…" };
const MESH_FOLDER_DESTINATION: ExportDestination = {
  value: "folder",
  label: "Save to Mold folder",
};
/**
 * Which button opened the turntable's options sheet. The destination is
 * already decided by then — Share… or Save to Mold folder — so the sheet
 * opens with that choice selected and the user can still change it there.
 */
const pendingMeshDestination = ref<"share" | "folder">("share");
const exportDestinations = computed<ExportDestination[]>(() => {
  if (!mesh.value || !nativeShell.value) return [];
  return pendingMeshDestination.value === "folder"
    ? [MESH_FOLDER_DESTINATION, MESH_SHARE_DESTINATION]
    : [MESH_SHARE_DESTINATION, MESH_FOLDER_DESTINATION];
});
/**
 * The export picker: every container the host advertises, and ONE Turntable
 * entry standing for the animated ones. Two verbs follow it — Share… and
 * Save to Mold folder — instead of a button per container per destination,
 * which is what pushed the media off a phone screen.
 */
const meshChoices = computed(() =>
  meshExportChoices(meshGeometryExports.value, meshAnimationExports.value),
);
const meshFormat = ref("");
watch(
  meshChoices,
  (choices) => {
    if (!choices.some((choice) => choice.value === meshFormat.value)) {
      meshFormat.value = choices[0]?.value ?? "";
    }
  },
  { immediate: true },
);
/**
 * The primary verb. A native shell shares (the shell runs the export and
 * opens the system share sheet); a browser downloads, and says which file.
 */
const meshPrimaryLabel = computed(() => {
  if (nativeShell.value) return "Share…";
  return meshFormat.value === "turntable"
    ? "Export turntable…"
    : `Export as ${meshFormat.value.toUpperCase()}`;
});
const pipeline = computed(() => (video.value ? (props.item.metadata.pipeline ?? null) : null));
const canReuse = computed(() => !props.item.metadata_synthetic);
// A mesh has no raster to stage as conditioning, whatever the owner allows.
const canUseSource = computed(
  () => props.canUseAsSource && !video.value && !audio.value && !mesh.value,
);
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

// ── Details sheet ──────────────────────────────────────────────────────────
/*
 * The media owns the whole screen; every detail and action lives in a sheet
 * that peeks one line above the bottom edge and swipes up. The body stays
 * mounted while collapsed — it is translated out of view, not removed — so
 * an action keeps its identity across the toggle and nothing has to remount.
 */
const mediaKind = computed<ViewerMediaKind>(() =>
  mesh.value ? "mesh" : audio.value ? "audio" : video.value ? "video" : "image",
);
const kindLabel = computed(() => viewerKindLabel(mediaKind.value));
/** What the staged media reported about itself, cleared with every print. */
const meshStats = ref<{ vertexCount: number; triangleCount: number } | null>(null);
const mediaDurationMs = ref<number | null>(null);
const peekSummary = computed(() =>
  viewerPeekSummary(mediaKind.value, props.item, {
    mesh: meshStats.value,
    durationMs: mediaDurationMs.value,
  }),
);
/** The handle points at the body it opens; two viewers must not collide. */
const sheetBodyId = `gallery-viewer-sheet-body-${useId()}`;
const sheetExpanded = ref(false);
const sheetDragging = ref(false);
const sheetDrag = ref(0);
const sheetRoot = ref<HTMLElement | null>(null);
const sheetHandle = ref<HTMLElement | null>(null);
const sheetBody = ref<HTMLElement | null>(null);
let sheetPointerId: number | null = null;
let sheetStartX = 0;
let sheetStartY = 0;
let sheetTravel = 0;
let sheetSuppressClick = false;
let sheetDragStartedInBody = false;
/** The collapsed peek in px, measured once the sheet has been laid out. */
let sheetPeek = 0;

function collapseSheet(): void {
  if (!sheetExpanded.value) return;
  sheetExpanded.value = false;
  // Focus was inside the body that just went out of view. The handle is what
  // replaces it on screen, so it is what replaces it in the focus order.
  sheetHandle.value?.focus?.();
}

function toggleSheet(): void {
  if (sheetExpanded.value) collapseSheet();
  else sheetExpanded.value = true;
}

/**
 * Whether the LIST should keep this drag. Only a drag that began inside the
 * scrolling body can be the list scrolling back to its top; a pull on the
 * handle is the sheet's own grip and always closes it.
 */
function sheetDragBelongsToBody(): boolean {
  return sheetDragStartedInBody && (sheetBody.value?.scrollTop ?? 0) > 0;
}

/**
 * How far the sheet can travel: its own height, less the whole collapsed
 * peek. Measuring the peek from the collapsed geometry rather than from the
 * handle alone keeps the safe-area band below it in the sum, so a drag ends
 * exactly where the sheet rests instead of a home-indicator's worth short.
 */
function measureSheetTravel(): number {
  const sheet = sheetRoot.value;
  if (!sheet) return 0;
  if (!sheetExpanded.value) {
    const showing = Math.round(window.innerHeight - sheet.getBoundingClientRect().top);
    if (showing > 0) sheetPeek = showing;
  }
  const peek = sheetPeek || (sheetHandle.value?.offsetHeight ?? 0);
  return Math.max(0, sheet.offsetHeight - peek);
}

function beginSheetDrag(event: PointerEvent): void {
  sheetSuppressClick = false;
  if (
    event.isPrimary === false ||
    (event.pointerType === "mouse" && event.button !== 0) ||
    // A text field owns its own caret drag; nothing else here does.
    (event.target instanceof Element &&
      !!event.target.closest("input, textarea, select, [contenteditable='true']"))
  ) {
    return;
  }
  sheetPointerId = event.pointerId;
  sheetStartX = event.clientX;
  sheetStartY = event.clientY;
  sheetDragStartedInBody =
    event.target instanceof Node && !!sheetBody.value?.contains(event.target);
  sheetTravel = measureSheetTravel();
  // Capture, so a finger that lifts over the media still finishes the drag
  // here instead of stranding the sheet halfway open.
  try {
    sheetRoot.value?.setPointerCapture?.(event.pointerId);
  } catch {
    // A shell without pointer capture just keeps the default targeting.
  }
}

function trackSheetDrag(event: PointerEvent): void {
  if (sheetPointerId !== event.pointerId) return;
  const deltaX = event.clientX - sheetStartX;
  const deltaY = event.clientY - sheetStartY;
  if (Math.abs(deltaY) <= Math.abs(deltaX)) {
    // Not the sheet's drag after all: hand the transition back so it slides
    // home rather than snapping there.
    sheetDrag.value = 0;
    sheetDragging.value = false;
    return;
  }
  if (sheetExpanded.value && sheetDragBelongsToBody()) return;
  sheetDragging.value = true;
  sheetDrag.value = sheetExpanded.value
    ? Math.max(0, Math.min(sheetTravel, deltaY))
    : Math.min(0, Math.max(-sheetTravel, deltaY));
  if (event.cancelable) event.preventDefault();
}

function finishSheetDrag(event: PointerEvent): void {
  if (sheetPointerId !== event.pointerId) return;
  const deltaX = event.clientX - sheetStartX;
  const deltaY = event.clientY - sheetStartY;
  resetSheetDrag();
  const gesture = resolveSheetGesture({
    deltaX,
    deltaY,
    expanded: sheetExpanded.value,
    scrolled: sheetDragBelongsToBody(),
  });
  if (gesture === "none") return;
  // A drag that ends on a control must not also fire that control.
  sheetSuppressClick = true;
  sheetExpanded.value = gesture === "expand";
}

function resetSheetDrag(event?: PointerEvent): void {
  if (event && sheetPointerId !== event.pointerId) return;
  if (sheetPointerId !== null) {
    try {
      sheetRoot.value?.releasePointerCapture?.(sheetPointerId);
    } catch {
      // Already released with the pointer; nothing to undo.
    }
  }
  sheetPointerId = null;
  sheetDrag.value = 0;
  sheetDragging.value = false;
}

/** Swallow the click a resolved drag would otherwise deliver to a button. */
function guardSheetClick(event: MouseEvent): void {
  if (!sheetSuppressClick) return;
  sheetSuppressClick = false;
  event.preventDefault();
  event.stopPropagation();
}

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
    // A mesh, like a clip, is fetched whole by its viewer and must not take
    // the legacy blob path.
    allowLegacyBlob: !isVideoItem(props.item) && !isMeshItem(props.item),
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

/**
 * Every staged medium reports itself here. A gallery row records neither a
 * mesh's triangle count nor a clip's running time, so the peek takes them
 * from the viewer that just loaded the file: `MeshViewer` hands over its
 * stats, a media element carries its duration.
 */
function mediaReady(detail?: unknown): void {
  loading.value = false;
  if (detail instanceof Event) {
    const element = detail.target;
    if (element instanceof HTMLMediaElement && Number.isFinite(element.duration)) {
      mediaDurationMs.value = element.duration * 1000;
    }
    return;
  }
  if (detail && typeof detail === "object" && "triangleCount" in detail) {
    const stats = detail as { vertexCount?: number; triangleCount?: number };
    meshStats.value = {
      vertexCount: stats.vertexCount ?? 0,
      triangleCount: stats.triangleCount ?? 0,
    };
  }
}

function mediaFailed(): void {
  loading.value = false;
  loadError.value = video.value
    ? "Couldn’t play this video from the host."
    : "Couldn’t load the full print from this host.";
}

/**
 * Escape is the phone's Back: it undoes the last thing that opened. With
 * the details sheet up that is the sheet; the viewer itself goes on the next
 * one.
 */
function cancelViewer(): void {
  if (sheetExpanded.value) {
    collapseSheet();
    return;
  }
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
    !!target.closest(
      // `[data-gesture='own']` is the generic opt-out for a surface that
      // interprets drags itself. The stage arms its swipe on
      // `pointerdown.capture`, so it runs BEFORE any child handler and a
      // child cannot stop it — dragging a mesh to rotate it navigated the
      // gallery instead of turning the model. Any future canvas that owns
      // its own drag (a pannable map, a curve editor) opts out the same way.
      "button, input, textarea, select, a, [contenteditable='true'], [data-gesture='own']",
    )
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

// A new print is a new subject: the sheet gets out of its way again.
watch(
  () => props.item.filename,
  () => {
    collapseSheet();
    resetSheetDrag();
    // The next print's details start at the top, not at this one's offset.
    if (sheetBody.value) sheetBody.value.scrollTop = 0;
    actionStatus.value = "";
    meshStats.value = null;
    mediaDurationMs.value = null;
  },
);

async function fullImageBase64(): Promise<string> {
  const response = props.mediaUrlOverride
    ? await fetch(props.mediaUrlOverride)
    : await apiFetchTo(props.target, galleryMediaPath(props.item.filename, "host"));
  return blobToBase64(await response.blob());
}

async function performImageAction(action: "copy" | "save"): Promise<void> {
  // A stored mesh is glTF, not a raster: neither Photos nor the clipboard
  // takes it, and its poster is the gallery's, not the print itself.
  if (video.value || mesh.value || actionBusy.value) return;
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

/**
 * Where a "Save to Mold folder" export landed, as the shell reports it. Only
 * what the status shows crosses the bridge; the path stays native.
 */
interface MoldFolderSave {
  /** The final name, numbered past a collision. */
  filename: string;
  /** `Files ▸ Mold ▸ chair.stl` or `Downloads/Mold/chair.stl`. */
  label: string;
}

/**
 * The one export request a native shell runs for a print, whichever door it
 * leaves through. `share_exported_animation` opens the system share sheet;
 * `save_export_to_mold_folder` writes the on-device Mold folder. Both take
 * the identical arguments and reuse key, so a download staged for a share the
 * user backed out of is saved without a second fetch, and the shell checks
 * the bytes against the container the filename claims on either path.
 */
function nativeExportArguments(filename: string, request: Record<string, unknown>) {
  return {
    url: `${props.target.baseUrl}${videoExportPath(props.item.filename)}`,
    apiKey: props.target.apiKey,
    request,
    filename,
    reuseKey: `${props.target.baseUrl}\n${props.item.filename}\n${JSON.stringify(request)}`,
  };
}

/**
 * One geometry transcode of the stored GLB — or, on a phone, the stored GLB
 * itself. The body is the bare format: a mesh export has no playback options.
 *
 * On the native shells this is the SAME command a turntable takes: the shell
 * runs the export itself, checks the bytes against the container the filename
 * claims, and opens the system share sheet, so OBJ, STL, PLY and GLB land in
 * Files, AirDrop or any app that accepts them. A WebView `navigator.share`
 * cannot do that — it has no media type for geometry and falls back to a
 * browser download inside the app. The browser path stays exactly that, for
 * the mobile UI opened outside a shell.
 */
async function performMeshExport(format: string): Promise<void> {
  if (exportBusy.value) return;
  exportBusy.value = true;
  exportError.value = "";
  actionStatus.value = "";
  try {
    const path = videoExportPath(props.item.filename);
    const filename = meshExportFilename(props.item.filename, format);
    const request = { format };
    if (nativeShell.value) {
      const outcome = await invoke<"shared" | "cancelled">(
        "share_exported_animation",
        nativeExportArguments(filename, request),
      );
      if (outcome === "cancelled") return;
      actionStatus.value = "Export ready to share";
      return;
    }
    const response = await apiFetchTo(props.target, path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    downloadVideoExport(await response.blob(), filename);
    actionStatus.value = "Mesh exported";
  } catch (error) {
    // A geometry export never opens the options sheet, so `exportError` (the
    // sheet's own slot) would be invisible here: the footer status line the
    // tap is watching is where the failure has to land.
    actionStatus.value = error instanceof Error ? error.message : String(error);
  } finally {
    exportBusy.value = false;
  }
}

/**
 * The other half of the pair: the same transcode, written into the Mold
 * folder instead of handed to the share sheet. The status names where it
 * went in the words the shell's own file browser uses.
 */
async function performMeshSave(format: string): Promise<void> {
  if (exportBusy.value || !nativeShell.value) return;
  exportBusy.value = true;
  exportError.value = "";
  actionStatus.value = "";
  try {
    const saved = await invoke<MoldFolderSave>(
      "save_export_to_mold_folder",
      nativeExportArguments(meshExportFilename(props.item.filename, format), { format }),
    );
    actionStatus.value = `Saved to ${saved.label}`;
  } catch (error) {
    actionStatus.value = error instanceof Error ? error.message : String(error);
  } finally {
    exportBusy.value = false;
  }
}

/**
 * The picked format decides the route, the tapped button decides where the
 * file goes. A turntable carries playback options, so it stops at the export
 * sheet the phone already has — with the destination already chosen.
 */
async function runMeshExport(destination: "share" | "folder"): Promise<void> {
  const format = meshFormat.value;
  if (!format) return;
  if (format === "turntable") {
    pendingMeshDestination.value = destination;
    await openVideoExport();
    return;
  }
  if (destination === "folder") await performMeshSave(format);
  else await performMeshExport(format);
}

async function openVideoExport(): Promise<void> {
  exportOpen.value = true;
  exportError.value = "";
  if (mesh.value) {
    // A turntable's containers are the host's advertised ANIMATED mesh
    // exports; `/api/gallery/export-options` answers for clips only.
    exportCapabilities.value = {
      ...DEFAULT_VIDEO_EXPORT_CAPABILITIES,
      formats: meshAnimationExports.value,
    };
    return;
  }
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

/**
 * `destination` is the sheet's pick when it offered one (a mesh turntable on
 * a phone): `folder` writes the Mold folder, anything else shares. A clip's
 * sheet offers none and shares as before.
 */
async function performVideoExport(
  options: VideoExportOptions,
  destination?: string,
): Promise<void> {
  if (exportBusy.value) return;
  exportBusy.value = true;
  exportError.value = "";
  // A stale "Saved to …" from an earlier export must not outlive this one.
  actionStatus.value = "";
  try {
    const path = videoExportPath(props.item.filename);
    const filename = videoExportFilename(props.item.filename, options.format);
    const native = nativeShell.value;
    if (native && destination === "folder") {
      const saved = await invoke<MoldFolderSave>(
        "save_export_to_mold_folder",
        nativeExportArguments(filename, { ...options }),
      );
      exportOpen.value = false;
      actionStatus.value = `Saved to ${saved.label}`;
      return;
    }
    if (native) {
      const outcome = await invoke<"shared" | "cancelled">(
        "share_exported_animation",
        nativeExportArguments(filename, { ...options }),
      );
      if (outcome === "cancelled") {
        // The export itself succeeded and stays staged under its reuse key;
        // the options sheet has done its job once the share sheet was shown,
        // so a dismissal closes it too — silently, there is nothing to report.
        exportOpen.value = false;
        return;
      }
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
    actionStatus.value = native
      ? "Export ready to share"
      : mesh.value
        ? "Mesh exported"
        : "Video exported";
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
      <MeshViewer
        v-else-if="mesh"
        :key="mediaLoadKey"
        class="gallery-viewer-media"
        :src="mediaUrl"
        :poster="thumbnailUrl"
        :alt="item.metadata.prompt || item.filename"
        data-test="gallery-viewer-mesh"
        @ready="mediaReady"
        @fail="mediaFailed"
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

    <button
      v-if="sheetExpanded"
      class="gallery-viewer-sheet-scrim"
      type="button"
      aria-label="Collapse print details"
      data-test="gallery-viewer-sheet-scrim"
      @click="collapseSheet"
    />

    <!-- Every detail and action lives here; the media keeps the screen. The
         body is translated out of view when collapsed, never unmounted. -->
    <section
      ref="sheetRoot"
      class="gallery-viewer-sheet"
      :class="{ 'is-expanded': sheetExpanded, 'is-dragging': sheetDragging }"
      role="region"
      aria-label="Print details"
      :style="{ '--viewer-sheet-drag': `${sheetDrag}px` }"
      data-test="gallery-viewer-sheet"
      @pointerdown="beginSheetDrag"
      @pointermove="trackSheetDrag"
      @pointerup="finishSheetDrag"
      @pointercancel="resetSheetDrag"
      @click.capture="guardSheetClick"
    >
      <button
        ref="sheetHandle"
        class="gallery-viewer-sheet-handle"
        type="button"
        :aria-expanded="sheetExpanded"
        :aria-controls="sheetBodyId"
        :aria-label="sheetExpanded ? 'Hide print details' : 'Show print details'"
        data-test="gallery-viewer-sheet-handle"
        @click="toggleSheet"
      >
        <span class="gallery-viewer-sheet-grabber" aria-hidden="true" />
        <span class="gallery-viewer-sheet-peek" data-test="gallery-viewer-sheet-peek">
          <span class="gallery-viewer-kind">{{ kindLabel }}</span>
          <span v-if="peekSummary" class="gallery-viewer-peek-fact">{{ peekSummary }}</span>
          <span class="gallery-viewer-peek-host">{{ hostName }}</span>
        </span>
      </button>

      <div
        :id="sheetBodyId"
        ref="sheetBody"
        class="gallery-viewer-details"
        data-test="gallery-viewer-sheet-body"
      >
        <div class="gallery-viewer-prompt">
          <span v-if="preparedPosition" data-test="gallery-viewer-batch">{{
            preparedPosition
          }}</span>
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
            v-if="upscaleEnabled && !audio && !mesh && !trashed"
            class="secondary-button gallery-viewer-upscale"
            type="button"
            data-test="gallery-viewer-upscale"
            @click="emit('upscale')"
          >
            {{ video ? "Framewise upscale…" : "Upscale…" }}
          </button>
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
          <!-- 3-D: GLB is the only stored container; the rest are transcodes of
             it, built from what THIS host advertises. One picker names the
             container — Turntable stands for the animated ones, which carry
             playback options — and the two verbs below it are where it goes. -->
          <div v-if="meshChoices.length" class="gallery-viewer-mesh-export">
            <span class="gallery-viewer-mesh-label">Export</span>
            <div data-test="gallery-viewer-mesh-format">
              <SegmentedControl
                v-model="meshFormat"
                :options="meshChoices"
                label="Export format"
                wrap
              />
            </div>
            <div class="gallery-viewer-mesh-verbs">
              <button
                class="secondary-button gallery-viewer-export"
                type="button"
                data-test="gallery-viewer-mesh-export"
                :disabled="!!actionBusy || exportBusy"
                @click="runMeshExport('share')"
              >
                {{ meshPrimaryLabel }}
              </button>
              <button
                v-if="nativeShell"
                class="secondary-button gallery-viewer-export"
                type="button"
                data-test="gallery-viewer-mesh-save"
                :disabled="!!actionBusy || exportBusy"
                @click="runMeshExport('folder')"
              >
                Save to Mold folder
              </button>
            </div>
          </div>
          <template v-if="!video && !audio && !mesh">
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
      </div>
    </section>
    <VideoExportDialog
      :open="exportOpen"
      :filename="item.filename"
      :formats="exportCapabilities.formats"
      :destinations="exportDestinations"
      :busy="exportBusy"
      :error="exportError"
      @close="exportOpen = false"
      @export="performVideoExport"
    />
    <MobileLibrarySheet
      :open="infoOpen"
      :title="viewerTitle"
      :focus-first-control="false"
      swipe-to-dismiss
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
/* One picker and two verbs, in the width of a single stacked button. */
.gallery-viewer-mesh-export {
  display: grid;
  min-width: 0;
  gap: 8px;
}

.gallery-viewer-mesh-label {
  color: rgba(245, 239, 255, 0.62);
  font-family: var(--font-utility);
  font-size: var(--text-edge-code);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.gallery-viewer-mesh-verbs {
  display: grid;
  min-width: 0;
  gap: 8px;
}

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

/* Audio has no raster to fill the stage: waveform above, transport below.
   The stage is the whole viewport now, so this has to claim its height the
   way an image does, or the transport floats against the header. */
.gallery-viewer-audio {
  display: flex;
  width: 100%;
  height: 100%;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  box-sizing: border-box;
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

/* Centred on the MEDIA, not on the padded stage: the header inset above and
   the sheet's peek below are not part of the picture the arrows page. */
.gallery-viewer-nav {
  position: absolute;
  z-index: 2;
  top: calc(50% + (var(--viewer-header-inset) - var(--viewer-peek)) / 2);
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
