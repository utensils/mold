<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";
import Icon from "@ui/components/Icon.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import MeshExportDialog from "@ui/components/MeshExportDialog.vue";
import AuthedMedia from "./AuthedMedia.vue";
import CollectionPicker from "../library/CollectionPicker.vue";
import TagEditor from "../library/TagEditor.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import {
  fetchGalleryMediaBytes,
  galleryMediaPath,
  type GallerySource,
} from "../../lib/gallery/media";
import { ipc } from "../../lib/ipc";
import { useToastStore } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { copyImageBytesToClipboard } from "../../lib/clipboard";
import { copyLocalOutputPath } from "../../lib/localOutputPath";
import { formatBytes, formatScheduler } from "../../lib/format";
import { modelDisplayNameForId } from "../../lib/models";
import { useHostModelsStore } from "../../stores/hostModels";
import type { ApiTarget } from "../../lib/api/client";
import type { GalleryImage } from "../../lib/api/types";
import { isUpscaledImage } from "../../lib/gallery/upscaled";
import {
  DEFAULT_VIDEO_EXPORT_CAPABILITIES,
  videoExportFilename,
  type VideoExportCapabilities,
  type VideoExportOptions,
} from "@studio/lib/videoExport";
import { strengthSemanticsForModel } from "@studio/lib/strengthSemantics";
import { identityProvenance } from "@studio/lib/identityConditioning";
import {
  displayTitle,
  purgeCountdownFromPurgeAt,
  validatePrintTitle,
  type MergedCollection,
} from "@studio/lib/libraryOrganization";
import type { TagCount } from "@studio/lib/api/galleryOrganization";
import { saveGalleryMedia, showSavedMediaToast } from "../../lib/mediaSave";
import { suggestedSaveName } from "../../lib/gallery/saveName";
import { formatGenerationTime } from "@studio/lib/generationTime";
import {
  meshAnimationExportFormats,
  meshExportFilename,
  meshFileExportFormats,
} from "../../lib/gallery/meshExport";
import {
  meshExportRequest,
  meshGeometryDefaults,
  takesGeometryOptions,
  type MeshExportGeometryCapabilities,
  type MeshGeometryOptions,
} from "@studio/lib/meshExport";

/** The print's organization across every copy (the Library's
 *  `organizationOf(entry)` union). Optional so callers that predate the
 *  Library organization keep mounting the Lightbox unchanged. */
export interface LightboxOrganization {
  title: string | null;
  favorite: boolean;
  tags: string[];
  collections: string[];
  trashedAt?: number | null;
  purgeAt?: number | null;
}

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    index: number;
    count: number;
    video: boolean;
    mesh?: boolean;
    /** Audio-only print: no raster to show, a transport instead. */
    audio?: boolean;
    /** The origin host can actually execute this media's upscale workflow. */
    upscaleEnabled?: boolean;
    source?: GallerySource;
    /** Origin host to fetch media from; null = the primary connection. */
    target?: ApiTarget | null;
    /** Blob-cache bucket, usually the origin host id. */
    cacheKey?: string | null;
    /** Origin host's friendly name for the metadata block. */
    hostLabel?: string | null;
    canReveal?: boolean;
    /** This print was stitched from a sequence (`metadata.chain` present), so
     *  reuse loads a clip rail instead of the One shot composer. */
    isSequence?: boolean;
    /** The producing job is (as far as we know without probing) still on its
     *  origin host — see `sequenceEditAvailability`. */
    canEditSequence?: boolean;
    /** Title / ♥ / tags / collections union for this print; null hides the
     *  editing rows (the title line still shows the display title). */
    organization?: LightboxOrganization | null;
    /** Some host holding this print can edit its organization. */
    canOrganize?: boolean;
    /** Delete moves to the trash (6 s undo) rather than hard-deleting. */
    canTrash?: boolean;
    /** The open print is in the trash: show the purge countdown and
     *  Restore / Delete forever instead of the Delete button. */
    trashed?: boolean;
    /** Host-merged collections for the "In collections" checklist. */
    collections?: MergedCollection[];
    /** Optional logical-print counts per collection slug. */
    collectionCounts?: ((slug: string) => number) | null;
    /** Host-merged tags for the tag editor's suggestions. */
    tagSuggestions?: TagCount[];
    /** `capabilities.mesh.export_formats` of the host that HOLDS this print.
     *  GLB is the only stored form; every entry here is a transcode that host
     *  offers, so the menu is never a client constant. */
    meshExportFormats?: string[];
    /** `capabilities.mesh.export_geometry` of the host that HOLDS this print
     *  — its own bounds, axes, origins and per-container defaults. Absent or
     *  null means that host predates the feature, which is the ONLY gate:
     *  the export then posts the bare `{ format }` it always has, because an
     *  older server drops unknown fields instead of refusing them. */
    meshExportGeometry?: MeshExportGeometryCapabilities | null;
  }>(),
  {
    mesh: false,
    meshExportFormats: () => [],
    meshExportGeometry: null,
    audio: false,
    upscaleEnabled: true,
    source: "host",
    target: null,
    cacheKey: null,
    hostLabel: null,
    canReveal: false,
    isSequence: false,
    canEditSequence: false,
    organization: null,
    canOrganize: false,
    canTrash: false,
    trashed: false,
    collections: () => [],
    collectionCounts: null,
    tagSuggestions: () => [],
  },
);
const emit = defineEmits<{
  close: [];
  prev: [];
  next: [];
  delete: [];
  useSource: [];
  /** One-shot reuse. The OWNER runs it: retained private source media is
   *  attached there, and doing it here would drop it. */
  reuse: [];
  reuseSequence: [];
  editSequence: [];
  /** Title edited in the aside (`null` clears it). */
  rename: [title: string | null];
  favorite: [value: boolean];
  tags: [change: { add: string[]; remove: string[] }];
  /** Collection membership toggled, or a new collection named. */
  collections: [change: { slug?: string; name?: string; checked: boolean }];
  restore: [];
  deleteForever: [];
  upscale: [];
}>();

const toasts = useToastStore();
const ui = useUiStore();
const contextMenu = useContextMenuStore();
const hostModels = useHostModelsStore();
const modelLabel = computed(() =>
  modelDisplayNameForId(props.item.metadata.model, hostModels.unionInstalled),
);

// ⇧⌘C copies the seed while the lightbox is open.
watch(
  () => ui.copySeedTick,
  () => void copy(String(props.item.metadata.seed)),
);

/** The open print's bytes live in the trash: resolve media, file paths,
 *  Reveal, and Save into `.trash/` so a newer same-name live file can never
 *  shadow them. Derived from the row itself with the view as fallback. */
const fromTrash = computed(() => props.trashed || props.item.trashed_at != null);

const confirmingDelete = ref(false);
const confirmingForever = ref(false);
const exportOpen = ref(false);
const exportBusy = ref(false);
const saveBusy = ref(false);
const exportError = ref("");
const exportCapabilities = ref<VideoExportCapabilities>(DEFAULT_VIDEO_EXPORT_CAPABILITIES);
watch(
  () => props.item.filename,
  () => {
    confirmingDelete.value = false;
    confirmingForever.value = false;
  },
);

// ── Title (Library organization, D5) ─────────────────────────────────────────
// The aside leads with an editable title; the raw filename drops to a mono
// detail row. Enter commits (emit `rename`), Escape reverts, blur commits a
// changed draft. The placeholder is the display fallback (prompt excerpt or
// filename stem) so an untitled print never reads as a literal "Untitled".
const currentTitle = computed(() => props.organization?.title ?? props.item.title ?? null);
const titleDraft = ref(currentTitle.value ?? "");
const titleEditing = ref(false);
const titleError = ref<string | null>(null);
const titleEl = ref<HTMLInputElement | null>(null);
watch([currentTitle, () => props.item.filename], () => {
  if (!titleEditing.value) titleDraft.value = currentTitle.value ?? "";
  titleError.value = null;
});
const titlePlaceholder = computed(() => displayTitle({ ...props.item, title: null }));
const headline = computed(() => displayTitle({ ...props.item, title: currentTitle.value }));

async function startTitleEdit() {
  if (!props.canOrganize) return;
  titleEditing.value = true;
  titleDraft.value = currentTitle.value ?? "";
  await nextTick();
  titleEl.value?.focus();
  titleEl.value?.select();
}

function commitTitle() {
  if (!titleEditing.value) return;
  const check = validatePrintTitle(titleDraft.value);
  if (!check.ok) {
    titleError.value = check.reason;
    return;
  }
  titleError.value = null;
  titleEditing.value = false;
  if ((check.value ?? null) !== (currentTitle.value ?? null)) emit("rename", check.value);
}

function revertTitle() {
  titleDraft.value = currentTitle.value ?? "";
  titleError.value = null;
  titleEditing.value = false;
}

function onTitleKeydown(event: KeyboardEvent) {
  if (event.key === "Enter") {
    event.preventDefault();
    commitTitle();
  } else if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    revertTitle();
  }
}

// ── ♥ / tags / collections ──────────────────────────────────────────────────
const isFavorite = computed(() => props.organization?.favorite === true);
const tags = computed(() => props.organization?.tags ?? []);
const inCollections = computed(() => props.organization?.collections ?? []);
const purge = computed(() =>
  props.trashed ? purgeCountdownFromPurgeAt(props.organization?.purgeAt ?? null, Date.now()) : null,
);
const showOrganization = computed(() => props.canOrganize && props.organization !== null);

// Focus the close button on open and hand focus back to the opener on teardown,
// so the lightbox is keyboard-operable and doesn't strand focus when dismissed.
const closeBtn = ref<HTMLButtonElement | null>(null);
// The lightbox is an overlay in its own right: it joins the shared register
// for its whole life, so a question it opens (delete forever, the export
// sheets) sits ABOVE it and takes Escape on its own.
useOverlayStack(ref(true), "lightbox");
let restoreFocusEl: HTMLElement | null = null;
onMounted(() => {
  restoreFocusEl = document.activeElement as HTMLElement | null;
  closeBtn.value?.focus();
});
onBeforeUnmount(() => restoreFocusEl?.focus?.());

/** How long the render took, when the print knows (additive metadata). */
const took = computed(() => formatGenerationTime(meta.value.generation_time_ms));
const meta = computed(() => props.item.metadata);
// An LTX-2 print's `strength` is source preservation, not denoise (#1055).
// Family resolves through the live inventory (sequences record strength but
// no `pipeline`), with the model-id name markers as the offline fallback.
const strengthCaption = computed(() => {
  const model = meta.value?.model;
  const family = hostModels.unionInstalled.find((entry) => entry.name === model)?.family;
  return strengthSemanticsForModel(model, family).label;
});
const upscaled = computed(() => isUpscaledImage(props.item));
// `POST /api/gallery/export/:filename` reads the LIVE gallery only; a trashed
// print's bytes sit under `.trash`, where the route cannot see them.
const canExportVideo = computed(
  () => props.video && !fromTrash.value && props.item.filename.toLowerCase().endsWith(".mp4"),
);

/** Full LoRA stack; a legacy single `lora`/`lora_scale` pair becomes one row. */
const loraStack = computed(() => {
  const m = meta.value;
  if (m.loras?.length) return m.loras;
  return m.lora ? [{ path: m.lora, scale: m.lora_scale ?? 1.0 }] : [];
});

/** Face-identity provenance (#1224): names and digests only — saved metadata
 * never carries the photo itself, which is exactly why the digest is shown. */
const identity = computed(() => identityProvenance(meta.value));

const schedulerName = computed(() => formatScheduler(meta.value.scheduler));
const frames = computed(() => meta.value.frames ?? meta.value.video_frames ?? null);
const fps = computed(() => meta.value.fps ?? meta.value.video_fps ?? null);
const pipeline = computed(() => (props.video ? meta.value.pipeline : null));
const fileFormat = computed(() => props.item.format ?? meta.value.output_format ?? null);
const fileSize = computed(() =>
  props.item.size_bytes != null ? formatBytes(props.item.size_bytes) : null,
);

const when = computed(() =>
  new Date(props.item.timestamp * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }),
);

function primaryAction() {
  if (props.isSequence) {
    if (props.canEditSequence) emit("editSequence");
    else emit("reuseSequence");
    return;
  }
  // Hand the one-shot back to the owner rather than prefilling here. This used
  // to call `composer.set` directly, which INVALIDATES retained-source
  // authority — so the most visible reuse control silently dropped a print's
  // private source media while the right-click item kept it. Owners route this
  // through the same `reuseSettings` the context menu uses, which ships the
  // full metadata AND the retained inventory.
  emit("reuse");
}

async function copy(text: string) {
  await navigator.clipboard.writeText(text);
  toasts.push("Copied");
}

async function copyImage() {
  try {
    const target = props.target;
    await copyImageBytesToClipboard(
      galleryMediaPath(props.item.filename, props.source, false, fromTrash.value),
      target ? { fetchImage: (p) => fetchGalleryMediaBytes(p, target) } : undefined,
    );
    toasts.push("Image copied");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

function imageMenu(): MenuEntry[] {
  return [
    {
      label: "Copy image",
      disabled: props.video || props.audio || props.mesh,
      action: () => void copyImage(),
    },
    // A trashed print is on its way to being purged: it is neither a recipe
    // to pick up nor a photo that will still be there. The grid's own tile
    // menu is already gated this way; Upscale and the delete block below read
    // the same `fromTrash`.
    ...(fromTrash.value
      ? []
      : [
          {
            label: "Use as source",
            disabled: props.audio || props.mesh,
            action: () => emit("useSource"),
          },
        ]),
    {
      label: "Copy file path",
      action: () =>
        void copyLocalOutputPath(props.item.filename, fromTrash.value)
          .then(() => toasts.push("File path copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          ),
    },
    { label: "Copy prompt", action: () => void copy(meta.value.prompt) },
    { label: "Copy seed", action: () => void copy(String(meta.value.seed)) },
    { separator: true },
    { label: "Reveal in file manager", disabled: !props.canReveal, action: () => void reveal() },
  ];
}

async function reveal() {
  try {
    await ipc.revealOutputFile(props.item.filename, fromTrash.value);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

function onDelete() {
  if (props.canTrash) {
    // The parent owns the 6 s undo toast.
    emit("delete");
    return;
  }
  if (!confirmingDelete.value) {
    confirmingDelete.value = true;
    return;
  }
  emit("delete");
  toasts.push("Deleted print");
}

function confirmDeleteForever() {
  confirmingForever.value = false;
  emit("deleteForever");
}

async function saveMedia() {
  if (saveBusy.value) return;
  saveBusy.value = true;
  try {
    const saved = await saveGalleryMedia(
      props.target,
      props.item.filename,
      suggestedSaveName({ ...props.item, title: currentTitle.value }),
      null,
      fromTrash.value,
    );
    showSavedMediaToast(toasts, saved);
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  } finally {
    saveBusy.value = false;
  }
}

// ── 3-D exports ─────────────────────────────────────────────────────────────
// The holding host is the authority on what it can transcode a stored GLB
// into. Direct containers get one entry each; animated turntables share the
// export sheet's playback options, so they collapse into one entry that opens
// it with just those containers. The server lists the stored container
// (`glb`) first so a CLI can name it; Save already hands over that exact
// file, so it is not an export here. A trashed print is under `.trash`,
// which the export route cannot read, so nothing is offered for it.
const meshFileExports = computed(() =>
  props.mesh && !fromTrash.value
    ? meshFileExportFormats(props.meshExportFormats).filter((format) => format !== "glb")
    : [],
);
const meshAnimationExports = computed(() =>
  props.mesh && !fromTrash.value ? meshAnimationExportFormats(props.meshExportFormats) : [],
);

// The geometry sheet, kept on its own pair of refs so the turntable's
// `VideoExportDialog` is untouched: the two sheets answer different questions
// and can never be open at once.
const meshGeometryOpen = ref(false);
const meshGeometryFormat = ref("");

/**
 * A geometry container the holding host advertises defaults for gets the
 * options sheet; everything else — a turntable, the stored GLB, a container
 * this host does not scale, and EVERY container on a host that predates the
 * block — exports straight through with the body this client has always sent.
 */
function exportMesh(format: string) {
  if (exportBusy.value) return;
  if (takesGeometryOptions(format) && meshGeometryDefaults(props.meshExportGeometry, format)) {
    exportError.value = "";
    meshGeometryFormat.value = format;
    meshGeometryOpen.value = true;
    return;
  }
  void runMeshExport(format, null);
}

/** The dialog's answer, or the direct path's absence of one. */
function performMeshGeometryExport(geometry: MeshGeometryOptions) {
  void runMeshExport(meshGeometryFormat.value, geometry);
}

async function runMeshExport(format: string, geometry: MeshGeometryOptions | null) {
  if (exportBusy.value) return;
  exportBusy.value = true;
  exportError.value = "";
  try {
    const saved = await saveGalleryMedia(
      props.target,
      props.item.filename,
      meshExportFilename(suggestedSaveName({ ...props.item, title: currentTitle.value }), format),
      meshExportRequest(format, geometry),
      fromTrash.value,
    );
    meshGeometryOpen.value = false;
    showSavedMediaToast(toasts, saved);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // A failure inside the sheet is reported IN the sheet, beside the knobs
    // that may need changing; the one-click path has nowhere but a toast.
    if (meshGeometryOpen.value) exportError.value = message;
    else toasts.push(message, "error");
  } finally {
    exportBusy.value = false;
  }
}

/** The turntable sheet needs no capability probe: the advertised containers
 * arrived with the host's own capabilities on connect. */
function openMeshAnimationExport() {
  exportError.value = "";
  exportCapabilities.value = {
    ...DEFAULT_VIDEO_EXPORT_CAPABILITIES,
    formats: meshAnimationExports.value,
  };
  exportOpen.value = true;
}

async function openVideoExport() {
  exportOpen.value = true;
  exportError.value = "";
  try {
    const { apiJson, apiJsonTo } = await import("../../lib/api/client");
    exportCapabilities.value = props.target
      ? await apiJsonTo<VideoExportCapabilities>(props.target, "/api/gallery/export-options")
      : await apiJson<VideoExportCapabilities>("/api/gallery/export-options");
  } catch (error) {
    exportCapabilities.value = DEFAULT_VIDEO_EXPORT_CAPABILITIES;
    exportError.value =
      error instanceof Error ? error.message : "Couldn’t read export options from this host.";
  }
}

async function performVideoExport(options: VideoExportOptions) {
  if (exportBusy.value) return;
  exportBusy.value = true;
  exportError.value = "";
  try {
    const saved = await saveGalleryMedia(
      props.target,
      props.item.filename,
      videoExportFilename(
        suggestedSaveName({ ...props.item, title: currentTitle.value }),
        options.format,
      ),
      options,
    );
    exportOpen.value = false;
    showSavedMediaToast(toasts, saved);
  } catch (error) {
    exportError.value = error instanceof Error ? error.message : String(error);
  } finally {
    exportBusy.value = false;
  }
}
</script>

<template>
  <!-- `fixed`, deliberately: the Lightbox covers the WHOLE window, title bar
       and sidebar included. Never convert it to the `absolute inset-0` that
       ModalPanel and DrawerPanel use — `LibraryView`'s root is `relative`, so
       it would silently shrink to the grid pane. -->
  <div
    class="lightbox-scrim fixed inset-0 z-40 flex flex-col"
    role="dialog"
    aria-modal="true"
    :aria-label="`Picture ${index + 1} of ${count}`"
  >
    <!-- header: filename · how it was made, in mono · the print's actions -->
    <div
      class="flex h-[52px] shrink-0 items-center gap-2.5 border-b border-border px-3.5"
      data-test="lightbox-header"
    >
      <span
        class="min-w-0 truncate font-mono text-xs text-fg-2"
        data-test="lightbox-filename"
        :title="item.filename"
      >
        {{ item.filename }}
      </span>
      <span class="shrink-0 font-mono text-micro text-fg-dim">
        {{ meta.width }}×{{ meta.height }} · {{ modelLabel }} · {{ index + 1 }} / {{ count }}
      </span>
      <span class="flex-1" />
      <button
        v-if="showOrganization"
        type="button"
        class="ms-toolbar-button h-[28px]"
        :class="{ 'lightbox-fav--on': isFavorite }"
        :aria-pressed="isFavorite"
        :aria-label="isFavorite ? 'Unfavorite' : 'Favorite'"
        :title="isFavorite ? 'Unfavorite (F)' : 'Favorite (F)'"
        data-test="lightbox-favorite"
        @click="emit('favorite', !isFavorite)"
      >
        <span class="font-mono" :class="isFavorite ? 'text-star' : 'text-fg-dim'">★</span>
        {{ isFavorite ? "Favourited" : "Favourite" }}
      </button>
      <button
        type="button"
        data-test="save-media"
        class="ms-toolbar-button h-[28px]"
        :disabled="saveBusy"
        @click="saveMedia"
      >
        {{ saveBusy ? "Saving…" : "Save a copy" }}
      </button>
      <button
        v-if="!fromTrash"
        type="button"
        data-test="lightbox-primary-action"
        class="ms-toolbar-button ms-toolbar-button--on h-[28px] font-semibold"
        @click="primaryAction"
      >
        <Icon name="reuse" :size="13" />
        {{
          isSequence ? (canEditSequence ? "Edit clip" : "Duplicate as new") : "Use these settings"
        }}
      </button>
      <button
        ref="closeBtn"
        type="button"
        class="flex h-[28px] w-[28px] items-center justify-center rounded-control text-fg-2 transition-colors duration-100 hover:bg-surface hover:text-fg"
        title="Close (Esc)"
        aria-label="Close"
        @click="emit('close')"
      >
        <Icon name="close" :size="15" />
      </button>
    </div>

    <div class="flex min-h-0 flex-1">
      <!-- media pane -->
      <div
        class="relative flex min-w-0 flex-1 items-center justify-center p-6"
        @click.self="emit('close')"
      >
        <div
          data-test="lightbox-media"
          class="relative flex h-full w-full items-center justify-center overflow-hidden"
          @contextmenu="contextMenu.open($event, imageMenu())"
        >
          <div v-if="audio" class="flex w-full max-w-2xl flex-col items-center gap-4 p-4">
            <AuthedMedia
              :path="galleryMediaPath(item.filename, source, true, fromTrash)"
              :target="target"
              :cache-key="cacheKey"
              :alt="meta.prompt"
              class="!object-contain"
            />
            <AuthedMedia
              :path="galleryMediaPath(item.filename, source, false, fromTrash)"
              :target="target"
              :cache-key="cacheKey"
              audio
              controls
              :alt="meta.prompt"
            />
          </div>
          <AuthedMedia
            v-else
            :path="galleryMediaPath(item.filename, source, false, fromTrash)"
            :target="target"
            :cache-key="cacheKey"
            :video="video"
            :mesh="mesh"
            :poster-path="galleryMediaPath(item.filename, source, true, fromTrash)"
            :controls="video"
            :alt="meta.prompt"
            class="!object-contain"
          />
          <span
            v-if="upscaled"
            data-test="upscaled-badge"
            class="ms-lib-upscaled absolute top-2 left-2"
          >
            Upscaled
          </span>
        </div>
        <button
          type="button"
          class="absolute top-1/2 left-3.5 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-control border border-border bg-surface text-fg-2 transition-colors duration-100 hover:text-fg disabled:opacity-30"
          :disabled="index === 0"
          aria-label="Previous picture"
          @click="emit('prev')"
        >
          <Icon name="chevron-left" :size="20" />
        </button>
        <button
          type="button"
          class="absolute top-1/2 right-3.5 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-control border border-border bg-surface text-fg-2 transition-colors duration-100 hover:text-fg disabled:opacity-30"
          :disabled="index === count - 1"
          aria-label="Next picture"
          @click="emit('next')"
        >
          <Icon name="chevron-right" :size="20" />
        </button>
      </div>

      <!-- aside: words used · how it was made · tags · the secondary actions -->
      <aside
        class="flex w-[var(--mold-shell-inspector-w)] shrink-0 flex-col gap-3.5 overflow-y-auto border-l border-border bg-bg-deep p-4"
      >
        <!-- Title lead line: editable when a host can organize, else the
             display title (title ?? prompt excerpt ?? filename stem). -->
        <div class="flex flex-col gap-1">
          <div class="flex items-center gap-2" data-test="lightbox-title-row">
            <input
              v-if="canOrganize"
              ref="titleEl"
              v-model="titleDraft"
              data-selectable
              data-test="lightbox-title"
              type="text"
              class="lightbox-title min-w-0 flex-1"
              :class="{ 'lightbox-title--editing': titleEditing }"
              :placeholder="titlePlaceholder"
              :aria-invalid="titleError !== null"
              aria-label="Picture title"
              :title="currentTitle ?? titlePlaceholder"
              @focus="startTitleEdit"
              @blur="commitTitle"
              @keydown="onTitleKeydown"
            />
            <span
              v-else
              class="lightbox-title min-w-0 flex-1 truncate"
              data-test="lightbox-title"
              :title="headline"
            >
              {{ headline }}
            </span>
          </div>
          <p
            v-if="titleError"
            class="text-micro text-error"
            data-test="lightbox-title-error"
            role="alert"
          >
            {{ titleError }}
          </p>
          <p
            v-if="purge"
            class="font-mono text-micro text-fg-2"
            data-test="lightbox-purge"
            :data-kind="purge.kind"
          >
            <template v-if="purge.kind === 'purges'">
              In the trash · purges in <b class="text-warning">{{ purge.days }} d</b>
            </template>
            <template v-else-if="purge.kind === 'today'">
              In the trash · purges <b class="text-warning">today</b>
            </template>
            <template v-else>In the trash · kept until you empty it</template>
          </p>
        </div>

        <!-- Words used -->
        <div class="flex flex-col gap-1.5">
          <span class="flex items-center gap-2">
            <span class="ms-group-label uppercase">Words used</span>
            <span class="flex-1" />
            <button
              v-if="meta.prompt"
              type="button"
              data-test="copy-prompt"
              class="flex items-center gap-1 font-mono text-micro text-fg-dim hover:text-fg"
              aria-label="Copy prompt"
              title="Copy prompt"
              @click="copy(meta.prompt)"
            >
              <Icon name="copy" :size="12" />
              copy
            </button>
          </span>
          <p
            data-selectable
            data-test="lightbox-prompt"
            class="whitespace-pre-wrap text-sm leading-body text-fg"
            :title="meta.prompt"
          >
            {{ meta.prompt }}
          </p>
          <p
            v-if="meta.original_prompt"
            data-test="lightbox-original"
            data-selectable
            class="text-micro text-fg-2"
            :title="meta.original_prompt"
          >
            <span class="text-fg-dim">Before Write more for me</span> {{ meta.original_prompt }}
          </p>
          <p
            v-if="meta.negative_prompt"
            data-test="lightbox-negative"
            data-selectable
            class="text-micro text-fg-2"
            :title="meta.negative_prompt"
          >
            <span class="text-fg-dim">Kept out</span> {{ meta.negative_prompt }}
          </p>
          <p
            v-if="meta.batch_id && meta.batch_index && meta.batch_count"
            data-test="lightbox-batch"
            data-selectable
            class="text-micro text-fg-2"
            :title="meta.batch_id"
          >
            <span class="text-fg-dim">Prepared batch</span>
            {{ meta.batch_index }} of {{ meta.batch_count }} · {{ meta.batch_id }}
          </p>
        </div>

        <!-- How it was made: plain label left, mono truth right -->
        <div class="flex flex-col gap-2">
          <span class="ms-group-label uppercase">How it was made</span>
          <dl class="flex flex-col gap-1.5">
            <div class="lightbox-fact">
              <dt>Style</dt>
              <dd class="truncate">{{ modelLabel }}</dd>
            </div>
            <div class="lightbox-fact">
              <dt>Size</dt>
              <dd>{{ meta.width }} × {{ meta.height }}</dd>
            </div>
            <div class="lightbox-fact">
              <dt>Detail</dt>
              <dd>{{ meta.steps }} passes</dd>
            </div>
            <div class="lightbox-fact">
              <dt>Stick to my words</dt>
              <dd>{{ meta.guidance.toFixed(1) }}</dd>
            </div>
            <div class="lightbox-fact">
              <dt>Repeat this look</dt>
              <dd>
                <button
                  type="button"
                  class="font-mono hover:text-accent"
                  title="Copy seed"
                  @click="copy(String(meta.seed))"
                >
                  seed {{ meta.seed }} ⧉
                </button>
              </dd>
            </div>
            <div v-if="schedulerName" class="lightbox-fact" data-test="lightbox-scheduler">
              <dt>Scheduler</dt>
              <dd>{{ schedulerName }}</dd>
            </div>
            <div v-if="meta.cfg_plus" class="lightbox-fact" data-test="lightbox-cfg-plus">
              <dt>CFG++</dt>
              <dd>on</dd>
            </div>
            <div v-if="meta.strength != null" class="lightbox-fact" data-test="lightbox-strength">
              <dt>{{ strengthCaption }}</dt>
              <dd>{{ meta.strength.toFixed(2) }}</dd>
            </div>
            <div v-if="frames" class="lightbox-fact" data-test="lightbox-video">
              <dt>Length</dt>
              <dd>
                {{ frames }} frames<template v-if="fps"> · {{ fps }} fps</template>
              </dd>
            </div>
            <div v-if="pipeline" class="lightbox-fact" data-test="lightbox-pipeline">
              <dt>Pipeline</dt>
              <dd>{{ pipeline }}</dd>
            </div>
            <div v-if="took" class="lightbox-fact" data-test="lightbox-took">
              <dt>Took</dt>
              <dd>{{ took }}</dd>
            </div>
            <div
              v-for="l in loraStack"
              :key="l.path"
              class="lightbox-fact"
              data-test="lightbox-lora"
            >
              <dt>Add-on look</dt>
              <dd class="truncate" :title="l.path">{{ l.path }} × {{ l.scale.toFixed(2) }}</dd>
            </div>
            <div v-if="identity" class="lightbox-fact" data-test="lightbox-identity-photo">
              <dt>Face photo</dt>
              <dd class="truncate" :title="identity.sha256 ?? undefined">
                {{ identity.name ?? "Identity photo"
                }}<template v-if="identity.shortSha"> · {{ identity.shortSha }}</template>
              </dd>
            </div>
            <div v-if="identity" class="lightbox-fact" data-test="lightbox-identity">
              <dt>Face strength</dt>
              <dd>{{ identity.weight }} · from step {{ identity.startStep }}</dd>
            </div>
            <div v-if="fileSize" class="lightbox-fact" data-test="lightbox-file-size">
              <dt>File size</dt>
              <dd>{{ fileSize }}</dd>
            </div>
            <div v-if="fileFormat" class="lightbox-fact" data-test="lightbox-format">
              <dt>Format</dt>
              <dd>{{ fileFormat.toUpperCase() }}</dd>
            </div>
            <div class="lightbox-fact">
              <dt>Made</dt>
              <dd>{{ when }}</dd>
            </div>
            <div v-if="hostLabel" class="lightbox-fact" data-test="lightbox-host">
              <dt>Made on</dt>
              <dd class="truncate">{{ hostLabel }}</dd>
            </div>
          </dl>
          <span
            v-if="meta.version"
            class="font-mono text-micro text-fg-dim"
            data-test="lightbox-version"
          >
            mold {{ meta.version }}
          </span>
          <span v-if="item.metadata_synthetic" class="font-mono text-micro text-fg-dim"
            >SYNTHETIC METADATA</span
          >
        </div>

        <template v-if="showOrganization">
          <div class="flex flex-col gap-1.5">
            <span class="ms-group-label uppercase">Tags</span>
            <TagEditor
              :model-value="tags"
              :suggestions="tagSuggestions"
              aria-label="Picture tags"
              data-test="lightbox-tags"
              @add="(name) => emit('tags', { add: [name], remove: [] })"
              @remove="(name) => emit('tags', { add: [], remove: [name] })"
            />
          </div>
          <div class="flex flex-col gap-1.5">
            <span class="ms-group-label uppercase">Albums</span>
            <CollectionPicker
              :collections="collections"
              :selected="inCollections"
              :counts="collectionCounts"
              aria-label="In albums"
              data-test="lightbox-collections"
              @toggle="(slug, checked) => emit('collections', { slug, checked })"
              @create="(name) => emit('collections', { name, checked: true })"
            />
          </div>
        </template>

        <!-- the secondary actions -->
        <div class="mt-auto flex flex-col gap-2 pt-2">
          <button
            v-if="canEditSequence && !fromTrash"
            type="button"
            data-test="lightbox-duplicate-sequence"
            class="ms-toolbar-button justify-center"
            @click="emit('reuseSequence')"
          >
            Duplicate as new
          </button>
          <div class="flex gap-2">
            <button
              v-if="!fromTrash"
              type="button"
              data-test="lightbox-use-source"
              class="ms-toolbar-button flex-1 justify-center"
              :disabled="audio || mesh"
              @click="emit('useSource')"
            >
              Start from this photo
            </button>
            <button
              v-if="upscaleEnabled && !audio && !mesh && !trashed"
              type="button"
              data-test="lightbox-upscale"
              class="ms-toolbar-button flex-1 justify-center"
              @click="emit('upscale')"
            >
              {{ video ? "Framewise upscale…" : "Make bigger…" }}
            </button>
          </div>
          <!-- 3-D transcodes, straight from the holding host's own
               `mesh.export_formats`. GLB is what is stored; these are made on
               request. -->
          <div
            v-if="meshFileExports.length > 0 || meshAnimationExports.length > 0"
            class="flex flex-wrap gap-2"
            data-test="mesh-exports"
          >
            <button
              v-for="format in meshFileExports"
              :key="format"
              type="button"
              :data-test="`mesh-export-${format}`"
              class="ms-toolbar-button flex-1 justify-center"
              :disabled="exportBusy"
              @click="exportMesh(format)"
            >
              Export as {{ format.toUpperCase() }}…
            </button>
            <button
              v-if="meshAnimationExports.length > 0"
              type="button"
              data-test="mesh-export-animation"
              class="ms-toolbar-button flex-1 justify-center"
              @click="openMeshAnimationExport"
            >
              Export turntable…
            </button>
          </div>
          <div class="flex gap-2">
            <button
              v-if="canExportVideo"
              type="button"
              data-test="export-video"
              class="ms-toolbar-button flex-1 justify-center"
              @click="openVideoExport"
            >
              Export format…
            </button>
            <button
              v-if="canReveal"
              type="button"
              class="ms-toolbar-button flex-1 justify-center"
              @click="reveal"
            >
              Show the file
            </button>
            <template v-if="trashed">
              <button
                type="button"
                data-test="lightbox-restore"
                class="ms-toolbar-button flex-1 justify-center"
                @click="emit('restore')"
              >
                Restore
              </button>
              <button
                type="button"
                data-test="lightbox-delete-forever"
                class="ms-toolbar-button ms-toolbar-button--danger-hover flex-1 justify-center"
                @click="confirmingForever = true"
              >
                Delete forever
              </button>
            </template>
            <button
              v-else
              type="button"
              data-test="lightbox-delete"
              class="ms-toolbar-button flex-1 justify-center"
              :class="
                confirmingDelete ? 'ms-toolbar-button--danger' : 'ms-toolbar-button--danger-hover'
              "
              @blur="confirmingDelete = false"
              @click="onDelete"
            >
              {{
                canTrash
                  ? "Move to trash"
                  : confirmingDelete
                    ? "Delete? Can't be undone."
                    : "Delete"
              }}
            </button>
          </div>
        </div>
      </aside>
    </div>

    <ConfirmDialog
      :open="confirmingForever"
      :title="`Delete “${headline}” forever?`"
      message="This can't be undone."
      confirm-label="Delete forever"
      danger
      @confirm="confirmDeleteForever"
      @cancel="confirmingForever = false"
    />
    <VideoExportDialog
      :open="exportOpen"
      :filename="item.filename"
      :formats="exportCapabilities.formats"
      :busy="exportBusy"
      :error="exportError"
      @close="exportOpen = false"
      @export="performVideoExport"
    />
    <!-- Geometry options, only where the holding host advertises them. The
         Library lightbox shows the mesh through `AuthedMedia`, not a
         `MeshViewer`, so there is no bounding box to name real extents with
         and the sheet falls back to naming the knob itself. -->
    <MeshExportDialog
      v-if="meshExportGeometry"
      :open="meshGeometryOpen"
      :filename="item.filename"
      :format="meshGeometryFormat"
      :capabilities="meshExportGeometry"
      :bounds="null"
      :busy="exportBusy"
      :error="exportError"
      @close="meshGeometryOpen = false"
      @export="performMeshGeometryExport"
    />
  </div>
</template>

<style scoped>
/* The lightbox is the whole window on a near-solid crust. */
.lightbox-scrim {
  background: color-mix(in srgb, var(--mold-bg-crust) 92%, transparent);
}

/* Title lead line: a quiet underline that turns accent while editing. */
.lightbox-title {
  display: block;
  height: 30px;
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-md);
  font-weight: 600;
  color: var(--mold-text);
  background: transparent;
  border: 0;
  border-bottom: 1.5px solid transparent;
  padding: 0 0 2px;
  outline: none;
  line-height: 26px;
  transition: border-color var(--mold-dur-quick) var(--mold-ease-out);
}

input.lightbox-title::placeholder {
  color: var(--mold-text-dim);
  font-weight: 500;
}

input.lightbox-title:hover {
  border-bottom-color: var(--mold-border);
}

input.lightbox-title:focus,
.lightbox-title--editing {
  border-bottom-color: var(--mold-blue);
}

.lightbox-fav--on {
  border-color: var(--mold-star);
}

/* One fact: plain words left, the technical truth right, in mono. */
.lightbox-fact {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}
.lightbox-fact dt {
  flex-shrink: 0;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
.lightbox-fact dd {
  min-width: 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  text-align: right;
}
</style>
