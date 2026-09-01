<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
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
import { useComposerStore } from "../../stores/composer";
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
  }>(),
  {
    mesh: false,
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

const router = useRouter();
const composer = useComposerStore();
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
let restoreFocusEl: HTMLElement | null = null;
onMounted(() => {
  restoreFocusEl = document.activeElement as HTMLElement | null;
  closeBtn.value?.focus();
});
onBeforeUnmount(() => restoreFocusEl?.focus?.());

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
const canExportVideo = computed(
  () => props.video && props.item.filename.toLowerCase().endsWith(".mp4"),
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
  // Ship the full metadata — `applyPrefillToForm` restores every serialized
  // knob (negative prompt, LoRA stack, scheduler, strength, video params, …)
  // and prefers the pre-upscale generation canvas over the raster size.
  composer.set({ metadata: meta.value });
  emit("close");
  void router.push("/create");
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
    {
      label: "Use as source",
      disabled: props.audio || props.mesh,
      action: () => emit("useSource"),
    },
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
  <div
    class="lightbox-scrim fixed inset-0 z-40 flex items-center justify-center p-10"
    role="dialog"
    aria-modal="true"
    :aria-label="`Print ${index + 1} of ${count}`"
    @click.self="emit('close')"
  >
    <div
      class="ms-fade-up flex max-h-[86vh] w-full max-w-[1000px] overflow-hidden rounded-card-lg border border-edge bg-bench shadow-raised"
    >
      <!-- media pane -->
      <div class="relative flex min-w-0 flex-1 items-center justify-center bg-print-surface p-5">
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
          class="absolute top-1/2 left-3.5 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full bg-black/50 text-on-media transition-opacity duration-100 hover:bg-black/70 disabled:opacity-30"
          :disabled="index === 0"
          aria-label="Previous print"
          @click="emit('prev')"
        >
          <Icon name="chevron-left" :size="22" />
        </button>
        <button
          type="button"
          class="absolute top-1/2 right-3.5 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full bg-black/50 text-on-media transition-opacity duration-100 hover:bg-black/70 disabled:opacity-30"
          :disabled="index === count - 1"
          aria-label="Next print"
          @click="emit('next')"
        >
          <Icon name="chevron-right" :size="22" />
        </button>
      </div>

      <!-- details pane -->
      <aside class="flex w-80 shrink-0 flex-col p-6">
        <div class="mb-4 flex items-center gap-2.5">
          <span class="lightbox-kicker">Print details</span>
          <div class="flex-1" />
          <span class="data-mono text-caption text-ink-3">{{ index + 1 }} / {{ count }}</span>
          <button
            ref="closeBtn"
            type="button"
            class="flex h-[30px] w-[30px] items-center justify-center rounded-full bg-[color-mix(in_srgb,var(--rebate)_9%,transparent)] text-body-lg text-ink-2 transition-colors duration-100 hover:text-ink"
            title="Close (Esc)"
            aria-label="Close"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div class="min-h-0 flex-1 overflow-y-auto">
          <!-- Title lead line: editable when a host can organize, else the
               display title (title ?? prompt excerpt ?? filename stem). -->
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
              aria-label="Print title"
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
            <button
              v-if="showOrganization"
              type="button"
              class="lightbox-fav flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-chrome border transition-colors duration-100"
              :class="
                isFavorite
                  ? 'lightbox-fav--on border-safelight text-safelight'
                  : 'border-edge text-ink-3 hover:text-ink'
              "
              :aria-pressed="isFavorite"
              :aria-label="isFavorite ? 'Unfavorite' : 'Favorite'"
              :title="isFavorite ? 'Unfavorite (F)' : 'Favorite (F)'"
              data-test="lightbox-favorite"
              @click="emit('favorite', !isFavorite)"
            >
              <Icon name="heart" :size="15" />
            </button>
          </div>
          <p
            v-if="titleError"
            class="mt-1 text-caption text-stop"
            data-test="lightbox-title-error"
            role="alert"
          >
            {{ titleError }}
          </p>
          <span
            class="data-mono mt-1.5 block truncate text-caption text-ink-3"
            data-test="lightbox-filename"
            :title="item.filename"
          >
            {{ item.filename }}
          </span>
          <p
            v-if="purge"
            class="mt-2 font-utility text-[11px] text-ink-2"
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
          <div class="mt-3 flex items-start gap-2">
            <p
              data-selectable
              data-test="lightbox-prompt"
              class="min-w-0 flex-1 whitespace-pre-wrap text-body text-ink"
              :title="meta.prompt"
            >
              {{ meta.prompt }}
            </p>
            <button
              v-if="meta.prompt"
              type="button"
              data-test="copy-prompt"
              class="flex h-[30px] shrink-0 items-center gap-1.5 rounded-chrome border border-edge px-2 font-utility text-[11px] text-ink-2 transition-colors hover:text-ink"
              aria-label="Copy prompt"
              title="Copy prompt"
              @click="copy(meta.prompt)"
            >
              <Icon name="copy" :size="14" />
              Copy
            </button>
          </div>
          <template v-if="showOrganization">
            <p class="lightbox-kicker mt-4 mb-1.5">Tags</p>
            <TagEditor
              :model-value="tags"
              :suggestions="tagSuggestions"
              aria-label="Print tags"
              data-test="lightbox-tags"
              @add="(name) => emit('tags', { add: [name], remove: [] })"
              @remove="(name) => emit('tags', { add: [], remove: [name] })"
            />
            <p class="lightbox-kicker mt-4 mb-1.5">In collections</p>
            <CollectionPicker
              :collections="collections"
              :selected="inCollections"
              :counts="collectionCounts"
              aria-label="In collections"
              data-test="lightbox-collections"
              @toggle="(slug, checked) => emit('collections', { slug, checked })"
              @create="(name) => emit('collections', { name, checked: true })"
            />
          </template>
          <p
            v-if="meta.original_prompt"
            data-test="lightbox-original"
            data-selectable
            class="mt-2 text-caption text-ink-2"
            :title="meta.original_prompt"
          >
            <span class="text-ink-3">Original</span> {{ meta.original_prompt }}
          </p>
          <p
            v-if="meta.negative_prompt"
            data-test="lightbox-negative"
            data-selectable
            class="mt-2 text-caption text-ink-2"
            :title="meta.negative_prompt"
          >
            <span class="text-ink-3">Negative</span> {{ meta.negative_prompt }}
          </p>
          <p
            v-if="meta.batch_id && meta.batch_index && meta.batch_count"
            data-test="lightbox-batch"
            data-selectable
            class="mt-2 text-caption text-ink-2"
            :title="meta.batch_id"
          >
            <span class="text-ink-3">Prepared batch</span>
            {{ meta.batch_index }} of {{ meta.batch_count }} · {{ meta.batch_id }}
          </p>

          <dl class="mt-4 space-y-2.5 font-utility">
            <div class="flex justify-between gap-2">
              <dt class="text-caption text-ink-3">Model</dt>
              <dd class="data-mono truncate text-caption text-ink">{{ modelLabel }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-caption text-ink-3">Seed</dt>
              <dd>
                <button
                  type="button"
                  class="data-mono text-caption text-ink hover:text-safelight"
                  title="Copy seed"
                  @click="copy(String(meta.seed))"
                >
                  {{ meta.seed }} ⧉
                </button>
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-caption text-ink-3">Dimensions</dt>
              <dd class="data-mono text-caption text-ink">{{ meta.width }}×{{ meta.height }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-caption text-ink-3">Steps · guidance</dt>
              <dd class="data-mono text-caption text-ink">
                {{ meta.steps }} · {{ meta.guidance.toFixed(1) }}
              </dd>
            </div>
            <div
              v-if="schedulerName"
              class="flex justify-between gap-2"
              data-test="lightbox-scheduler"
            >
              <dt class="text-caption text-ink-3">Scheduler</dt>
              <dd class="data-mono text-caption text-ink">{{ schedulerName }}</dd>
            </div>
            <div
              v-if="meta.cfg_plus"
              class="flex justify-between gap-2"
              data-test="lightbox-cfg-plus"
            >
              <dt class="text-caption text-ink-3">CFG++</dt>
              <dd class="data-mono text-caption text-ink">on</dd>
            </div>
            <div
              v-if="meta.strength != null"
              class="flex justify-between gap-2"
              data-test="lightbox-strength"
            >
              <dt class="text-caption text-ink-3">{{ strengthCaption }}</dt>
              <dd class="data-mono text-caption text-ink">{{ meta.strength.toFixed(2) }}</dd>
            </div>
            <div v-if="frames" class="flex justify-between gap-2" data-test="lightbox-video">
              <dt class="text-caption text-ink-3">Frames</dt>
              <dd class="data-mono text-caption text-ink">
                {{ frames }}<template v-if="fps"> · {{ fps }} fps</template>
              </dd>
            </div>
            <div v-if="pipeline" class="flex justify-between gap-2" data-test="lightbox-pipeline">
              <dt class="text-caption text-ink-3">Pipeline</dt>
              <dd class="data-mono text-caption text-ink">{{ pipeline }}</dd>
            </div>
            <div
              v-for="l in loraStack"
              :key="l.path"
              class="flex justify-between gap-2"
              data-test="lightbox-lora"
            >
              <dt class="text-caption text-ink-3">LoRA</dt>
              <dd class="data-mono truncate text-caption text-ink" :title="l.path">
                {{ l.path }} × {{ l.scale.toFixed(2) }}
              </dd>
            </div>
            <div
              v-if="identity"
              class="flex justify-between gap-2"
              data-test="lightbox-identity-photo"
            >
              <dt class="text-caption text-ink-3">Identity photo</dt>
              <dd
                class="data-mono truncate text-caption text-ink"
                :title="identity.sha256 ?? undefined"
              >
                {{ identity.name ?? "Identity photo"
                }}<template v-if="identity.shortSha"> · {{ identity.shortSha }}</template>
              </dd>
            </div>
            <div v-if="identity" class="flex justify-between gap-2" data-test="lightbox-identity">
              <dt class="text-caption text-ink-3">Identity strength</dt>
              <dd class="data-mono text-caption text-ink">
                {{ identity.weight }} · from step {{ identity.startStep }}
              </dd>
            </div>
            <div v-if="fileSize" class="flex justify-between gap-2" data-test="lightbox-file-size">
              <dt class="text-caption text-ink-3">File size</dt>
              <dd class="data-mono text-caption text-ink">{{ fileSize }}</dd>
            </div>
            <div v-if="fileFormat" class="flex justify-between gap-2" data-test="lightbox-format">
              <dt class="text-caption text-ink-3">Format</dt>
              <dd class="data-mono text-caption text-ink">{{ fileFormat.toUpperCase() }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt class="text-caption text-ink-3">Created</dt>
              <dd class="text-caption text-ink">{{ when }}</dd>
            </div>
            <div v-if="hostLabel" class="flex justify-between gap-2" data-test="lightbox-host">
              <dt class="text-caption text-ink-3">Host</dt>
              <dd class="data-mono truncate text-caption text-ink">{{ hostLabel }}</dd>
            </div>
          </dl>
          <span
            v-if="meta.version"
            class="data-mono mt-2 block text-caption text-ink-3"
            data-test="lightbox-version"
          >
            mold {{ meta.version }}
          </span>
          <span v-if="item.metadata_synthetic" class="edge-code mt-2 block"
            >SYNTHETIC METADATA</span
          >
        </div>

        <button
          type="button"
          data-test="lightbox-primary-action"
          class="mt-4 flex h-11 w-full items-center justify-center gap-2 rounded-[10px] bg-safelight text-body-lg font-bold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
          @click="primaryAction"
        >
          <Icon name="reuse" :size="15" />
          {{
            isSequence
              ? canEditSequence
                ? "Edit sequence"
                : "Duplicate as new"
              : "Reuse these settings"
          }}
        </button>
        <button
          v-if="canEditSequence"
          type="button"
          data-test="lightbox-duplicate-sequence"
          class="border-ce mt-2.5 h-10 w-full rounded-control border text-body font-semibold text-ink-2 transition-colors duration-100 hover:text-ink"
          @click="emit('reuseSequence')"
        >
          Duplicate as new
        </button>
        <div class="mt-2.5 flex gap-2.5">
          <button
            type="button"
            data-test="lightbox-use-source"
            class="border-ce h-10 flex-1 rounded-control border text-body font-semibold text-ink-2 transition-colors duration-100 hover:text-ink"
            :disabled="audio"
            @click="emit('useSource')"
          >
            Use as source
          </button>
          <button
            type="button"
            data-test="save-media"
            class="border-ce h-10 flex-1 rounded-control border text-body font-semibold text-ink-2 transition-colors duration-100 hover:text-ink"
            :disabled="saveBusy"
            @click="saveMedia"
          >
            {{ saveBusy ? "Saving…" : audio ? "Save audio" : video ? "Save video" : "Save image" }}
          </button>
        </div>
        <button
          v-if="upscaleEnabled && !audio && !trashed"
          type="button"
          data-test="lightbox-upscale"
          class="border-ce mt-2.5 h-10 w-full rounded-control border text-body font-semibold text-ink-2 transition-colors duration-100 hover:text-ink"
          @click="emit('upscale')"
        >
          {{ video ? "Framewise upscale…" : "Upscale…" }}
        </button>
        <div class="mt-2 flex gap-2.5">
          <button
            v-if="canExportVideo"
            type="button"
            data-test="export-video"
            class="border-edge h-8 flex-1 rounded-control border text-caption text-ink-2 transition-colors duration-100 hover:text-ink"
            @click="openVideoExport"
          >
            Export format…
          </button>
          <button
            v-if="canReveal"
            type="button"
            class="border-edge h-8 flex-1 rounded-control border text-caption text-ink-2 transition-colors duration-100 hover:text-ink"
            @click="reveal"
          >
            Reveal in file manager
          </button>
          <template v-if="trashed">
            <button
              type="button"
              data-test="lightbox-restore"
              class="border-edge h-8 flex-1 rounded-control border text-caption text-ink-2 transition-colors duration-100 hover:text-ink"
              @click="emit('restore')"
            >
              Restore
            </button>
            <button
              type="button"
              data-test="lightbox-delete-forever"
              class="border-edge h-8 flex-1 rounded-control border text-caption text-ink-2 transition-colors duration-100 hover:text-stop"
              @click="confirmingForever = true"
            >
              Delete forever
            </button>
          </template>
          <button
            v-else
            type="button"
            data-test="lightbox-delete"
            class="border-edge h-8 flex-1 rounded-control border text-caption transition-colors duration-100"
            :class="
              confirmingDelete
                ? 'border-stop bg-stop font-semibold text-on-accent'
                : 'text-ink-2 hover:text-stop'
            "
            @blur="confirmingDelete = false"
            @click="onDelete"
          >
            {{
              canTrash ? "Move to trash" : confirmingDelete ? "Delete? Can't be undone." : "Delete"
            }}
          </button>
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
  </div>
</template>

<style scoped>
.lightbox-scrim {
  background: rgba(6, 5, 10, 0.82);
  backdrop-filter: blur(6px);
}

.lightbox-kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

/* Title lead line: display 16/600, a quiet underline that turns accent
   while editing. 34px tall so it is a real target. */
.lightbox-title {
  display: block;
  height: 34px;
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 600;
  color: var(--rebate);
  background: transparent;
  border: 0;
  border-bottom: 1.5px solid transparent;
  padding: 0 0 2px;
  outline: none;
  line-height: 30px;
  transition: border-color var(--dur-quick) var(--ease);
}

input.lightbox-title::placeholder {
  color: var(--ink-3);
  font-weight: 500;
}

input.lightbox-title:hover {
  border-bottom-color: var(--edge);
}

input.lightbox-title:focus,
.lightbox-title--editing {
  border-bottom-color: var(--safelight);
}

/* The filled heart: the registry ships one outline glyph; the active state
   fills it with the current (accent) color. */
.lightbox-fav--on :deep(svg) {
  fill: currentColor;
}

.ms-lib-upscaled {
  font-family: var(--f-mono);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: color-mix(in srgb, var(--rebate) 88%, black);
  color: var(--on-accent);
  padding: 2px 6px;
  border-radius: 5px;
}
</style>
