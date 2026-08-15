<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import VideoExportDialog from "@ui/components/VideoExportDialog.vue";
import AuthedMedia from "./AuthedMedia.vue";
import { galleryMediaPath, type GallerySource } from "../../lib/gallery/media";
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
import { saveGalleryMedia, showSavedMediaToast } from "../../lib/mediaSave";

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    index: number;
    count: number;
    video: boolean;
    /** Audio-only print: no raster to show, a transport instead. */
    audio?: boolean;
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
  }>(),
  {
    audio: false,
    source: "host",
    target: null,
    cacheKey: null,
    hostLabel: null,
    canReveal: false,
    isSequence: false,
    canEditSequence: false,
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

const confirmingDelete = ref(false);
const exportOpen = ref(false);
const exportBusy = ref(false);
const saveBusy = ref(false);
const exportError = ref("");
const exportCapabilities = ref<VideoExportCapabilities>(DEFAULT_VIDEO_EXPORT_CAPABILITIES);
watch(
  () => props.item.filename,
  () => (confirmingDelete.value = false),
);

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
      galleryMediaPath(props.item.filename, props.source),
      target
        ? {
            fetchImage: async (p) => {
              const { apiFetchTo } = await import("../../lib/api/client");
              return new Uint8Array(await (await apiFetchTo(target, p)).arrayBuffer());
            },
          }
        : undefined,
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
      disabled: props.video || props.audio,
      action: () => void copyImage(),
    },
    {
      label: "Copy file path",
      action: () =>
        void copyLocalOutputPath(props.item.filename)
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
    await ipc.revealOutputFile(props.item.filename);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

function onDelete() {
  if (!confirmingDelete.value) {
    confirmingDelete.value = true;
    return;
  }
  emit("delete");
  toasts.push("Deleted print");
}

async function saveMedia() {
  if (saveBusy.value) return;
  saveBusy.value = true;
  try {
    const saved = await saveGalleryMedia(props.target, props.item.filename);
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
      videoExportFilename(props.item.filename, options.format),
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
              :path="galleryMediaPath(item.filename, source, true)"
              :target="target"
              :cache-key="cacheKey"
              :alt="meta.prompt"
              class="!object-contain"
            />
            <AuthedMedia
              :path="galleryMediaPath(item.filename, source)"
              :target="target"
              :cache-key="cacheKey"
              audio
              controls
              :alt="meta.prompt"
            />
          </div>
          <AuthedMedia
            v-else
            :path="galleryMediaPath(item.filename, source)"
            :target="target"
            :cache-key="cacheKey"
            :video="video"
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
          <span class="data-mono block truncate text-caption text-ink-3" :title="item.filename">
            {{ item.filename }}
          </span>
          <p data-selectable class="mt-3 text-body text-ink" :title="meta.prompt">
            {{ meta.prompt }}
          </p>
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
          <button
            type="button"
            class="border-edge h-8 flex-1 rounded-control border text-caption transition-colors duration-100"
            :class="
              confirmingDelete
                ? 'border-stop bg-stop font-semibold text-on-accent'
                : 'text-ink-2 hover:text-stop'
            "
            @blur="confirmingDelete = false"
            @click="onDelete"
          >
            {{ confirmingDelete ? "Delete? Can't be undone." : "Delete" }}
          </button>
        </div>
      </aside>
    </div>
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
