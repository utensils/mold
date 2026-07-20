<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import AuthedMedia from "./AuthedMedia.vue";
import { galleryMediaPath, type GallerySource } from "../../lib/gallery/media";
import { ipc } from "../../lib/ipc";
import { useComposerStore } from "../../stores/composer";
import { useToastStore } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { copyImageBytesToClipboard } from "../../lib/clipboard";
import { formatBytes, formatScheduler } from "../../lib/format";
import type { ApiTarget } from "../../lib/api/client";
import type { GalleryImage } from "../../lib/api/types";

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    index: number;
    count: number;
    video: boolean;
    source?: GallerySource;
    /** Origin host to fetch media from; null = the primary connection. */
    target?: ApiTarget | null;
    /** Blob-cache bucket, usually the origin host id. */
    cacheKey?: string | null;
    /** Origin host's friendly name for the metadata block. */
    hostLabel?: string | null;
    canReveal?: boolean;
  }>(),
  { source: "host", target: null, cacheKey: null, hostLabel: null, canReveal: false },
);
const emit = defineEmits<{ close: []; prev: []; next: []; delete: [] }>();

const router = useRouter();
const composer = useComposerStore();
const toasts = useToastStore();
const ui = useUiStore();
const contextMenu = useContextMenuStore();

// ⇧⌘C copies the seed while the lightbox is open.
watch(
  () => ui.copySeedTick,
  () => void copy(String(props.item.metadata.seed)),
);

const confirmingDelete = ref(false);
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

/** Full LoRA stack; a legacy single `lora`/`lora_scale` pair becomes one row. */
const loraStack = computed(() => {
  const m = meta.value;
  if (m.loras?.length) return m.loras;
  return m.lora ? [{ path: m.lora, scale: m.lora_scale ?? 1.0 }] : [];
});

const schedulerName = computed(() => formatScheduler(meta.value.scheduler));
const frames = computed(() => meta.value.frames ?? meta.value.video_frames ?? null);
const fps = computed(() => meta.value.fps ?? meta.value.video_fps ?? null);
const fileFormat = computed(() => props.item.format ?? meta.value.output_format ?? null);
const fileSize = computed(() =>
  props.item.size_bytes != null ? formatBytes(props.item.size_bytes) : null,
);

const edgeCode = computed(() => {
  const m = meta.value;
  return [
    m.model.toUpperCase().replace(":", "·"),
    `S ${m.seed}`,
    `${m.steps}/${m.steps}`,
    `${m.width}×${m.height}`,
  ].join("  ");
});

const when = computed(() =>
  new Date(props.item.timestamp * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }),
);

function reuseSettings() {
  // Ship the full metadata — `applyPrefillToForm` restores every serialized
  // knob (negative prompt, LoRA stack, scheduler, strength, video params, …)
  // and prefers the pre-upscale generation canvas over the raster size.
  composer.set({ metadata: meta.value });
  emit("close");
  void router.push("/generate");
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
    { label: "Copy image", disabled: props.video, action: () => void copyImage() },
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
</script>

<template>
  <div
    class="fixed inset-0 z-40 flex bg-black/70"
    role="dialog"
    aria-modal="true"
    :aria-label="`Print ${index + 1} of ${count}`"
    @click.self="emit('close')"
  >
    <button
      ref="closeBtn"
      type="button"
      class="absolute top-4 left-4 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-black/55 text-body-lg text-on-media transition-colors duration-100 hover:bg-black/80"
      title="Close (Esc)"
      aria-label="Close"
      @click="emit('close')"
    >
      ✕
    </button>
    <div class="m-6 flex min-w-0 flex-1 flex-col">
      <div
        data-test="lightbox-media"
        class="relative min-h-0 flex-1 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
        @contextmenu="contextMenu.open($event, imageMenu())"
      >
        <AuthedMedia
          :path="galleryMediaPath(item.filename, source)"
          :target="target"
          :cache-key="cacheKey"
          :video="video"
          :controls="video"
          :alt="meta.prompt"
          class="!object-contain"
        />
      </div>
      <div class="mt-2 flex items-center justify-between">
        <span class="edge-code">{{ edgeCode }}</span>
        <span class="data-mono text-caption text-ink-3">{{ index + 1 }} / {{ count }}</span>
      </div>
    </div>

    <aside
      class="border-edge my-6 mr-6 flex w-72 shrink-0 flex-col rounded-chrome border bg-bench p-4"
    >
      <span class="data-mono truncate text-caption text-ink-3" :title="item.filename">
        {{ item.filename }}
      </span>
      <p
        data-selectable
        class="mt-3 max-h-40 overflow-y-auto text-body text-ink"
        :title="meta.prompt"
      >
        {{ meta.prompt }}
      </p>
      <p
        v-if="meta.original_prompt"
        data-test="lightbox-original"
        data-selectable
        class="mt-2 max-h-24 overflow-y-auto text-caption text-ink-2"
        :title="meta.original_prompt"
      >
        <span class="text-ink-3">Original</span> {{ meta.original_prompt }}
      </p>
      <p
        v-if="meta.negative_prompt"
        data-test="lightbox-negative"
        data-selectable
        class="mt-2 max-h-24 overflow-y-auto text-caption text-ink-2"
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

      <dl class="mt-4 space-y-1.5">
        <div class="flex justify-between gap-2">
          <dt class="text-caption text-ink-3">Model</dt>
          <dd class="data-mono truncate text-caption text-ink">{{ meta.model }}</dd>
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
          <dt class="text-caption text-ink-3">Size</dt>
          <dd class="data-mono text-caption text-ink">{{ meta.width }}×{{ meta.height }}</dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-caption text-ink-3">Steps · guidance</dt>
          <dd class="data-mono text-caption text-ink">
            {{ meta.steps }} · {{ meta.guidance.toFixed(1) }}
          </dd>
        </div>
        <div v-if="schedulerName" class="flex justify-between gap-2" data-test="lightbox-scheduler">
          <dt class="text-caption text-ink-3">Scheduler</dt>
          <dd class="data-mono text-caption text-ink">{{ schedulerName }}</dd>
        </div>
        <div v-if="meta.cfg_plus" class="flex justify-between gap-2" data-test="lightbox-cfg-plus">
          <dt class="text-caption text-ink-3">CFG++</dt>
          <dd class="data-mono text-caption text-ink">on</dd>
        </div>
        <div
          v-if="meta.strength != null"
          class="flex justify-between gap-2"
          data-test="lightbox-strength"
        >
          <dt class="text-caption text-ink-3">img2img strength</dt>
          <dd class="data-mono text-caption text-ink">{{ meta.strength.toFixed(2) }}</dd>
        </div>
        <div v-if="frames" class="flex justify-between gap-2" data-test="lightbox-video">
          <dt class="text-caption text-ink-3">Frames</dt>
          <dd class="data-mono text-caption text-ink">
            {{ frames }}<template v-if="fps"> · {{ fps }} fps</template>
          </dd>
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
          <dt class="text-caption text-ink-3">Available on</dt>
          <dd class="data-mono truncate text-caption text-ink">{{ hostLabel }}</dd>
        </div>
      </dl>
      <span
        v-if="meta.version"
        class="data-mono mt-2 text-caption text-ink-3"
        data-test="lightbox-version"
      >
        mold {{ meta.version }}
      </span>
      <span v-if="item.metadata_synthetic" class="edge-code mt-2">SYNTHETIC METADATA</span>

      <div class="flex-1" />

      <button
        type="button"
        class="h-8 w-full rounded-control bg-safelight text-body font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
        @click="reuseSettings"
      >
        Reuse settings
      </button>
      <button
        v-if="canReveal"
        type="button"
        class="border-edge mt-2 h-8 w-full rounded-control border text-body text-ink-2 transition-colors duration-100 hover:text-ink"
        @click="reveal"
      >
        Reveal in file manager
      </button>
      <button
        type="button"
        class="border-edge mt-2 h-8 w-full rounded-control border text-body transition-colors duration-100"
        :class="
          confirmingDelete
            ? 'border-stop bg-stop font-semibold text-on-accent'
            : 'text-ink-2 hover:text-stop'
        "
        @blur="confirmingDelete = false"
        @click="onDelete"
      >
        {{ confirmingDelete ? "Delete print? This can't be undone." : "Delete" }}
      </button>
    </aside>

    <button
      type="button"
      class="absolute top-1/2 left-2 -translate-y-1/2 rounded-control px-2 py-4 text-ink-2 hover:text-ink"
      :disabled="index === 0"
      aria-label="Previous print"
      @click="emit('prev')"
    >
      ←
    </button>
    <button
      type="button"
      class="absolute top-1/2 right-80 -translate-y-1/2 rounded-control px-2 py-4 text-ink-2 hover:text-ink"
      :disabled="index === count - 1"
      aria-label="Next print"
      @click="emit('next')"
    >
      →
    </button>
  </div>
</template>
