<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import AuthedMedia from "./AuthedMedia.vue";
import { mediaPath } from "../../lib/gallery/media";
import { ipc } from "../../lib/ipc";
import { useComposerStore } from "../../stores/composer";
import { useToastStore } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import type { GalleryImage } from "../../lib/api/types";

const props = defineProps<{
  item: GalleryImage;
  index: number;
  count: number;
  video: boolean;
}>();
const emit = defineEmits<{ close: []; prev: []; next: []; delete: [] }>();

const router = useRouter();
const composer = useComposerStore();
const toasts = useToastStore();
const ui = useUiStore();

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
  const m = meta.value;
  composer.set({
    prompt: m.prompt,
    model: m.model,
    seed: m.seed,
    width: m.width,
    height: m.height,
    steps: m.steps,
    guidance: m.guidance,
  });
  emit("close");
  void router.push("/generate");
}

async function copy(text: string) {
  await navigator.clipboard.writeText(text);
  toasts.push("Copied");
}

// Reveal in Finder only exists when the engine writes to this disk.
const canReveal = ref(false);
void ipc.getOutputDir().then((dir) => (canReveal.value = dir !== null));

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
      class="absolute top-4 left-4 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-black/55 text-body-lg text-rebate transition-colors duration-100 hover:bg-black/80"
      title="Close (Esc)"
      aria-label="Close"
      @click="emit('close')"
    >
      ✕
    </button>
    <div class="m-6 flex min-w-0 flex-1 flex-col">
      <div
        class="relative min-h-0 flex-1 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
      >
        <AuthedMedia
          :path="mediaPath(item.filename)"
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
        <div v-if="meta.lora" class="flex justify-between gap-2">
          <dt class="text-caption text-ink-3">LoRA</dt>
          <dd class="data-mono truncate text-caption text-ink">
            {{ meta.lora }} {{ meta.lora_scale?.toFixed(2) ?? "" }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-caption text-ink-3">Created</dt>
          <dd class="text-caption text-ink">{{ when }}</dd>
        </div>
      </dl>
      <span v-if="item.metadata_synthetic" class="edge-code mt-2">SYNTHETIC METADATA</span>

      <div class="flex-1" />

      <button
        type="button"
        class="h-8 w-full rounded-control bg-safelight text-body font-semibold text-[#141110] transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
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
        Reveal in Finder
      </button>
      <button
        type="button"
        class="border-edge mt-2 h-8 w-full rounded-control border text-body transition-colors duration-100"
        :class="
          confirmingDelete
            ? 'border-stop bg-stop font-semibold text-[#141110]'
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
