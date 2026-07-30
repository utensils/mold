<script setup lang="ts">
/*
 * Source-image picker (spec §05 overlays, §06 Create). Upload from disk or
 * reuse a gallery print.
 *
 * Mold Studio migration: this was a hand-rolled `fixed inset-0` panel with an
 * emoji title, Tailwind utility colors, and no Escape handling. It now runs on
 * the @ui panels — ModalPanel at desk width, full SheetPanel on phones, which
 * is how ModelDetailDrawer and the Advanced sheet choose — plus
 * `useOverlayFocus` for the Tab trap and opener restoration the kit leaves to
 * the host.
 */
import { computed, markRaw, onBeforeUnmount, onMounted, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { listGallery, thumbnailUrl, imageUrl } from "../api";
import { blobToBase64 } from "../lib/base64";
import { useOverlayFocus } from "../composables/useOverlayFocus";
import type { GalleryImage, SourceImageState } from "../types";

const props = withDefaults(
  defineProps<{ open: boolean; title?: string; multiple?: boolean }>(),
  {
    title: "Source image",
    multiple: true,
  },
);
const emit = defineEmits<{
  (e: "pick", v: SourceImageState[]): void;
  (e: "close"): void;
}>();

const tab = ref<"upload" | "gallery">("upload");
const entries = ref<GalleryImage[]>([]);
const stillEntries = computed(() =>
  entries.value.filter((entry) =>
    /\.(png|jpe?g)$/i.test(entry.filename.trim()),
  ),
);
const loading = ref(false);
const error = ref<string | null>(null);
const uploadError = ref<string | null>(null);
const dragging = ref(false);
const fileInput = ref<HTMLInputElement | null>(null);

const host = ref<HTMLElement | null>(null);
const isOpen = computed(() => props.open);
const { onKeydown } = useOverlayFocus(isOpen, host, () => emit("close"));

watch(
  () => props.open,
  (open) => {
    if (open) return;
    uploadError.value = null;
    dragging.value = false;
    if (fileInput.value) fileInput.value.value = "";
  },
);

const tabOptions = [
  { value: "upload" as const, label: "Upload" },
  { value: "gallery" as const, label: "From gallery" },
];

// Phone surfaces get the full-screen sheet; everything else the centered modal.
const isPhone = ref(false);
let mq: MediaQueryList | null = null;
function onMediaChange(event: MediaQueryListEvent) {
  isPhone.value = event.matches;
}

const panelComponent = computed(() =>
  markRaw(isPhone.value ? SheetPanel : ModalPanel),
);
const panelProps = computed(() =>
  isPhone.value
    ? { variant: "full" as const, title: props.title }
    : { width: 720, label: props.title },
);

onMounted(async () => {
  mq = window.matchMedia?.("(max-width: 639px)") ?? null;
  if (mq) {
    isPhone.value = mq.matches;
    mq.addEventListener?.("change", onMediaChange);
  }

  loading.value = true;
  try {
    entries.value = await listGallery();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});

onBeforeUnmount(() => mq?.removeEventListener?.("change", onMediaChange));

async function emitFiles(files: File[]) {
  if (!files.length) return;
  const images = files.filter(
    (file) =>
      file.type === "image/png" ||
      file.type === "image/jpeg" ||
      (!file.type && /\.(png|jpe?g)$/i.test(file.name.trim())),
  );
  if (!images.length) {
    uploadError.value = "Only PNG or JPEG images can be used here.";
    return;
  }
  uploadError.value = null;
  const selected = props.multiple ? images : images.slice(0, 1);
  const picked = await Promise.all(
    selected.map(async (file) => ({
      kind: "upload" as const,
      filename: file.name,
      base64: await blobToBase64(file),
    })),
  );
  emit("pick", picked);
  emit("close");
}

async function onFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  try {
    await emitFiles(Array.from(input.files ?? []));
  } finally {
    // Let selecting the same file fire `change` again, including after a
    // validation error where the picker remains open.
    input.value = "";
  }
}

async function onDrop(event: DragEvent) {
  event.preventDefault();
  dragging.value = false;
  await emitFiles(Array.from(event.dataTransfer?.files ?? []));
}

function onDragEnter(event: DragEvent) {
  event.preventDefault();
  dragging.value = true;
}

function onDragLeave(event: DragEvent) {
  // Only clear when the pointer actually leaves the drop target — dragleave
  // also fires when it crosses onto a child node.
  const next = event.relatedTarget;
  const target = event.currentTarget as HTMLElement;
  if (next instanceof Node && target.contains(next)) return;
  dragging.value = false;
}

async function pickFromGallery(item: GalleryImage) {
  const res = await fetch(imageUrl(item.filename));
  if (!res.ok) {
    error.value = `Fetch failed: ${res.status}`;
    return;
  }
  const blob = await res.blob();
  const b64 = await blobToBase64(blob);
  emit("pick", [{ kind: "gallery", filename: item.filename, base64: b64 }]);
  emit("close");
}
</script>

<template>
  <!-- Viewport-fixed host: the @ui panels are `position: absolute; inset: 0`
       so they fill their nearest positioned ancestor. Create is a long
       scrolling column, so mounting inline would draw the panel at the top of
       the document instead of where the user is looking. Same host pattern as
       ModelDetailDrawer. -->
  <div
    v-if="open"
    ref="host"
    class="ip-host"
    data-test="image-picker-host"
    @keydown="onKeydown"
  >
    <component
      :is="panelComponent"
      v-bind="panelProps"
      :open="open"
      @close="emit('close')"
    >
      <div class="ip">
        <!-- SheetPanel draws its own header + close control; the modal needs
             one. -->
        <header v-if="!isPhone" class="ip__head">
          <h2 class="ip__title">
            <Icon name="image" :size="17" />
            <span>{{ title }}</span>
          </h2>
          <button
            type="button"
            class="ip__close"
            aria-label="Close"
            data-test="image-picker-close"
            @click="emit('close')"
          >
            <Icon name="close" :size="15" />
          </button>
        </header>

        <SegmentedControl
          v-model="tab"
          :options="tabOptions"
          label="Image source"
          class="ip__seg"
        />

        <div v-if="tab === 'upload'" class="ip__body">
          <div
            class="ip__drop"
            :data-dragging="dragging ? 'true' : undefined"
            data-test="image-picker-drop"
            @dragover.prevent
            @dragenter="onDragEnter"
            @dragleave="onDragLeave"
            @drop="onDrop"
          >
            <button
              type="button"
              class="ip__drop-btn"
              data-test="image-picker-browse"
              @click="fileInput?.click()"
            >
              <span class="ip__drop-icon" aria-hidden="true">
                <Icon name="upload" :size="22" />
              </span>
              <span class="ip__drop-lead">Drop an image here</span>
              <span class="ip__drop-sub">or click to browse</span>
            </button>
            <input
              ref="fileInput"
              type="file"
              accept="image/png,image/jpeg"
              :multiple="multiple"
              class="ip__file"
              tabindex="-1"
              aria-hidden="true"
              @change="onFiles"
            />
          </div>
          <p v-if="uploadError" class="ip__status ip__status--error">
            {{ uploadError }}
          </p>
        </div>

        <div v-else class="ip__body">
          <p v-if="loading" class="ip__status">loading gallery…</p>
          <p v-else-if="error" class="ip__status ip__status--error">
            {{ error }}
          </p>
          <p v-else-if="!stillEntries.length" class="ip__status">
            no PNG or JPEG images available
          </p>
          <ul v-else class="ip__grid">
            <li v-for="item in stillEntries" :key="item.filename">
              <button
                type="button"
                class="ip__tile"
                :title="item.filename"
                @click="pickFromGallery(item)"
              >
                <img
                  :src="thumbnailUrl(item.filename)"
                  :alt="item.filename"
                  loading="lazy"
                />
              </button>
            </li>
          </ul>
        </div>
      </div>
    </component>
  </div>
</template>

<style scoped>
.ip-host {
  position: fixed;
  inset: 0;
  z-index: 40;
}

.ip {
  display: flex;
  flex-direction: column;
  gap: 14px;
  color: var(--rebate);
}

.ip__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.ip__title {
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 0;
  font-family: var(--f-display);
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.ip__title svg {
  color: var(--ink-3);
}

.ip__close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  transition:
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.ip__close:hover {
  color: var(--rebate);
  background: var(--surface);
}

.ip__body {
  min-height: 0;
  max-height: min(56vh, 460px);
  overflow-y: auto;
}

.ip__drop {
  border: 1px dashed var(--ce);
  border-radius: var(--radius-card);
  background: var(--bath);
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.ip__drop[data-dragging="true"] {
  border-color: var(--safelight);
  border-style: solid;
  background: var(--sel-bg);
}

.ip__drop-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  min-height: 184px;
  padding: 24px;
  border: 0;
  border-radius: inherit;
  background: transparent;
  color: var(--ink-2);
  font-family: var(--f-body);
  cursor: pointer;
}

.ip__drop-btn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: -4px;
}

.ip__drop-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  margin-bottom: 4px;
  border-radius: var(--radius-pill);
  background: var(--surface);
  color: var(--ink-3);
}

.ip__drop[data-dragging="true"] .ip__drop-icon {
  color: var(--safelight);
}

.ip__drop-lead {
  font-size: 14px;
  font-weight: 600;
  color: var(--rebate);
}

.ip__drop-sub {
  font-family: var(--f-mono);
  font-size: 11px;
  letter-spacing: 0.02em;
  color: var(--ink-3);
}

/* Visually hidden, not display:none — a hidden input cannot be opened by
   .click() in every engine, and Safari needs it laid out. */
.ip__file {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.ip__status {
  margin: 0;
  padding: 24px 0;
  font-family: var(--f-mono);
  font-size: 12px;
  letter-spacing: 0.02em;
  color: var(--ink-3);
}

.ip__status--error {
  color: var(--stop);
}

.ip__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

@media (min-width: 640px) {
  .ip__grid {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }
}

.ip__tile {
  display: block;
  width: 100%;
  padding: 0;
  overflow: hidden;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control);
  background: var(--bath);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    opacity var(--dur-quick) var(--ease);
}

.ip__tile:hover {
  border-color: var(--safelight);
  opacity: 0.85;
}

.ip__tile:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ip__tile img {
  display: block;
  width: 100%;
  height: 96px;
  object-fit: cover;
}
</style>
