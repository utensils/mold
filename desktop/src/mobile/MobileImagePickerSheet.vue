<script setup lang="ts">
import { computed, ref, watch } from "vue";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import type { GalleryImage } from "../lib/api/types";
import { galleryMediaPath } from "../lib/gallery/media";
import { MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES } from "../lib/generateValidation";
import { blobToBase64, fileToBase64, isStillImageFile } from "../lib/image";

export interface MobilePickedImage {
  filename: string;
  base64: string;
}

export interface MobileGallerySource {
  id: string;
  label: string;
  target: ApiTarget;
}

interface MobileGalleryEntry {
  image: GalleryImage;
  source: MobileGallerySource;
}

const props = withDefaults(
  defineProps<{
    open: boolean;
    target: ApiTarget | null;
    gallerySources?: MobileGallerySource[];
    title?: string;
    multiple?: boolean;
    maxBytes?: number;
    oversizeMessage?: string;
  }>(),
  {
    title: "Opening image",
    multiple: false,
    gallerySources: () => [],
    maxBytes: MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
    oversizeMessage: "The opening image must be 45 MiB or smaller on this phone.",
  },
);

const emit = defineEmits<{
  pick: [image: MobilePickedImage];
  "pick-many": [images: MobilePickedImage[]];
  close: [];
}>();

const tab = ref<"file" | "gallery">("file");
const input = ref<HTMLInputElement | null>(null);
const entries = ref<MobileGalleryEntry[]>([]);
const loading = ref(false);
const picking = ref(false);
const error = ref("");
const selectedGallery = ref<MobileGalleryEntry[]>([]);

const sources = computed<MobileGallerySource[]>(() => {
  if (props.gallerySources.length) return props.gallerySources;
  return props.target
    ? [{ id: props.target.baseUrl, label: "Selected machine", target: props.target }]
    : [];
});
const sourceSignature = computed(() =>
  sources.value
    .map(
      (source) =>
        `${source.id}\u0000${source.label}\u0000${source.target.baseUrl}\u0000${source.target.apiKey ?? ""}`,
    )
    .join("\u0001"),
);
const stillEntries = computed(() =>
  entries.value.filter((entry) => isStillImageFile(entry.image.filename)),
);
const showHostLabels = computed(() => sources.value.length > 1);

watch(
  [() => props.open, sourceSignature] as const,
  async ([open], _previous, onCleanup) => {
    let active = true;
    const controllers: AbortController[] = [];
    onCleanup(() => {
      active = false;
      for (const controller of controllers) controller.abort();
    });
    if (!open || sources.value.length === 0) {
      entries.value = [];
      selectedGallery.value = [];
      loading.value = false;
      return;
    }
    const requestedSources = [...sources.value];
    // Authority changes invalidate every captured target immediately. Never
    // leave a tile with an old key/instance clickable while its replacement
    // fleet is loading.
    entries.value = [];
    selectedGallery.value = [];
    error.value = "";
    loading.value = true;
    try {
      const results = await Promise.allSettled(
        requestedSources.map(async (source) => {
          const controller = new AbortController();
          controllers.push(controller);
          const timeout = setTimeout(() => controller.abort(), 10_000);
          try {
            const images = await apiJsonTo<GalleryImage[]>(source.target, "/api/gallery", {
              signal: controller.signal,
            });
            if (active) {
              entries.value = [
                ...entries.value,
                ...images.map((image) => ({ image, source })),
              ].sort((left, right) => right.image.timestamp - left.image.timestamp);
              // A healthy host must become usable without waiting for a slow
              // peer. Keep the remaining requests alive so their tiles join
              // the same sorted grid when they arrive.
              if (entries.value.length > 0) loading.value = false;
            }
            return { source, images };
          } finally {
            clearTimeout(timeout);
          }
        }),
      );
      if (!active) return;
      const failure = results.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failure && entries.value.length === 0) {
        error.value =
          failure.reason instanceof Error ? failure.reason.message : String(failure.reason);
      }
    } catch (cause) {
      if (!active) return;
      entries.value = [];
      error.value = cause instanceof Error ? cause.message : String(cause);
    } finally {
      if (active) loading.value = false;
    }
  },
  { immediate: true },
);

async function chooseFile(event: Event): Promise<void> {
  const element = event.target as HTMLInputElement;
  const files = [...(element.files ?? [])];
  element.value = "";
  if (!files.length) return;
  const selected = props.multiple ? files : files.slice(0, 1);
  if (
    selected.some(
      (file) =>
        !(
          file.type === "image/png" ||
          file.type === "image/jpeg" ||
          (!file.type && isStillImageFile(file.name))
        ),
    )
  ) {
    error.value = "Only PNG or JPEG photos can be used here.";
    return;
  }
  if (selected.some((file) => file.size === 0)) {
    error.value = "That image is empty.";
    return;
  }
  if (selected.reduce((total, file) => total + file.size, 0) > props.maxBytes) {
    error.value = props.oversizeMessage;
    return;
  }
  error.value = "";
  const picked = await Promise.all(
    selected.map(async (file) => ({
      filename: file.name,
      base64: await fileToBase64(file),
    })),
  );
  if (props.multiple) emit("pick-many", picked);
  else emit("pick", picked[0]!);
  emit("close");
}

async function chooseGallery(entry: MobileGalleryEntry): Promise<void> {
  if (props.multiple) {
    const key = `${entry.source.id}:${entry.image.filename}`;
    const index = selectedGallery.value.findIndex(
      (selected) => `${selected.source.id}:${selected.image.filename}` === key,
    );
    selectedGallery.value =
      index >= 0
        ? selectedGallery.value.filter((_, item) => item !== index)
        : [...selectedGallery.value, entry];
    return;
  }
  await emitGalleryEntries([entry]);
}

async function emitGalleryEntries(selected: MobileGalleryEntry[]): Promise<void> {
  if (picking.value) return;
  picking.value = true;
  error.value = "";
  try {
    let totalBytes = 0;
    const picked: MobilePickedImage[] = [];
    for (const entry of selected) {
      const response = await apiFetchTo(
        entry.source.target,
        galleryMediaPath(entry.image.filename, "host"),
      );
      const declared = Number(response.headers?.get("content-length") ?? Number.NaN);
      if (Number.isFinite(declared) && totalBytes + declared > props.maxBytes) {
        throw new Error(props.oversizeMessage);
      }
      const blob = await response.blob();
      if (blob.size === 0) throw new Error("That gallery image is empty.");
      totalBytes += blob.size;
      if (totalBytes > props.maxBytes) throw new Error(props.oversizeMessage);
      picked.push({
        filename: entry.image.filename,
        base64: await blobToBase64(blob),
      });
    }
    if (props.multiple) emit("pick-many", picked);
    else emit("pick", picked[0]!);
    emit("close");
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    picking.value = false;
  }
}

function galleryEntrySelected(entry: MobileGalleryEntry): boolean {
  return selectedGallery.value.some(
    (selected) =>
      selected.source.id === entry.source.id && selected.image.filename === entry.image.filename,
  );
}
</script>

<template>
  <div
    v-if="open"
    class="mobile-image-picker"
    role="dialog"
    aria-modal="true"
    :aria-label="title"
    data-test="mobile-image-picker"
  >
    <button
      type="button"
      class="mobile-image-picker-backdrop"
      :aria-label="`Close ${title.toLowerCase()} picker`"
      @click="emit('close')"
    />
    <section class="mobile-image-picker-panel">
      <header>
        <div>
          <strong>{{ title }}</strong>
          <p>Choose a PNG or JPEG from this phone or any available machine.</p>
        </div>
        <button
          type="button"
          class="secondary-button"
          data-test="mobile-image-picker-close"
          @click="emit('close')"
        >
          Done
        </button>
      </header>

      <div class="mobile-image-picker-tabs" role="tablist" aria-label="Image source">
        <button
          type="button"
          role="tab"
          :aria-selected="tab === 'file'"
          data-test="mobile-image-picker-file-tab"
          @click="tab = 'file'"
        >
          Local file
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="tab === 'gallery'"
          data-test="mobile-image-picker-gallery-tab"
          @click="tab = 'gallery'"
        >
          Gallery
        </button>
      </div>

      <div v-if="tab === 'file'" class="mobile-image-picker-file">
        <input
          ref="input"
          hidden
          type="file"
          accept="image/png,image/jpeg"
          :multiple="multiple"
          data-test="mobile-image-picker-input"
          @change="chooseFile"
        />
        <button
          type="button"
          class="secondary-button"
          data-test="mobile-image-picker-browse"
          @click="input?.click()"
        >
          Choose local image
        </button>
      </div>

      <div v-else class="mobile-image-picker-gallery">
        <p v-if="loading">Loading gallery…</p>
        <p v-else-if="!sources.length">Connect a machine first.</p>
        <p v-else-if="!stillEntries.length && !error">No PNG or JPEG prints yet.</p>
        <div v-else class="mobile-image-picker-grid">
          <button
            v-for="entry in stillEntries"
            :key="`${entry.source.id}:${entry.image.filename}`"
            type="button"
            :aria-label="`Use ${entry.image.filename} from ${entry.source.label}`"
            :disabled="picking"
            :aria-pressed="multiple ? galleryEntrySelected(entry) : undefined"
            :class="{ 'is-selected': galleryEntrySelected(entry) }"
            data-test="mobile-image-picker-gallery-item"
            @click="chooseGallery(entry)"
          >
            <AuthedMedia
              :path="galleryMediaPath(entry.image.filename, 'host', true)"
              :target="entry.source.target"
              :cache-key="entry.source.id"
              :alt="entry.image.filename"
            />
            <span
              v-if="showHostLabels"
              class="mobile-image-picker-host"
              data-test="mobile-image-picker-host"
            >
              {{ entry.source.label }}
            </span>
            <span
              v-if="multiple && galleryEntrySelected(entry)"
              class="mobile-image-picker-order"
              aria-hidden="true"
            >
              {{
                selectedGallery.findIndex(
                  (selected) =>
                    selected.source.id === entry.source.id &&
                    selected.image.filename === entry.image.filename,
                ) + 1
              }}
            </span>
          </button>
        </div>
        <div
          v-if="multiple && selectedGallery.length"
          class="mobile-image-picker-selection"
          data-test="mobile-image-picker-selection"
        >
          <span>{{ selectedGallery.length }} selected · kept in this order</span>
          <button
            type="button"
            class="primary-button"
            data-test="mobile-image-picker-confirm"
            :disabled="picking"
            @click="emitGalleryEntries(selectedGallery)"
          >
            Add selected
          </button>
        </div>
      </div>

      <p v-if="error" class="mobile-image-picker-error" role="alert">{{ error }}</p>
    </section>
  </div>
</template>

<style scoped>
.mobile-image-picker {
  position: fixed;
  inset: 0;
  z-index: 80;
}

.mobile-image-picker-backdrop {
  position: absolute;
  inset: 0;
  width: 100%;
  border: 0;
  background: rgb(0 0 0 / 62%);
}

.mobile-image-picker-panel {
  position: absolute;
  inset: auto 0 0;
  display: grid;
  max-height: min(82dvh, 720px);
  gap: 14px;
  overflow: auto;
  padding: 18px 16px calc(18px + env(safe-area-inset-bottom));
  border: 1px solid var(--edge);
  border-radius: 20px 20px 0 0;
  background: var(--bench);
}

.mobile-image-picker-panel header,
.mobile-image-picker-tabs {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.mobile-image-picker-panel header p {
  margin: 4px 0 0;
  color: var(--ink-3);
  font-size: 13px;
}

.mobile-image-picker-tabs {
  padding: 3px;
  border: 1px solid var(--edge);
  border-radius: 12px;
  background: var(--bath);
}

.mobile-image-picker-tabs button {
  min-height: 44px;
  flex: 1;
  border-radius: 9px;
}

.mobile-image-picker-tabs button[aria-selected="true"] {
  background: var(--safelight);
  color: var(--on-accent);
}

.mobile-image-picker-grid button.is-selected {
  outline: 2px solid var(--safelight);
  outline-offset: -2px;
}

.mobile-image-picker-order {
  position: absolute;
  top: 6px;
  right: 6px;
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--safelight);
  color: var(--on-accent);
  font-weight: 700;
}

.mobile-image-picker-selection {
  position: sticky;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--edge);
  background: var(--bench);
}

.mobile-image-picker-file {
  display: grid;
  min-height: 160px;
  place-items: center;
  border: 1px dashed var(--control-edge);
  border-radius: 14px;
}

.mobile-image-picker-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.mobile-image-picker-grid button {
  position: relative;
  min-height: 44px;
  overflow: hidden;
  aspect-ratio: 1;
  border: 1px solid var(--edge);
  border-radius: 10px;
  background: var(--print-surface);
}

.mobile-image-picker-host {
  position: absolute;
  inset: auto 4px 4px;
  overflow: hidden;
  padding: 2px 5px;
  border-radius: 6px;
  background: rgb(0 0 0 / 68%);
  color: white;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.mobile-image-picker-grid :deep(img) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.mobile-image-picker-error {
  margin: 0;
  color: var(--stop);
  font-size: 14px;
}
</style>
