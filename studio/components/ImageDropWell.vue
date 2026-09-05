<script setup lang="ts">
import { computed, ref } from "vue";

/**
 * The one source-image well shared by every Studio surface: a drop zone that
 * also opens a file picker on click/Enter/Space, an optional gallery escape
 * hatch, and a thumbnail preview with an explicit remove control. Surfaces own
 * what a picked File means (validation, probing, state) — this component only
 * acquires media and reports intent.
 */
const props = withDefaults(
  defineProps<{
    /** Raw base64 (no data-URI prefix) of the attached image, if any. */
    image?: string | null;
    mimeType?: string | null;
    /** Provenance label; with no `image` bytes it renders a reattach state. */
    filename?: string | null;
    placeholder?: string;
    accept?: string;
    /** Disables everything, removal included. */
    disabled?: boolean;
    /** Disables acquiring new media (drop, pick, gallery) but keeps removal. */
    pickDisabled?: boolean;
    required?: boolean;
    /** Renders the "from gallery" action; the host opens its own picker. */
    gallery?: boolean;
    galleryLabel?: string;
    alt?: string;
    /** Prefix for data-test hooks: `<id>-well|-file|-gallery|-remove`. */
    testId?: string;
    /** iPhone invariant: keeps every tappable control at least 44pt tall. */
    touchFriendly?: boolean;
    /** Platform-specific minimum touch target; Android uses 48dp. */
    touchTargetSize?: number;
    /** Delegates acquisition to a native shell instead of the hidden input. */
    nativePicker?: boolean;
    /**
     * Which well this is, for `imageDropRouting`. Rendered as
     * `data-drop-target` on the well root so a shell that intercepts the OS
     * drag (Tauri swallows it before any HTML5 `drop`) can name the well
     * under the cursor with `elementFromPoint(...).closest(…)` instead of
     * routing by model capability. Attachment STRIPS carry the attribute on
     * their own container — they are not this component.
     */
    dropTarget?: string | null;
  }>(),
  {
    image: null,
    mimeType: null,
    filename: null,
    placeholder: "Drop an image or click to pick",
    accept: "image/*",
    disabled: false,
    pickDisabled: false,
    required: false,
    gallery: false,
    galleryLabel: "Choose from gallery…",
    alt: "Attached image",
    testId: "image-well",
    touchFriendly: false,
    touchTargetSize: 44,
    nativePicker: false,
    dropTarget: null,
  },
);

const emit = defineEmits<{
  file: [file: File];
  gallery: [];
  clear: [];
  pick: [];
}>();

const input = ref<HTMLInputElement | null>(null);
const dragOver = ref(false);

const inert = computed(() => props.disabled || props.pickDisabled);
const previewUrl = computed(() =>
  props.image
    ? `data:${props.mimeType || "image/png"};base64,${props.image}`
    : null,
);
/** Attached provenance whose bytes were stripped for persistence. */
const needsReattach = computed(() => !props.image && !!props.filename);

function pick(): void {
  if (inert.value) return;
  if (props.nativePicker) emit("pick");
  else input.value?.click();
}
function replace(): void {
  if (inert.value) return;
  if (props.nativePicker) emit("pick");
  else if (props.gallery) emit("gallery");
  else input.value?.click();
}
function onChange(event: Event): void {
  const el = event.target as HTMLInputElement;
  const file = el.files?.[0];
  el.value = "";
  if (file) emit("file", file);
}
function onDragOver(): void {
  if (!inert.value) dragOver.value = true;
}
function onDrop(event: DragEvent): void {
  dragOver.value = false;
  if (inert.value) return;
  const file = event.dataTransfer?.files?.[0];
  if (file) emit("file", file);
}
</script>

<template>
  <div
    class="image-well"
    :class="{ 'image-well--touch': touchFriendly }"
    :data-drop-target="dropTarget || undefined"
    :style="
      touchFriendly
        ? { '--image-well-touch-target': `${touchTargetSize}px` }
        : undefined
    "
  >
    <input
      ref="input"
      type="file"
      :accept="accept"
      :disabled="inert"
      class="image-well__input"
      :data-test="`${testId}-file`"
      @change="onChange"
    />

    <figure v-if="previewUrl" class="image-well__preview">
      <img :src="previewUrl" :alt="alt" :data-test="`${testId}-preview`" />
      <figcaption v-if="filename">{{ filename }}</figcaption>
    </figure>

    <div
      v-else
      class="image-well__zone"
      :class="{
        'image-well__zone--over': dragOver,
        'image-well__zone--inert': inert,
      }"
      role="button"
      :tabindex="inert ? -1 : 0"
      :aria-disabled="inert || undefined"
      :aria-required="required || undefined"
      :aria-label="alt"
      :data-test="`${testId}-well`"
      @click="pick"
      @keydown.enter.prevent="pick"
      @keydown.space.prevent="pick"
      @dragover.prevent="onDragOver"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
    >
      <template v-if="needsReattach">
        <strong class="image-well__filename">{{ filename }}</strong>
        <span>Reattach original media — drop or click to pick</span>
      </template>
      <template v-else>{{ placeholder }}</template>
    </div>

    <div
      v-if="previewUrl || gallery || needsReattach"
      class="image-well__actions"
    >
      <span
        v-if="previewUrl"
        class="image-well__action-alias"
        :data-test="gallery ? `${testId}-gallery` : undefined"
        @click="replace"
      >
        <button
          type="button"
          class="image-well__action"
          :disabled="inert"
          :data-test="`${testId}-replace`"
          @click.stop="replace"
        >
          Replace photo
        </button>
      </span>
      <button
        v-else-if="gallery"
        type="button"
        class="image-well__action image-well__action--quiet"
        :disabled="inert"
        :data-test="`${testId}-gallery`"
        @click="emit('gallery')"
      >
        {{ galleryLabel }}
      </button>
      <button
        v-if="previewUrl || needsReattach"
        type="button"
        class="image-well__action"
        :disabled="disabled"
        :aria-label="`Remove ${alt.toLowerCase()}`"
        :data-test="`${testId}-remove`"
        @click="emit('clear')"
      >
        Remove
      </button>
    </div>
  </div>
</template>

<style scoped>
.image-well {
  display: grid;
  gap: 7px;
  color: var(--mold-text);
}
.image-well__input {
  display: none;
}
.image-well__action-alias {
  display: contents;
}
.image-well__zone {
  display: grid;
  gap: 4px;
  place-items: center;
  min-height: 88px;
  padding: 12px;
  border: 1px dashed var(--mold-border, #bbb);
  border-radius: 10px;
  color: var(--mold-text-dim, #737373);
  font-size: 12px;
  line-height: 1.45;
  text-align: center;
  cursor: pointer;
  transition:
    border-color 0.1s,
    color 0.1s;
}
.image-well__zone--over {
  border-color: var(--mold-blue, #b45309);
  color: var(--mold-blue, #b45309);
}
.image-well__zone--inert {
  cursor: default;
  opacity: 0.6;
}
.image-well__zone:focus-visible {
  outline: 2px solid var(--mold-blue, #b45309);
  outline-offset: 1px;
}
.image-well__filename {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text);
}
.image-well__preview {
  display: grid;
  gap: 5px;
  margin: 0;
  max-width: 100%;
  justify-self: start;
}
.image-well__preview img {
  display: block;
  max-width: 100%;
  max-height: 160px;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 10px;
}
.image-well__preview figcaption {
  overflow: hidden;
  color: var(--mold-text-dim, #737373);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.image-well__actions {
  display: flex;
  gap: 10px;
}
.image-well__action {
  min-height: 30px;
  padding: 5px 10px;
  border: 1px solid var(--mold-border, #bbb);
  border-radius: 8px;
  background: var(--mold-bg, transparent);
  color: var(--mold-text-dim, #737373);
  font-size: 12px;
  cursor: pointer;
}
.image-well__action--quiet {
  padding: 0;
  border: 0;
  background: none;
  text-decoration: underline;
  text-underline-offset: 2px;
}
.image-well__action:hover {
  color: var(--mold-text);
}
.image-well__action:disabled {
  opacity: 0.5;
  cursor: default;
}
.image-well--touch .image-well__preview {
  width: 100%;
  justify-self: stretch;
}
.image-well--touch .image-well__preview img {
  width: 100%;
  max-height: min(55vh, 440px);
  object-fit: contain;
  background: var(--color-print-surface, #111);
}
.image-well--touch .image-well__actions {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.image-well--touch .image-well__action {
  min-height: var(--image-well-touch-target, 44px);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: 14px;
  font-weight: 700;
  text-decoration: none;
}
</style>
