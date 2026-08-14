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
  },
);

const emit = defineEmits<{ file: [file: File]; gallery: []; clear: [] }>();

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
  if (!inert.value) input.value?.click();
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
  <div class="image-well" :class="{ 'image-well--touch': touchFriendly }">
    <input
      ref="input"
      type="file"
      :accept="accept"
      :disabled="inert"
      class="image-well__input"
      :data-test="`${testId}-file`"
      @change="onChange"
    />

    <div v-if="previewUrl" class="image-well__preview">
      <img :src="previewUrl" :alt="alt" />
      <button
        type="button"
        class="image-well__clear"
        :disabled="disabled"
        :aria-label="`Remove ${alt.toLowerCase()}`"
        :data-test="`${testId}-remove`"
        @click="emit('clear')"
      >
        ✕
      </button>
    </div>

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

    <div v-if="gallery || needsReattach" class="image-well__actions">
      <button
        v-if="gallery"
        type="button"
        class="image-well__gallery"
        :disabled="inert"
        :data-test="`${testId}-gallery`"
        @click="emit('gallery')"
      >
        {{ galleryLabel }}
      </button>
      <button
        v-if="needsReattach"
        type="button"
        class="image-well__gallery"
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
  color: var(--ink, currentColor);
}
.image-well__input {
  display: none;
}
.image-well__zone {
  display: grid;
  gap: 4px;
  place-items: center;
  min-height: 88px;
  padding: 12px;
  border: 1px dashed var(--edge, #bbb);
  border-radius: 10px;
  color: var(--ink-3, #737373);
  font-size: 12px;
  line-height: 1.45;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.1s, color 0.1s;
}
.image-well__zone--over {
  border-color: var(--safelight, #b45309);
  color: var(--safelight, #b45309);
}
.image-well__zone--inert {
  cursor: default;
  opacity: 0.6;
}
.image-well__zone:focus-visible {
  outline: 2px solid var(--safelight, #b45309);
  outline-offset: 1px;
}
.image-well__filename {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ink, currentColor);
}
.image-well__preview {
  position: relative;
  display: inline-block;
  max-width: 100%;
  justify-self: start;
}
.image-well__preview img {
  display: block;
  max-width: 100%;
  max-height: 160px;
  border: 1px solid var(--edge, #bbb);
  border-radius: 10px;
}
.image-well__clear {
  position: absolute;
  top: 4px;
  right: 4px;
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border: 1px solid var(--edge, #bbb);
  border-radius: 6px;
  background: var(--bench, rgba(255, 255, 255, 0.85));
  color: var(--ink-2, inherit);
  font-size: 11px;
  cursor: pointer;
}
.image-well__clear:hover {
  color: var(--stop, #b42318);
}
.image-well__actions {
  display: flex;
  gap: 12px;
}
.image-well__gallery {
  padding: 0;
  border: 0;
  background: none;
  color: var(--ink-3, #737373);
  font-size: 12px;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
}
.image-well__gallery:hover {
  color: var(--ink, currentColor);
}
.image-well__gallery:disabled,
.image-well__clear:disabled {
  opacity: 0.5;
  cursor: default;
}
.image-well--touch .image-well__gallery {
  min-height: 44px;
}
.image-well--touch .image-well__clear {
  width: 44px;
  height: 44px;
  font-size: 14px;
}
</style>
