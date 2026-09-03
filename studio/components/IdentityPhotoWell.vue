<script setup lang="ts">
import { computed } from "vue";
import ImageDropWell from "./ImageDropWell.vue";
import {
  ID_IMAGE_ACCEPT,
  IDENTITY_PHOTO_HINT,
  IDENTITY_PHOTO_PLACEHOLDER,
  IDENTITY_SECTION_LABEL,
} from "../lib/identityConditioning";

/**
 * The identity (PuLID) photo well shared by every Studio surface. Wording,
 * accepted formats, and the inline refusal all come from
 * `@studio/lib/identityConditioning` so no surface restates the policy; the
 * surface owns only what a picked File means and where the gallery pick opens.
 *
 * The whole block is capability-gated by the caller — a host or checkpoint
 * without identity support must render nothing at all rather than a disabled
 * control for a feature it does not have.
 */
const props = withDefaults(
  defineProps<{
    /** Raw base64 (no data-URI prefix); null renders the empty well. */
    image?: string | null;
    mimeType?: string | null;
    /** Provenance label; with no bytes it renders the reattach state. */
    filename?: string | null;
    /** One inline reason the identity partition would be refused. */
    error?: string | null;
    /** Renders the "from gallery" action; the host opens its own picker. */
    gallery?: boolean;
    disabled?: boolean;
    touchFriendly?: boolean;
    touchTargetSize?: number;
    /** Delegates acquisition to the platform shell while retaining this UI. */
    nativePicker?: boolean;
  }>(),
  {
    image: null,
    mimeType: null,
    filename: null,
    error: null,
    gallery: false,
    disabled: false,
    touchFriendly: false,
    touchTargetSize: 44,
    nativePicker: false,
  },
);

const emit = defineEmits<{
  file: [file: File];
  gallery: [];
  clear: [];
  pick: [];
}>();

const accept = ID_IMAGE_ACCEPT;
const sectionLabel = IDENTITY_SECTION_LABEL;
const placeholder = IDENTITY_PHOTO_PLACEHOLDER;
const hint = IDENTITY_PHOTO_HINT;
const attached = computed(() => Boolean(props.image || props.filename));
</script>

<template>
  <div
    class="idw"
    :class="{ 'idw--touch': touchFriendly }"
    data-test="identity-photo-well"
  >
    <div class="idw__head">
      <span class="idw__label">{{ sectionLabel }}</span>
      <span class="idw__label idw__label--muted">Optional</span>
      <span class="idw__rule" aria-hidden="true" />
    </div>
    <ImageDropWell
      :image="image"
      :mime-type="mimeType"
      :filename="filename"
      :placeholder="placeholder"
      :accept="accept"
      :disabled="disabled"
      :gallery="gallery"
      :touch-friendly="touchFriendly"
      :touch-target-size="touchTargetSize"
      :native-picker="nativePicker"
      alt="Identity photo"
      test-id="identity"
      @file="emit('file', $event)"
      @gallery="emit('gallery')"
      @clear="emit('clear')"
      @pick="emit('pick')"
    />
    <p
      v-if="error"
      class="idw__error"
      role="alert"
      data-test="identity-conditioning-error"
    >
      {{ error }}
    </p>
    <p v-else-if="!attached" class="idw__hint" data-test="identity-hint">
      {{ hint }}
    </p>
  </div>
</template>

<style scoped>
.idw {
  display: grid;
  gap: 7px;
  min-width: 0;
}
.idw__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.idw__label {
  color: var(--mold-text-dim, #737373);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.idw__label--muted {
  opacity: 0.8;
}
.idw__rule {
  flex: 1;
  border-top: 1px solid var(--mold-border, #ddd);
}
.idw__hint,
.idw__error {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--mold-text-dim, #737373);
}
.idw__error {
  color: var(--mold-error, #b42318);
}
</style>
