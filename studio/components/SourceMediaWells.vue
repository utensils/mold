<script setup lang="ts">
import { computed } from "vue";
import type { SourceMediaPlan } from "../lib/sourceMediaPlan";
import ImageDropWell from "./ImageDropWell.vue";

/**
 * The primary-form image conditioning block shared by desktop, web, and
 * iPhone: a source well plus an optional closing-frame well, with wording
 * that follows the model's `SourceMediaPlan`. Qwen edit's primary Target and
 * MiniMax H3 FL2VA boundaries render through the exact same wells — the
 * surface only maps slots to its own state and pickers. Attachment plans
 * without a primary well keep their surface-owned strip.
 */
export type SourceMediaSlot = "source" | "end";

export interface SourceMediaImage {
  /** Raw base64 without a data-URI prefix; empty/null renders reattach. */
  data: string | null;
  mimeType?: string | null;
  filename?: string | null;
}

const props = withDefaults(
  defineProps<{
    plan: SourceMediaPlan;
    source?: SourceMediaImage | null;
    endFrame?: SourceMediaImage | null;
    disabled?: boolean;
    touchFriendly?: boolean;
    /** One conditioning error at a time, in the server's own order. */
    error?: string | null;
    /** The engine takes only PNG/JPEG as source/keyframe conditioning. */
    accept?: string;
    /** Prefix for surface-specific test hooks while keeping one component. */
    testIdPrefix?: string;
  }>(),
  {
    source: null,
    endFrame: null,
    disabled: false,
    touchFriendly: false,
    error: null,
    accept: "image/png,image/jpeg",
    testIdPrefix: "",
  },
);

const emit = defineEmits<{
  file: [slot: SourceMediaSlot, file: File];
  gallery: [slot: SourceMediaSlot];
  clear: [slot: SourceMediaSlot];
}>();

const wells = computed(() =>
  props.plan.kind === "single" || props.plan.kind === "h3-boundaries"
    ? props.plan
    : props.plan.kind === "attachments" && props.plan.primary === "target"
      ? props.plan
      : null,
);
const h3 = computed(() => props.plan.kind === "h3-boundaries");
const target = computed(
  () => props.plan.kind === "attachments" && props.plan.primary === "target",
);
const video = computed(
  () => h3.value || (props.plan.kind === "single" && props.plan.video),
);
const required = computed(() =>
  props.plan.kind === "single"
    ? props.plan.required
    : props.plan.kind === "h3-boundaries"
      ? props.plan.requiredEndpoint === "first"
      : props.plan.kind === "attachments"
        ? props.plan.required
        : false,
);
const sourceLabel = computed(() =>
  h3.value ? "First frame" : target.value ? "Target" : "Source",
);
const sourcePlaceholder = computed(() =>
  target.value
    ? "Drop the image to edit or click to pick"
    : video.value
      ? "Drop the opening frame or click to pick"
      : "Drop an image or click to pick",
);
const sourceAlt = computed(
  () =>
    props.source?.filename ||
    (h3.value ? "First frame" : target.value ? "Edit target" : "Source image"),
);
/** H3's reviewed first-frame-only runtime refuses a closing frame; a restored
 * one stays visible so it can be removed, but never re-acquired. */
const endIncompatible = computed(
  () => h3.value && required.value && !!props.endFrame,
);
const showEndWell = computed(() => {
  if (props.plan.kind === "single") return props.plan.endFrame;
  if (props.plan.kind === "h3-boundaries")
    return props.plan.requiredEndpoint !== "first" || !!props.endFrame;
  return false;
});
const endLabel = computed(() => (h3.value ? "Last frame" : "End frame"));
const endAlt = computed(() => props.endFrame?.filename || endLabel.value);
const endHint = computed(() =>
  endIncompatible.value
    ? "This runtime accepts a first frame only — remove the closing frame."
    : "Renders a first/last-frame clip: the source opens it, this closes it.",
);
</script>

<template>
  <div
    v-if="wells"
    class="smw"
    :class="{ 'smw--touch': touchFriendly }"
    data-test="source-media-wells"
  >
    <div class="smw__head">
      <span class="smw__label">{{ sourceLabel }}</span>
      <span
        v-if="required"
        class="smw__label smw__label--required"
        data-test="source-required-badge"
        >Required</span
      >
      <span class="smw__rule" aria-hidden="true" />
    </div>
    <ImageDropWell
      :image="source?.data || null"
      :mime-type="source?.mimeType ?? null"
      :filename="source?.filename ?? null"
      :placeholder="sourcePlaceholder"
      :accept="accept"
      :disabled="disabled"
      :required="required"
      gallery
      :touch-friendly="touchFriendly"
      :alt="sourceAlt"
      :test-id="`${testIdPrefix}source`"
      @file="emit('file', 'source', $event)"
      @gallery="emit('gallery', 'source')"
      @clear="emit('clear', 'source')"
    />

    <p
      v-if="error"
      class="smw__error"
      role="alert"
      data-test="source-conditioning-error"
    >
      {{ error }}
    </p>

    <template v-if="showEndWell">
      <div class="smw__head smw__head--end">
        <span class="smw__label">{{ endLabel }}</span>
        <span class="smw__label smw__label--muted">{{
          endIncompatible ? "Incompatible" : "Optional"
        }}</span>
        <span class="smw__rule" aria-hidden="true" />
      </div>
      <ImageDropWell
        :image="endFrame?.data || null"
        :mime-type="endFrame?.mimeType ?? null"
        :filename="endFrame?.filename ?? null"
        placeholder="Drop the closing frame or click to pick"
        :accept="accept"
        :disabled="disabled"
        :pick-disabled="endIncompatible"
        gallery
        :touch-friendly="touchFriendly"
        :alt="endAlt"
        :test-id="`${testIdPrefix}end-frame`"
        @file="emit('file', 'end', $event)"
        @gallery="emit('gallery', 'end')"
        @clear="emit('clear', 'end')"
      />
      <p class="smw__hint" data-test="end-frame-hint">{{ endHint }}</p>
    </template>
  </div>
</template>

<style scoped>
.smw {
  display: grid;
  gap: 7px;
  min-width: 0;
}
.smw__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.smw__head--end {
  margin-top: 8px;
}
.smw__label {
  color: var(--mold-text-dim, #737373);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.smw__label--required {
  color: var(--mold-blue, #b45309);
}
.smw__label--muted {
  opacity: 0.8;
}
.smw__rule {
  flex: 1;
  border-top: 1px solid var(--mold-border, #ddd);
}
.smw__hint,
.smw__error {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--mold-text-dim, #737373);
}
.smw__error {
  color: var(--mold-error, #b42318);
}
</style>
