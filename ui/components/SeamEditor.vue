<script setup lang="ts">
/*
 * Seam editor — the shared CONTENT of the seam pill's click target
 * (popover on desktop/web, bottom sheet on iPhone; mockups 2b / 3b).
 * Three teaching rows (diagram + word + description + check) and a fade
 * length stepper that clamps to the server's fade_frames_max.
 */
import { computed } from "vue";
import SeamGlyph from "./SeamGlyph.vue";
import Stepper from "./Stepper.vue";
import {
  transitionDescription,
  transitionLabel,
  type SequenceTransition,
} from "../lib/seam";

const props = withDefaults(
  defineProps<{
    transition: SequenceTransition;
    fadeFrames: number;
    /** Motion-tail frames of the active model; 0 relabels smooth "Join". */
    motionTail: number;
    fps: number;
    /** Server cap (`chain-limits.fade_frames_max`); 32 when unknown. */
    fadeFramesMax?: number;
    /** "join · opening → clip 2" context line. */
    fromLabel?: string;
    toLabel?: string;
    /** Desktop/web: mention the ⌥ apply-to-every-seam shortcut. */
    showApplyAllHint?: boolean;
    /** iPhone: 56px rows / 44pt stepper targets. */
    large?: boolean;
  }>(),
  {
    fadeFramesMax: 32,
    showApplyAllHint: false,
    large: false,
  },
);

const emit = defineEmits<{
  "update:transition": [value: SequenceTransition];
  "update:fadeFrames": [value: number];
  /** ⌥-modified selection: apply this transition to every seam. */
  "apply-all": [value: SequenceTransition];
}>();

const OPTIONS: SequenceTransition[] = ["smooth", "cut", "fade"];

const context = computed(() =>
  props.fromLabel && props.toLabel
    ? `join · ${props.fromLabel} → ${props.toLabel}`
    : null,
);

const fadeSeconds = computed(() =>
  props.fps > 0 ? (props.fadeFrames / props.fps).toFixed(2) : null,
);

function pick(option: SequenceTransition, event: MouseEvent) {
  if (event.altKey) {
    emit("apply-all", option);
    return;
  }
  emit("update:transition", option);
}

// Roving tabindex — arrow keys move the selection, matching the ARIA radio
// pattern SegmentedControl uses.
function onOptionKeydown(event: KeyboardEvent) {
  const delta =
    event.key === "ArrowDown" || event.key === "ArrowRight"
      ? 1
      : event.key === "ArrowUp" || event.key === "ArrowLeft"
        ? -1
        : 0;
  if (delta === 0) return;
  event.preventDefault();
  const index = OPTIONS.indexOf(props.transition);
  const next = OPTIONS[(index + delta + OPTIONS.length) % OPTIONS.length];
  if (next) emit("update:transition", next);
}
</script>

<template>
  <div class="ms-seam-editor" :class="{ 'ms-seam-editor--large': large }">
    <div v-if="context" class="ms-seam-editor__context">{{ context }}</div>
    <div
      class="ms-seam-editor__options"
      role="radiogroup"
      aria-label="Transition"
    >
      <button
        v-for="option in OPTIONS"
        :key="option"
        type="button"
        class="ms-seam-editor__row"
        role="radio"
        :aria-checked="option === transition"
        :data-on="option === transition ? 'true' : undefined"
        :tabindex="option === transition ? 0 : -1"
        @click="pick(option, $event)"
        @keydown="onOptionKeydown"
      >
        <span class="ms-seam-editor__diagram" aria-hidden="true">
          <SeamGlyph class="ms-seam-editor__glyph" :transition="option" />
        </span>
        <span class="ms-seam-editor__text">
          <span class="ms-seam-editor__label">{{
            transitionLabel(option, motionTail)
          }}</span>
          <span class="ms-seam-editor__desc">{{
            transitionDescription(option, motionTail)
          }}</span>
        </span>
        <span
          v-if="option === transition"
          class="ms-seam-editor__check"
          aria-hidden="true"
          >✓</span
        >
      </button>
    </div>

    <template v-if="transition === 'fade'">
      <div class="ms-seam-editor__divider" />
      <div class="ms-seam-editor__fade-row">
        <span class="ms-seam-editor__fade-label">Fade length</span>
        <Stepper
          :model-value="fadeFrames"
          :min="1"
          :max="fadeFramesMax"
          label="Fade length in frames"
          :format="(v: number) => `${v}f`"
          @update:model-value="emit('update:fadeFrames', $event)"
        />
      </div>
      <div class="ms-seam-editor__footnote">
        <template v-if="fadeSeconds"
          >{{ fadeSeconds }}s @ {{ fps }}fps · </template
        >max {{ fadeFramesMax }}f<template v-if="showApplyAllHint">
          · ⌥ applies to every seam</template
        >
      </div>
    </template>
    <div v-else-if="showApplyAllHint" class="ms-seam-editor__footnote">
      ⌥ applies to every seam
    </div>
  </div>
</template>

<style scoped>
.ms-seam-editor {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 250px;
}

.ms-seam-editor__context {
  font-family: var(--mold-font-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
  padding: 4px 8px 6px;
}

.ms-seam-editor__options {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.ms-seam-editor--large .ms-seam-editor__options {
  gap: 8px;
}

.ms-seam-editor__row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 7px 8px;
  border: 0;
  background: transparent;
  border-radius: var(--mold-radius-1);
  cursor: pointer;
  text-align: left;
  color: var(--mold-text);
  font-family: var(--mold-font-sans);
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-seam-editor--large .ms-seam-editor__row {
  min-height: 56px;
  padding: 10px 12px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
}

.ms-seam-editor__row:hover {
  background: var(--mold-surface);
}

.ms-seam-editor__row[data-on="true"] {
  background: var(--mold-surface);
}

.ms-seam-editor--large .ms-seam-editor__row[data-on="true"] {
  border-color: var(--mold-blue);
  background: var(--mold-accent-tint);
}

.ms-seam-editor__row:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

/* The same circular glyph badge as the seam pill, so the editor teaches
   with the exact mark the rail shows. */
.ms-seam-editor__diagram {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  border: 1px solid var(--mold-border-control);
  border-radius: 50%;
  background: var(--mold-bg-deep);
  color: var(--mold-text-2);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-seam-editor--large .ms-seam-editor__diagram {
  width: 40px;
  height: 40px;
  flex-basis: 40px;
}

.ms-seam-editor__glyph {
  display: block;
  width: 14px;
  height: 14px;
  fill: currentColor;
}

.ms-seam-editor--large .ms-seam-editor__glyph {
  width: 17px;
  height: 17px;
}

.ms-seam-editor__row[data-on="true"] .ms-seam-editor__diagram {
  border-color: var(--mold-blue);
  background: color-mix(in srgb, var(--mold-blue) 12%, transparent);
  color: var(--mold-blue);
}

.ms-seam-editor__text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.ms-seam-editor__label {
  font-size: 12.5px;
}

.ms-seam-editor--large .ms-seam-editor__label {
  font-size: 16px;
}

.ms-seam-editor__row[data-on="true"] .ms-seam-editor__label {
  color: var(--mold-blue);
}

.ms-seam-editor__desc {
  font-family: var(--mold-font-mono);
  font-size: 9.5px;
  color: var(--mold-text-dim);
}

.ms-seam-editor--large .ms-seam-editor__desc {
  font-size: 10.5px;
}

.ms-seam-editor__check {
  color: var(--mold-blue);
  font-size: 12px;
  font-weight: 800;
}

.ms-seam-editor--large .ms-seam-editor__check {
  font-size: 18px;
}

.ms-seam-editor__divider {
  height: 1px;
  background: var(--mold-border);
  margin: 6px 8px;
}

.ms-seam-editor__fade-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 4px 8px 2px;
}

.ms-seam-editor__fade-label {
  font-size: 12px;
  color: var(--mold-text-2);
}

.ms-seam-editor--large .ms-seam-editor__fade-label {
  font-size: 16px;
}

.ms-seam-editor__footnote {
  padding: 2px 8px 6px;
  font-family: var(--mold-font-mono);
  font-size: 9.5px;
  color: var(--mold-text-dim);
}
</style>
