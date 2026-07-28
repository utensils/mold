<script setup lang="ts">
/*
 * Seam editor — the shared CONTENT of the seam pill's click target
 * (popover on desktop/web, bottom sheet on iPhone; mockups 2b / 3b).
 * Three teaching rows (diagram + word + description + check) and a fade
 * length stepper that clamps to the server's fade_frames_max.
 */
import { computed } from "vue";
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
    /** Motion-tail frames of the active model; 0 relabels smooth "Join clips". */
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
          <span
            v-if="option === 'smooth'"
            class="ms-seam-editor__line-smooth"
          />
          <span v-else-if="option === 'cut'" class="ms-seam-editor__line-cut" />
          <span v-else class="ms-seam-editor__line-fade" />
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
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
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
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  text-align: left;
  color: var(--rebate);
  font-family: var(--f-body);
  transition: background var(--dur-quick) var(--ease);
}

.ms-seam-editor--large .ms-seam-editor__row {
  min-height: 56px;
  padding: 10px 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-card);
  background: var(--bath);
}

.ms-seam-editor__row:hover {
  background: var(--surface);
}

.ms-seam-editor__row[data-on="true"] {
  background: var(--surface);
}

.ms-seam-editor--large .ms-seam-editor__row[data-on="true"] {
  border-color: var(--sel-border);
  background: var(--sel-bg);
}

.ms-seam-editor__row:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-seam-editor__diagram {
  position: relative;
  display: block;
  width: 44px;
  height: 22px;
  flex: 0 0 44px;
  border-radius: 2px;
  background: var(--print);
  overflow: hidden;
}

.ms-seam-editor--large .ms-seam-editor__diagram {
  width: 56px;
  height: 30px;
  flex-basis: 56px;
}

.ms-seam-editor__line-smooth {
  position: absolute;
  top: calc(50% - 1px);
  left: 0;
  right: 0;
  height: 2px;
  background: var(--halide);
}

.ms-seam-editor__line-cut {
  position: absolute;
  top: 0;
  bottom: 0;
  left: calc(50% - 1px);
  width: 2px;
  background: var(--stop);
}

.ms-seam-editor__line-fade {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 20%,
    var(--safelight) 50%,
    transparent 80%
  );
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
  color: var(--sel-ink);
}

.ms-seam-editor__desc {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}

.ms-seam-editor--large .ms-seam-editor__desc {
  font-size: 10.5px;
}

.ms-seam-editor__check {
  color: var(--safelight);
  font-size: 12px;
  font-weight: 800;
}

.ms-seam-editor--large .ms-seam-editor__check {
  font-size: 18px;
}

.ms-seam-editor__divider {
  height: 1px;
  background: var(--edge);
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
  color: var(--ink-2);
}

.ms-seam-editor--large .ms-seam-editor__fade-label {
  font-size: 16px;
}

.ms-seam-editor__footnote {
  padding: 2px 8px 6px;
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}
</style>
