<script setup lang="ts">
/*
 * Seam pill — the resting transition control between two clips on the rail
 * (spec §06, mockup 2a "text first"). Names the transition in words with a
 * mini diagram as reinforcement; fade length rides along in the pill.
 * `motionTail` is REQUIRED so the zero-tail "Join clips" label can never be
 * skipped (the old SpliceMark defaulted it and mislabeled LTX-Video seams).
 */
import { computed } from "vue";
import { transitionLabel, type SequenceTransition } from "../lib/seam";

const props = withDefaults(
  defineProps<{
    transition: SequenceTransition;
    /** Motion-tail frames of the active model; 0 = LTX-Video "Join clips". */
    motionTail: number;
    /** Shown as `· Nf` when the transition is fade. */
    fadeFrames?: number;
    /** Marks the seam whose editor popover/sheet is open. */
    active?: boolean;
    disabled?: boolean;
    /** iPhone: bump the hit target to the 44pt floor. */
    large?: boolean;
  }>(),
  { active: false, disabled: false, large: false },
);

const emit = defineEmits<{ click: [event: MouseEvent] }>();

const label = computed(() =>
  transitionLabel(props.transition, props.motionTail),
);
const fadeSuffix = computed(() =>
  props.transition === "fade" && props.fadeFrames != null
    ? `${props.fadeFrames}f`
    : null,
);
</script>

<template>
  <button
    type="button"
    class="ms-seam"
    :class="{ 'ms-seam--large': large }"
    :data-on="active ? 'true' : undefined"
    :data-transition="transition"
    :disabled="disabled"
    :aria-label="`Transition: ${label}${fadeSuffix ? ` ${fadeSuffix}` : ''}`"
    @click="emit('click', $event)"
  >
    <span class="ms-seam__diagram" aria-hidden="true">
      <span v-if="transition === 'smooth'" class="ms-seam__line-smooth" />
      <span v-else-if="transition === 'cut'" class="ms-seam__line-cut" />
      <span v-else class="ms-seam__line-fade" />
    </span>
    <span class="ms-seam__label">{{ label }}</span>
    <span v-if="fadeSuffix" class="ms-seam__frames">{{ fadeSuffix }}</span>
    <svg
      class="ms-seam__chevron"
      viewBox="0 0 24 24"
      width="11"
      height="11"
      fill="none"
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      aria-hidden="true"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  </button>
</template>

<style scoped>
.ms-seam {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 9px;
  flex: 0 0 auto;
  white-space: nowrap;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 10px;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.ms-seam--large {
  min-height: 44px;
  padding: 0 14px;
  font-size: 12px;
}

.ms-seam:hover:not(:disabled) {
  color: var(--rebate);
  border-color: var(--ink-3);
}

.ms-seam[data-on="true"] {
  border-color: var(--sel-border);
  background: var(--sel-bg);
  color: var(--sel-ink);
  box-shadow: var(--sel-ring);
}

.ms-seam:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-seam:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-seam__diagram {
  position: relative;
  display: block;
  width: 22px;
  height: 12px;
  border-radius: 2px;
  background: var(--print);
  overflow: hidden;
  flex: 0 0 auto;
}

.ms-seam--large .ms-seam__diagram {
  width: 26px;
  height: 14px;
}

.ms-seam__line-smooth {
  position: absolute;
  top: 5px;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--halide);
}

.ms-seam--large .ms-seam__line-smooth {
  top: 6px;
}

.ms-seam__line-cut {
  position: absolute;
  top: 0;
  bottom: 0;
  left: calc(50% - 1px);
  width: 2px;
  background: var(--stop);
}

.ms-seam__line-fade {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 20%,
    var(--safelight) 50%,
    transparent 80%
  );
}

.ms-seam__frames {
  color: var(--ink-3);
}

.ms-seam[data-on="true"] .ms-seam__frames {
  color: var(--sel-ink);
}

.ms-seam__chevron {
  color: var(--ink-3);
}
</style>
