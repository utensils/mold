<script setup lang="ts">
/*
 * Seam pill — the resting transition control between two clips on the rail
 * a circular glyph
 * badge with the transition named in words beneath it; fade length rides
 * along in the caption. `motionTail` is REQUIRED so the zero-tail
 * "Join" label can never be skipped (the old SpliceMark defaulted it
 * and mislabeled LTX-Video seams).
 */
import { computed } from "vue";
import SeamGlyph from "./SeamGlyph.vue";
import { transitionLabel, type SequenceTransition } from "../lib/seam";

const props = withDefaults(
  defineProps<{
    transition: SequenceTransition;
    /** Motion-tail frames of the active model; 0 = LTX-Video "Join". */
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
    @contextmenu.prevent="emit('click', $event)"
  >
    <span class="ms-seam__diagram" aria-hidden="true">
      <SeamGlyph class="ms-seam__glyph" :transition="transition" />
    </span>
    <span class="ms-seam__caption">
      <span class="ms-seam__label">{{ label }}</span>
      <span v-if="fadeSuffix" class="ms-seam__frames">{{ fadeSuffix }}</span>
    </span>
  </button>
</template>

<style scoped>
.ms-seam {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  padding: 2px;
  flex: 0 0 auto;
  background: transparent;
  border: 0;
  border-radius: 8px;
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: 10px;
  line-height: 1;
  cursor: pointer;
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-seam--large {
  min-width: 44px;
  min-height: 44px;
  gap: 6px;
  padding: 4px;
  font-size: 12px;
}

.ms-seam:hover:not(:disabled) {
  color: var(--mold-text);
}

.ms-seam:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-seam:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-seam__diagram {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  flex: 0 0 auto;
  border: 1px solid var(--mold-border-control);
  border-radius: 50%;
  background: var(--mold-bg);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out),
    transform var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-seam--large .ms-seam__diagram {
  width: 38px;
  height: 38px;
}

.ms-seam__glyph {
  display: block;
  width: 14px;
  height: 14px;
  fill: currentColor;
}

.ms-seam--large .ms-seam__glyph {
  width: 16px;
  height: 16px;
}

.ms-seam:hover:not(:disabled) .ms-seam__diagram {
  border-color: var(--mold-text-dim);
  transform: scale(1.04);
}

.ms-seam[data-on="true"],
.ms-seam[data-transition="fade"] {
  color: var(--mold-blue);
}

.ms-seam[data-on="true"] .ms-seam__diagram,
.ms-seam[data-transition="fade"] .ms-seam__diagram {
  border-color: var(--mold-blue);
  background: color-mix(in srgb, var(--mold-blue) 12%, transparent);
}

.ms-seam[data-on="true"] .ms-seam__diagram {
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

/* A label that outgrows the seam's narrow column wraps centered instead of
   ellipsizing or spilling over the neighboring tiles. */
.ms-seam__caption {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 1px 3px;
  min-width: 0;
  max-width: 64px;
  text-align: center;
  line-height: 1.2;
}

.ms-seam__frames {
  flex: 0 0 auto;
}
</style>
