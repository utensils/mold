<script setup lang="ts">
/*
 * Progress ring — SVG ring for the generate canvas (spec §09). The dash
 * offset is computed directly from `value`: no tween, no transition —
 * progress tracks the reported value exactly. The ring always sits on
 * media (the shimmering print bed, which never inverts), so the track is
 * a fixed white alpha and the label uses --on-media, not --rebate.
 */
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    /** Progress percentage, 0–100 (clamped). */
    value: number;
    /** Rendered square size in px. */
    size?: number;
    /** Show the centered percent readout. */
    showLabel?: boolean;
  }>(),
  { size: 104, showLabel: false },
);

/** Geometry is fixed in the 80×80 viewBox: r = 34. */
const CIRCUMFERENCE = 2 * Math.PI * 34;

const clamped = computed(() => Math.min(100, Math.max(0, props.value)));

const dashOffset = computed(() => CIRCUMFERENCE * (1 - clamped.value / 100));
</script>

<template>
  <div
    class="ms-ring"
    role="progressbar"
    :aria-valuenow="Math.round(clamped)"
    aria-valuemin="0"
    aria-valuemax="100"
    :style="{ width: `${size}px`, height: `${size}px` }"
  >
    <svg :width="size" :height="size" viewBox="0 0 80 80" class="ms-ring__svg">
      <circle class="ms-ring__track" cx="40" cy="40" r="34" stroke-width="5" />
      <circle
        class="ms-ring__value"
        cx="40"
        cy="40"
        r="34"
        stroke-width="5"
        stroke-linecap="round"
        :stroke-dasharray="CIRCUMFERENCE"
        :stroke-dashoffset="dashOffset"
      />
    </svg>
    <span v-if="showLabel" class="ms-ring__label"
      >{{ Math.round(clamped) }}%</span
    >
  </div>
</template>

<style scoped>
.ms-ring {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.ms-ring__svg {
  display: block;
  transform: rotate(-90deg);
}

.ms-ring__track {
  fill: none;
  /* Media-locked: the ring lives on the dark print bed in every theme. */
  stroke: rgba(255, 255, 255, 0.14);
}

.ms-ring__value {
  fill: none;
  stroke: var(--safelight);
}

.ms-ring__label {
  position: absolute;
  font-family: var(--f-mono);
  font-size: 16px;
  font-weight: 600;
  color: var(--on-media);
}
</style>
