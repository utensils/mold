<script setup lang="ts">
/*
 * Progress bar — the thin activity/telemetry bar (prototype activity strip,
 * host detail, downloads). Width is set directly from `value` — no width
 * transition, progress tracks the reported value exactly. Tones map to the
 * fixed accent jobs: accent = actions (safelight), info = telemetry
 * (halide), plus the status trio.
 */
import { computed } from "vue";

export type ProgressBarTone =
  "accent" | "info" | "success" | "warning" | "danger";

const props = withDefaults(
  defineProps<{
    /** Progress percentage, 0–100 (clamped). */
    value: number;
    tone?: ProgressBarTone;
    /** Track height in px. */
    height?: number;
    /** Accessible name for the bar. */
    label?: string;
  }>(),
  { tone: "accent", height: 6 },
);

const clamped = computed(() => Math.min(100, Math.max(0, props.value)));
</script>

<template>
  <div
    class="ms-bar"
    role="progressbar"
    :aria-label="label"
    :aria-valuenow="Math.round(clamped)"
    aria-valuemin="0"
    aria-valuemax="100"
    :style="{ height: `${height}px` }"
  >
    <span
      class="ms-bar__fill"
      :class="`ms-bar__fill--${tone}`"
      :style="{ width: `${clamped}%` }"
    />
  </div>
</template>

<style scoped>
.ms-bar {
  background: var(--mold-bg-deep);
  /* Micro-radius below the control scale, per the prototype telemetry bars. */
  border-radius: 4px;
  overflow: hidden;
}

.ms-bar__fill {
  display: block;
  height: 100%;
}

.ms-bar__fill--accent {
  background: var(--mold-blue);
}

.ms-bar__fill--info {
  background: var(--mold-sapphire);
}

.ms-bar__fill--success {
  background: var(--mold-success);
}

.ms-bar__fill--warning {
  background: var(--mold-warning);
}

.ms-bar__fill--danger {
  background: var(--mold-error);
}
</style>
