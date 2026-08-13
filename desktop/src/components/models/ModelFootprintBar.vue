<script setup lang="ts">
import { computed } from "vue";

/**
 * Relative model-footprint meter used by every desktop model table. It is a
 * comparison within the current list, not transfer progress, so the shared
 * hover and accessibility copy makes that distinction explicit everywhere.
 */
const props = defineProps<{
  percent: number;
  sizeLabel?: string | null;
  descriptionId: string;
}>();

const clamped = computed(() => Math.min(100, Math.max(0, props.percent)));
const explanation = computed(() => {
  const size = props.sizeLabel ? `${props.sizeLabel}. ` : "";
  return `${size}Relative on-disk footprint compared with the largest model in this list. This is not download progress.`;
});
</script>

<template>
  <span
    class="model-footprint"
    role="meter"
    :aria-label="explanation"
    :aria-valuenow="Math.round(clamped)"
    aria-valuemin="0"
    aria-valuemax="100"
    :title="explanation"
    data-test="model-footprint-bar"
  >
    <span class="model-footprint__fill" :style="{ width: `${clamped}%` }" />
  </span>
  <span :id="descriptionId" class="sr-only" data-test="model-footprint-description">
    {{ explanation }}
  </span>
</template>

<style scoped>
.model-footprint {
  display: block;
  min-width: 32px;
  height: 6px;
  flex: 1 1 auto;
  overflow: hidden;
  border-radius: 999px;
  background: var(--bath);
  cursor: help;
}

.model-footprint__fill {
  display: block;
  height: 100%;
  background: var(--halide);
}
</style>
