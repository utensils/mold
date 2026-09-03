<script setup lang="ts">
import { computed } from "vue";

import Tooltip from "@ui/components/Tooltip.vue";

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
  <Tooltip :text="explanation" class="model-footprint-anchor">
    <span
      class="model-footprint"
      role="meter"
      :aria-label="explanation"
      :aria-valuenow="Math.round(clamped)"
      aria-valuemin="0"
      aria-valuemax="100"
      data-test="model-footprint-bar"
    >
      <span class="model-footprint__fill" :style="{ width: `${clamped}%` }" />
    </span>
  </Tooltip>
  <span :id="descriptionId" class="sr-only" data-test="model-footprint-description">
    {{ explanation }}
  </span>
</template>

<style scoped>
/* The tooltip wrapper takes over the meter's old flex role in the row. */
.model-footprint-anchor {
  min-width: 32px;
  flex: 1 1 auto;
}

.model-footprint {
  display: block;
  width: 100%;
  height: 6px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--mold-bg-deep);
}

.model-footprint__fill {
  display: block;
  height: 100%;
  background: var(--mold-sapphire);
}
</style>
