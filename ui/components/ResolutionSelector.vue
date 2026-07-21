<script setup lang="ts">
/*
 * Resolution selector — megapixel steps (spec §03). A segmented control over
 * the MP budget plus a mono resolved-pixels line derived from the current
 * shape's aspect ratio, so users always see the real output dimensions.
 */
import { computed } from "vue";
import SegmentedControl from "./SegmentedControl.vue";
import { RESOLUTIONS, dimsLabel } from "../lib/resolution";

export interface ResolutionOption {
  mp: number;
  label: string;
  sub?: string;
}

const props = withDefaults(
  defineProps<{
    /** Megapixel budget. */
    modelValue: number;
    /** Aspect ratio (width / height) the pixels resolve against. */
    ratio: number;
    options?: readonly ResolutionOption[];
    /** Accessible name for the group. */
    label?: string;
    disabled?: boolean;
  }>(),
  { options: () => RESOLUTIONS, label: "Resolution", disabled: false },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const segments = computed(() =>
  props.options.map((o) => ({ value: o.mp, label: o.label, sub: o.sub })),
);

const resolved = computed(
  () => `${dimsLabel(props.modelValue, props.ratio)} px`,
);
</script>

<template>
  <div class="ms-res">
    <SegmentedControl
      :model-value="modelValue"
      :options="segments"
      :label="label"
      :disabled="disabled"
      @update:model-value="emit('update:modelValue', $event)"
    />
    <div class="ms-res__dims">{{ resolved }}</div>
  </div>
</template>

<style scoped>
.ms-res__dims {
  margin-top: 7px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
</style>
