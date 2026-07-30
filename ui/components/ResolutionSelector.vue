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
  width?: number;
  height?: number;
}

const props = withDefaults(
  defineProps<{
    /** Megapixel budget. */
    modelValue: number;
    /** Aspect ratio (width / height) the pixels resolve against. */
    ratio: number;
    options?: readonly ResolutionOption[];
    /** Explicit authoritative pixels for custom/source-derived resolutions.
     * When both are present they win over the selected preset projection. */
    resolvedWidth?: number;
    resolvedHeight?: number;
    /** Optional badge beside the resolved pixels, e.g. Source or Downscaled. */
    customLabel?: string | undefined;
    /** Optional explanatory status rendered below the resolved pixels. */
    status?: string | undefined;
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

const resolved = computed(() => {
  if (
    Number.isFinite(props.resolvedWidth) &&
    Number.isFinite(props.resolvedHeight) &&
    Number(props.resolvedWidth) > 0 &&
    Number(props.resolvedHeight) > 0
  ) {
    return `${props.resolvedWidth}×${props.resolvedHeight} px`;
  }
  const selected = props.options.find(
    (option) => option.mp === props.modelValue,
  );
  return selected?.width && selected.height
    ? `${selected.width}×${selected.height} px`
    : `${dimsLabel(props.modelValue, props.ratio)} px`;
});
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
    <div class="ms-res__resolved">
      <div class="ms-res__dims">{{ resolved }}</div>
      <span v-if="customLabel" class="ms-res__label">{{ customLabel }}</span>
    </div>
    <div v-if="status" class="ms-res__status" role="status">{{ status }}</div>
  </div>
</template>

<style scoped>
.ms-res__dims {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.ms-res__resolved {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 7px;
}

.ms-res__label {
  border: 1px solid var(--sel-border);
  border-radius: 999px;
  padding: 1px 6px;
  background: var(--sel-bg);
  color: var(--sel-ink);
  font-family: var(--f-mono);
  font-size: 9px;
  font-weight: 600;
  line-height: 1.4;
}

.ms-res__status {
  margin-top: 4px;
  color: var(--ink-3);
  font-size: 10px;
  line-height: 1.4;
}
</style>
