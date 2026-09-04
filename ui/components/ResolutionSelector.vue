<script setup lang="ts">
/*
 * Resolution selector — megapixel steps. A segmented control over
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
  /** Stable selection key. Defaults to `mp` for megapixel-budget callers;
   * pixel ladders pass their own id so two sizes of equal area stay
   * distinct. */
  id?: string;
}

const props = withDefaults(
  defineProps<{
    /** Selected option: a megapixel budget, or an explicit option id. */
    modelValue: number | string;
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

const emit = defineEmits<{ "update:modelValue": [value: number | string] }>();

const segments = computed(() =>
  props.options.map((o) => ({
    value: o.id ?? o.mp,
    label: o.label,
    sub: o.sub,
  })),
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
    (option) => (option.id ?? option.mp) === props.modelValue,
  );
  if (selected?.width && selected.height) {
    return `${selected.width}×${selected.height} px`;
  }
  const mp = typeof props.modelValue === "number" ? props.modelValue : 0;
  return `${dimsLabel(mp, props.ratio)} px`;
});
</script>

<template>
  <div class="ms-res">
    <SegmentedControl
      :model-value="modelValue"
      :options="segments"
      :label="label"
      :disabled="disabled"
      :wrap="segments.length > 3"
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
  font-family: var(--mold-font-mono);
  font-size: 11px;
  color: var(--mold-text-dim);
}

.ms-res__resolved {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 7px;
}

.ms-res__label {
  border: 1px solid var(--mold-blue);
  border-radius: 999px;
  padding: 1px 6px;
  background: var(--mold-accent-tint);
  color: var(--mold-blue);
  font-family: var(--mold-font-mono);
  font-size: 9px;
  font-weight: 600;
  line-height: 1.4;
}

.ms-res__status {
  margin-top: 4px;
  color: var(--mold-text-dim);
  font-size: 10px;
  line-height: 1.4;
}
</style>
