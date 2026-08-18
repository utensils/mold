<script setup lang="ts">
/*
 * Shape picker — aspect-ratio chips (spec §03). Each option is a chip with a
 * proportional swatch (the swatch IS the ratio, never text) above a mono
 * label. Keyboard: arrow keys move the selection (roving tabindex).
 */
import { computed } from "vue";
import { ASPECTS, swatchDims, type AspectOption } from "../lib/resolution";

const props = withDefaults(
  defineProps<{
    modelValue: string;
    options?: readonly AspectOption[];
    /** Accessible name for the group. */
    label?: string;
    disabled?: boolean;
    /** The active chip is the CLOSEST match to a custom size from Advanced,
     * not an exact preset — rendered with an "≈" prefix. */
    approximate?: boolean;
  }>(),
  { options: () => ASPECTS, disabled: false, approximate: false },
);

const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const activeIndex = computed(() =>
  props.options.findIndex((o) => o.id === props.modelValue),
);

function swatchStyle(ratio: number) {
  const { width, height } = swatchDims(ratio);
  return { width: `${width}px`, height: `${height}px` };
}

function pick(id: string) {
  if (props.disabled) return;
  emit("update:modelValue", id);
}

function onKeydown(event: KeyboardEvent) {
  if (props.disabled) return;
  const delta =
    event.key === "ArrowRight" || event.key === "ArrowDown"
      ? 1
      : event.key === "ArrowLeft" || event.key === "ArrowUp"
        ? -1
        : 0;
  if (delta === 0) return;
  event.preventDefault();
  const count = props.options.length;
  const next = (activeIndex.value + delta + count) % count;
  const option = props.options[next];
  if (option) emit("update:modelValue", option.id);
}
</script>

<template>
  <div
    class="ms-shape"
    role="radiogroup"
    :aria-label="label"
    :aria-disabled="disabled || undefined"
    @keydown="onKeydown"
  >
    <button
      v-for="option in options"
      :key="option.id"
      type="button"
      class="ms-shape__btn"
      role="radio"
      :aria-checked="option.id === modelValue"
      :data-shape="option.id"
      :data-on="option.id === modelValue ? 'true' : undefined"
      :data-approximate="
        option.id === modelValue && approximate ? 'true' : undefined
      "
      :tabindex="option.id === modelValue ? 0 : -1"
      :disabled="disabled"
      :title="
        option.id === modelValue && approximate
          ? 'Closest match to your custom size (set in Advanced)'
          : undefined
      "
      :aria-label="
        option.id === modelValue && approximate
          ? `${option.label} — closest match to your custom size (set in Advanced)`
          : undefined
      "
      @click="pick(option.id)"
    >
      <span class="ms-shape__well" aria-hidden="true">
        <span class="ms-shape__swatch" :style="swatchStyle(option.ratio)" />
      </span>
      <span class="ms-shape__label">
        {{ option.id === modelValue && approximate ? "≈" : ""
        }}{{ option.label }}
      </span>
    </button>
  </div>
</template>

<style scoped>
.ms-shape {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
}

.ms-shape__btn {
  width: 52px;
  height: 60px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}

.ms-shape__btn:hover:not([data-on="true"]):not(:disabled) {
  color: var(--rebate);
}

.ms-shape__btn[data-on="true"] {
  border-color: var(--sel-border);
  color: var(--sel-ink);
  background: var(--sel-bg);
  box-shadow: var(--sel-ring);
}

.ms-shape__btn[data-on="true"] .ms-shape__swatch {
  background: var(--sel-fill);
}

.ms-shape__btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-shape__btn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-shape__well {
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ms-shape__swatch {
  display: block;
  border: 1.5px solid currentColor;
  border-radius: 2px;
  transition: background var(--dur-quick) var(--ease);
}

.ms-shape__label {
  font-family: var(--f-mono);
  font-size: 10px;
}
</style>
