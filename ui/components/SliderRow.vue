<script setup lang="ts">
/*
 * Slider row — labeled range control (prototype Detail / Prompt strength).
 * Header row pairs the label with a mono accent readout; the full-width
 * range input sits below. The readout defaults to the raw value and can be
 * overridden with a formatted string (e.g. "28 steps").
 */
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    modelValue: number;
    min: number;
    max: number;
    step?: number;
    label: string;
    /** Formatted readout; defaults to String(modelValue). */
    valueLabel?: string;
    disabled?: boolean;
  }>(),
  { step: 1 },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const readout = computed(() => props.valueLabel ?? String(props.modelValue));

function onInput(event: Event) {
  emit("update:modelValue", Number((event.target as HTMLInputElement).value));
}
</script>

<template>
  <div class="ms-slider">
    <div class="ms-slider__head">
      <span class="ms-slider__label">{{ label }}</span>
      <span class="ms-slider__value">{{ readout }}</span>
    </div>
    <input
      class="ms-slider__input"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :value="modelValue"
      :aria-label="label"
      :aria-valuetext="readout"
      :disabled="disabled"
      @input="onInput"
    />
  </div>
</template>

<style scoped>
.ms-slider__head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.ms-slider__label {
  font-family: var(--f-body);
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-2);
}

.ms-slider__value {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--safelight);
}

/* Track — 4px bar; --radius-pill clamps to fully round on this height. */
.ms-slider__input {
  -webkit-appearance: none;
  appearance: none;
  display: block;
  width: 100%;
  height: 4px;
  border-radius: var(--radius-pill);
  background: var(--ce);
}

.ms-slider__input::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--safelight);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  transition: background var(--dur-quick) var(--ease);
}

.ms-slider__input::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border: 0;
  border-radius: 50%;
  background: var(--safelight);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  transition: background var(--dur-quick) var(--ease);
}

.ms-slider__input:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-slider__input:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}
</style>
