<script setup lang="ts">
/*
 * Lightroom-style thumbnail size control. The opposing wedges communicate
 * dense-to-large without adding toolbar copy; the native range input preserves
 * keyboard, pointer, and assistive-technology behavior.
 */
import { computed } from "vue";

const props = defineProps<{
  modelValue: number;
  min: number;
  max: number;
  step: number;
}>();

const emit = defineEmits<{ "update:modelValue": [value: number] }>();
const valueText = computed(() => `${props.modelValue} px`);

function onInput(event: Event) {
  emit("update:modelValue", Number((event.target as HTMLInputElement).value));
}
</script>

<template>
  <label class="ms-thumbnail-size" :title="`Thumbnail size: ${valueText}`">
    <svg
      class="ms-thumbnail-size__wedge ms-thumbnail-size__wedge--small"
      data-test="thumbnail-size-small"
      viewBox="0 0 42 24"
      aria-hidden="true"
    >
      <path d="M2 17 40 11v8H2Z" />
    </svg>
    <svg
      class="ms-thumbnail-size__wedge ms-thumbnail-size__wedge--large"
      data-test="thumbnail-size-large"
      viewBox="0 0 48 24"
      aria-hidden="true"
    >
      <path d="M2 10 46 3v16H2Z" />
    </svg>
    <input
      class="ms-thumbnail-size__input"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :value="modelValue"
      aria-label="Thumbnail size"
      :aria-valuetext="valueText"
      @input="onInput"
    />
  </label>
</template>

<style scoped>
.ms-thumbnail-size {
  position: relative;
  display: inline-flex;
  width: 136px;
  height: 34px;
  flex: 0 0 136px;
  align-items: center;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: color-mix(in srgb, var(--bath) 82%, transparent);
  color: var(--ink-3);
}

.ms-thumbnail-size__wedge {
  position: absolute;
  top: 5px;
  height: 24px;
  fill: currentColor;
  pointer-events: none;
}

.ms-thumbnail-size__wedge--small {
  left: 10px;
  width: 42px;
}

.ms-thumbnail-size__wedge--large {
  right: 9px;
  width: 48px;
}

.ms-thumbnail-size__input {
  position: absolute;
  inset: 0 7px;
  width: calc(100% - 14px);
  height: 34px;
  margin: 0;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  cursor: ew-resize;
}

.ms-thumbnail-size__input::-webkit-slider-runnable-track {
  height: 100%;
  background: transparent;
}

.ms-thumbnail-size__input::-webkit-slider-thumb {
  width: 7px;
  height: 28px;
  margin-top: 3px;
  -webkit-appearance: none;
  border: 0;
  border-radius: var(--radius-pill);
  background: var(--rebate);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.ms-thumbnail-size__input::-moz-range-track {
  height: 100%;
  border: 0;
  background: transparent;
}

.ms-thumbnail-size__input::-moz-range-thumb {
  width: 7px;
  height: 28px;
  border: 0;
  border-radius: var(--radius-pill);
  background: var(--rebate);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.ms-thumbnail-size__input:focus-visible {
  border-radius: var(--radius-control);
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-thumbnail-size:hover {
  color: var(--ink-2);
}
</style>
