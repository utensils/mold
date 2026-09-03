<script setup lang="ts">
/*
 * Lightroom-style thumbnail size control. One ramp spanning the whole track
 * communicates dense-to-large without adding toolbar copy; the native range
 * input preserves keyboard, pointer, and assistive-technology behavior.
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
      class="ms-thumbnail-size__ramp"
      data-test="thumbnail-size-ramp"
      viewBox="0 0 114 24"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path d="M0 17L114 3L114 19L0 19Z" />
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
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: color-mix(in srgb, var(--mold-bg-deep) 82%, transparent);
  color: var(--mold-text-dim);
}

/*
 * Spans the thumb's full travel (the 7px-wide thumb centers at 10.5px and
 * 123.5px), so the ramp reads as one glyph the thumb rides along instead of
 * separate marks with dead track between them.
 */
.ms-thumbnail-size__ramp {
  position: absolute;
  top: 5px;
  left: 10px;
  right: 10px;
  height: 24px;
  fill: currentColor;
  pointer-events: none;
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
  border-radius: 999px;
  background: var(--mold-text);
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
  border-radius: 999px;
  background: var(--mold-text);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.ms-thumbnail-size__input:focus-visible {
  border-radius: var(--mold-radius-2);
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-thumbnail-size:hover {
  color: var(--mold-text-2);
}
</style>
