<script setup lang="ts">
/*
 * Thumbnail size — a 13px grid glyph beside a bare 74×4 track, no border and
 * no ground of its own, so it sits inside a 40px view toolbar instead of
 * filling it. The filled part of the track shows the current size; the native
 * range input rides invisibly over it and keeps keyboard, pointer, and
 * assistive-technology behaviour.
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
/** How far along its travel the size sits, as a percentage of the track. */
const fraction = computed(() => {
  const span = props.max - props.min;
  if (span <= 0) return 0;
  return Math.min(
    100,
    Math.max(0, ((props.modelValue - props.min) / span) * 100),
  );
});

function onInput(event: Event) {
  emit("update:modelValue", Number((event.target as HTMLInputElement).value));
}
</script>

<template>
  <label class="ms-thumbnail-size" :title="`Thumbnail size: ${valueText}`">
    <svg
      class="ms-thumbnail-size__glyph"
      data-test="thumbnail-size-glyph"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="4" width="7" height="7" rx="1" />
      <rect x="14" y="4" width="6" height="6" rx="1" />
    </svg>
    <span class="ms-thumbnail-size__track">
      <span
        class="ms-thumbnail-size__fill"
        data-test="thumbnail-size-fill"
        :style="{ width: `${fraction}%` }"
      />
      <span
        class="ms-thumbnail-size__knob"
        :style="{ left: `calc(${fraction}% - 7px)` }"
      />
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
    </span>
  </label>
</template>

<style scoped>
.ms-thumbnail-size {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 7px;
  color: var(--mold-text-dim);
}

.ms-thumbnail-size__glyph {
  flex: 0 0 auto;
}

.ms-thumbnail-size__track {
  position: relative;
  display: block;
  width: 74px;
  height: 4px;
  flex: 0 0 74px;
  background: var(--mold-surface);
}

.ms-thumbnail-size__fill {
  display: block;
  height: 100%;
  background: var(--mold-blue);
}

.ms-thumbnail-size__knob {
  position: absolute;
  top: -5px;
  width: 14px;
  height: 14px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
  pointer-events: none;
}

/* The real control: invisible, but the only thing that takes the pointer,
   the keyboard, and the accessibility tree. The visible track is a 4px
   hairline, so this is the whole hit area — 24px, centred on the track
   (4px tall from top 0, so -10px puts the two centres together). */
.ms-thumbnail-size__input {
  position: absolute;
  top: -10px;
  left: -7px;
  width: calc(100% + 14px);
  height: 24px;
  margin: 0;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  opacity: 0;
  cursor: ew-resize;
}

.ms-thumbnail-size__input::-webkit-slider-thumb {
  width: 14px;
  height: 24px;
  -webkit-appearance: none;
  border: 0;
  background: transparent;
}

.ms-thumbnail-size__input::-moz-range-thumb {
  width: 14px;
  height: 24px;
  border: 0;
  background: transparent;
}

.ms-thumbnail-size:focus-within .ms-thumbnail-size__track {
  outline: 2px solid var(--mold-blue);
  outline-offset: 3px;
}

.ms-thumbnail-size:hover {
  color: var(--mold-text-2);
}
</style>
