<script setup lang="ts">
/*
 * Slider row — a labelled range control. The head pairs the plain word with a
 * mono readout, the 4px track sits below it, and an optional `low`/`high` pair
 * closes the group with the two ends named in words. The readout defaults to
 * the raw value and can be overridden with a formatted string.
 */
import { computed, ref, useId } from "vue";

interface SliderMark {
  value: number;
  label: string;
  title?: string;
}

const props = withDefaults(
  defineProps<{
    modelValue: number;
    min: number;
    max: number;
    step?: number;
    label: string;
    /** Formatted readout; defaults to String(modelValue). */
    valueLabel?: string;
    ariaValueText?: string;
    /** The two ends of the range in plain words, e.g. Loose → Literal. */
    low?: string;
    high?: string;
    disabled?: boolean;
    marks?: readonly SliderMark[];
    /** Fraction of the full range captured by a mark during pointer drag. */
    snapThresholdRatio?: number;
  }>(),
  { step: 1 },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const readout = computed(() => props.valueLabel ?? String(props.modelValue));
const datalistId = `ms-slider-marks-${useId()}`;
const positionedMarks = computed(() => {
  const span = props.max - props.min;
  if (span <= 0) return [];
  return (props.marks ?? [])
    .filter((mark) => mark.value >= props.min && mark.value <= props.max)
    .map((mark) => ({
      ...mark,
      left: `${((mark.value - props.min) / span) * 100}%`,
    }));
});

const pointerDragging = ref(false);
const pointerValue = ref<number | null>(null);

function snappedValue(value: number): number {
  if (positionedMarks.value.length <= 1) return value;
  const snapDistance = Math.max(
    props.step,
    (props.max - props.min) * (props.snapThresholdRatio ?? 0.015),
  );
  const nearest = positionedMarks.value.reduce<
    (typeof positionedMarks.value)[number] | null
  >((best, mark) => {
    if (Math.abs(mark.value - value) > snapDistance) return best;
    if (!best || Math.abs(mark.value - value) < Math.abs(best.value - value)) {
      return mark;
    }
    return best;
  }, null);
  return nearest?.value ?? value;
}

function onInput(event: Event) {
  const value = Number((event.target as HTMLInputElement).value);
  if (pointerDragging.value) pointerValue.value = value;
  emit("update:modelValue", value);
}

function onPointerDown() {
  pointerDragging.value = true;
  pointerValue.value = null;
}

function onPointerUp(event: PointerEvent) {
  finishPointerDrag(Number((event.target as HTMLInputElement).value));
}

function finishPointerDrag(fallbackValue: number) {
  if (!pointerDragging.value) return;
  pointerDragging.value = false;
  const value = pointerValue.value ?? fallbackValue;
  pointerValue.value = null;
  const snapped = snappedValue(value);
  if (snapped !== value) emit("update:modelValue", snapped);
}

// Mobile range controls can commit with `change` before WebKit delivers the
// final pointerup to the input. Finish the same drag here so touch snapping is
// not dependent on that browser-specific event order. Keyboard changes do not
// set pointerDragging and therefore retain exact frame-grid movement.
function onChange(event: Event) {
  finishPointerDrag(Number((event.target as HTMLInputElement).value));
}

function onPointerCancel() {
  pointerDragging.value = false;
  pointerValue.value = null;
}
</script>

<template>
  <div class="ms-slider">
    <div class="ms-slider__head">
      <span class="ms-slider__label">{{ label }}</span>
      <span class="ms-slider__value">{{ readout }}</span>
    </div>
    <div
      class="ms-slider__track"
      :class="{ 'ms-slider__track--marked': positionedMarks.length > 1 }"
    >
      <div
        v-if="positionedMarks.length > 1"
        class="ms-slider__marks"
        aria-hidden="true"
      >
        <span
          v-for="mark in positionedMarks"
          :key="mark.value"
          class="ms-slider__mark"
          :style="{ left: mark.left }"
          :title="mark.title"
        >
          <i />
          <b>{{ mark.label }}</b>
        </span>
      </div>
      <input
        class="ms-slider__input"
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        :list="positionedMarks.length > 1 ? datalistId : undefined"
        :aria-label="label"
        :aria-valuetext="ariaValueText ?? readout"
        :disabled="disabled"
        @pointerdown="onPointerDown"
        @pointerup="onPointerUp"
        @pointercancel="onPointerCancel"
        @input="onInput"
        @change="onChange"
      />
      <datalist v-if="positionedMarks.length > 1" :id="datalistId">
        <option
          v-for="mark in positionedMarks"
          :key="mark.value"
          :value="mark.value"
        />
      </datalist>
    </div>
    <div v-if="low || high" class="ms-slider__ends">
      <span>{{ low }}</span>
      <span>{{ high }}</span>
    </div>
  </div>
</template>

<style scoped>
.ms-slider__head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.ms-slider__label {
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-xs);
  font-weight: 600;
  color: var(--mold-text-2);
}

.ms-slider__value {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-slider__ends {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

/* A square 4px track: the theme's radii start at the thumb. */
.ms-slider__input {
  -webkit-appearance: none;
  appearance: none;
  display: block;
  width: 100%;
  height: 4px;
  background: var(--mold-surface);
}

.ms-slider__track--marked .ms-slider__input {
  grid-row: 2;
}

.ms-slider__input::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-slider__input::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border: 0;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text);
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-slider__input:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-slider__input:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.ms-slider__track {
  position: relative;
}

.ms-slider__track--marked {
  display: grid;
  grid-template-rows: 20px auto;
}

.ms-slider__marks {
  grid-row: 1;
  position: relative;
  margin: 0 8px;
  height: 20px;
  pointer-events: none;
}

.ms-slider__mark {
  position: absolute;
  top: 0;
  display: grid;
  justify-items: center;
  transform: translateX(-50%);
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
}

.ms-slider__mark i {
  order: 2;
  width: 1px;
  height: 6px;
  margin-top: 4px;
  background: currentColor;
}

.ms-slider__mark b {
  order: 1;
  font-size: 8px; /* literal: tick captions ride a 4px track and collide at micro */
  font-weight: 500;
  line-height: 1;
}
</style>
