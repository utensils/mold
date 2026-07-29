<script setup lang="ts">
/*
 * Stepper — discrete count control (prototype Batch). Bordered rounded row
 * with − / + buttons flanking a mono value. The wrapper is the spinbutton
 * (single tab stop); ArrowUp/ArrowDown adjust, buttons soft-disable at the
 * bounds via aria-disabled so layout never shifts.
 */
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    modelValue: number;
    min: number;
    max: number;
    /** Accessible name for the control. */
    label?: string;
    /** Optional value renderer (e.g. fade frames show "8f"). */
    format?: (value: number) => string;
  }>(),
  {},
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const atMin = computed(() => props.modelValue <= props.min);
const atMax = computed(() => props.modelValue >= props.max);
const valueText = computed(() =>
  props.format ? props.format(props.modelValue) : String(props.modelValue),
);
const decreaseLabel = computed(() =>
  props.label ? `Decrease ${props.label}` : "Decrease",
);
const increaseLabel = computed(() =>
  props.label ? `Increase ${props.label}` : "Increase",
);

function set(next: number) {
  const clamped = Math.min(props.max, Math.max(props.min, next));
  if (clamped !== props.modelValue) emit("update:modelValue", clamped);
}

function decrement() {
  if (atMin.value) return;
  set(props.modelValue - 1);
}

function increment() {
  if (atMax.value) return;
  set(props.modelValue + 1);
}

function onKeydown(event: KeyboardEvent) {
  const delta =
    event.key === "ArrowUp" ? 1 : event.key === "ArrowDown" ? -1 : 0;
  if (delta === 0) return;
  event.preventDefault();
  set(props.modelValue + delta);
}
</script>

<template>
  <div
    class="ms-stepper"
    role="spinbutton"
    tabindex="0"
    :aria-valuenow="modelValue"
    :aria-valuemin="min"
    :aria-valuemax="max"
    :aria-valuetext="valueText"
    :aria-label="label"
    @keydown="onKeydown"
  >
    <button
      type="button"
      class="ms-stepper__btn"
      tabindex="-1"
      :aria-label="decreaseLabel"
      :aria-disabled="atMin || undefined"
      @click="decrement"
    >
      −
    </button>
    <span class="ms-stepper__value" aria-hidden="true">{{ valueText }}</span>
    <button
      type="button"
      class="ms-stepper__btn"
      tabindex="-1"
      :aria-label="increaseLabel"
      :aria-disabled="atMax || undefined"
      @click="increment"
    >
      +
    </button>
  </div>
</template>

<style scoped>
.ms-stepper {
  display: inline-flex;
  align-items: center;
  gap: 1px;
  min-height: 38px;
  padding: 2px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  box-shadow: inset 0 1px color-mix(in srgb, var(--rebate) 4%, transparent);
}

.ms-stepper:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-stepper__btn {
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 18px;
  line-height: 1;
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  transition:
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.ms-stepper__btn:hover:not([aria-disabled="true"]) {
  color: var(--rebate);
  background: color-mix(in srgb, var(--rebate) 7%, transparent);
}

.ms-stepper__btn:active:not([aria-disabled="true"]) {
  background: color-mix(in srgb, var(--rebate) 12%, transparent);
}

.ms-stepper__btn[aria-disabled="true"] {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-stepper__btn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-stepper__value {
  min-width: 52px;
  padding: 0 4px;
  text-align: center;
  font-family: var(--f-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  white-space: nowrap;
}
</style>
