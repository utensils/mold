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
  }>(),
  {},
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const atMin = computed(() => props.modelValue <= props.min);
const atMax = computed(() => props.modelValue >= props.max);

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
    :aria-label="label"
    @keydown="onKeydown"
  >
    <button
      type="button"
      class="ms-stepper__btn"
      tabindex="-1"
      aria-label="Decrease"
      :aria-disabled="atMin || undefined"
      @click="decrement"
    >
      −
    </button>
    <span class="ms-stepper__value" aria-hidden="true">{{ modelValue }}</span>
    <button
      type="button"
      class="ms-stepper__btn"
      tabindex="-1"
      aria-label="Increase"
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
  gap: 2px;
  padding: 2px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
}

.ms-stepper:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-stepper__btn {
  width: 30px;
  height: 30px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 18px;
  line-height: 1;
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.ms-stepper__btn:hover:not([aria-disabled="true"]) {
  color: var(--rebate);
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
  width: 26px;
  text-align: center;
  font-family: var(--f-mono);
  font-size: 13px;
}
</style>
