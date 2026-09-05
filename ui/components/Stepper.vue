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
    /** Omit for a control with no product-imposed upper bound. */
    max?: number | undefined;
    /** Accessible name for the control. */
    label?: string;
    /** Optional value renderer (e.g. fade frames show "8f"). */
    format?: (value: number) => string;
    /** Allow direct numeric entry in addition to the step buttons. */
    editable?: boolean;
    /**
     * Bare variant for nesting inside another bordered control (the
     * composer's Make chip): no box of its own, 22px buttons, so it fits a
     * 28px chip instead of painting a 38px box over it.
     */
    compact?: boolean;
  }>(),
  { editable: false, compact: false },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const atMin = computed(() => props.modelValue <= props.min);
const atMax = computed(
  () => props.max !== undefined && props.modelValue >= props.max,
);
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
  if (!Number.isFinite(next)) return;
  const integer = Math.floor(next);
  const clamped = Math.min(
    props.max ?? Number.MAX_SAFE_INTEGER,
    Math.max(props.min, integer),
  );
  if (clamped !== props.modelValue) emit("update:modelValue", clamped);
}

function commitInput(event: Event) {
  const input = event.target as HTMLInputElement;
  const next = Number(input.value);
  if (input.value.trim() === "" || !Number.isFinite(next)) {
    input.value = String(props.modelValue);
    return;
  }
  set(next);
  input.value = String(
    Math.min(
      props.max ?? Number.MAX_SAFE_INTEGER,
      Math.max(props.min, Math.floor(next)),
    ),
  );
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
  // Editable inputs own their in-progress value and native arrow behavior.
  // Handling the bubbled event here would apply the last committed prop and
  // overwrite text the user has not blurred yet.
  if (props.editable) return;
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
    :class="{ 'ms-stepper--compact': compact }"
    :role="editable ? undefined : 'spinbutton'"
    :tabindex="editable ? undefined : 0"
    :aria-valuenow="editable ? undefined : modelValue"
    :aria-valuemin="editable ? undefined : min"
    :aria-valuemax="editable ? undefined : max"
    :aria-valuetext="editable ? undefined : valueText"
    :aria-label="editable ? undefined : label"
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
    <input
      v-if="editable"
      class="ms-stepper__value ms-stepper__input"
      type="number"
      inputmode="numeric"
      pattern="[0-9]*"
      :aria-label="label"
      :min="min"
      :max="max"
      :value="modelValue"
      @change="commitInput"
      @blur="commitInput"
      @keydown.enter="($event.target as HTMLInputElement).blur()"
    />
    <span v-else class="ms-stepper__value" aria-hidden="true">{{
      valueText
    }}</span>
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
  background: var(--mold-bg-deep);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  box-shadow: inset 0 1px color-mix(in srgb, var(--mold-text) 4%, transparent);
}

.ms-stepper:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-stepper__btn {
  width: 32px;
  height: 32px;
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  font-family: var(--mold-font-sans);
  font-size: 18px;
  line-height: 1;
  border-radius: var(--mold-radius-1);
  cursor: pointer;
  transition:
    color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-stepper__btn:hover:not([aria-disabled="true"]) {
  color: var(--mold-text);
  background: color-mix(in srgb, var(--mold-text) 7%, transparent);
}

.ms-stepper__btn:active:not([aria-disabled="true"]) {
  background: color-mix(in srgb, var(--mold-text) 12%, transparent);
}

.ms-stepper__btn[aria-disabled="true"] {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-stepper__btn:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-stepper__value {
  min-width: 52px;
  padding: 0 4px;
  text-align: center;
  font-family: var(--mold-font-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  white-space: nowrap;
}

.ms-stepper__input {
  width: 52px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--mold-text);
  appearance: textfield;
}

.ms-stepper__input:focus-visible {
  border-radius: var(--mold-radius-1);
  outline: 2px solid var(--mold-blue);
  outline-offset: 1px;
}

.ms-stepper__input::-webkit-inner-spin-button,
.ms-stepper__input::-webkit-outer-spin-button {
  margin: 0;
  appearance: none;
}

/* Compact: the host control draws the box; this only lays out − value +. */
.ms-stepper--compact {
  gap: 0;
  min-height: 0;
  padding: 0;
  background: transparent;
  border: 0;
  box-shadow: none;
  border-radius: 0;
}

.ms-stepper--compact .ms-stepper__btn {
  width: 22px;
  height: 22px;
  font-size: 15px;
}

.ms-stepper--compact .ms-stepper__value {
  min-width: 28px;
  font-size: var(--mold-fs-xs);
}

.ms-stepper--compact .ms-stepper__input {
  width: 28px;
}
</style>
