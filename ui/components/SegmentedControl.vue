<script setup lang="ts" generic="T extends string | number">
/*
 * Segmented control — 2–4 exclusive options (spec §03). Accent-tinted active
 * segment with ring. Options may carry a sub-line (e.g. resolution "Draft").
 * Keyboard: arrow keys move the selection (roving tabindex).
 */
import { computed } from "vue";

export interface SegmentOption<V> {
  value: V;
  label: string;
  sub?: string | undefined;
}

const props = withDefaults(
  defineProps<{
    modelValue: T;
    options: readonly SegmentOption<T>[];
    /** Accessible name for the group. */
    label?: string;
    disabled?: boolean;
    /** Tighter segment padding for dense chrome (e.g. 52px header rows). */
    compact?: boolean;
    /**
     * Let segments flow onto more than one row. A size ladder can be five or
     * six pixel-labelled options, which cannot fit one row in a 340 px
     * inspector or on a phone; 2–4 option controls keep the single row.
     */
    wrap?: boolean;
  }>(),
  { disabled: false, compact: false, wrap: false },
);

const emit = defineEmits<{ "update:modelValue": [value: T] }>();

const activeIndex = computed(() =>
  props.options.findIndex(
    (o: SegmentOption<T>) => o.value === props.modelValue,
  ),
);

function pick(value: T) {
  if (props.disabled) return;
  emit("update:modelValue", value);
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
  if (option) emit("update:modelValue", option.value);
}
</script>

<template>
  <div
    class="ms-seg"
    :class="{ 'ms-seg--compact': compact, 'ms-seg--wrap': wrap }"
    role="radiogroup"
    :aria-label="label"
    :aria-disabled="disabled || undefined"
    @keydown="onKeydown"
  >
    <button
      v-for="option in options"
      :key="String(option.value)"
      type="button"
      class="ms-seg__btn"
      role="radio"
      :aria-checked="option.value === modelValue"
      :data-on="option.value === modelValue ? 'true' : undefined"
      :tabindex="option.value === modelValue ? 0 : -1"
      :disabled="disabled"
      @click="pick(option.value)"
    >
      <span class="ms-seg__label">{{ option.label }}</span>
      <span v-if="option.sub" class="ms-seg__sub">{{ option.sub }}</span>
    </button>
  </div>
</template>

<style scoped>
.ms-seg {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: var(--mold-bg-deep);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
}

.ms-seg__btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1px;
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  padding: 7px 8px;
  border-radius: var(--mold-radius-1);
  font-family: var(--mold-font-sans);
  font-size: 12px;
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-seg--compact .ms-seg__btn {
  padding: 4px 12px;
}

.ms-seg--wrap {
  flex-wrap: wrap;
}

/* A wrapped row keeps each segment at its own content width — equal 1/N
 * columns would shrink a six-entry ladder below its own pixel label. */
.ms-seg--wrap .ms-seg__btn {
  flex: 1 1 auto;
  min-width: 0;
  min-height: 44px;
  justify-content: center;
}

.ms-seg__btn:hover:not([data-on="true"]):not(:disabled) {
  color: var(--mold-text);
}

.ms-seg__btn[data-on="true"] {
  background: var(--mold-accent-tint);
  color: var(--mold-blue);
  font-weight: 700;
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

.ms-seg__btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-seg__btn:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-seg__sub {
  font-size: 9px;
  color: var(--mold-text-dim);
}
</style>
