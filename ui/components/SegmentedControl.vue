<script setup lang="ts" generic="T extends string | number">
/*
 * Segmented control — 2–4 exclusive options. Accent-tinted active
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
    /** Label and sub side by side — tabs carrying a mono count. */
    inline?: boolean;
    /**
     * How the active segment is painted. `accent` (the default) tints and
     * rings it — the treatment for a control that picks a MODE, and what
     * nav rows, the Quality rows and the mesh detail ladder use. `neutral`
     * fills it with `--mold-surface-2` in ordinary ink, for a control that
     * picks a SETTING; the mock uses it for the toolbar's output kind and
     * for Keep | Surprise me, and the accent stays one thing.
     */
    variant?: "accent" | "neutral";
  }>(),
  {
    disabled: false,
    compact: false,
    wrap: false,
    inline: false,
    variant: "accent",
  },
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
    :class="{
      'ms-seg--compact': compact,
      'ms-seg--wrap': wrap,
      'ms-seg--inline': inline,
      'ms-seg--neutral': variant === 'neutral',
    }"
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
      <span class="ms-seg__label" :data-label="option.label">{{
        option.label
      }}</span>
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
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
}

/* `nowrap` is load-bearing, not cosmetic. `flex: 1` is `flex: 1 1 0%`, so a
 * segment's floor is its MIN-CONTENT width — with wrappable text that is only
 * its longest word, and a tight row silently broke "Still picture" onto two
 * lines and doubled the control's height. Nowrap raises the floor to the whole
 * label, so the row's other flex children yield instead. */
.ms-seg__btn {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1px;
  white-space: nowrap;
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  padding: 7px 8px;
  border-radius: var(--mold-radius-1);
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

/* Compact is the TOOLBAR size: the whole control is one `--mold-ctl-md`
 * tall (border + 2px inset + segment), the height of the chips beside it. A
 * padded segment grew with the theme's type scale until, in a serif theme,
 * the control filled the 40px bar and its border sat on the bar's rule. */
.ms-seg--compact {
  padding: 2px;
}
.ms-seg--compact .ms-seg__btn {
  height: calc(var(--mold-ctl-md) - 4px - 2 * var(--mold-bw));
  padding: 0 12px;
  justify-content: center;
}

/* The selected segment is bold, and bold is wider — so picking a segment
 * used to resize the control and shift its neighbours. Each label carries a
 * hidden bold ghost of itself, so a segment is always as wide as its bold
 * self and the control never changes width on selection. */
.ms-seg__label::after {
  content: attr(data-label);
  display: block;
  height: 0;
  overflow: hidden;
  visibility: hidden;
  font-weight: 700;
  user-select: none;
  pointer-events: none;
}

.ms-seg--wrap {
  flex-wrap: wrap;
}

.ms-seg--inline .ms-seg__btn {
  flex-direction: row;
  gap: 6px;
}

.ms-seg--inline .ms-seg__sub {
  font-family: var(--mold-font-mono);
  font-size: 10.5px;
  color: inherit;
  opacity: 0.8;
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
  box-shadow: inset 0 0 0 var(--mold-bw) var(--mold-blue);
}

/* Picks a setting, not a mode: a raised fill in ordinary ink, no ring. */
.ms-seg--neutral .ms-seg__btn[data-on="true"] {
  background: var(--mold-surface-2);
  color: var(--mold-text);
  font-weight: 600;
  box-shadow: none;
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
