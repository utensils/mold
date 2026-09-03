<script setup lang="ts">
/*
 * Chip — rounded pill toggle button (spec §03). Style/filter chips with an
 * accent-tinted active state and ring. The parent owns toggle/deselect
 * semantics; the chip only reports clicks and mirrors `active` in
 * aria-pressed.
 */
const props = withDefaults(
  defineProps<{
    active?: boolean;
    disabled?: boolean;
    /** The 24px filter chip (README §04): tighter padding, micro type. */
    compact?: boolean;
  }>(),
  { active: false, disabled: false, compact: false },
);

const emit = defineEmits<{ click: [event: MouseEvent] }>();

function onClick(event: MouseEvent) {
  if (props.disabled) return;
  emit("click", event);
}
</script>

<template>
  <button
    type="button"
    class="ms-chip"
    :class="{ 'ms-chip--compact': compact }"
    :aria-pressed="active"
    :data-on="active ? 'true' : undefined"
    :disabled="disabled"
    @click="onClick"
  >
    <slot />
  </button>
</template>

<style scoped>
.ms-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--mold-border-control);
  background: transparent;
  color: var(--mold-text-2);
  padding: 6px 13px;
  border-radius: var(--mold-radius-2);
  font-family: var(--mold-font-sans);
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out),
    box-shadow var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-chip--compact {
  height: 24px;
  padding: 0 9px;
  font-size: var(--mold-fs-micro);
}

.ms-chip:hover:not([data-on="true"]):not(:disabled) {
  border-color: var(--mold-text-dim);
}

.ms-chip[data-on="true"] {
  border-color: var(--mold-blue);
  color: var(--mold-blue);
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

.ms-chip:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-chip:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
</style>
