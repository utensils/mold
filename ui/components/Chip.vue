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
  }>(),
  { active: false, disabled: false },
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
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 13px;
  border-radius: var(--radius-pill);
  font-family: var(--f-body);
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}

.ms-chip:hover:not([data-on="true"]):not(:disabled) {
  border-color: var(--ink-3);
}

.ms-chip[data-on="true"] {
  border-color: var(--sel-border);
  color: var(--sel-ink);
  background: var(--sel-bg);
  box-shadow: var(--sel-ring);
}

.ms-chip:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-chip:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
