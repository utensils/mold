<script setup lang="ts">
/*
 * A switch. Shared by every shell, so its contract is `role="switch"` plus the
 * `commit` emit — never a class name: desktop's Tailwind aliases (`bg-accent`,
 * `bg-fg-dim`) do not exist on web, and the retired ones are banned outright.
 */
const props = defineProps<{
  modelValue: boolean;
  disabled?: boolean | undefined;
  /** Accessible name — the switch has no visible <label> beside it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: boolean): void }>();

function toggle() {
  if (props.disabled) return;
  emit("commit", !props.modelValue);
}
</script>

<template>
  <button
    type="button"
    role="switch"
    class="ms-toggle"
    :class="{ 'ms-toggle--on': props.modelValue }"
    :aria-checked="props.modelValue"
    :aria-label="ariaLabel"
    :disabled="disabled"
    @click="toggle"
  >
    <span class="ms-toggle__knob" aria-hidden="true" />
  </button>
</template>

<style scoped>
.ms-toggle {
  position: relative;
  width: 36px;
  height: 20px;
  flex: none;
  border: none;
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface-2);
  cursor: pointer;
  transition: background-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-toggle--on {
  background: var(--mold-blue);
}
.ms-toggle:disabled {
  opacity: 0.4;
  cursor: default;
}
.ms-toggle__knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-text-dim);
  transition:
    left var(--mold-dur-quick) var(--mold-ease-out),
    background-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-toggle--on .ms-toggle__knob {
  left: 18px;
  background: var(--mold-on-accent);
}
</style>
