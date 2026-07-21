<script setup lang="ts">
/*
 * Switch toggle — boolean control (prototype upscale toggle). 44×26 track
 * with a 20px knob that slides 18px right when on; the on state tints the
 * track with the selection set and lights the knob with the accent.
 * Space/Enter toggle (handled on keydown with preventDefault so the native
 * button activation never double-fires).
 */
const props = withDefaults(
  defineProps<{
    modelValue: boolean;
    disabled?: boolean;
    /** Accessible name for the switch. */
    label?: string | undefined;
  }>(),
  { disabled: false },
);

const emit = defineEmits<{ "update:modelValue": [value: boolean] }>();

function toggle() {
  if (props.disabled) return;
  emit("update:modelValue", !props.modelValue);
}

function onKeydown(event: KeyboardEvent) {
  if (event.key !== " " && event.key !== "Enter") return;
  event.preventDefault();
  toggle();
}
</script>

<template>
  <button
    type="button"
    class="ms-switch"
    role="switch"
    :aria-checked="modelValue"
    :aria-label="label"
    :data-on="modelValue ? 'true' : undefined"
    :disabled="disabled"
    @click="toggle"
    @keydown="onKeydown"
  >
    <span class="ms-switch__knob" aria-hidden="true"></span>
  </button>
</template>

<style scoped>
/* --radius-pill clamps to fully round (13px) on the 26px track. */
.ms-switch {
  position: relative;
  width: 44px;
  height: 26px;
  flex: 0 0 auto;
  padding: 0;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  background: var(--bench);
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}

.ms-switch__knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--ink-3);
  transition:
    transform var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.ms-switch[data-on="true"] {
  background: var(--sel-bg);
  border-color: var(--sel-border);
}

.ms-switch[data-on="true"] .ms-switch__knob {
  transform: translateX(18px);
  background: var(--safelight);
}

.ms-switch:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ms-switch:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
