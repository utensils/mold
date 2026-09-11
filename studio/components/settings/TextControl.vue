<script setup lang="ts">
const props = defineProps<{
  modelValue: string;
  placeholder?: string | undefined;
  wide?: boolean | undefined;
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> beside it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();

/** Only a real change: this writes on blur, so tabbing past must be silent. */
function commit(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  if (value !== props.modelValue) emit("commit", value);
}
</script>

<template>
  <input
    class="ms-setting-field"
    :class="{ 'ms-setting-field--wide': wide }"
    type="text"
    data-selectable
    :value="modelValue"
    :placeholder="placeholder"
    :disabled="disabled"
    :aria-label="ariaLabel"
    @change="commit"
    @keydown.enter="($event.target as HTMLInputElement).blur()"
  />
</template>

<style scoped>
.ms-setting-field {
  width: 176px;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-setting-field--wide {
  width: 288px;
}
.ms-setting-field::placeholder {
  color: var(--mold-text-dim);
}
.ms-setting-field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-setting-field:disabled {
  opacity: 0.4;
}
</style>
