<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  modelValue: string;
  options: { value: string; label: string }[];
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> beside it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();

/** A value the options do not list still shows — a select that renders
 *  blank for what the machine actually has hides the one fact that matters. */
const listed = computed(() =>
  props.options.some((option) => option.value === props.modelValue)
    ? props.options
    : [{ value: props.modelValue, label: props.modelValue }, ...props.options],
);
</script>

<template>
  <select
    class="ms-setting-field"
    :value="modelValue"
    :disabled="disabled"
    :aria-label="ariaLabel"
    @change="emit('commit', ($event.target as HTMLSelectElement).value)"
  >
    <option v-for="o in listed" :key="o.value" :value="o.value">
      {{ o.label }}
    </option>
  </select>
</template>

<style scoped>
.ms-setting-field {
  height: var(--mold-ctl-lg, 32px);
  max-width: min(224px, 100%);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-setting-field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-setting-field:disabled {
  opacity: 0.4;
}
</style>
