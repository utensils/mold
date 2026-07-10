<script setup lang="ts">
defineProps<{
  modelValue: string;
  options: { value: string; label: string }[];
  disabled?: boolean | undefined;
  /** Accessible name — the control has no visible <label> associated with it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();
</script>

<template>
  <select
    :value="modelValue"
    :disabled="disabled"
    :aria-label="ariaLabel"
    class="border-edge h-7 max-w-56 rounded-control border bg-bath px-2 text-body text-ink disabled:opacity-40"
    @change="emit('commit', ($event.target as HTMLSelectElement).value)"
  >
    <option v-for="o in options" :key="o.value" :value="o.value">{{ o.label }}</option>
  </select>
</template>
