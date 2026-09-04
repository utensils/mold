<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean;
  disabled?: boolean | undefined;
  /** Accessible name — the switch has no visible <label> associated with it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: boolean): void }>();
</script>

<template>
  <button
    type="button"
    role="switch"
    :aria-checked="props.modelValue"
    :aria-label="ariaLabel"
    :disabled="disabled"
    class="rounded-control relative h-5 w-9 transition-colors duration-150 disabled:opacity-40"
    :class="props.modelValue ? 'bg-accent' : 'bg-surface-2'"
    @click="emit('commit', !props.modelValue)"
  >
    <span
      class="rounded-inner absolute top-0.5 h-4 w-4 transition-[left] duration-150"
      :class="props.modelValue ? 'bg-on-accent' : 'bg-fg-dim'"
      :style="{ left: props.modelValue ? '18px' : '2px' }"
    />
  </button>
</template>
