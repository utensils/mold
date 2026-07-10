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
    class="relative h-5 w-9 rounded-full transition-colors duration-150 disabled:opacity-40"
    :class="
      props.modelValue ? 'bg-safelight' : 'bg-[color-mix(in_srgb,var(--rebate)_18%,transparent)]'
    "
    @click="emit('commit', !props.modelValue)"
  >
    <span
      class="absolute top-0.5 h-4 w-4 rounded-full bg-rebate transition-[left] duration-150"
      :style="{ left: props.modelValue ? '18px' : '2px' }"
    />
  </button>
</template>
