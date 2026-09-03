<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{
  modelValue: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean | undefined;
  /** Accessible name — the slider has no visible <label> associated with it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: number): void }>();

// Track the drag locally; commit on release so we don't spam the config API.
const live = ref(props.modelValue);
watch(
  () => props.modelValue,
  (v) => (live.value = v),
);
</script>

<template>
  <div class="flex items-center gap-2">
    <input
      type="range"
      :value="live"
      :min="min"
      :max="max"
      :step="step"
      :disabled="disabled"
      :aria-label="ariaLabel"
      class="w-36 accent-accent disabled:opacity-40"
      @input="live = Number(($event.target as HTMLInputElement).value)"
      @change="emit('commit', live)"
    />
    <span class="font-mono text-xs w-10 text-right text-micro text-fg-2">{{ live }}</span>
  </div>
</template>
