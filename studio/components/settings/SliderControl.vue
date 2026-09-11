<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{
  modelValue: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean | undefined;
  /** Accessible name — the slider has no visible <label> beside it. */
  ariaLabel?: string | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: number): void }>();

// Track the drag locally; commit on release so a drag is one write, not fifty.
const live = ref(props.modelValue);
watch(
  () => props.modelValue,
  (value) => (live.value = value),
);
</script>

<template>
  <div class="ms-slider">
    <input
      type="range"
      :value="live"
      :min="min"
      :max="max"
      :step="step"
      :disabled="disabled"
      :aria-label="ariaLabel"
      @input="live = Number(($event.target as HTMLInputElement).value)"
      @change="emit('commit', live)"
    />
    <span class="ms-slider__value">{{ live }}</span>
  </div>
</template>

<style scoped>
.ms-slider {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-2);
}
.ms-slider input {
  width: 144px;
  accent-color: var(--mold-blue);
}
.ms-slider input:disabled {
  opacity: 0.4;
}
.ms-slider__value {
  width: 40px;
  text-align: right;
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
</style>
