<script setup lang="ts">
import { computed } from "vue";

export interface HostFilterChip {
  key: string;
  label: string;
  count: number;
}

const props = defineProps<{
  chips: HostFilterChip[];
  /** "all" or a chip key. */
  modelValue: string;
  /** Override when All collapses prints that exist on multiple hosts. */
  allCount?: number;
  /** Accessible name for the chip group (also reused for kind chips). */
  ariaLabel?: string;
}>();
const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const total = computed(
  () => props.allCount ?? props.chips.reduce((sum, chip) => sum + chip.count, 0),
);
const options = computed(() => [{ key: "all", label: "All", count: total.value }, ...props.chips]);
</script>

<template>
  <div
    class="flex rounded-control border border-border-control bg-bg-deep p-0.5"
    role="tablist"
    :aria-label="ariaLabel ?? 'Made on'"
  >
    <button
      v-for="option in options"
      :key="option.key"
      type="button"
      role="tab"
      class="flex items-center gap-1 rounded-control px-2.5 py-1 text-micro transition-colors"
      :class="
        modelValue === option.key
          ? 'bg-bg font-medium text-fg shadow-sm'
          : 'text-fg-dim hover:text-fg'
      "
      :aria-selected="modelValue === option.key"
      @click="emit('update:modelValue', option.key)"
    >
      <span class="max-w-32 truncate">{{ option.label }}</span>
      <span class="font-mono text-xs text-fg-dim">{{ option.count }}</span>
    </button>
  </div>
</template>
