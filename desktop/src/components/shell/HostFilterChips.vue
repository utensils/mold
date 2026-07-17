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
}>();
const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const total = computed(
  () => props.allCount ?? props.chips.reduce((sum, chip) => sum + chip.count, 0),
);
const options = computed(() => [{ key: "all", label: "All", count: total.value }, ...props.chips]);
</script>

<template>
  <div
    class="border-edge flex rounded-control border bg-bath p-0.5"
    role="tablist"
    aria-label="Gallery source"
  >
    <button
      v-for="option in options"
      :key="option.key"
      type="button"
      role="tab"
      class="flex items-center gap-1 rounded-control px-2.5 py-1 text-caption transition-colors"
      :class="
        modelValue === option.key
          ? 'bg-bench font-medium text-ink shadow-sm'
          : 'text-ink-3 hover:text-ink'
      "
      :aria-selected="modelValue === option.key"
      @click="emit('update:modelValue', option.key)"
    >
      <span class="max-w-32 truncate">{{ option.label }}</span>
      <span class="data-mono text-ink-3">{{ option.count }}</span>
    </button>
  </div>
</template>
